from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v01 as engine
from . import fact_sheet_text_composer_v03 as phase52

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v03.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.4"
SCHEMA_VERSION = "fact-sheet-text-composer-0.4"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"

GRAMMAR_PROFILES: dict[str, dict[str, str]] = {
    "feminine": {
        "subject_pronoun": "she",
        "object_pronoun": "her",
        "possessive_pronoun": "her",
        "reflexive_pronoun": "herself",
    },
    "masculine": {
        "subject_pronoun": "he",
        "object_pronoun": "him",
        "possessive_pronoun": "his",
        "reflexive_pronoun": "himself",
    },
    "neutral": {
        "subject_pronoun": "they",
        "object_pronoun": "them",
        "possessive_pronoun": "their",
        "reflexive_pronoun": "themself",
    },
}

# Re-export the Phase-5.2 semantic projections.  The shared torso helper now
# includes signed turn_direction, while Phase-5.2 also preserves the validated
# gaze-caption-semantics projection from 4B.3.
_torso_fact = phase52._torso_fact
_gaze_fact = phase52._gaze_fact

_BASE_PROJECTION = engine._projection
_BASE_CAPTION_AUDIT = engine._caption_audit


def _grammar_profile(subject_class: str | None) -> tuple[str | None, dict[str, str] | None]:
    value = str(subject_class or "").strip().lower()
    if value in {"woman", "female", "feminine"}:
        return "feminine", GRAMMAR_PROFILES["feminine"]
    if value in {"man", "male", "masculine"}:
        return "masculine", GRAMMAR_PROFILES["masculine"]
    if value in {"person", "neutral", "nonbinary", "non-binary"}:
        return "neutral", GRAMMAR_PROFILES["neutral"]
    return None, None


def _projection(
    sheet: dict[str, Any],
    *,
    trigger_token: str | None = None,
    subject_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    projection, audit = _BASE_PROJECTION(
        sheet,
        trigger_token=trigger_token,
        subject_class=subject_class,
    )
    subject = projection.get("subject") if isinstance(projection.get("subject"), dict) else {}
    trigger = engine._clean(subject.get("trigger_token"))
    profile_name, profile = _grammar_profile(subject_class)
    if trigger:
        subject["primary_reference_policy"] = "trigger_as_first_grammatical_subject_then_pronouns"
        subject["trigger_remention_policy"] = (
            "normally_use_pronouns; repeat the exact trigger or its possessive form only when needed "
            "to keep a body part, possession, action, or relationship unambiguously attached to the primary subject"
        )
        if profile_name and profile:
            subject["grammar_profile"] = profile_name
            subject.update(profile)
    projection["subject"] = subject
    audit["trigger_subject_binding_projected"] = bool(trigger)
    audit["subject_grammar_profile"] = profile_name
    audit["trigger_remention_for_attachment_allowed"] = bool(trigger)
    return projection, audit


def _laterality_pairs(text: str) -> set[tuple[str, str]]:
    return {
        (match.group(1).lower(), match.group(2).lower())
        for match in engine.ANATOMICAL_LATERALITY_RE.finditer(str(text or ""))
    }


def _authorized_laterality_pairs(projection: dict[str, Any]) -> set[tuple[str, str]]:
    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    allowed: set[tuple[str, str]] = set()
    for value in configuration:
        if isinstance(value, str):
            allowed.update(_laterality_pairs(value))
    return allowed


def _trigger_pattern(trigger: str, *, ignore_case: bool = False) -> re.Pattern[str]:
    flags = re.I if ignore_case else 0
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(trigger)}(?![A-Za-z0-9_])", flags)


def _apply_trigger_binding_audit(caption: str, projection: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    subject = projection.get("subject") if isinstance(projection.get("subject"), dict) else {}
    trigger = engine._clean(subject.get("trigger_token"))
    if not trigger:
        return audit

    violations = list(audit.get("violations") or [])
    warnings = list(audit.get("warnings") or [])
    exact_matches = list(_trigger_pattern(trigger).finditer(caption))
    ci_matches = list(_trigger_pattern(trigger, ignore_case=True).finditer(caption))
    starts_with_trigger = bool(re.match(rf"^{re.escape(trigger)}(?=$|[^A-Za-z0-9_])", caption))

    if not exact_matches:
        if ci_matches:
            violations.append("primary_trigger_case_changed")
        else:
            violations.append("primary_trigger_missing")
    if not starts_with_trigger:
        violations.append("primary_trigger_not_first")

    generic_after_trigger = re.match(
        rf"^{re.escape(trigger)}\s*[,;:\-]\s*(?:a|the)\s+(?:woman|man|person|subject)\b",
        caption,
        re.I,
    )
    if generic_after_trigger:
        violations.append("generic_primary_subject_after_trigger")

    occurrence_count = len(exact_matches)
    if occurrence_count > 3:
        warnings.append(f"frequent_trigger_remention:{occurrence_count}")

    audit["violations"] = sorted(set(violations))
    audit["warnings"] = sorted(set(warnings))
    audit["trigger_binding"] = {
        "trigger": trigger,
        "exact_occurrences": occurrence_count,
        "starts_with_trigger": starts_with_trigger,
        "grammar_profile": subject.get("grammar_profile"),
        "expected_subject_pronoun": subject.get("subject_pronoun"),
        "expected_possessive_pronoun": subject.get("possessive_pronoun"),
        "trigger_remention_allowed": True,
    }
    return audit


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    """Keep Phase-5 audits while adding specialist laterality and trigger binding.

    Earlier composer versions correctly rejected every anatomical left/right
    phrase because Qwen did not own laterality.  Phase 4B.4 now projects only
    DWPose-bound laterality into body.configuration.  Permit only the exact
    side/body-part pairs already present there; any new side label remains a
    hard violation.

    Trigger binding follows the useful part of Fizgig Custom 1: the exact
    trigger must open the caption as the grammatical subject.  Unlike the old
    single-trigger rule, later exact/possessive rementions are allowed when
    they help keep body parts or possessions attached to the primary subject.
    """
    audit = _BASE_CAPTION_AUDIT(caption, projection)
    violations = list(audit.get("violations") or [])

    if "unauthorized_anatomical_laterality" in violations:
        used = _laterality_pairs(caption)
        allowed = _authorized_laterality_pairs(projection)
        if used and used.issubset(allowed):
            violations = [v for v in violations if v != "unauthorized_anatomical_laterality"]
            audit["violations"] = sorted(set(violations))
            audit["authorized_anatomical_laterality"] = sorted(
                f"{side}_{part}" for side, part in used
            )

    return _apply_trigger_binding_audit(caption, projection, audit)


def main() -> int:
    # Build Phase 5.3 on the validated Phase-5.2 composer rather than jumping
    # back to v01.  That preserves gaze caption semantics and all existing
    # text-only behavior while adding signed torso/laterality wording and the
    # original Fizgig-style trigger/pronoun subject contract.
    engine._projection = _projection
    engine._caption_audit = _caption_audit
    phase52.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase52.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase52.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase52.SCHEMA_VERSION = SCHEMA_VERSION
    phase52.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase52.main()


if __name__ == "__main__":
    raise SystemExit(main())
