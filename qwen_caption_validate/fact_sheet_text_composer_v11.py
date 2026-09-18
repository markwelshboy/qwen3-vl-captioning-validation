from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v10 as phase59

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3.1"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v07.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.11"
SCHEMA_VERSION = "fact-sheet-text-composer-0.11"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3.1"

_BASE_PROJECTION = phase59._projection
_BASE_CAPTION_AUDIT = phase59._caption_audit
_BASE_RETRY_PROMPT = phase59._retry_prompt

_SHOT_SCALE_RE = re.compile(
    r"\b(?:"
    r"extreme[- ]close[- ]up|"
    r"medium[- ]close[- ]up|"
    r"medium[- ]wide(?:\s+shot)?|"
    r"close[- ]up|"
    r"medium\s+shot|"
    r"full[- ]body(?:\s+(?:shot|framing))?|"
    r"waist[- ]up(?:\s+(?:shot|framing))?|"
    r"three[- ]quarter(?:\s+(?:shot|framing))?"
    r")\b",
    re.I,
)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _normalized_phrase(value: str) -> str:
    text = str(value or "").casefold().replace("-", " ")
    return " ".join(text.split())


def _source_framing(sheet: dict[str, Any]) -> dict[str, Any] | None:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    framing = facts.get("framing") if isinstance(facts.get("framing"), dict) else {}
    composer = framing.get("composer_framing") if isinstance(framing.get("composer_framing"), dict) else {}
    text = _clean(composer.get("composer_text"))
    if not text:
        return None

    source = _clean(composer.get("source"))
    out: dict[str, Any] = {
        "composer_text": text,
        "surface_source": source or "unknown",
    }

    scale = framing.get("standard_shot_scale") if isinstance(framing.get("standard_shot_scale"), dict) else {}
    if source == "standard_shot_scale" and scale.get("status") == "candidate":
        label = _clean(scale.get("label"))
        if label:
            out["shot_scale_label"] = label

    span = framing.get("anatomical_span") if isinstance(framing.get("anatomical_span"), dict) else {}
    if source == "anatomical_span":
        upper = _clean(span.get("upper_anchor"))
        lower = _clean(span.get("lower_anchor"))
        if upper and lower:
            out["anatomical_span"] = {"upper": upper, "lower": lower}

    return out


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

    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    framing = _source_framing(sheet)

    # The inherited engine knows only the retired legacy extent enum. Phase
    # 4B.16 demotes that enum, so replace framing wholesale with the production
    # composer surface.
    if framing:
        authoritative["framing"] = framing
    else:
        authoritative.pop("framing", None)

    projection["authoritative_facts"] = authoritative
    audit["production_framing_projected"] = framing is not None
    audit["production_framing_composer_text"] = framing.get("composer_text") if framing else None
    audit["legacy_extent_projection_disabled"] = True
    audit["framing_is_first_visual_fact_contract"] = True
    return projection, audit


def _projected_framing(projection: dict[str, Any]) -> dict[str, Any]:
    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    return authoritative.get("framing") if isinstance(authoritative.get("framing"), dict) else {}


def _shot_scale_token(text: str) -> str:
    normalized = _normalized_phrase(text)
    if normalized.startswith("extreme close up"):
        return "extreme_close_up"
    if normalized.startswith("medium close up"):
        return "medium_close_up"
    if normalized.startswith("medium wide"):
        return "medium_wide"
    if normalized.startswith("close up"):
        return "close_up"
    if normalized.startswith("medium shot"):
        return "medium"
    if normalized.startswith("full body"):
        return "full_body"
    if normalized.startswith("waist up"):
        return "waist_up"
    if normalized.startswith("three quarter"):
        return "three_quarter"
    return normalized.replace(" ", "_")


def _used_shot_scales(caption: str) -> set[str]:
    return {_shot_scale_token(m.group(0)) for m in _SHOT_SCALE_RE.finditer(caption)}


def _phrase_word_position(caption: str, phrase: str) -> int | None:
    caption_norm = _normalized_phrase(caption)
    phrase_norm = _normalized_phrase(phrase)
    idx = caption_norm.find(phrase_norm)
    if idx < 0:
        return None
    return len(caption_norm[:idx].split())


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = _BASE_CAPTION_AUDIT(caption, projection)
    violations = list(audit.get("violations") or [])

    framing = _projected_framing(projection)
    expected_text = _clean(framing.get("composer_text"))
    source = _clean(framing.get("surface_source"))
    expected_scale = _clean(framing.get("shot_scale_label"))
    used_scales = _used_shot_scales(caption)

    position = _phrase_word_position(caption, expected_text) if expected_text else None
    if expected_text and position is None:
        violations.append("authoritative_framing_missing")
    elif expected_text and position is not None and position > 14:
        violations.append("authoritative_framing_not_near_opening")

    if source == "anatomical_span" and used_scales:
        violations.append("shot_scale_language_without_authority")

    if source == "standard_shot_scale" and expected_scale:
        unauthorized = sorted(used_scales - {expected_scale})
        if unauthorized:
            violations.append("unauthorized_shot_scale:" + ",".join(unauthorized))

    audit["violations"] = sorted(set(violations))
    audit["authoritative_framing_text"] = expected_text
    audit["authoritative_framing_source"] = source
    audit["authoritative_framing_word_position"] = position
    audit["authorized_shot_scale"] = expected_scale
    audit["used_shot_scales"] = sorted(used_scales)
    audit["framing_first_visual_fact_contract"] = True
    return audit


def _retry_prompt(original_prompt: str, caption: str, violations: list[str]) -> str:
    prompt = _BASE_RETRY_PROMPT(original_prompt, caption, violations)
    extra: list[str] = []

    if "authoritative_framing_missing" in violations:
        extra.append(
            "Restore authoritative_facts.framing.composer_text faithfully in the first sentence."
        )
    if "authoritative_framing_not_near_opening" in violations:
        extra.append(
            "Move the authoritative framing to the beginning: immediately after the trigger/subject, before pose, body configuration, torso, head, gaze, appearance, or scene facts."
        )
    if "shot_scale_language_without_authority" in violations:
        extra.append(
            "Remove all photographic shot-scale labels. The framing authority supplies a literal anatomical crop description instead."
        )
    if any(v.startswith("unauthorized_shot_scale:") for v in violations):
        extra.append(
            "Remove every shot-scale term except the one explicitly supplied in authoritative_facts.framing."
        )

    if not extra:
        return prompt
    return prompt + "\nFRAMING REVISION:\n- " + "\n- ".join(extra)


def main() -> int:
    # Preserve the complete Phase-5.9/5.10 safety stack, including crouched
    # stance depth, and replace only its framing projection/audit/prompt.
    phase59._projection = _projection
    phase59._caption_audit = _caption_audit
    phase59._retry_prompt = _retry_prompt
    phase59.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase59.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase59.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase59.SCHEMA_VERSION = SCHEMA_VERSION
    phase59.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase59.main()


if __name__ == "__main__":
    raise SystemExit(main())
