from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v13 as phase512

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3.2"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v10.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.14"
SCHEMA_VERSION = "fact-sheet-text-composer-0.14"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3.2"

# Freeze the validated Phase-5.12 stack before main() patches module globals.
_BASE_PROJECTION = phase512._projection
_BASE_CAPTION_AUDIT = phase512._caption_audit
_BASE_RETRY_PROMPT = phase512._retry_prompt

_CAMERA_HOLD_SUFFIX_RE = re.compile(
    r"(?:\s*,\s*|\s+)(?:while\s+)?"
    r"(?:holding|gripping|carrying)\s+(?:a\s+|the\s+)?camera\b.*$",
    re.I,
)
_CAMERA_CAPTURE_MECHANISM_RE = re.compile(
    r"\b(?:hold(?:s|ing)?|grip(?:s|ping)?|carry(?:ing|ies)?)\s+(?:a\s+|the\s+)?camera\b"
    r"|\b(?:tak(?:e|es|ing)|captur(?:e|es|ing))\s+(?:a\s+|the\s+)?"
    r"(?:photo|photograph|picture|selfie)\b"
    r"|\bphotograph(?:s|ing)?\s+(?:herself|himself|themself|themselves)\b",
    re.I,
)
_NEGATIVE_BODY_GEOMETRY_RE = re.compile(
    r"\bno\s+(?:visible\s+)?(?:body\s+|torso\s+)?(?:turn|tilt|lean|bend)\b"
    r"|\bno\s+(?:visible\s+)?(?:turn|tilt)(?:\s+or\s+(?:turn|tilt))?\b"
    r"|\bwithout\s+(?:any\s+)?(?:visible\s+)?(?:body\s+|torso\s+)?(?:turn|tilt|lean|bend)\b",
    re.I,
)
_MIRROR_TAUTOLOGY_RE = re.compile(
    r"\b(?:the\s+)?mirror(?:\s+surface)?\s+reflects?\s+(?:the\s+)?scene\b",
    re.I,
)
_META_COMPOSITION_RE = re.compile(
    r"\b(?:the\s+)?(?:mirror\s+selfie\s+)?composition\s+"
    r"(?:frames?|focuses?|emphasiz(?:es|ing)|highlights?|captures?)\b"
    r"|\b(?:the\s+)?framing\s+(?:focuses?|emphasiz(?:es|ing)|highlights?|captures?)\b",
    re.I,
)
_INTERPRETIVE_CAUSAL_RE = re.compile(
    r"\b(?:contribut(?:e|es|ed|ing)|add(?:s|ed|ing)?|giv(?:e|es|ing)|"
    r"creat(?:e|es|ed|ing)|evok(?:e|es|ed|ing)|suggest(?:s|ed|ing)?)\b"
    r"[^.!?]{0,120}\b(?:mood|expression|feeling|atmosphere)\b",
    re.I,
)
_SELF_CONTAINED_MOMENT_RE = re.compile(r"\bself[- ]contained\s+moment\b", re.I)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _capture_authorized(projection_or_authoritative: dict[str, Any]) -> bool:
    if not isinstance(projection_or_authoritative, dict):
        return False

    authoritative = projection_or_authoritative
    if isinstance(projection_or_authoritative.get("authoritative_facts"), dict):
        authoritative = projection_or_authoritative["authoritative_facts"]

    capture = (
        authoritative.get("capture")
        if isinstance(authoritative.get("capture"), dict)
        else {}
    )
    return capture.get("subtype") in {"direct_selfie", "mirror_selfie"}


def _strip_unauthorized_capture_mechanism(text: str) -> str | None:
    """Remove capture-mechanism semantics while preserving local body geometry."""
    value = " ".join(str(text or "").split())
    if not value:
        return None

    # The common useful form is "arm extended forward, holding the camera".
    # Keep the local arm geometry and remove only the capture claim.
    value = _CAMERA_HOLD_SUFFIX_RE.sub("", value).strip(" ,.;")

    # If the remaining phrase still states a capture mechanism, the relation is
    # capture-owned and must be withheld entirely.
    if _CAMERA_CAPTURE_MECHANISM_RE.search(value):
        return None

    return value or None


def _capture_safe_configuration(
    values: Any,
    *,
    capture_authorized: bool,
) -> tuple[list[Any], list[str]]:
    if not isinstance(values, list):
        return [], []

    if capture_authorized:
        return copy.deepcopy(values), []

    out: list[Any] = []
    withheld: list[str] = []
    for item in values:
        if not isinstance(item, str):
            out.append(copy.deepcopy(item))
            continue

        cleaned = _strip_unauthorized_capture_mechanism(item)
        if cleaned != " ".join(item.split()):
            withheld.append(item)
        if cleaned:
            out.append(cleaned)

    return out, withheld


def _suppress_mirror_neutral_torso(body: dict[str, Any]) -> bool:
    torso = (
        body.get("torso_orientation")
        if isinstance(body.get("torso_orientation"), dict)
        else {}
    )
    if torso.get("camera_orientation") != "frontal":
        return False

    body.pop("torso_orientation", None)
    return True


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
    authoritative = copy.deepcopy(authoritative)

    capture = (
        authoritative.get("capture")
        if isinstance(authoritative.get("capture"), dict)
        else {}
    )
    subtype = _clean(capture.get("subtype"))
    capture_ok = subtype in {"direct_selfie", "mirror_selfie"}
    mirror_mode = subtype == "mirror_selfie"

    body = (
        authoritative.get("body")
        if isinstance(authoritative.get("body"), dict)
        else {}
    )
    body = copy.deepcopy(body)

    original_configuration = (
        body.get("configuration")
        if isinstance(body.get("configuration"), list)
        else []
    )
    configuration, withheld_capture_relations = _capture_safe_configuration(
        original_configuration,
        capture_authorized=capture_ok,
    )
    if original_configuration:
        if configuration:
            body["configuration"] = configuration
        else:
            body.pop("configuration", None)

    mirror_frontal_torso_suppressed = False
    if mirror_mode:
        mirror_frontal_torso_suppressed = _suppress_mirror_neutral_torso(body)

    if body:
        authoritative["body"] = body
    else:
        authoritative.pop("body", None)

    holistic = _clean(projection.get("holistic_context_non_authoritative"))
    holistic_capture_withheld = bool(
        not capture_ok
        and holistic
        and _CAMERA_CAPTURE_MECHANISM_RE.search(holistic)
    )
    if holistic_capture_withheld:
        projection.pop("holistic_context_non_authoritative", None)

    projection["authoritative_facts"] = authoritative
    audit["capture_mechanism_domain_firewall_enforced"] = True
    audit["capture_mechanism_relations_sanitized"] = withheld_capture_relations
    audit["holistic_capture_mechanism_withheld"] = holistic_capture_withheld
    audit["mirror_neutral_frontal_torso_suppressed"] = (
        mirror_frontal_torso_suppressed
    )
    return projection, audit


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = _BASE_CAPTION_AUDIT(caption, projection)
    violations = list(audit.get("violations") or [])

    capture_ok = _capture_authorized(projection)
    capture_mechanism_used = bool(_CAMERA_CAPTURE_MECHANISM_RE.search(caption))
    if capture_mechanism_used and not capture_ok:
        violations.append("capture_mechanism_language_without_capture_authority")

    negative_geometry = [
        " ".join(m.group(0).split())
        for m in _NEGATIVE_BODY_GEOMETRY_RE.finditer(caption)
    ]
    if negative_geometry:
        violations.append("unsupported_negative_body_geometry_language")

    mirror_tautology = [
        " ".join(m.group(0).split())
        for m in _MIRROR_TAUTOLOGY_RE.finditer(caption)
    ]
    if mirror_tautology:
        violations.append("mirror_reflection_tautology")

    meta_composition = [
        " ".join(m.group(0).split())
        for m in _META_COMPOSITION_RE.finditer(caption)
    ]
    if meta_composition:
        violations.append("meta_composition_narration")

    interpretive_causal = [
        " ".join(m.group(0).split())
        for m in _INTERPRETIVE_CAUSAL_RE.finditer(caption)
    ]
    if interpretive_causal:
        violations.append("unsupported_setting_to_mood_causality")

    self_contained = [
        " ".join(m.group(0).split())
        for m in _SELF_CONTAINED_MOMENT_RE.finditer(caption)
    ]
    if self_contained:
        violations.append("interpretive_moment_narration")

    audit["violations"] = sorted(set(violations))
    audit["capture_mechanism_language_used"] = capture_mechanism_used
    audit["capture_mechanism_authorized"] = capture_ok
    audit["negative_body_geometry_phrases"] = negative_geometry
    audit["mirror_reflection_tautology_phrases"] = mirror_tautology
    audit["meta_composition_phrases"] = meta_composition
    audit["setting_to_mood_causality_phrases"] = interpretive_causal
    audit["interpretive_moment_phrases"] = self_contained
    audit["caption_surface_narration_guard_enforced"] = True
    return audit


def _retry_prompt(
    original_prompt: str,
    caption: str,
    violations: list[str],
) -> str:
    prompt = _BASE_RETRY_PROMPT(original_prompt, caption, violations)
    extra: list[str] = []

    unauthorized_pose = [
        violation.split(":", 1)[1]
        for violation in violations
        if violation.startswith("unauthorized_broad_pose:") and ":" in violation
    ]
    if unauthorized_pose:
        extra.append(
            "Remove the unsupported broad-pose/posture claim(s) "
            + ", ".join(unauthorized_pose)
            + " entirely. Do not replace them with another posture or with a negative/neutral posture statement. If a scene/object phrase depends grammatically on the unsupported posture (for example 'lying on a gray fabric surface'), remove that dependent relation too rather than preserving a dangling preposition."
        )

    if "capture_mechanism_language_without_capture_authority" in violations:
        extra.append(
            "Remove all claims that the subject is holding a camera, taking a photo, or otherwise operating the capture device. Preserve only local body geometry that is independently supplied, such as an arm extending forward."
        )
    if "unsupported_negative_body_geometry_language" in violations:
        extra.append(
            "Remove negative/neutral completion such as 'no visible turn or tilt'. Absence of a turn, tilt, lean, or bend is not a caption fact."
        )
    if "mirror_reflection_tautology" in violations:
        extra.append(
            "Remove tautological mirror narration such as 'the mirror reflects the scene'. The authoritative 'mirror selfie' phrase already carries that capture information."
        )
    if "meta_composition_narration" in violations:
        extra.append(
            "Remove meta-caption narration about what the composition or framing focuses on, emphasizes, highlights, captures, or frames. State concrete visible facts directly."
        )
    if "unsupported_setting_to_mood_causality" in violations:
        extra.append(
            "Remove any causal claim that the setting, lighting, or background creates or contributes to the subject's mood or expression. Keep the visible setting and visible expression as separate facts."
        )
    if "interpretive_moment_narration" in violations:
        extra.append(
            "Remove interpretive phrases such as 'self-contained moment'. End on concrete visible facts instead."
        )

    if not extra:
        return prompt
    return prompt + "\nCAPTION-SURFACE REVISION:\n- " + "\n- ".join(extra)


def main() -> int:
    # Preserve the full Phase-5.12 semantic-surface + Phase-5.11 selfie/mirror
    # stack. Extend only capture-domain ownership, mirror neutral geometry, and
    # anti-narration surface discipline.
    phase512._projection = _projection
    phase512._caption_audit = _caption_audit
    phase512._retry_prompt = _retry_prompt
    phase512.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase512.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase512.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase512.SCHEMA_VERSION = SCHEMA_VERSION
    phase512.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase512.main()


if __name__ == "__main__":
    raise SystemExit(main())
