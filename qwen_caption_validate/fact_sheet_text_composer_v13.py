from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v12 as phase511

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3.2"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v09.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.13"
SCHEMA_VERSION = "fact-sheet-text-composer-0.13"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3.2"

# Freeze the validated Phase-5.11 stack before main() patches module globals.
_BASE_PROJECTION = phase511._projection
_BASE_CAPTION_AUDIT = phase511._caption_audit
_BASE_RETRY_PROMPT = phase511._retry_prompt

_COORDINATE_POLICY_LEAK_RE = re.compile(
    r"\b(?:strictly\s+)?frame[- ]relative\b|"
    r"\bdepicted[- ]frame(?:\s+perspective|\s+coordinates?)?\b|"
    r"\banatomical\s+laterality\b|"
    r"\b(?:turn|facing)\s+direction\s+(?:is\s+)?(?:not\s+)?specified\b|"
    r"\bno\s+(?:visible\s+)?(?:turn|facing)\s+direction\s+(?:is\s+)?specified\b|"
    r"\bno\s+anatomical\s+(?:side|laterality)\s+(?:is\s+)?(?:specified|indicated)\b",
    re.I,
)

_MIRROR_PHONE_HARDWARE_PATTERNS = (
    re.compile(r"\s+with\s+multiple\s+rear\s+cameras?\b", re.I),
    re.compile(r"\s+with\s+(?:a\s+)?triple[- ]camera\s+module\b", re.I),
    re.compile(r"\s+with\s+three\s+rear\s+cameras?\b", re.I),
    re.compile(r"\s+with\s+multiple\s+(?:rear\s+)?camera\s+lenses\b", re.I),
    re.compile(r"\s+showing\s+(?:the\s+)?(?:multiple|three)\s+rear\s+cameras?\b", re.I),
)

_MIRROR_LOW_VALUE_DEVICE_RELATION_RE = re.compile(
    r"\bhead\b.*\b(?:phone|smartphone|device)\b.*\b(?:align(?:ed|ment)?|orientation|camera)\b|"
    r"\b(?:phone|smartphone|device)\b.*\bhead\b.*\b(?:align(?:ed|ment)?|orientation|camera)\b",
    re.I,
)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _natural_head_surface(head: dict[str, Any]) -> dict[str, Any] | None:
    """Collapse head-axis bookkeeping into one natural caption-facing phrase.

    Center/frontal are useful specialist states but low-value caption facts.
    They were causing prose such as "centered horizontally and vertically with
    a frontal yaw orientation".  Keep only semantically directional head pose.
    """
    if not isinstance(head, dict):
        return None

    horizontal = _clean(head.get("horizontal"))
    vertical = _clean(head.get("vertical"))
    strength = _clean(head.get("yaw_strength"))

    if horizontal == "center":
        horizontal = None
    if vertical == "center":
        vertical = None
    if strength == "frontal":
        strength = None

    parts: list[str] = []
    if horizontal in {"frame_left", "frame_right"}:
        direction = horizontal.replace("_", " ")
        if strength in {"strong_turn", "strong", "large_turn"}:
            parts.append(f"head turned strongly toward {direction}")
        elif strength in {"moderate_turn", "moderate"}:
            parts.append(f"head turned clearly toward {direction}")
        else:
            parts.append(f"head turned toward {direction}")

    if vertical == "up":
        parts.append("head tilted upward")
    elif vertical == "down":
        parts.append("head tilted downward")

    if not parts:
        return None

    if len(parts) == 1:
        composer_text = parts[0]
    else:
        composer_text = parts[0] + " and " + parts[1].removeprefix("head ")

    return {
        "composer_text": composer_text,
        "horizontal": horizontal,
        "vertical": vertical,
        "turn_strength": strength,
    }


def _compact_neutral_torso(torso: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(torso, dict):
        return None

    out = copy.deepcopy(torso)
    orientation = _clean(out.get("camera_orientation"))
    if orientation == "frontal":
        # "frontal, angled about 10 degrees" is technically compatible but
        # semantically awkward.  Once the categorical class is frontal, small
        # yaw magnitudes are diagnostic rather than useful caption detail.
        out.pop("yaw_magnitude_deg", None)
        out.pop("approx_yaw_deg", None)
        out.pop("turn_direction", None)

    return out or None


def _strip_mirror_phone_hardware(text: str) -> str:
    value = str(text or "")
    for pattern in _MIRROR_PHONE_HARDWARE_PATTERNS:
        value = pattern.sub("", value)
    value = re.sub(r"\s{2,}", " ", value)
    value = re.sub(r"\s+([,.;])", r"\1", value)
    return value.strip()


def _mirror_semantic_economy(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_mirror_phone_hardware(value)
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            cleaned = _mirror_semantic_economy(item)
            if isinstance(cleaned, str) and _MIRROR_LOW_VALUE_DEVICE_RELATION_RE.search(cleaned):
                continue
            if cleaned not in (None, "", [], {}):
                out.append(cleaned)
        return out
    if isinstance(value, dict):
        return {
            key: _mirror_semantic_economy(item)
            for key, item in value.items()
            if _mirror_semantic_economy(item) not in (None, "", [], {})
        }
    return copy.deepcopy(value)


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

    raw_head = authoritative.get("head") if isinstance(authoritative.get("head"), dict) else {}
    head_surface = _natural_head_surface(raw_head)
    if head_surface:
        authoritative["head"] = head_surface
    else:
        authoritative.pop("head", None)

    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    torso = body.get("torso_orientation") if isinstance(body.get("torso_orientation"), dict) else {}
    compact_torso = _compact_neutral_torso(torso)
    if compact_torso:
        body["torso_orientation"] = compact_torso
    else:
        body.pop("torso_orientation", None)
    if body:
        authoritative["body"] = body
    else:
        authoritative.pop("body", None)

    capture = authoritative.get("capture") if isinstance(authoritative.get("capture"), dict) else {}
    mirror_mode = capture.get("subtype") == "mirror_selfie"
    if mirror_mode:
        authoritative = _mirror_semantic_economy(authoritative)

    projection["authoritative_facts"] = authoritative
    audit["neutral_head_bookkeeping_suppressed"] = bool(raw_head) and not bool(head_surface)
    audit["head_surface_text"] = head_surface.get("composer_text") if head_surface else None
    audit["frontal_torso_numeric_yaw_suppressed"] = bool(
        torso and torso.get("camera_orientation") == "frontal"
    )
    audit["mirror_semantic_economy_applied"] = mirror_mode
    audit["mirror_low_value_device_alignment_filtered"] = mirror_mode
    audit["mirror_phone_hardware_classification_cues_suppressed"] = mirror_mode
    return projection, audit


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = _BASE_CAPTION_AUDIT(caption, projection)
    violations = list(audit.get("violations") or [])

    coordinate_policy_leaks = [
        " ".join(match.group(0).split())
        for match in _COORDINATE_POLICY_LEAK_RE.finditer(caption)
    ]
    if coordinate_policy_leaks:
        violations.append("coordinate_or_authority_policy_language_leak")

    # Once neutral head bookkeeping has been removed from projection there is
    # no reason to let the model invent pseudo-technical "frontal yaw" prose.
    if re.search(r"\bfrontal\s+yaw(?:\s+orientation)?\b", caption, re.I):
        violations.append("head_axis_instrumentation_language_leak")

    audit["violations"] = sorted(set(violations))
    audit["coordinate_policy_leak_phrases"] = coordinate_policy_leaks
    audit["semantic_surface_cleanup_enforced"] = True
    return audit


def _retry_prompt(original_prompt: str, caption: str, violations: list[str]) -> str:
    prompt = _BASE_RETRY_PROMPT(original_prompt, caption, violations)
    extra: list[str] = []

    if "coordinate_or_authority_policy_language_leak" in violations:
        extra.append(
            "Remove all explanations of coordinate systems, omitted laterality, unspecified turn direction, or evidence policy. Apply frame-relative wording silently; never say 'frame-relative', 'depicted-frame perspective', 'anatomical laterality', or 'no turn direction specified'."
        )
    if "head_axis_instrumentation_language_leak" in violations:
        extra.append(
            "Replace technical head-axis wording such as 'frontal yaw orientation' with ordinary photographic language, or omit the head clause if no directional head fact is supplied."
        )

    if not extra:
        return prompt
    return prompt + "\nSEMANTIC-SURFACE REVISION:\n- " + "\n- ".join(extra)


def main() -> int:
    # Preserve the full Phase-5.11 selfie/mirror + Phase-5.10 safety stack.
    # Change only the caption-facing semantic surface and audit.
    phase511._projection = _projection
    phase511._caption_audit = _caption_audit
    phase511._retry_prompt = _retry_prompt
    phase511.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase511.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase511.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase511.SCHEMA_VERSION = SCHEMA_VERSION
    phase511.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase511.main()


if __name__ == "__main__":
    raise SystemExit(main())
