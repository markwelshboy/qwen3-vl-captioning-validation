from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v11 as phase510

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3.2"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v08.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.12"
SCHEMA_VERSION = "fact-sheet-text-composer-0.12"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3.2"

# Freeze the complete validated Phase-5.10 stack before main() rebinds it.
_BASE_PROJECTION = phase510._projection
_BASE_CAPTION_AUDIT = phase510._caption_audit
_BASE_RETRY_PROMPT = phase510._retry_prompt

_SELFIE_RE = re.compile(r"\bselfie\b|\bselfie[- ]style\b", re.I)
_MIRROR_SELFIE_RE = re.compile(r"\bmirror[- ]selfie\b", re.I)
_MIRROR_ANATOMICAL_LATERALITY_RE = re.compile(
    r"\b(left|right)\s+"
    r"(fist|hand|arm|forearm|wrist|elbow|shoulder|hip|knee|leg|thigh|calf|ankle|foot|eye|ear)\b",
    re.I,
)

# Structured keys whose value is anatomical side, not image-frame direction.
_MIRROR_ANATOMICAL_SIDE_KEYS = {
    "support_side",
    "elevated_side",
    "anatomical_side",
    "anatomical_side_internal",
    "anatomical_side_nearer",
    "clear_nearer_shoulder_anatomical_side",
}


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _source_capture(sheet: dict[str, Any]) -> dict[str, Any] | None:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    capture = facts.get("capture") if isinstance(facts.get("capture"), dict) else {}
    if not capture.get("available"):
        return None
    if capture.get("family") != "selfie":
        return None
    subtype = _clean(capture.get("subtype"))
    if subtype not in {"direct_selfie", "mirror_selfie"}:
        return None
    return capture


def _capture_projection(capture: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(capture, dict):
        return None

    subtype = _clean(capture.get("subtype"))
    composer_text = _clean(capture.get("composer_text"))
    if subtype == "mirror_selfie":
        return {
            "family": "selfie",
            "subtype": "mirror_selfie",
            "composer_text": composer_text or "mirror selfie",
            "spatial_surface_mode": "depicted_frame",
            "directional_reference_system": "frame_relative_only",
            "anatomical_laterality_policy": "withhold",
        }

    if subtype == "direct_selfie":
        facts = [
            _clean(v)
            for v in (capture.get("direct_capture_facts") or capture.get("promoted_fact_candidates") or [])
            if _clean(v)
        ]
        base_text = composer_text or "selfie-style capture"
        composition = [v for v in facts if v.casefold() != base_text.casefold()]
        out: dict[str, Any] = {
            "family": "selfie",
            "subtype": "direct_selfie",
            "composer_text": base_text,
            "spatial_surface_mode": "direct_camera_frame",
        }
        if composition:
            out["composition_facts"] = composition
        return out

    return None


def _neutralize_anatomical_string(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        part = match.group(2)
        return part

    value = _MIRROR_ANATOMICAL_LATERALITY_RE.sub(repl, str(text or ""))
    return " ".join(value.split())


def _mirror_surface_value(value: Any) -> Any:
    if isinstance(value, str):
        return _neutralize_anatomical_string(value)
    if isinstance(value, list):
        return [_mirror_surface_value(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if (
                key_text in _MIRROR_ANATOMICAL_SIDE_KEYS
                or "anatomical_side" in key_text
            ):
                continue
            out[key] = _mirror_surface_value(item)
        return out
    return copy.deepcopy(value)


def _collect_anatomical_pairs(value: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if isinstance(value, str):
        for match in _MIRROR_ANATOMICAL_LATERALITY_RE.finditer(value):
            side = match.group(1).lower()
            part = match.group(2).lower()
            if part == "fist":
                part = "hand"
            pairs.add((side, part))
    elif isinstance(value, list):
        for item in value:
            pairs.update(_collect_anatomical_pairs(item))
    elif isinstance(value, dict):
        for item in value.values():
            pairs.update(_collect_anatomical_pairs(item))
    return pairs


def _projected_capture(projection: dict[str, Any]) -> dict[str, Any]:
    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    return (
        authoritative.get("capture")
        if isinstance(authoritative.get("capture"), dict)
        else {}
    )


def _authorized_direct_capture_laterality(
    projection: dict[str, Any],
) -> set[tuple[str, str]]:
    capture = _projected_capture(projection)
    if capture.get("subtype") != "direct_selfie":
        return set()
    composition = (
        capture.get("composition_facts")
        if isinstance(capture.get("composition_facts"), list)
        else []
    )
    allowed: set[tuple[str, str]] = set()
    for text in composition:
        if isinstance(text, str):
            allowed.update(phase510._extended_laterality_pairs(text))
    return allowed


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

    capture_source = _source_capture(sheet)
    capture = _capture_projection(capture_source)
    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    authoritative = copy.deepcopy(authoritative)

    if capture:
        authoritative["capture"] = capture
    else:
        authoritative.pop("capture", None)

    mirror_mode = bool(capture and capture.get("subtype") == "mirror_selfie")
    pre_surface_pairs = _collect_anatomical_pairs(authoritative) if mirror_mode else set()

    if mirror_mode:
        # Collapse all caption-facing anatomical laterality to one visible-image
        # coordinate system.  Explicit frame_left/frame_right tokens survive
        # because they are not anatomical-side phrases.
        authoritative = _mirror_surface_value(authoritative)

    projection["authoritative_facts"] = authoritative

    # Selfie semantics are specialist-owned.  Do not leave an ungoverned
    # "selfie" hint in holistic context, including on withheld Qwen-only cases.
    holistic = _clean(projection.get("holistic_context_non_authoritative"))
    holistic_selfie_removed = bool(holistic and _SELFIE_RE.search(holistic))
    if holistic_selfie_removed:
        projection.pop("holistic_context_non_authoritative", None)

    audit["capture_style_projected"] = bool(capture)
    audit["capture_subtype"] = capture.get("subtype") if capture else None
    audit["mirror_depicted_frame_mode"] = mirror_mode
    audit["mirror_surface_neutralized_anatomical_laterality"] = sorted(
        f"{side}_{part}" for side, part in pre_surface_pairs
    )
    audit["holistic_selfie_context_withheld"] = holistic_selfie_removed
    audit["selfie_capture_is_specialist_owned"] = True
    return projection, audit


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = _BASE_CAPTION_AUDIT(caption, projection)
    violations = list(audit.get("violations") or [])

    capture = _projected_capture(projection)
    subtype = _clean(capture.get("subtype"))
    selfie_used = bool(_SELFIE_RE.search(caption))
    mirror_used = bool(_MIRROR_SELFIE_RE.search(caption))
    mirror_anatomical_pairs = _collect_anatomical_pairs(caption)

    if not subtype and selfie_used:
        violations.append("selfie_language_without_capture_authority")

    if subtype == "direct_selfie":
        if not selfie_used:
            violations.append("direct_selfie_capture_missing")
        if mirror_used:
            violations.append("mirror_selfie_without_mirror_capture_authority")

        # Phase-5.10 authorizes anatomical laterality only from body
        # configuration. Direct-selfie capture may independently authorize a
        # nearer-shoulder phrase; extend that one audit boundary without
        # changing body-specialist ownership.
        if "unauthorized_anatomical_laterality" in violations:
            used = phase510._extended_laterality_pairs(caption)
            allowed = phase510._authorized_extended_laterality(projection)
            allowed.update(_authorized_direct_capture_laterality(projection))
            if used and used.issubset(allowed):
                violations = [
                    v for v in violations
                    if v != "unauthorized_anatomical_laterality"
                ]

    if subtype == "mirror_selfie":
        if not mirror_used:
            violations.append("mirror_selfie_capture_missing")
        if mirror_anatomical_pairs:
            violations.append("mirror_selfie_anatomical_laterality_forbidden")
        # Never let the generic inherited laterality exception authorize a
        # mirrored anatomical side. One visible-image coordinate system only.
        if mirror_anatomical_pairs and "unauthorized_anatomical_laterality" not in violations:
            violations.append("unauthorized_anatomical_laterality")

    audit["violations"] = sorted(set(violations))
    audit["capture_subtype"] = subtype
    audit["selfie_language_used"] = selfie_used
    audit["mirror_selfie_language_used"] = mirror_used
    audit["mirror_anatomical_laterality_used"] = sorted(
        f"{side}_{part}" for side, part in mirror_anatomical_pairs
    )
    audit["mirror_depicted_frame_contract_enforced"] = subtype == "mirror_selfie"
    return audit


def _retry_prompt(original_prompt: str, caption: str, violations: list[str]) -> str:
    prompt = _BASE_RETRY_PROMPT(original_prompt, caption, violations)
    extra: list[str] = []

    if "selfie_language_without_capture_authority" in violations:
        extra.append(
            "Remove all selfie/mirror-selfie wording. No authoritative capture-style fact is supplied."
        )
    if "direct_selfie_capture_missing" in violations:
        extra.append(
            "Restore the authoritative direct-selfie capture semantic using natural selfie/selfie-style wording."
        )
    if "mirror_selfie_without_mirror_capture_authority" in violations:
        extra.append(
            "Remove 'mirror selfie'; the authoritative capture subtype is direct_selfie."
        )
    if "mirror_selfie_capture_missing" in violations:
        extra.append(
            "State that the image is a mirror selfie. This is an authoritative capture-style fact."
        )
    if (
        "mirror_selfie_anatomical_laterality_forbidden" in violations
        or "unauthorized_anatomical_laterality" in violations
    ):
        extra.append(
            "For a mirror selfie, remove anatomical left/right from every hand, arm, wrist, elbow, shoulder, hip, knee, leg, ankle, foot, eye, and ear phrase. Preserve explicit frame-left/frame-right only when it is supplied by AUTHORITATIVE FACTS; otherwise use side-neutral wording."
        )

    if not extra:
        return prompt
    return prompt + "\nSELFIE / MIRROR REVISION:\n- " + "\n- ".join(extra)


def main() -> int:
    # Preserve the complete Phase-5.10 framing/crouch/torso/gaze/trigger safety
    # stack. Extend only its projection, audit, retry prompt, and schema paths.
    phase510._projection = _projection
    phase510._caption_audit = _caption_audit
    phase510._retry_prompt = _retry_prompt
    phase510.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase510.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase510.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase510.SCHEMA_VERSION = SCHEMA_VERSION
    phase510.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase510.main()


if __name__ == "__main__":
    raise SystemExit(main())
