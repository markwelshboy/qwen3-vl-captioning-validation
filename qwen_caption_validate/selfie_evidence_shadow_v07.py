from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from . import selfie_evidence_shadow_v06 as base

SCHEMA_VERSION = "selfie-evidence-shadow-0.7"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "selfie-evidence-shadow-v0.7"

# Capture the v0.6 implementation before main() monkey-patches the module.
# Calling base._neutral_mirror_selfie_semantic dynamically from the wrapper
# would recurse once base is rebound to this v0.7 function.
_BASE_NEUTRAL_MIRROR_SELFIE_SEMANTIC = base._neutral_mirror_selfie_semantic

_REAR_CAMERA_RE = re.compile(
    r"\bmultiple\s+rear\s+cameras?\b"
    r"|\brear[- ]facing\s+cameras?\b"
    r"|\brear\s+camera(?:s|\s+module)?\b"
    r"|\bback\s+of\s+(?:the\s+)?(?:phone|smartphone)\b",
    re.I,
)


def _neutral_mirror_selfie_semantic(
    gestalt_record: dict[str, Any] | None,
) -> dict[str, Any]:
    out = dict(_BASE_NEUTRAL_MIRROR_SELFIE_SEMANTIC(gestalt_record))
    if out.get("supported"):
        out["precision_policy"] = (
            "Explicit neutral mirror-selfie language, neutral selfie plus an independent "
            "mirror/reflection cue, or neutral selfie plus visible rear-phone-camera "
            "semantics can publish mirror_selfie in v0.7."
        )
        return out

    if not isinstance(gestalt_record, dict):
        return out

    acquisition = (
        gestalt_record.get("acquisition")
        if isinstance(gestalt_record.get("acquisition"), dict)
        else {}
    )
    strings = base._flatten_strings(acquisition)

    generic_selfie = [s for s in strings if base._GENERIC_SELFIE_RE.search(s)]
    phone = [s for s in strings if base._PHONE_RE.search(s)]
    rear_camera = [s for s in strings if _REAR_CAMERA_RE.search(s)]

    out["rear_phone_camera_text"] = rear_camera

    if generic_selfie and phone and rear_camera:
        out.update(
            {
                "grade": "strong",
                "supported": True,
                "reason": (
                    "neutral_selfie_semantic_plus_visible_rear_phone_camera_semantic"
                ),
                "authority": (
                    "unprompted_qwen_selfie_semantic_plus_visible_phone_back_detail"
                ),
                "precision_policy": (
                    "A neutral selfie semantic plus explicit visible rear-camera/back-of-phone "
                    "detail is treated as reflection-mediated capture. Phone presence alone is "
                    "never sufficient."
                ),
            }
        )
    else:
        out["precision_policy"] = (
            "Mirror selfie remains withheld unless explicit neutral mirror semantics, "
            "neutral selfie plus mirror/reflection semantics, or neutral selfie plus "
            "visible rear-phone-camera semantics are present."
        )

    return out


def main() -> int:
    original_argv = list(sys.argv)
    old_semantic = base._neutral_mirror_selfie_semantic
    old_schema = base.SCHEMA_VERSION
    old_output = base.DEFAULT_OUTPUT_SUBDIR
    try:
        base._neutral_mirror_selfie_semantic = _neutral_mirror_selfie_semantic
        base.SCHEMA_VERSION = SCHEMA_VERSION
        base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
        return base.main()
    finally:
        base._neutral_mirror_selfie_semantic = old_semantic
        base.SCHEMA_VERSION = old_schema
        base.DEFAULT_OUTPUT_SUBDIR = old_output
        sys.argv[:] = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
