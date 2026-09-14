from __future__ import annotations

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

# Re-export the Phase-5.2 semantic projections.  The shared torso helper now
# includes signed turn_direction, while Phase-5.2 also preserves the validated
# gaze-caption-semantics projection from 4B.3.
_torso_fact = phase52._torso_fact
_gaze_fact = phase52._gaze_fact

_BASE_CAPTION_AUDIT = engine._caption_audit


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


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    """Keep the Phase-5 audit, but permit specialist-authorized body laterality.

    Earlier composer versions correctly rejected every anatomical left/right
    phrase because Qwen did not own laterality.  Phase 4B.4 now projects only
    DWPose-bound laterality into body.configuration.  Permit only the exact
    side/body-part pairs already present there; any new side label remains a
    hard violation.
    """
    audit = _BASE_CAPTION_AUDIT(caption, projection)
    violations = list(audit.get("violations") or [])
    if "unauthorized_anatomical_laterality" not in violations:
        return audit

    used = _laterality_pairs(caption)
    allowed = _authorized_laterality_pairs(projection)
    if used and used.issubset(allowed):
        violations = [v for v in violations if v != "unauthorized_anatomical_laterality"]
        audit["violations"] = sorted(set(violations))
        audit["authorized_anatomical_laterality"] = sorted(
            f"{side}_{part}" for side, part in used
        )
    return audit


def main() -> int:
    # Build Phase 5.3 on the validated Phase-5.2 composer rather than jumping
    # back to v01.  That preserves gaze caption semantics and all existing
    # text-only behavior while adding signed torso/laterality wording.
    engine._caption_audit = _caption_audit
    phase52.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase52.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase52.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase52.SCHEMA_VERSION = SCHEMA_VERSION
    phase52.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase52.main()


if __name__ == "__main__":
    raise SystemExit(main())
