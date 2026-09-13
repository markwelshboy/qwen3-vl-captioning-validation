from __future__ import annotations

from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v01 as base
from .fact_sheet_text_composer_torso_projection import compact_orientation, torso_fact

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v02.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.3"
SCHEMA_VERSION = "fact-sheet-text-composer-0.3"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"


def _compact_orientation(value: Any) -> dict[str, Any] | None:
    return compact_orientation(value, base._clean)


def _torso_fact(body: dict[str, Any]) -> dict[str, Any] | None:
    return torso_fact(body, base._clean)


def _gaze_fact(facts: dict[str, Any]) -> dict[str, Any] | None:
    raw = facts.get("gaze") if isinstance(facts.get("gaze"), dict) else {}
    semantics = raw.get("caption_semantics") if isinstance(raw.get("caption_semantics"), dict) else {}
    if not semantics.get("publishable"):
        return None

    out: dict[str, Any] = {}
    horizontal = semantics.get("horizontal") if isinstance(semantics.get("horizontal"), dict) else {}
    vertical = semantics.get("vertical") if isinstance(semantics.get("vertical"), dict) else {}
    camera = semantics.get("camera_relationship") if isinstance(semantics.get("camera_relationship"), dict) else {}

    if horizontal.get("publishable"):
        value = base._clean(horizontal.get("composer_value"))
        if value:
            out["horizontal"] = value
    if vertical.get("publishable"):
        value = base._clean(vertical.get("composer_value"))
        if value:
            out["vertical"] = value
    if camera.get("publishable"):
        value = base._clean(camera.get("composer_value"))
        if value:
            out["camera_relationship"] = value
    return out or None


def main() -> int:
    # Phase 5.2 consumes the governed identity-policy fact sheet. Torso and gaze
    # are projected through their caption-facing semantic layers so diagnostic
    # measurements cannot leak directly into prose.
    base._torso_fact = _torso_fact
    base._gaze_fact = _gaze_fact
    base.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    base.DEFAULT_PROMPT = DEFAULT_PROMPT
    base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
