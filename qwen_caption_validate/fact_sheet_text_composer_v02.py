from __future__ import annotations

from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v01 as base
from .fact_sheet_text_composer_torso_projection import compact_orientation, torso_fact

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.2"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v02.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.2"
SCHEMA_VERSION = "fact-sheet-text-composer-0.2"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.2.2"


def _compact_orientation(value: Any) -> dict[str, Any] | None:
    return compact_orientation(value, base._clean)


def _torso_fact(body: dict[str, Any]) -> dict[str, Any] | None:
    return torso_fact(body, base._clean)


def main() -> int:
    base._torso_fact = _torso_fact
    base.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    base.DEFAULT_PROMPT = DEFAULT_PROMPT
    base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
