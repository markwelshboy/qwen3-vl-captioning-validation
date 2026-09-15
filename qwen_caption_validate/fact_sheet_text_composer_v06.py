from __future__ import annotations

from pathlib import Path

from . import fact_sheet_text_composer_v05 as phase54

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v05.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.6"
SCHEMA_VERSION = "fact-sheet-text-composer-0.6"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"

# Re-export the validated Phase-5.4 projection/audit stack.  v06 changes only
# the composer contract: support/contact may no longer be inferred from a
# raised leg or ordinary biomechanics, while specialist-bound laterality and
# useful raised-leg geometry remain publishable.
_projection = phase54._projection
_caption_audit = phase54._caption_audit
_torso_fact = phase54._torso_fact
_gaze_fact = phase54._gaze_fact


def main() -> int:
    phase54.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase54.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase54.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase54.SCHEMA_VERSION = SCHEMA_VERSION
    phase54.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase54.main()


if __name__ == "__main__":
    raise SystemExit(main())
