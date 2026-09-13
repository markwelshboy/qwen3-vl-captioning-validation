from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import caption_policy_identity_v02 as mod


def test_identity_policy_v02_defaults_to_phase4b3_input():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.EXPECTED_INPUT_SCHEMA == "caption-fact-sheet-0.2.3"
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.3"
