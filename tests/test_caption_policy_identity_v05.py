from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import caption_policy_identity_v05 as mod


def test_phase4b6_identity_policy_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.6"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.EXPECTED_INPUT_SCHEMA == "caption-fact-sheet-0.2.6"
