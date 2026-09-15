from pathlib import Path

from qwen_caption_validate import caption_policy_identity_v06 as mod


def test_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.11"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.EXPECTED_INPUT_SCHEMA == "caption-fact-sheet-0.2.11"
