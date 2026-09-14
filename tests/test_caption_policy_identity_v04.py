from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import caption_policy_identity_v04 as mod


def test_phase4c_consumes_phase4b5_namespace():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.5"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.EXPECTED_INPUT_SCHEMA == "caption-fact-sheet-0.2.5"
