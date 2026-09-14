from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import caption_policy_identity_v03 as mod


def test_identity_policy_v03_consumes_phase4b4_fact_sheets():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.4"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.EXPECTED_INPUT_SCHEMA == "caption-fact-sheet-0.2.4"
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.3"
