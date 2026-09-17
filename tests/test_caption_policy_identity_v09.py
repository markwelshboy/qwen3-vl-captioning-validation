from pathlib import Path

from qwen_caption_validate import caption_policy_identity_v09 as mod


def test_phase4c_v09_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.14"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.EXPECTED_INPUT_SCHEMA == "caption-fact-sheet-0.2.14"


def test_identity_policy_preserves_crouched_stance_depth_fact():
    source = {
        "schema_version": "caption-fact-sheet-0.2.14",
        "facts": {
            "body": {
                "crouched_stance_depth": {
                    "classification": "moderate_lowering",
                    "semantic_relation": "hips_slightly_lowered",
                    "composer_text": "hips slightly lowered",
                }
            },
            "visual": {"appearance": []},
        },
        "audit": {},
    }
    out = mod.base.apply_character_identity_policy(source)
    depth = out["facts"]["body"]["crouched_stance_depth"]
    assert depth["classification"] == "moderate_lowering"
    assert depth["semantic_relation"] == "hips_slightly_lowered"
    assert depth["composer_text"] == "hips slightly lowered"
