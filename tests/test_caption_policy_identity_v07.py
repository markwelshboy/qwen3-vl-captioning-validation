from pathlib import Path

from qwen_caption_validate import caption_policy_identity_v07 as mod


def test_phase4b12_identity_policy_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.12"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.EXPECTED_INPUT_SCHEMA == "caption-fact-sheet-0.2.12"


def test_bilateral_knee_specialist_fact_and_adjudication_survive_identity_policy():
    source = {
        "schema_version": "caption-fact-sheet-0.2.12",
        "status": "ok",
        "image_key": "imageblind-01_00066",
        "facts": {
            "body": {
                "pose_candidate": {
                    "text": "crouching",
                    "composer_text": "crouching",
                    "promotion_status": "accepted_specialist_adjudicated_candidate",
                },
                "configuration": [
                    {
                        "text": "one knee slightly bent",
                        "composer_text": "both knees bent",
                        "normalized_text": "both knees bent",
                        "promotion_status": "accepted_specialist_bilateral_knee_flexion_candidate",
                        "specialist_owner": "sam3d_v16_bilateral_knee_flexion_specialist",
                    }
                ],
                "bilateral_knee_flexion_adjudication": {
                    "status": "adjudicated",
                    "composer_authoritative": True,
                    "canonical_relation": "both knees bent",
                },
            },
            "visual": {"appearance": []},
        },
        "audit": {"warnings": [], "invariants": {}},
    }

    out = mod.base.apply_character_identity_policy(source)
    body = out["facts"]["body"]
    assert out["schema_version"] == "caption-fact-sheet-0.3"
    assert body["configuration"][0]["composer_text"] == "both knees bent"
    assert body["configuration"][0]["specialist_owner"] == "sam3d_v16_bilateral_knee_flexion_specialist"
    assert body["bilateral_knee_flexion_adjudication"]["canonical_relation"] == "both knees bent"
