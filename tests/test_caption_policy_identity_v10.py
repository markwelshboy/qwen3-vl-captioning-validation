from qwen_caption_validate import caption_policy_identity_v10 as identity


def test_identity_policy_preserves_phase4b16_framing_and_head_support():
    source = {
        "schema_version": "caption-fact-sheet-0.2.16",
        "status": "ok",
        "image_key": "imageblind-01_00061",
        "facts": {
            "framing": {
                "authority": "deterministic_observation",
                "anatomical_span": {
                    "upper_anchor": "head",
                    "lower_anchor": "shoulders",
                },
                "standard_shot_scale": {
                    "status": "candidate",
                    "label": "medium_close_up",
                    "composer_text": "medium close-up",
                },
                "composer_framing": {
                    "source": "standard_shot_scale",
                    "composer_text": "medium close-up",
                },
            },
            "body": {
                "configuration": [
                    {
                        "composer_text": "chin resting on the left fist, with the forearm beneath/supporting the pose"
                    }
                ],
                "head_support_adjudication": {
                    "status": "adjudicated",
                    "composer_authoritative": True,
                    "laterality_binding": {
                        "status": "bound",
                        "anatomical_side": "left",
                    },
                },
            },
            "visual": {"appearance": []},
        },
        "audit": {"warnings": [], "invariants": {}},
    }

    out = identity._apply_identity_policy(source)

    assert out["schema_version"] == "caption-fact-sheet-0.3.1"
    assert out["facts"]["framing"] == source["facts"]["framing"]
    assert out["facts"]["body"]["head_support_adjudication"] == source["facts"]["body"]["head_support_adjudication"]
    assert out["facts"]["body"]["configuration"] == source["facts"]["body"]["configuration"]
    assert out["audit"]["invariants"]["phase4b16_production_framing_is_preserved"] is True
    assert out["audit"]["invariants"]["head_support_adjudication_is_preserved"] is True
