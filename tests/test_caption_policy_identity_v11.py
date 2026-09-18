from qwen_caption_validate import caption_policy_identity_v11 as identity


def test_identity_policy_v11_preserves_capture_framing_and_head_support():
    source = {
        "schema_version": "caption-fact-sheet-0.2.17",
        "status": "ok",
        "image_key": "imageblind-01_00031",
        "facts": {
            "capture": {
                "available": True,
                "family": "selfie",
                "subtype": "mirror_selfie",
                "composer_text": "mirror selfie",
                "spatial_surface_mode": "depicted_frame",
                "composer_spatial_contract": {
                    "directional_reference_system": "frame_relative_only",
                    "anatomical_laterality": "withhold",
                    "physical_reflection_laterality": "withhold",
                },
            },
            "framing": {
                "authority": "deterministic_observation",
                "anatomical_span": {
                    "upper_anchor": "head",
                    "lower_anchor": "hips",
                },
                "standard_shot_scale": {
                    "status": "candidate",
                    "label": "medium",
                    "composer_text": "medium shot",
                },
                "composer_framing": {
                    "source": "standard_shot_scale",
                    "composer_text": "medium shot",
                },
            },
            "body": {
                "configuration": [
                    {
                        "composer_text": "right hand holding a phone",
                    }
                ],
                "head_support_adjudication": {
                    "status": "not_applicable",
                    "composer_authoritative": False,
                },
            },
            "visual": {"appearance": []},
        },
        "audit": {"warnings": [], "invariants": {}},
    }

    out = identity._apply_identity_policy(source)

    assert out["schema_version"] == "caption-fact-sheet-0.3.2"
    assert out["facts"]["capture"] == source["facts"]["capture"]
    assert out["facts"]["framing"] == source["facts"]["framing"]
    assert out["facts"]["body"]["configuration"] == source["facts"]["body"]["configuration"]
    assert out["audit"]["invariants"]["phase4b17_capture_style_is_preserved"] is True
    assert out["audit"]["invariants"]["mirror_selfie_surface_policy_is_preserved"] is True
