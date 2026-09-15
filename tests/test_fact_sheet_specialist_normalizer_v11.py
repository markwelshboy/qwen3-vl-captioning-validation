from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v11 as mod


def _bound(text, relation, side, authority):
    return {
        "text": text,
        "composer_text": f"{side} foot {'slightly lifted' if relation == 'foot_lifted' else 'planted'}",
        "promotion_status": "accepted_specialist_lateralized_candidate",
        "specialist_owner": "dwpose_anatomical_relation_binding",
        "laterality_binding": {
            "anatomical_side": side,
            "semantic_relation": relation,
            "authority": authority,
        },
    }


def test_defaults():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.10"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.10"


def test_00014_like_ankle_height_pair_is_withheld_not_reinterpreted():
    weak = mod.ANKLE_HEIGHT_ONLY_AUTHORITY
    sheet = {
        "facts": {
            "body": {
                "configuration": [
                    {"text": "both hands holding visible fabric", "composer_text": "both hands holding visible fabric"},
                    _bound("one foot planted", "foot_planted", "left", weak),
                    _bound("other foot slightly lifted off floor", "foot_lifted", "right", weak),
                ],
                "support_geometry": {
                    "available": True,
                    "composer_eligible": True,
                    "overall_shape": "mostly_upright_over_support_leg",
                    "support_side": "left",
                    "elevated_side": "right",
                    "elevated_relation": "foot_lifted",
                },
                "anatomical_laterality": {
                    "relation_bindings": [
                        {"semantic_relation": "foot_planted", "authority": weak},
                        {"semantic_relation": "foot_lifted", "authority": weak},
                    ]
                },
            }
        }
    }
    out = mod._apply_support_contact_truth_gate(sheet)
    body = out["facts"]["body"]
    assert [item["composer_text"] for item in body["configuration"]] == ["both hands holding visible fabric"]
    adjudication = body["support_contact_adjudication"]
    assert adjudication["status"] == "withheld_unverified_ground_contact"
    assert adjudication["applied"] is True
    assert len(adjudication["removed_configuration_relations"]) == 2
    assert body["support_geometry"]["composer_eligible"] is False
    assert body["support_geometry"]["publication_status"] == "withheld_unverified_ground_contact"
    assert all(
        binding["publication_status"] == "withheld_unverified_ground_contact"
        for binding in body["anatomical_laterality"]["relation_bindings"]
    )


def test_00066_like_crouch_keeps_pose_and_torso_but_drops_weak_foot_pair():
    weak = mod.ANKLE_HEIGHT_ONLY_AUTHORITY
    sheet = {
        "facts": {
            "body": {
                "pose_candidate": {"text": "crouching", "composer_text": "crouching"},
                "broad_pose_adjudication": {"canonical_pose_text": "crouching", "composer_authoritative": True},
                "torso_relation_adjudication": {"status": "supported", "composer_authoritative": True},
                "configuration": [
                    {"text": "torso bent forward", "composer_text": "upper body bent forward from the hips"},
                    {"text": "one knee slightly bent", "composer_text": "one knee slightly bent"},
                    _bound("one foot planted", "foot_planted", "left", weak),
                    _bound("other foot slightly lifted", "foot_lifted", "right", weak),
                ],
            }
        }
    }
    out = mod._apply_support_contact_truth_gate(sheet)
    body = out["facts"]["body"]
    assert body["pose_candidate"]["composer_text"] == "crouching"
    assert body["broad_pose_adjudication"]["canonical_pose_text"] == "crouching"
    assert body["torso_relation_adjudication"]["status"] == "supported"
    assert [item["composer_text"] for item in body["configuration"]] == [
        "upper body bent forward from the hips",
        "one knee slightly bent",
    ]


def test_00064_like_raised_knee_support_binding_is_preserved():
    sheet = {
        "facts": {
            "body": {
                "configuration": [
                    {
                        "text": "one knee raised",
                        "composer_text": "left knee raised high",
                        "laterality_binding": {
                            "anatomical_side": "left",
                            "semantic_relation": "knee_raised",
                            "authority": "dwpose_bilateral_thigh_angle_from_vertical",
                        },
                    },
                    _bound(
                        "one foot planted",
                        "foot_planted",
                        "right",
                        "opposite_of_dwpose_bound_raised_knee_with_observed_leg_chain",
                    ),
                ]
            }
        }
    }
    out = mod._apply_support_contact_truth_gate(sheet)
    body = out["facts"]["body"]
    assert [item["composer_text"] for item in body["configuration"]] == [
        "left knee raised high",
        "right foot planted",
    ]
    assert body["support_contact_adjudication"]["status"] == "not_applicable"


def test_future_stronger_ground_contact_specialist_is_not_silently_suppressed():
    sheet = {
        "facts": {
            "body": {
                "configuration": [
                    _bound("other foot lifted", "foot_lifted", "right", "explicit_visible_ground_clearance_specialist")
                ]
            }
        }
    }
    out = mod._apply_support_contact_truth_gate(sheet)
    body = out["facts"]["body"]
    assert len(body["configuration"]) == 1
    assert body["support_contact_adjudication"]["status"] == "defer_to_stronger_specialist"
    assert body["support_contact_adjudication"]["applied"] is False
