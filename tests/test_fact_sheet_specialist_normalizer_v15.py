from qwen_caption_validate import fact_sheet_specialist_normalizer_v15 as mod


def _sheet(pose="crouching"):
    return {
        "image_key": "imageblind-01_00066",
        "policy": {"mode": "pose_guided"},
        "facts": {
            "body": {
                "pose_candidate": {
                    "text": pose,
                    "composer_text": pose,
                    "normalized_text": pose,
                    "promotion_status": "accepted_specialist_broad_pose_candidate",
                },
                "configuration": [
                    {
                        "text": "both knees bent",
                        "composer_text": "both knees bent",
                        "normalized_text": "both knees bent",
                    }
                ],
            }
        },
    }


def _projected(
    *,
    pose="crouching",
    knee_authority=0.95,
    left_knee=123.5,
    right_knee=111.2,
    left_hip=75.1,
    right_hip=69.1,
    left_thigh=34.3,
    right_thigh=36.9,
    left_ext=0.881,
    right_ext=0.825,
):
    return {
        "pose": pose,
        "region_support": {"knees": knee_authority},
        "geometry": {
            "asymmetric_lower_body": {
                "per_side": {
                    "left": {
                        "knee_flexion_deg": left_knee,
                        "hip_flexion_deg": left_hip,
                        "thigh_axis_from_image_down_deg": left_thigh,
                        "leg_extension_ratio": left_ext,
                    },
                    "right": {
                        "knee_flexion_deg": right_knee,
                        "hip_flexion_deg": right_hip,
                        "thigh_axis_from_image_down_deg": right_thigh,
                        "leg_extension_ratio": right_ext,
                    },
                }
            }
        },
    }


def test_00066_like_geometry_promotes_typed_depth_fact_without_support_claims():
    out = mod._apply_crouched_stance_depth_authority(
        _sheet(), projected=_projected()
    )
    body = out["facts"]["body"]
    depth = body["crouched_stance_depth"]
    adj = body["crouched_stance_depth_adjudication"]

    assert depth["classification"] == "moderate_lowering"
    assert depth["semantic_relation"] == "hips_slightly_lowered"
    assert depth["composer_text"] == "hips slightly lowered"
    assert depth["promotion_status"].startswith("accepted_specialist_")
    assert adj["status"] == "adjudicated"
    assert adj["composer_authoritative"] is True
    assert adj["broad_pose_created"] is False
    assert adj["support_contact_claim_created"] is False
    assert adj["support_side_claim_created"] is False


def test_standing_pose_cannot_gain_depth_fact_even_with_00066_geometry():
    out = mod._apply_crouched_stance_depth_authority(
        _sheet(pose="standing"), projected=_projected()
    )
    body = out["facts"]["body"]

    assert "crouched_stance_depth" not in body
    assert body["crouched_stance_depth_adjudication"]["status"] == "not_applicable"


def test_deep_geometry_withholds_slight_lowering_modifier():
    out = mod._apply_crouched_stance_depth_authority(
        _sheet(),
        projected=_projected(
            left_knee=76.0,
            right_knee=72.0,
            left_thigh=62.0,
            right_thigh=58.0,
            left_ext=0.66,
            right_ext=0.63,
        ),
    )
    body = out["facts"]["body"]

    assert "crouched_stance_depth" not in body
    adj = body["crouched_stance_depth_adjudication"]
    assert adj["status"] == "confirmed_crouched_stance_without_depth_modifier"
    assert adj["composer_authoritative"] is False


def test_low_knee_authority_withholds_depth_modifier():
    out = mod._apply_crouched_stance_depth_authority(
        _sheet(), projected=_projected(knee_authority=0.60)
    )
    body = out["facts"]["body"]

    assert "crouched_stance_depth" not in body
    assert body["crouched_stance_depth_adjudication"]["composer_authoritative"] is False
