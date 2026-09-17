from qwen_caption_validate import crouched_stance_depth_shadow_v01 as mod


def _sheet(*, route="pose_guided", pose="crouching"):
    return {
        "policy": {"mode": route},
        "facts": {
            "body": {
                "pose_candidate": {
                    "composer_text": pose,
                    "normalized_text": pose,
                    "text": pose,
                }
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


def test_00066_like_geometry_qualifies_hips_slightly_lowered():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00066",
        sheet=_sheet(),
        projected=_projected(),
    )

    assert record["status"] == "candidate_hips_slightly_lowered"
    assert record["depth_classification"] == "moderate_lowering"
    assert record["would_qualify_hips_slightly_lowered"] is True
    assert record["lexicalization"]["base_predicate"] == "holds a crouched stance"
    assert (
        record["lexicalization"]["qualified_predicate"]
        == "holds a crouched stance with her hips slightly lowered"
    )
    assert record["aggregate"]["mean_thigh_axis_from_image_down_deg"] == 35.6
    assert record["aggregate"]["mean_knee_flexion_deg"] == 117.35
    assert record["aggregate"]["mean_leg_extension_ratio"] == 0.853


def test_standing_control_is_not_applicable_even_if_geometry_were_passed():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00014",
        sheet=_sheet(pose="standing"),
        projected=_projected(
            pose="standing",
            knee_authority=0.931,
            left_knee=146.0,
            right_knee=162.0,
            left_hip=137.8,
            right_hip=158.0,
            left_thigh=22.6,
            right_thigh=5.3,
            left_ext=0.956,
            right_ext=0.988,
        ),
    )

    assert record["status"] == "not_applicable"
    assert record["reason"] == "canonical_broad_pose_not_crouching"
    assert record["would_qualify_hips_slightly_lowered"] is False


def test_deeper_crouch_keeps_stance_wording_but_withholds_slight_modifier():
    record = mod.evaluate_shadow(
        image_key="deep-control",
        sheet=_sheet(),
        projected=_projected(
            left_knee=76.0,
            right_knee=72.0,
            left_thigh=62.0,
            right_thigh=58.0,
            left_ext=0.66,
            right_ext=0.63,
        ),
    )

    assert record["status"] == "candidate_crouched_stance_only"
    assert record["depth_classification"] == "deeper_or_more_compressed"
    assert record["would_qualify_hips_slightly_lowered"] is False
    assert record["lexicalization"]["base_predicate"] == "holds a crouched stance"


def test_low_authority_keeps_stance_wording_but_withholds_depth_modifier():
    record = mod.evaluate_shadow(
        image_key="low-authority-control",
        sheet=_sheet(),
        projected=_projected(knee_authority=0.60),
    )

    assert record["status"] == "candidate_crouched_stance_only"
    assert record["qualifier_gates"]["sam3d_knee_authority_high"] is False
    assert record["would_qualify_hips_slightly_lowered"] is False


def test_public_pose_disagreement_blocks_depth_modifier():
    record = mod.evaluate_shadow(
        image_key="pose-disagreement-control",
        sheet=_sheet(),
        projected=_projected(pose="standing"),
    )

    assert record["status"] == "candidate_crouched_stance_only"
    assert record["depth_classification"] == "unresolved"
    assert record["reason"] == "sam3d_public_pose_does_not_confirm_crouching"
    assert record["would_qualify_hips_slightly_lowered"] is False


def test_non_pose_route_abstains_before_lexical_projection():
    record = mod.evaluate_shadow(
        image_key="route-control",
        sheet=_sheet(route="framing_only"),
        projected=_projected(),
    )

    assert record["status"] == "route_abstain"
    assert record["reason"] == "route_outside_pose_bearing_scope"
    assert record["would_qualify_hips_slightly_lowered"] is False
