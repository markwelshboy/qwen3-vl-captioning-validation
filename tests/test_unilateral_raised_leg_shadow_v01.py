from qwen_caption_validate import unilateral_raised_leg_shadow_v01 as mod


def _sheet(route="pose_guided", config=None):
    return {
        "policy": {"mode": route},
        "facts": {
            "body": {
                "configuration": [
                    {"composer_text": text} for text in (config or ["one knee raised"])
                ]
            }
        },
    }


def _points(left_ankle_y=0.5, right_ankle_y=3.0):
    return {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (1.0, 0.0),
        "left_hip": (0.0, 1.0),
        "right_hip": (1.0, 1.0),
        "left_knee": (-0.5, 1.0),
        "right_knee": (1.0, 2.0),
        "left_ankle": (0.0, left_ankle_y),
        "right_ankle": (1.0, right_ankle_y),
    }


def _projected(
    *,
    pose="standing",
    left_knee=57.7,
    right_knee=156.7,
    left_thigh=102.2,
    right_thigh=14.2,
    left_ext=0.484,
    right_ext=0.979,
):
    return {
        "pose": pose,
        "geometry": {
            "asymmetric_lower_body": {
                "per_side": {
                    "left": {
                        "knee_flexion_deg": left_knee,
                        "hip_flexion_deg": 46.9,
                        "thigh_axis_from_image_down_deg": left_thigh,
                        "leg_extension_ratio": left_ext,
                    },
                    "right": {
                        "knee_flexion_deg": right_knee,
                        "hip_flexion_deg": 128.7,
                        "thigh_axis_from_image_down_deg": right_thigh,
                        "leg_extension_ratio": right_ext,
                    },
                }
            }
        },
    }


def test_00064_like_topology_proposes_left_high_raised_knee():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00064",
        sheet=_sheet(),
        projected=_projected(),
        dwpose_points=_points(),
    )
    assert record["status"] == "candidate_would_change"
    assert record["candidate_side"] == "left"
    assert record["proposed_relation"] == "left knee raised high with thigh nearly horizontal"
    assert record["side_evaluations"]["left"]["qualifies"] is True
    assert record["side_evaluations"]["right"]["qualifies"] is False


def test_specific_left_high_relation_is_confirmed_not_duplicated():
    record = mod.evaluate_shadow(
        image_key="x",
        sheet=_sheet(config=["left knee raised high with thigh nearly horizontal"]),
        projected=_projected(),
        dwpose_points=_points(),
    )
    assert record["status"] == "candidate_already_present"
    assert record["would_change"] is False


def test_ordinary_standing_control_does_not_qualify():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00014",
        sheet=_sheet(route="pose_allowed"),
        projected=_projected(
            left_knee=146.0,
            right_knee=162.0,
            left_thigh=22.6,
            right_thigh=5.3,
            left_ext=0.956,
            right_ext=0.988,
        ),
        dwpose_points=_points(left_ankle_y=3.0, right_ankle_y=3.1),
    )
    assert record["status"] == "not_candidate"
    assert record.get("candidate_side") is None


def test_crouching_control_is_outside_standing_specific_rollout():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00066",
        sheet=_sheet(),
        projected=_projected(
            pose="crouching",
            left_knee=123.5,
            right_knee=111.2,
            left_thigh=34.3,
            right_thigh=36.9,
            left_ext=0.881,
            right_ext=0.825,
        ),
        dwpose_points=_points(left_ankle_y=3.0, right_ankle_y=2.8),
    )
    assert record["status"] == "not_candidate"
    assert record["reason"] == "public_pose_not_standing"


def test_requires_substantial_distal_elevation():
    record = mod.evaluate_shadow(
        image_key="x",
        sheet=_sheet(),
        projected=_projected(),
        dwpose_points=_points(left_ankle_y=2.6, right_ankle_y=3.0),
    )
    assert record["status"] == "not_candidate"
    assert record["side_evaluations"]["left"]["gates"]["candidate_distal_leg_substantially_elevated"] is False
