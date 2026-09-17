from qwen_caption_validate import raised_leg_support_shadow_v01 as mod


def test_direct_left_raised_right_support_binding():
    sheet = {
        "route": "pose_guided",
        "body": {"configuration": ["one knee raised"]},
        "analysis": {
            "dwpose_named_joints": {
                "left_hip": {"x": 0.45, "y": 0.55, "accepted": True, "in_frame": True},
                "left_knee": {"x": 0.48, "y": 0.30, "accepted": True, "in_frame": True},
                "left_ankle": {"x": 0.58, "y": 0.54, "accepted": True, "in_frame": True},
                "right_hip": {"x": 0.55, "y": 0.56, "accepted": True, "in_frame": True},
                "right_knee": {"x": 0.56, "y": 0.76, "accepted": True, "in_frame": True},
                "right_ankle": {"x": 0.57, "y": 0.96, "accepted": True, "in_frame": True},
            },
            "sam3d": {
                "left_knee_angle_deg": 58.0,
                "right_knee_angle_deg": 157.0,
                "region_support": {"knees": 0.41},
            },
        },
    }
    result = mod.evaluate_sheet(sheet)
    assert result["decision"] == "candidate_would_bind"
    assert result["raised_side"] == "left"
    assert result["support_side"] == "right"
    assert result["raised_high"] is True


def test_no_opposite_side_inference_without_both_direct_leg_chains():
    sheet = {
        "route": "pose_guided",
        "body": {"configuration": ["one knee raised"]},
        "analysis": {
            "dwpose_named_joints": {
                "left_hip": [0.45, 0.55],
                "left_knee": [0.48, 0.30],
                "left_ankle": [0.58, 0.54],
                "right_hip": [0.55, 0.56],
                "right_knee": [0.56, 0.76],
            }
        },
    }
    result = mod.evaluate_sheet(sheet)
    assert result["decision"] == "abstain"
    assert result["reason"] == "both_full_dwpose_leg_chains_not_observed"


def test_non_pose_route_abstains():
    sheet = {
        "route": "framing_only",
        "body": {"configuration": ["one knee raised"]},
    }
    result = mod.evaluate_sheet(sheet)
    assert result["decision"] == "abstain"
    assert result["reason"] == "route_not_pose_bearing"


def test_no_raised_leg_semantic_candidate_abstains():
    sheet = {
        "route": "pose_guided",
        "body": {"configuration": ["both knees bent"]},
    }
    result = mod.evaluate_sheet(sheet)
    assert result["decision"] == "abstain"
    assert result["reason"] == "no_unilateral_raised_leg_semantic_candidate"
