from __future__ import annotations

from qwen_caption_validate import framing_semantics_shadow_v03 as mod


def _policy(landmarks: list[str], *, bbox_h: float = 0.8, bbox_w: float = 0.5) -> dict:
    return {
        "image_key": "synthetic",
        "visibility": {
            "observed_landmarks": landmarks,
            "observed_bbox": {"height_fraction": bbox_h, "width_fraction": bbox_w},
            "extent_hint": "legacy",
            "broad_pose_supported": False,
        },
        "policy": {"mode": "configuration"},
        "geometry": {
            "configuration_score": 1,
            "configuration_cues": ["visible_arm_relationship"],
            "pose_complexity_score": 0,
        },
        "sam3d": {"available": False, "projected_selected_joint_count": 0},
    }


def test_noncontiguous_knees_cannot_turn_head_shoulders_crop_into_medium_wide() -> None:
    policy = _policy(
        [
            "nose", "left_eye", "right_eye", "neck",
            "left_shoulder", "right_shoulder",
            "left_knee", "right_knee",
        ],
        bbox_h=0.55,
        bbox_w=0.40,
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["upper_anchor"] == "head"
    assert out["anatomical_span"]["lower_anchor"] == "shoulders"
    assert "knees" in out["anatomical_span"]["noncontiguous_observations"]
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert out["standard_shot_scale"]["rejected_candidate"]["label"] == "medium_wide"


def test_head_shoulders_close_up_remains_valid() -> None:
    policy = _policy(
        ["nose", "left_eye", "right_eye", "neck", "left_shoulder", "right_shoulder"],
        bbox_h=0.72,
        bbox_w=0.70,
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["lower_anchor"] == "shoulders"
    assert out["standard_shot_scale"]["label"] == "close_up"
    assert out["standard_shot_scale"]["span_coherent"] is True


def test_head_hips_medium_remains_valid() -> None:
    policy = _policy(
        [
            "nose", "left_eye", "right_eye", "neck",
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
        ],
        bbox_h=0.72,
        bbox_w=0.5,
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["lower_anchor"] == "hips"
    assert out["standard_shot_scale"]["label"] == "medium"


def test_head_knees_medium_wide_remains_valid() -> None:
    policy = _policy(
        [
            "nose", "left_eye", "right_eye", "neck",
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
        ],
        bbox_h=0.8,
        bbox_w=0.45,
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["lower_anchor"] == "knees"
    assert out["standard_shot_scale"]["label"] == "medium_wide"
