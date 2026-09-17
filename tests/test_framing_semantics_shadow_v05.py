from __future__ import annotations

from qwen_caption_validate import framing_semantics_shadow_v05 as mod


def _policy(landmarks: list[str]) -> dict:
    return {
        "image_key": "synthetic",
        "visibility": {
            "observed_landmarks": landmarks,
            "observed_bbox": {"height_fraction": 0.8, "width_fraction": 0.5},
            "extent_hint": "legacy",
            "broad_pose_supported": False,
        },
        "policy": {"mode": "framing_only"},
        "geometry": {
            "configuration_score": 1,
            "configuration_cues": ["visible_arm_relationship"],
            "pose_complexity_score": 0,
        },
        "sam3d": {"available": False, "projected_selected_joint_count": 0},
    }


def _uniface(height_fraction: float) -> dict:
    return {
        "status": "ok",
        "face": {
            "score": 0.99,
            "bbox_xyxy": [100.0, 100.0, 400.0, 100.0 + 1000.0 * height_fraction],
            "selection_strategy": "dwpose_body_membership_head_scale",
        },
    }


def test_clean_head_shoulders_medium_close_survives() -> None:
    policy = _policy([
        "nose", "left_eye", "right_eye", "neck",
        "left_shoulder", "right_shoulder",
    ])
    out = mod.evaluate(policy, uniface=_uniface(0.24), image_size=(1000, 1000))
    assert out["standard_shot_scale"]["label"] == "medium_close_up"


def test_partial_hip_below_head_shoulders_withholds_close_family() -> None:
    policy = _policy([
        "nose", "left_eye", "right_eye", "neck",
        "left_shoulder", "right_shoulder",
        "left_hip",
    ])
    out = mod.evaluate(policy, uniface=_uniface(0.24), image_size=(1000, 1000))
    scale = out["standard_shot_scale"]
    assert out["anatomical_span"]["lower_anchor"] == "shoulders"
    assert out["anatomical_span"]["lower_partial"] == "hips"
    assert scale["status"] == "withheld"
    assert "partial_hip_observation_below_head_shoulders_core" in scale["basis"]
    assert scale["rejected_candidate"]["label"] == "medium_close_up"


def test_noncontiguous_lower_observation_withholds_close_family() -> None:
    policy = _policy([
        "nose", "left_eye", "right_eye", "neck",
        "left_shoulder", "right_shoulder",
        "left_hip",
        "left_knee", "right_knee",
    ])
    out = mod.evaluate(policy, uniface=_uniface(0.24), image_size=(1000, 1000))
    scale = out["standard_shot_scale"]
    assert "knees" in out["anatomical_span"]["noncontiguous_observations"]
    assert scale["status"] == "withheld"
    assert any(x.startswith("noncontiguous_anatomical_observations=") for x in scale["basis"])


def test_head_only_partial_shoulders_can_still_use_extreme_close_up() -> None:
    policy = _policy([
        "nose", "left_eye", "right_eye", "neck",
        "left_shoulder",
    ])
    out = mod.evaluate(policy, uniface=_uniface(0.72), image_size=(1000, 1000))
    assert out["anatomical_span"]["lower_anchor"] == "head"
    assert out["anatomical_span"]["lower_partial"] == "shoulders"
    assert out["standard_shot_scale"]["label"] == "extreme_close_up"
    assert "shown in an extreme close-up" in out["proposed_opening_template"]


def test_extreme_close_up_uses_an_not_a() -> None:
    policy = _policy(["nose", "left_eye", "right_eye", "neck"])
    out = mod.evaluate(policy, uniface=_uniface(0.72), image_size=(1000, 1000))
    assert out["proposed_opening_template"].startswith("[[trigger]] is shown in an extreme close-up")
