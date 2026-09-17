from __future__ import annotations

from qwen_caption_validate import framing_semantics_shadow_v04 as mod


def _policy(landmarks: list[str], *, bbox_h: float = 0.8, bbox_w: float = 0.5) -> dict:
    return {
        "image_key": "synthetic",
        "visibility": {
            "observed_landmarks": landmarks,
            "observed_bbox": {"height_fraction": bbox_h, "width_fraction": bbox_w},
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


def _uniface_for_height(height_fraction: float, *, width_fraction: float = 0.30) -> dict:
    width = height = 1000
    fw = width * width_fraction
    fh = height * height_fraction
    return {
        "status": "ok",
        "image_key": "synthetic",
        "face": {
            "score": 0.99,
            "bbox_xyxy": [100.0, 100.0, 100.0 + fw, 100.0 + fh],
            "selection_strategy": "dwpose_body_membership_head_scale",
        },
    }


def _head_shoulders_policy() -> dict:
    return _policy([
        "nose", "left_eye", "right_eye", "neck",
        "left_shoulder", "right_shoulder",
        "right_elbow",
    ])


def _head_only_policy() -> dict:
    return _policy(["nose", "left_eye", "right_eye", "neck"])


def test_head_shoulders_uses_uniface_face_height_for_medium_close_up() -> None:
    out = mod.evaluate(
        _head_shoulders_policy(),
        uniface=_uniface_for_height(0.24),
        image_size=(1000, 1000),
    )
    assert out["anatomical_span"]["upper_anchor"] == "head"
    assert out["anatomical_span"]["lower_anchor"] == "shoulders"
    assert out["standard_shot_scale"]["label"] == "medium_close_up"
    assert out["face_scale_geometry"]["height_fraction"] == 0.24
    assert "target_bound_uniface_retinaface_height_fraction" in out["standard_shot_scale"]["basis"]


def test_head_shoulders_uses_uniface_face_height_for_close_up() -> None:
    out = mod.evaluate(
        _head_shoulders_policy(),
        uniface=_uniface_for_height(0.343),
        image_size=(1000, 1000),
    )
    assert out["standard_shot_scale"]["label"] == "close_up"
    assert out["standard_shot_scale"]["face_scale_authority"] == "observed_face_detector"


def test_head_shoulders_abstains_inside_calibration_gap() -> None:
    out = mod.evaluate(
        _head_shoulders_policy(),
        uniface=_uniface_for_height(0.285),
        image_size=(1000, 1000),
    )
    scale = out["standard_shot_scale"]
    assert scale["status"] == "withheld"
    assert "head_shoulders_face_height_in_abstention_band" in scale["basis"]


def test_head_only_close_and_extreme_are_separated_by_face_height() -> None:
    close = mod.evaluate(
        _head_only_policy(),
        uniface=_uniface_for_height(0.60),
        image_size=(1000, 1000),
    )
    extreme = mod.evaluate(
        _head_only_policy(),
        uniface=_uniface_for_height(0.72),
        image_size=(1000, 1000),
    )
    assert close["anatomical_span"]["lower_anchor"] == "head"
    assert close["standard_shot_scale"]["label"] == "close_up"
    assert extreme["standard_shot_scale"]["label"] == "extreme_close_up"


def test_head_only_abstains_inside_calibration_gap() -> None:
    out = mod.evaluate(
        _head_only_policy(),
        uniface=_uniface_for_height(0.66),
        image_size=(1000, 1000),
    )
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert "head_only_face_height_in_abstention_band" in out["standard_shot_scale"]["basis"]


def test_no_face_withholds_close_family_scale() -> None:
    out = mod.evaluate(
        _head_shoulders_policy(),
        uniface={"status": "no_face"},
        image_size=(1000, 1000),
    )
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert "target_bound_uniface_face_unavailable" in out["standard_shot_scale"]["basis"]
    assert out["face_scale_geometry"]["reason"] == "uniface_face_unavailable"


def test_non_close_span_keeps_v03_scale_logic() -> None:
    policy = _policy([
        "nose", "left_eye", "right_eye", "neck",
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
    ], bbox_h=0.72, bbox_w=0.50)
    out = mod.evaluate(
        policy,
        uniface=_uniface_for_height(0.15),
        image_size=(1000, 1000),
    )
    assert out["anatomical_span"]["lower_anchor"] == "hips"
    assert out["standard_shot_scale"]["label"] == "medium"


def test_v02_local_configuration_routing_survives_v04() -> None:
    out = mod.evaluate(
        _head_shoulders_policy(),
        uniface=_uniface_for_height(0.34),
        image_size=(1000, 1000),
    )
    gate = out["pose_gate_shadow"]
    assert gate["broad_pose_supported"] is False
    assert gate["local_configuration_supported"] is True
    assert gate["proposed_mode"] == "configuration"
