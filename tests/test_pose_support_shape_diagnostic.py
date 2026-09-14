from __future__ import annotations

from qwen_caption_validate import pose_support_shape_diagnostic as mod


def _policy() -> dict:
    return {
        "image_size": [400, 400],
        "policy": {"mode": "pose_guided"},
        "pose_relevance": "high",
        "sources": {"dwpose": "/tmp/example.dwpose.json"},
        "_source_path": "/tmp/example.perception_policy.json",
    }


def test_leg_metrics_separate_knee_flexion_from_global_vertical_extent():
    torso = 100.0
    straight = {
        "right_hip": (0.0, 0.0),
        "right_knee": (0.0, 100.0),
        "right_ankle": (0.0, 200.0),
    }
    bent_same_extent = {
        "right_hip": (0.0, 0.0),
        "right_knee": (80.0, 100.0),
        "right_ankle": (0.0, 200.0),
    }

    straight_metrics = mod._leg_metrics(straight, "right", torso)
    bent_metrics = mod._leg_metrics(bent_same_extent, "right", torso)

    assert straight_metrics["knee_angle_deg"] == 180.0
    assert bent_metrics["knee_angle_deg"] < 120.0
    assert straight_metrics["hip_to_ankle_vertical_drop_over_torso"] == 2.0
    assert bent_metrics["hip_to_ankle_vertical_drop_over_torso"] == 2.0
    assert bent_metrics["chain_straightness_euclidean_over_path"] < straight_metrics["chain_straightness_euclidean_over_path"]


def test_low_pelvis_reduces_vertical_drop_over_torso_even_with_same_torso_scale():
    upright = {
        "right_hip": (0.0, 0.0),
        "right_knee": (0.0, 100.0),
        "right_ankle": (0.0, 200.0),
    }
    compressed = {
        "right_hip": (0.0, 100.0),
        "right_knee": (60.0, 150.0),
        "right_ankle": (0.0, 200.0),
    }

    upright_metrics = mod._leg_metrics(upright, "right", 100.0)
    compressed_metrics = mod._leg_metrics(compressed, "right", 100.0)

    assert upright_metrics["hip_to_ankle_vertical_drop_over_torso"] == 2.0
    assert compressed_metrics["hip_to_ankle_vertical_drop_over_torso"] == 1.0
    assert compressed_metrics["knee_angle_deg"] < upright_metrics["knee_angle_deg"]


def test_build_diagnostic_is_threshold_free(monkeypatch):
    points = {
        "left_shoulder": (100.0, 50.0),
        "right_shoulder": (200.0, 50.0),
        "left_hip": (110.0, 150.0),
        "right_hip": (190.0, 150.0),
        "left_knee": (110.0, 240.0),
        "right_knee": (250.0, 220.0),
        "left_ankle": (110.0, 340.0),
        "right_ankle": (190.0, 340.0),
    }
    monkeypatch.setattr(mod, "_dwpose_points", lambda record, width, height: points)

    record = mod.build_diagnostic(image_key="example", policy=_policy(), dwpose_record={})

    assert record["route_mode"] == "pose_guided"
    assert record["interpretation"]["classification"] is None
    assert record["interpretation"]["thresholds_applied"] is False
    assert record["global"]["lower_ankle_side_in_frame"] in {"left", "right"}
    assert record["global"]["pelvis_mid_to_lower_ankle_vertical_drop_over_torso"] is not None
    assert record["legs"]["right"]["knee_angle_deg"] < record["legs"]["left"]["knee_angle_deg"]
