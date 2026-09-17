from __future__ import annotations

from qwen_caption_validate import framing_semantics_shadow_v02 as mod


def _policy(
    landmarks: list[str],
    *,
    legacy_broad: bool = False,
    legacy_mode: str = "framing_only",
    config_score: int = 0,
    config_cues: list[str] | None = None,
    complexity_score: int = 0,
    sam_available: bool = False,
    sam_count: int = 0,
) -> dict:
    return {
        "image_key": "synthetic",
        "visibility": {
            "observed_landmarks": landmarks,
            "observed_bbox": {"height_fraction": 0.8, "width_fraction": 0.6},
            "extent_hint": "legacy",
            "broad_pose_supported": legacy_broad,
        },
        "policy": {"mode": legacy_mode},
        "geometry": {
            "configuration_score": config_score,
            "configuration_cues": list(config_cues or []),
            "pose_complexity_score": complexity_score,
        },
        "sam3d": {
            "available": sam_available,
            "projected_selected_joint_count": sam_count,
        },
    }


def test_00061_close_portrait_keeps_local_configuration_observer() -> None:
    policy = _policy(
        [
            "nose", "neck", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_wrist", "right_elbow",
        ],
        legacy_mode="framing_only",
        config_score=1,
        config_cues=["visible_arm_relationship"],
        complexity_score=2,
    )
    out = mod.evaluate(policy)
    gate = out["pose_gate_shadow"]
    assert gate["broad_pose_supported"] is False
    assert gate["local_configuration_supported"] is True
    assert gate["proposed_mode"] == "configuration"
    assert gate["local_configuration_gate"]["direct_qualifying_cues"] == ["visible_arm_relationship"]


def test_00085_loses_broad_pose_but_keeps_local_arm_configuration() -> None:
    policy = _policy(
        [
            "neck",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_hip", "right_hip", "right_knee",
        ],
        legacy_broad=True,
        legacy_mode="pose_guided",
        config_score=1,
        config_cues=["visible_arm_relationship"],
        complexity_score=3,
        sam_available=True,
        sam_count=12,
    )
    out = mod.evaluate(policy)
    gate = out["pose_gate_shadow"]
    assert gate["broad_pose_supported"] is False
    assert gate["local_configuration_supported"] is True
    assert gate["proposed_mode"] == "configuration"
    assert gate["changed_from_legacy"] is True
    assert out["anatomical_span"]["upper_anchor"] == "shoulders"
    assert out["anatomical_span"]["lower_anchor"] == "hips"


def test_pure_close_portrait_without_local_relationship_stays_framing_only() -> None:
    policy = _policy(
        ["nose", "neck", "left_eye", "right_eye", "left_shoulder", "right_shoulder"],
        legacy_mode="framing_only",
        config_score=0,
        config_cues=[],
    )
    out = mod.evaluate(policy)
    gate = out["pose_gate_shadow"]
    assert gate["broad_pose_supported"] is False
    assert gate["local_configuration_supported"] is False
    assert gate["proposed_mode"] == "framing_only"


def test_existing_multi_cue_configuration_path_is_preserved() -> None:
    policy = _policy(
        ["nose", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        legacy_mode="configuration",
        config_score=2,
        config_cues=["shoulder_line_tilt", "visible_torso_without_leg_support"],
    )
    out = mod.evaluate(policy)
    gate = out["pose_gate_shadow"]
    assert gate["broad_pose_supported"] is False
    assert gate["local_configuration_supported"] is True
    assert gate["proposed_mode"] == "configuration"


def test_00087_bilateral_hips_and_knees_still_allow_broad_pose() -> None:
    policy = _policy(
        [
            "neck", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
        ],
        legacy_broad=True,
        legacy_mode="pose_guided",
        config_score=1,
        config_cues=["visible_arm_relationship"],
        complexity_score=3,
        sam_available=True,
        sam_count=12,
    )
    out = mod.evaluate(policy)
    gate = out["pose_gate_shadow"]
    assert gate["broad_pose_supported"] is True
    assert gate["proposed_mode"] == "pose_guided"
