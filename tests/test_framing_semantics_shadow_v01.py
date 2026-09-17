from __future__ import annotations

from qwen_caption_validate import framing_semantics_shadow_v01 as mod


def _policy(
    landmarks: list[str],
    *,
    bbox_h: float = 0.8,
    bbox_w: float = 0.5,
    legacy_extent: str = "legacy",
    legacy_broad: bool = False,
    legacy_mode: str = "configuration",
    config_score: int = 2,
    complexity_score: int = 0,
    sam_available: bool = False,
    sam_count: int = 0,
) -> dict:
    return {
        "image_key": "synthetic",
        "visibility": {
            "observed_landmarks": landmarks,
            "observed_bbox": {"height_fraction": bbox_h, "width_fraction": bbox_w},
            "extent_hint": legacy_extent,
            "broad_pose_supported": legacy_broad,
        },
        "policy": {"mode": legacy_mode},
        "geometry": {
            "configuration_score": config_score,
            "pose_complexity_score": complexity_score,
        },
        "sam3d": {
            "available": sam_available,
            "projected_selected_joint_count": sam_count,
        },
    }


def _fact(pose: str, authoritative: bool = True) -> dict:
    return {
        "facts": {
            "body": {
                "broad_pose_adjudication": {
                    "composer_authoritative": authoritative,
                    "canonical_pose_text": pose,
                }
            }
        }
    }


def test_00085_shape_stops_at_hips_and_withholds_broad_pose() -> None:
    policy = _policy(
        [
            "neck",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "right_knee",
        ],
        legacy_extent="three_quarter_or_long",
        legacy_broad=True,
        legacy_mode="pose_guided",
        config_score=2,
        complexity_score=2,
        sam_available=True,
        sam_count=12,
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["upper_anchor"] == "shoulders"
    assert out["anatomical_span"]["lower_anchor"] == "hips"
    assert out["anatomical_span"]["lower_partial"] == "knees"
    assert out["anatomical_span"]["composer_text"] == "framed from around the shoulders through the hips"
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert out["pose_gate_shadow"]["broad_pose_supported"] is False
    assert out["pose_gate_shadow"]["proposed_mode"] == "configuration"
    assert out["pose_gate_shadow"]["changed_from_legacy"] is True


def test_00087_shape_stops_at_knees_but_keeps_broad_pose() -> None:
    policy = _policy(
        [
            "neck",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle",
        ],
        legacy_extent="full_length",
        legacy_broad=True,
        legacy_mode="pose_guided",
        config_score=2,
        complexity_score=2,
        sam_available=True,
        sam_count=12,
    )
    out = mod.evaluate(policy, _fact("squatting"))
    assert out["anatomical_span"]["upper_anchor"] == "shoulders"
    assert out["anatomical_span"]["lower_anchor"] == "knees"
    assert out["anatomical_span"]["lower_partial"] == "ankles"
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert out["pose_gate_shadow"]["broad_pose_supported"] is True
    assert out["pose_gate_shadow"]["proposed_mode"] == "pose_guided"


def test_complete_standing_body_can_receive_full_body_shot_candidate() -> None:
    policy = _policy(
        [
            "nose", "left_eye", "right_eye", "neck",
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ],
        legacy_extent="full_length",
        legacy_broad=True,
        legacy_mode="pose_allowed",
        config_score=1,
    )
    out = mod.evaluate(policy, _fact("standing"))
    assert out["anatomical_span"]["upper_anchor"] == "head"
    assert out["anatomical_span"]["lower_anchor"] == "ankles"
    assert out["standard_shot_scale"]["label"] == "full_body"
    assert out["pose_gate_shadow"]["broad_pose_supported"] is True


def test_close_up_candidate_requires_head_and_shoulders_without_hips() -> None:
    policy = _policy(
        ["nose", "left_eye", "right_eye", "neck", "left_shoulder", "right_shoulder"],
        bbox_h=0.75,
        bbox_w=0.7,
        config_score=0,
        legacy_mode="framing_only",
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["upper_anchor"] == "head"
    assert out["anatomical_span"]["lower_anchor"] == "shoulders"
    assert out["standard_shot_scale"]["label"] == "close_up"


def test_extreme_close_up_candidate_for_face_dominant_crop() -> None:
    policy = _policy(
        ["nose", "left_eye", "right_eye"],
        bbox_h=0.6,
        bbox_w=0.58,
        config_score=0,
        legacy_mode="framing_only",
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["upper_anchor"] == "head"
    assert out["anatomical_span"]["lower_anchor"] == "head"
    assert out["standard_shot_scale"]["label"] == "extreme_close_up"


def test_medium_candidate_for_head_through_hips_without_knees() -> None:
    policy = _policy(
        [
            "nose", "left_eye", "right_eye", "neck",
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
        ],
        bbox_h=0.7,
        config_score=2,
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["upper_anchor"] == "head"
    assert out["anatomical_span"]["lower_anchor"] == "hips"
    assert out["standard_shot_scale"]["label"] == "medium"


def test_single_outlier_ankle_cannot_extend_core_span() -> None:
    policy = _policy(
        [
            "nose", "left_eye", "right_eye",
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_ankle",
        ],
        config_score=2,
    )
    out = mod.evaluate(policy)
    assert out["anatomical_span"]["lower_anchor"] == "hips"
    assert "ankles" in out["anatomical_span"]["noncontiguous_observations"]
