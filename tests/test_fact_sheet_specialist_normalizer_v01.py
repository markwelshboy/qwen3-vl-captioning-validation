from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "qwen_caption_validate" / "fact_sheet_specialist_normalizer_v01.py"
spec = importlib.util.spec_from_file_location("fact_sheet_specialist_normalizer_v01", MODULE_PATH)
facts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(facts)


def _phase4a(mode="configuration", pose=None, config=None, appearance=None):
    return {
        "schema_version": "caption-fact-sheet-0.1", "status": "ok", "image_key": "sample",
        "policy": {"mode": mode, "pose_relevance": "low"}, "sources": {},
        "facts": {
            "body": {
                "pose_candidate": ({"text": pose, "authority": "qwen_pose_hypothesis", "promotion_status": "candidate"} if pose else None),
                "configuration": [{"text": x, "authority": "route_scoped_candidate", "promotion_status": "pending_phase_4b"} for x in (config or [])],
            },
            "visual": {"appearance": appearance or [], "objects": [], "scene": [], "secondary_people": []},
        },
        "context_only": {}, "reserved_domains": {},
        "audit": {"violations": [], "warnings": [], "caption_ready": False, "phase": "4A"},
    }


def _policy(mode="configuration", extent="close_or_medium_close"):
    return {
        "image_key": "sample", "image_size": [1000, 1000], "sources": {}, "policy": {"mode": mode},
        "visibility": {
            "extent_hint": extent, "head": "strong", "shoulders": "strong", "torso": "partial",
            "hips": "absent", "knees": "absent", "feet": "absent", "broad_pose_supported": False,
            "observed_bbox": {"width_fraction": 0.4, "height_fraction": 0.6},
        },
    }


def _laterality():
    return facts._laterality_from_points({"left_shoulder": (1, 1), "right_shoulder": (2, 1), "right_wrist": (2, 3)}, "dwpose.json")


def _torso(available=False):
    return {
        "available": available, "source": "sam3d.npz" if available else None,
        "authority": "sam3d_reconstruction_observation_gated" if available else "unavailable",
        "torso_camera_orientation": "three_quarter" if available else None,
        "torso_yaw_magnitude_deg": 48.0 if available else None,
        "reconstruction_is_observation": False, "can_promote_broad_pose": False,
    }


def _build(phase4a, policy, head_gaze):
    return facts._build_fact_sheet(
        phase4a, policy, head_gaze,
        phase4a_path=Path("phase4a.json"), policy_path=Path("policy.json"), head_gaze_path=Path("head.json"),
        laterality=_laterality(), torso_geometry=_torso(False),
    )


def test_sunglasses_null_gaze_without_nulling_head():
    record = {
        "status": "ok",
        "head": {
            "available": True, "primary_source": "uniface_resnet50", "yaw_deg": -2.25, "pitch_deg": -9.73,
            "frame_horizontal": "center", "vertical": "center", "yaw_strength": "frontal", "authority": "corroborated",
            "axis_authority": {"yaw": {"authority": "corroborated"}, "pitch": {"authority": "corroborated"}, "roll": {"authority": "corroborated"}},
        },
        "gaze_observability": {"eligible": False, "authority": "unavailable", "reasons": ["sunglasses_obscure_eye_gaze"]},
        "gaze": {"available": False, "publishable": False, "primary_source": "l2cs", "reason": "eye_gaze_not_observable"},
    }
    sheet = _build(_phase4a(mode="framing_only"), _policy(mode="framing_only"), record)
    assert sheet["facts"]["head_pose"]["horizontal"]["value"] == "center"
    assert sheet["facts"]["head_pose"]["vertical"]["value"] == "center"
    assert sheet["facts"]["gaze"]["publishable"] is False
    assert sheet["facts"]["gaze"]["reason"] == "eye_gaze_not_observable"
    assert "horizontal" not in sheet["facts"]["gaze"]


def test_head_axis_authority_is_independent_but_gaze_can_publish():
    record = {
        "status": "ok",
        "head": {
            "available": True, "primary_source": "uniface_resnet50", "yaw_deg": -45.7, "pitch_deg": -35.5,
            "frame_horizontal": "frame_right", "vertical": "down", "yaw_strength": "strong_turn", "authority": "reduced",
            "axis_authority": {
                "yaw": {"authority": "corroborated", "abs_error_deg": 4.0, "primary_class": "frame_right", "secondary_class": "frame_right"},
                "pitch": {"authority": "reduced", "abs_error_deg": 27.8, "primary_class": "down", "secondary_class": "center"},
                "roll": {"authority": "reduced"},
            },
        },
        "gaze_observability": {"eligible": True, "authority": "high", "reasons": [], "warnings": []},
        "gaze": {"available": True, "publishable": True, "primary_source": "l2cs", "authority": "high", "horizontal": "frame_right", "vertical": "down", "camera_relationship": "off_camera"},
    }
    sheet = _build(_phase4a(mode="framing_only"), _policy(mode="framing_only"), record)
    head = sheet["facts"]["head_pose"]
    assert head["horizontal"]["value"] == "frame_right"
    assert head["vertical"]["value"] is None
    assert head["vertical"]["candidate_value"] == "down"
    assert sheet["facts"]["gaze"]["horizontal"] == "frame_right"
    assert sheet["facts"]["gaze"]["vertical"] == "down"
    assert sheet["facts"]["gaze"]["camera_relationship"] == "off_camera"


def test_qwen_anatomical_laterality_is_neutralized_not_swapped():
    sheet = _build(_phase4a(config=["right hand holding phone near face", "left arm bent across torso"]), _policy(), {"status": "missing"})
    config = sheet["facts"]["body"]["configuration"]
    assert config[0]["normalized_text"] == "hand holding phone near face"
    assert config[1]["normalized_text"] == "arm bent across torso"
    assert all(x["promotion_status"] == "accepted_unlateralized_candidate" for x in config)
    assert sheet["facts"]["body"]["anatomical_laterality"]["sides"]["right"]["visible_joints"] == ["shoulder", "wrist"]


def test_visual_accessory_side_is_removed_but_semantic_survives():
    appearance = [{"text": "pink smartwatch on left wrist", "promotion_status": "held_for_laterality_normalization", "authority": "qwen_semantic_candidate"}]
    sheet = _build(_phase4a(appearance=appearance), _policy(), {"status": "missing"})
    item = sheet["facts"]["visual"]["appearance"][0]
    assert item["normalized_text"] == "pink smartwatch on wrist"
    assert item["composer_text"] == "pink smartwatch on wrist"


def test_head_orientation_leakage_is_held():
    sheet = _build(_phase4a(config=["neck tilted slightly forward"]), _policy(), {"status": "missing"})
    item = sheet["facts"]["body"]["configuration"][0]
    assert item["promotion_status"] == "held_for_head_pose_normalization"
    assert item["composer_text"] is None


def test_framing_and_pose_hypothesis_remain_separate_from_specialists():
    sheet = facts._build_fact_sheet(
        _phase4a(mode="pose_guided", pose="standing with one leg raised", config=["one knee raised"]),
        _policy(mode="pose_guided", extent="full_length"), {"status": "missing"},
        phase4a_path=Path("phase4a.json"), policy_path=Path("policy.json"), head_gaze_path=None,
        laterality=_laterality(), torso_geometry=_torso(True),
    )
    assert sheet["facts"]["framing"]["extent"] == "full_length"
    assert sheet["facts"]["body"]["pose_candidate"]["authority"] == "qwen_pose_hypothesis"
    assert sheet["facts"]["body"]["torso_geometry"]["torso_camera_orientation"] == "three_quarter"
    assert sheet["facts"]["body"]["torso_geometry"]["can_promote_broad_pose"] is False
    assert sheet["audit"]["phase"] == "4B"
    assert sheet["audit"]["caption_ready"] is False
