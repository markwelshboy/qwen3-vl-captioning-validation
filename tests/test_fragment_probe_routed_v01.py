from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "qwen_caption_validate" / "fragment_probe_routed_v01.py"
spec = importlib.util.spec_from_file_location("fragment_probe_routed_v01", MODULE_PATH)
routed = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(routed)


def test_framing_only_never_calls_pose_model():
    spec = routed._route_spec("framing_only")
    assert spec["model_call"] is False
    assert spec["pose_candidate_permitted"] is False
    assert spec["body_relationships_permitted"] is False
    assert spec["sam3d_guidance_permitted"] is False


def test_configuration_structurally_drops_broad_pose_candidate():
    parsed = routed._parse_fact_sheet(
        "POSE CANDIDATE\n- seated\n\nBODY RELATIONSHIPS\n- hands clasped\n"
    )
    extraction, violations = routed._normalize_extraction(parsed, "configuration")
    assert extraction["pose_candidate"] is None
    assert [r["text"] for r in extraction["body_relationships"]] == ["hands clasped"]
    assert "pose_candidate_not_permitted_for_route" in violations


def test_pose_allowed_keeps_candidate_as_hypothesis():
    parsed = routed._parse_fact_sheet(
        "POSE CANDIDATE\n- standing with one leg raised\n\n"
        "BODY RELATIONSHIPS\n- one knee raised\n"
    )
    extraction, violations = routed._normalize_extraction(parsed, "pose_allowed")
    assert extraction["pose_candidate"] == {
        "text": "standing with one leg raised",
        "authority": "hypothesis",
    }
    assert violations == []


def test_sam3d_whitelist_excludes_head_limbs_contacts_and_raw_coordinates():
    diagnostic = {
        "body_camera_relation": {
            "orientation_band": "three_quarter",
            "yaw_deg": -47.24,
            "pitch_deg": 31.0,
        },
        "dwpose_visibility_gate": {"body_yaw_observation_gate": True},
        "face_camera_relation": {"yaw_deg": 55.0},
        "compound_pose_hint": {"head_relation": "turned_toward_camera"},
        "body_frame_landmarks": {"left_knee": [1.0, 2.0, 3.0]},
        "support": {"feet_planted": True},
    }
    facts = routed._sam3d_model_facts(diagnostic)
    assert facts == {
        "torso_camera_orientation": "three_quarter",
        "torso_yaw_magnitude_deg": 47.2,
    }


def test_sam3d_guidance_requires_observed_body_yaw_gate():
    diagnostic = {
        "body_camera_relation": {"orientation_band": "side_on", "yaw_deg": 82.0},
        "dwpose_visibility_gate": {"body_yaw_observation_gate": False},
    }
    assert routed._sam3d_model_facts(diagnostic) == {}


def test_pose_guided_prompt_names_only_whitelisted_sam3d_facts():
    facts = {
        "torso_camera_orientation": "three_quarter",
        "torso_yaw_magnitude_deg": 47.2,
    }
    prompt = routed._effective_prompt("pose_guided", "BASE", "CONFIG", facts)
    assert "torso camera orientation: three quarter" in prompt
    assert "torso yaw magnitude: 47.2 degrees" in prompt
    assert "hidden anatomy" in prompt
    assert "raw mesh" not in prompt
