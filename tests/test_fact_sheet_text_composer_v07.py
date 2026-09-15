from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v07 as mod


def test_phase56_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.7"
    assert mod.DEFAULT_PROMPT.name == "fact_sheet_text_composer_v06.txt"


def test_articulated_relative_orientation_marks_upper_torso_closer_to_frontal():
    torso = {
        "mode": "articulated",
        "body_orientation": {
            "camera_orientation": "oblique",
            "approx_yaw_deg": 35,
            "turn_direction": "frame_left",
        },
        "upper_torso_orientation": {
            "camera_orientation": "slightly_angled",
            "approx_yaw_deg": 20,
            "turn_direction": "frame_left",
        },
        "relative_twist_magnitude_deg": 13.5,
    }
    result = mod._articulated_relative_orientation(torso)
    assert result is not None
    assert result["relationship"] == "upper_torso_closer_to_frontal"
    assert result["body_turn_direction"] == "frame_left"
    assert result["upper_torso_turn_direction"] == "frame_left"
    assert result["relative_twist_magnitude_deg"] == 13.5
    assert "closer to frontal" in result["composer_relation"]
    assert "less turned" in result["composer_relation"]


def test_unsigned_torso_frame_direction_is_audit_violation():
    projection = {
        "authoritative_facts": {
            "body": {
                "broad_pose": "standing",
                "torso_orientation": {
                    "camera_orientation": "frontal",
                    "yaw_magnitude_deg": 13.0,
                    "approx_yaw_deg": 15,
                },
            }
        },
        "subject": {},
        "omitted_review_conflict_domains": [],
    }
    caption = "She stands upright. Her torso is angled about 15 degrees from frontal, with a subtle turn toward frame left."
    audit = mod._caption_audit(caption, projection)
    assert "unauthorized_torso_turn_direction:frame_left" in audit["violations"]


def test_signed_torso_frame_direction_is_allowed():
    projection = {
        "authoritative_facts": {
            "body": {
                "broad_pose": "standing",
                "torso_orientation": {
                    "camera_orientation": "three_quarter",
                    "yaw_magnitude_deg": 49.1,
                    "approx_yaw_deg": 50,
                    "turn_direction": "frame_left",
                },
            }
        },
        "subject": {},
        "omitted_review_conflict_domains": [],
    }
    caption = "She stands with her torso turned about 50 degrees toward frame left."
    audit = mod._caption_audit(caption, projection)
    assert not any(v.startswith("unauthorized_torso_turn_direction") for v in audit["violations"])


def test_reversed_articulated_relationship_is_audit_violation():
    projection = {
        "authoritative_facts": {
            "body": {
                "broad_pose": "crouching",
                "torso_orientation": {
                    "mode": "articulated",
                    "body_orientation": {
                        "camera_orientation": "oblique",
                        "approx_yaw_deg": 35,
                        "turn_direction": "frame_left",
                    },
                    "upper_torso_orientation": {
                        "camera_orientation": "slightly_angled",
                        "approx_yaw_deg": 20,
                        "turn_direction": "frame_left",
                    },
                    "articulated_relative_orientation": {
                        "relationship": "upper_torso_closer_to_frontal",
                        "relative_twist_magnitude_deg": 13.5,
                    },
                },
            }
        },
        "subject": {},
        "omitted_review_conflict_domains": [],
    }
    caption = "She crouches. Her torso turns toward frame left, while her upper torso turns slightly more toward frame left."
    audit = mod._caption_audit(caption, projection)
    assert "articulated_torso_relationship_reversed" in audit["violations"]


def test_retry_prompt_is_surgical_for_missing_gaze_and_unsigned_torso():
    prompt = mod._retry_prompt(
        "BASE PROMPT",
        "She stands with her torso toward frame left, though her gaze is not specified.",
        [
            "gaze_language_without_publishable_gaze",
            "unauthorized_torso_turn_direction:frame_left",
        ],
    )
    assert "Remove every statement about gaze" in prompt
    assert "Remove any frame-left/frame-right torso direction" in prompt
    assert "preserving all valid detail" in prompt
    assert "Return only the revised caption" in prompt
