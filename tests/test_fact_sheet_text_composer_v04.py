from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v04 as mod


def test_phase53_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.4"
    assert mod.EXPECTED_FACT_SHEET_SCHEMA == "caption-fact-sheet-0.3"
    assert mod.SCHEMA_VERSION == "fact-sheet-text-composer-0.4"
    assert mod.DEFAULT_PROMPT.name == "fact_sheet_text_composer_v03.txt"


def test_signed_turn_direction_survives_compact_torso_projection():
    body = {
        "torso_geometry": {
            "available": True,
            "composer_eligible": True,
            "upper_torso_orientation": {
                "orientation_band": "three_quarter",
                "yaw_deg": -49.5,
                "yaw_magnitude_deg": 49.5,
                "approx_yaw_deg": 50,
                "turn_direction": "frame_right",
                "turn_direction_publishable": True,
            },
            "caption_orientation": {
                "mode": "combined",
                "preferred_turn_direction": "frame_right",
                "turn_direction_publishable": True,
            },
        }
    }
    assert mod._torso_fact(body) == {
        "camera_orientation": "three_quarter",
        "yaw_magnitude_deg": 49.5,
        "approx_yaw_deg": 50,
        "turn_direction": "frame_right",
    }


def test_near_frontal_orientation_does_not_invent_turn_direction():
    body = {
        "torso_geometry": {
            "available": True,
            "composer_eligible": True,
            "upper_torso_orientation": {
                "orientation_band": "frontal",
                "yaw_deg": 9.5,
                "yaw_magnitude_deg": 9.5,
                "approx_yaw_deg": 10,
                "turn_direction": None,
                "turn_direction_publishable": False,
            },
            "caption_orientation": {"mode": "combined"},
        }
    }
    fact = mod._torso_fact(body)
    assert fact is not None
    assert fact["camera_orientation"] == "frontal"
    assert "turn_direction" not in fact


def test_phase53_retains_phase52_gaze_caption_semantics_projection():
    facts = {
        "gaze": {
            "publishable": True,
            "horizontal": "frame_left",
            "camera_relationship": "uncertain",
            "caption_semantics": {
                "publishable": False,
                "horizontal": {
                    "publishable": False,
                    "composer_value": None,
                    "semantic_class": "near_center",
                },
                "vertical": {"publishable": False, "composer_value": None},
                "camera_relationship": {"publishable": False, "composer_value": None},
            },
        }
    }
    assert mod._gaze_fact(facts) is None


def test_specialist_authorized_anatomical_laterality_passes_audit():
    projection = {
        "authoritative_facts": {
            "body": {
                "configuration": [
                    "right hand resting on hip",
                    "left arm relaxed at side",
                ]
            }
        }
    }
    audit = mod._caption_audit(
        "A woman stands with her right hand resting on her hip while her left arm hangs relaxed at her side.",
        projection,
    )
    assert "unauthorized_anatomical_laterality" not in audit["violations"]
    assert audit["authorized_anatomical_laterality"] == ["left_arm", "right_hand"]


def test_unprojected_anatomical_laterality_still_fails_audit():
    projection = {
        "authoritative_facts": {
            "body": {"configuration": ["right hand resting on hip"]}
        }
    }
    audit = mod._caption_audit(
        "Her right hand rests on her hip while her left knee is bent.",
        projection,
    )
    assert "unauthorized_anatomical_laterality" in audit["violations"]
