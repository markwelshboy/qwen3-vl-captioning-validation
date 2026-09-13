from __future__ import annotations

from qwen_caption_validate import fact_sheet_text_composer_v03 as mod


def test_composer_suppresses_near_center_lateral_gaze():
    facts = {
        "gaze": {
            "available": True,
            "publishable": True,
            "horizontal": "frame_left",
            "yaw_deg": 14.6,
            "caption_semantics": {
                "publishable": False,
                "horizontal": {
                    "raw_value": "frame_left",
                    "raw_yaw_deg": 14.6,
                    "semantic_class": "near_center",
                    "composer_value": None,
                    "publishable": False,
                },
                "vertical": {
                    "semantic_class": "near_center",
                    "composer_value": None,
                    "publishable": False,
                },
                "camera_relationship": {
                    "composer_value": None,
                    "publishable": False,
                },
            },
        }
    }
    assert mod._gaze_fact(facts) is None


def test_composer_emits_only_caption_facing_gaze_fields():
    facts = {
        "gaze": {
            "available": True,
            "publishable": True,
            "horizontal": "frame_left",
            "yaw_deg": 31.0,
            "caption_semantics": {
                "publishable": True,
                "horizontal": {
                    "composer_value": "frame_left",
                    "publishable": True,
                },
                "vertical": {
                    "composer_value": None,
                    "publishable": False,
                },
                "camera_relationship": {
                    "composer_value": None,
                    "publishable": False,
                },
            },
        }
    }
    assert mod._gaze_fact(facts) == {"horizontal": "frame_left"}
