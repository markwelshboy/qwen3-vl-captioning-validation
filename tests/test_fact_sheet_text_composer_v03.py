from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v03 as mod


def test_phase52_defaults_to_identity_policy_fact_sheets():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.3"
    assert mod.EXPECTED_FACT_SHEET_SCHEMA == "caption-fact-sheet-0.3"
    assert mod.SCHEMA_VERSION == "fact-sheet-text-composer-0.3"


def test_phase52_reuses_frozen_torso_projection():
    body = {
        "torso_geometry": {
            "available": True,
            "composer_eligible": True,
            "upper_torso_orientation": {
                "orientation_band": "oblique",
                "yaw_magnitude_deg": 29.1,
                "approx_yaw_deg": 30,
            },
            "caption_orientation": {
                "mode": "upper_torso_dominant",
                "relative_twist_magnitude_deg": 6.6,
            },
        }
    }
    assert mod._torso_fact(body) == {
        "camera_orientation": "oblique",
        "yaw_magnitude_deg": 29.1,
        "approx_yaw_deg": 30,
    }
