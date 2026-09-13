from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v02 as mod


def _body(torso: dict) -> dict:
    return {"torso_geometry": torso}


def test_phase51_defaults_to_enriched_phase4b2_fact_sheets():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.2"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.2"
    assert mod.EXPECTED_FACT_SHEET_SCHEMA == "caption-fact-sheet-0.2.2"


def test_combined_torso_projection_is_flat_preferred_orientation():
    fact = mod._torso_fact(_body({
        "available": True,
        "composer_eligible": True,
        "body_root_orientation": {"orientation_band": "three_quarter", "yaw_magnitude_deg": 41.0, "approx_yaw_deg": 40},
        "upper_torso_orientation": {"orientation_band": "three_quarter", "yaw_magnitude_deg": 44.0, "approx_yaw_deg": 45},
        "caption_orientation": {"mode": "combined", "relative_twist_magnitude_deg": 3.0},
    }))
    assert fact == {
        "camera_orientation": "three_quarter",
        "yaw_magnitude_deg": 44.0,
        "approx_yaw_deg": 45,
    }


def test_upper_torso_dominant_oblique_hides_mode_and_components():
    fact = mod._torso_fact(_body({
        "available": True,
        "composer_eligible": True,
        "body_root_orientation": {"orientation_band": "slightly_angled", "yaw_magnitude_deg": 22.5, "approx_yaw_deg": 25},
        "upper_torso_orientation": {"orientation_band": "oblique", "yaw_magnitude_deg": 29.1, "approx_yaw_deg": 30},
        "caption_orientation": {"mode": "upper_torso_dominant", "relative_twist_magnitude_deg": 6.6},
    }))
    assert fact == {
        "camera_orientation": "oblique",
        "yaw_magnitude_deg": 29.1,
        "approx_yaw_deg": 30,
    }
    assert "mode" not in fact
    assert "preferred" not in fact


def test_articulated_torso_projection_preserves_two_orientation_fields():
    fact = mod._torso_fact(_body({
        "available": True,
        "composer_eligible": True,
        "body_root_orientation": {"orientation_band": "slightly_angled", "yaw_magnitude_deg": 22.0, "approx_yaw_deg": 20},
        "upper_torso_orientation": {"orientation_band": "three_quarter", "yaw_magnitude_deg": 44.0, "approx_yaw_deg": 45},
        "caption_orientation": {"mode": "articulated", "relative_twist_magnitude_deg": 22.0},
    }))
    assert fact is not None
    assert fact["mode"] == "articulated"
    assert fact["body_orientation"]["camera_orientation"] == "slightly_angled"
    assert fact["upper_torso_orientation"]["camera_orientation"] == "three_quarter"
    assert fact["preferred"]["approx_yaw_deg"] == 45
    assert fact["relative_twist_magnitude_deg"] == 22.0


def test_legacy_torso_projection_is_also_flat():
    fact = mod._torso_fact(_body({
        "available": True,
        "composer_eligible": True,
        "torso_camera_orientation": "oblique",
        "torso_yaw_magnitude_deg": 29.1,
    }))
    assert fact == {
        "camera_orientation": "oblique",
        "yaw_magnitude_deg": 29.1,
        "approx_yaw_deg": 30,
    }


def test_diagnostic_only_torso_remains_unavailable_to_composer():
    assert mod._torso_fact(_body({
        "available": True,
        "composer_eligible": False,
        "upper_torso_orientation": {"orientation_band": "three_quarter", "yaw_magnitude_deg": 45.0},
    })) is None
