from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v06 as mod


def test_phase4b5_defaults():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.5"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.5"


def test_raised_knee_and_opposite_planted_foot_bind_anatomical_sides():
    points = {
        "left_shoulder": (20.0, 0.0),
        "right_shoulder": (80.0, 0.0),
        "left_hip": (30.0, 100.0),
        "right_hip": (70.0, 100.0),
        "left_knee": (32.0, 150.0),
        "right_knee": (70.0, 120.0),
        "left_ankle": (32.0, 200.0),
        "right_ankle": (72.0, 150.0),
    }
    configuration = [
        {"text": "one knee raised", "composer_text": "one knee raised"},
        {"text": "one foot planted", "composer_text": "one foot planted"},
    ]
    out, bindings, warnings = mod._bind_leg_laterality(configuration, points)
    assert warnings == []
    assert out[0]["composer_text"] == "right knee raised"
    assert out[0]["laterality_binding"]["anatomical_side"] == "right"
    assert out[1]["composer_text"] == "left foot planted"
    assert out[1]["laterality_binding"]["anatomical_side"] == "left"
    assert [x["anatomical_side"] for x in bindings] == ["right", "left"]


def test_raised_knee_uses_thigh_angle_when_vertical_drop_margin_is_modest():
    points = {
        "left_shoulder": (20.0, 0.0),
        "right_shoulder": (80.0, 0.0),
        "left_hip": (30.0, 100.0),
        "right_hip": (70.0, 100.0),
        "left_knee": (30.0, 150.0),
        "right_knee": (100.0, 140.0),
        "left_ankle": (30.0, 205.0),
        "right_ankle": (110.0, 170.0),
    }
    configuration = [
        {"text": "one knee raised", "composer_text": "one knee raised"},
        {"text": "one foot planted", "composer_text": "one foot planted"},
    ]
    out, bindings, warnings = mod._bind_leg_laterality(configuration, points)
    assert warnings == []
    assert out[0]["composer_text"] == "right knee raised"
    assert out[0]["laterality_binding"]["authority"] == "dwpose_bilateral_thigh_angle_from_vertical"
    assert out[0]["laterality_binding"]["score_margin"] < mod.KNEE_RELATIVE_HEIGHT_MIN_MARGIN
    assert out[0]["laterality_binding"]["thigh_angle_margin_deg"] >= mod.KNEE_THIGH_ANGLE_MIN_MARGIN_DEG
    assert out[1]["composer_text"] == "left foot planted"
    assert [x["anatomical_side"] for x in bindings] == ["right", "left"]


def test_planted_and_lifted_feet_bind_from_bilateral_ankle_height():
    points = {
        "left_shoulder": (20.0, 0.0),
        "right_shoulder": (80.0, 0.0),
        "left_hip": (30.0, 100.0),
        "right_hip": (70.0, 100.0),
        "left_knee": (32.0, 150.0),
        "right_knee": (70.0, 140.0),
        "left_ankle": (32.0, 205.0),
        "right_ankle": (72.0, 165.0),
    }
    configuration = [
        {"text": "one foot planted", "composer_text": "one foot planted"},
        {"text": "other foot slightly lifted off floor", "composer_text": "other foot slightly lifted off floor"},
    ]
    out, bindings, warnings = mod._bind_leg_laterality(configuration, points)
    assert warnings == []
    assert out[0]["composer_text"] == "left foot planted"
    assert out[1]["composer_text"] == "right foot slightly lifted"
    assert [x["anatomical_side"] for x in bindings] == ["left", "right"]


def test_small_bilateral_height_difference_stays_unlateralized():
    points = {
        "left_shoulder": (20.0, 0.0),
        "right_shoulder": (80.0, 0.0),
        "left_hip": (30.0, 100.0),
        "right_hip": (70.0, 100.0),
        "left_knee": (32.0, 150.0),
        "right_knee": (70.0, 149.0),
        "left_ankle": (32.0, 200.0),
        "right_ankle": (72.0, 197.0),
    }
    configuration = [
        {"text": "one foot planted", "composer_text": "one foot planted"},
        {"text": "other foot slightly lifted off floor", "composer_text": "other foot slightly lifted off floor"},
    ]
    out, bindings, warnings = mod._bind_leg_laterality(configuration, points)
    assert out[0]["composer_text"] == "one foot planted"
    assert out[1]["composer_text"] == "other foot slightly lifted off floor"
    assert bindings == []
    assert "asymmetric_feet_anatomical_laterality_unresolved" in warnings


def test_qwen_side_word_is_not_authority_for_knee_binding():
    points = {
        "left_shoulder": (20.0, 0.0),
        "right_shoulder": (80.0, 0.0),
        "left_hip": (30.0, 100.0),
        "right_hip": (70.0, 100.0),
        "left_knee": (32.0, 150.0),
        "right_knee": (70.0, 120.0),
        "left_ankle": (32.0, 200.0),
        "right_ankle": (72.0, 150.0),
    }
    configuration = [{"text": "left knee raised", "composer_text": "knee raised"}]
    out, bindings, warnings = mod._bind_leg_laterality(configuration, points)
    assert warnings == []
    assert out[0]["composer_text"] == "right knee raised"
    assert out[0]["laterality_binding"]["source_side_label"] == "left"
    assert out[0]["laterality_binding"]["source_side_label_trusted"] is False
    assert bindings[0]["anatomical_side"] == "right"
