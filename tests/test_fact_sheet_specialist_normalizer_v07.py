from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v07 as mod


def _bound_item(text: str, composer_text: str, relation: str, side: str) -> dict:
    return {
        "text": text,
        "composer_text": composer_text,
        "laterality_binding": {
            "semantic_relation": relation,
            "anatomical_side": side,
            "source_side_label_trusted": False,
        },
    }


def test_phase4b6_defaults():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.6"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.6"


def test_vertical_support_leg_and_high_other_ankle_create_global_support_shape():
    points = {
        "left_shoulder": (20.0, 0.0),
        "right_shoulder": (80.0, 0.0),
        "left_hip": (60.0, 100.0),
        "right_hip": (40.0, 100.0),
        "left_knee": (10.0, 90.0),
        "right_knee": (42.0, 180.0),
        "left_ankle": (40.0, 150.0),
        "right_ankle": (42.0, 300.0),
    }
    configuration = [
        _bound_item("one knee raised", "left knee raised", "knee_raised", "left"),
        _bound_item("one foot planted", "right foot planted", "foot_planted", "right"),
    ]
    support = mod._support_geometry(configuration, points)
    assert support is not None
    assert support["composer_eligible"] is True
    assert support["overall_shape"] == "mostly_upright_over_support_leg"
    assert support["support_side"] == "right"
    assert support["elevated_side"] == "left"
    assert support["elevated_leg_height"] == "high"
    assert support["support_axis_angle_from_vertical_deg"] < mod.SUPPORT_AXIS_MOSTLY_UPRIGHT_MAX_DEG


def test_support_shape_scopes_forward_bend_and_strengthens_high_knee():
    configuration = [
        {"text": "torso bent forward", "composer_text": "torso bent forward"},
        _bound_item("one knee raised", "left knee raised", "knee_raised", "left"),
        _bound_item("one foot planted", "right foot planted", "foot_planted", "right"),
    ]
    support = {
        "composer_eligible": True,
        "overall_shape": "mostly_upright_over_support_leg",
        "support_side": "right",
        "elevated_side": "left",
        "elevated_leg_height": "high",
        "support_axis_angle_from_vertical_deg": 2.3,
        "elevated_ankle_height_gap_norm": 2.45,
    }
    out = mod._apply_support_shape_to_configuration(configuration, support)
    assert out[0]["composer_text"] == "upper body bent forward from the hips while overall stance remains mostly upright"
    assert out[0]["promotion_status"] == "accepted_specialist_shape_refined_candidate"
    assert out[1]["composer_text"] == "left knee raised high"
    assert out[1]["promotion_status"] == "accepted_specialist_shape_refined_candidate"
    assert out[2]["composer_text"] == "right foot planted"


def test_high_lifted_foot_relation_becomes_high_leg_relation_not_slight_lift():
    configuration = [
        _bound_item("one foot planted", "left foot planted", "foot_planted", "left"),
        _bound_item("other foot slightly lifted off floor", "right foot slightly lifted", "foot_lifted", "right"),
    ]
    support = {
        "composer_eligible": True,
        "overall_shape": "mostly_upright_over_support_leg",
        "support_side": "left",
        "elevated_side": "right",
        "elevated_leg_height": "high",
        "support_axis_angle_from_vertical_deg": 3.0,
        "elevated_ankle_height_gap_norm": 1.8,
    }
    out = mod._apply_support_shape_to_configuration(configuration, support)
    assert out[1]["composer_text"] == "right leg lifted high with foot well off the floor"


def test_nonvertical_support_axis_does_not_refine_local_body_language():
    configuration = [
        {"text": "torso bent forward", "composer_text": "torso bent forward"},
        _bound_item("one knee raised", "left knee raised", "knee_raised", "left"),
        _bound_item("one foot planted", "right foot planted", "foot_planted", "right"),
    ]
    support = {
        "composer_eligible": False,
        "overall_shape": "unresolved",
        "support_side": "right",
        "elevated_side": "left",
        "elevated_leg_height": "high",
    }
    out = mod._apply_support_shape_to_configuration(configuration, support)
    assert out == configuration
