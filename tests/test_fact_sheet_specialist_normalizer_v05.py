from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v05 as mod


def _asymmetric_arm_points() -> dict:
    return {
        "left_shoulder": (20.0, 0.0),
        "left_elbow": (20.0, 55.0),
        "left_wrist": (20.0, 120.0),
        "left_hip": (30.0, 100.0),
        "right_shoulder": (80.0, 0.0),
        "right_elbow": (90.0, 45.0),
        "right_wrist": (72.0, 98.0),
        "right_hip": (70.0, 100.0),
    }


def test_phase4b4_uses_separate_output_namespace():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.4"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.4"


def test_signed_yaw_maps_to_frame_turn_direction_with_frontal_deadband():
    assert mod._turn_direction(49.5) == "frame_left"
    assert mod._turn_direction(-49.5) == "frame_right"
    assert mod._turn_direction(15.0) is None
    assert mod._turn_direction(-14.9) is None


def test_phase4b4_adds_preferred_signed_torso_direction():
    sheet = {
        "schema_version": "caption-fact-sheet-0.2.3",
        "facts": {
            "body": {
                "torso_geometry": {
                    "available": True,
                    "composer_eligible": True,
                    "body_root_orientation": {
                        "yaw_deg": -52.0,
                        "orientation_band": "three_quarter",
                    },
                    "upper_torso_orientation": {
                        "yaw_deg": -49.5,
                        "orientation_band": "three_quarter",
                    },
                    "caption_orientation": {"mode": "combined"},
                }
            }
        },
        "audit": {"warnings": [], "invariants": {}},
    }
    out, warnings = mod._apply_signed_torso_direction(sheet)
    assert warnings == []
    torso = out["facts"]["body"]["torso_geometry"]
    assert torso["upper_torso_orientation"]["turn_direction"] == "frame_right"
    assert torso["upper_torso_orientation"]["turn_direction_publishable"] is True
    assert torso["caption_orientation"]["preferred_turn_direction"] == "frame_right"
    assert torso["caption_orientation"]["turn_direction_publishable"] is True


def test_hand_on_hip_relation_binds_to_dwpose_anatomical_side_and_explicit_relaxed_arm():
    configuration = [
        {"text": "right hand on hip", "composer_text": "hand on hip"},
        {"text": "left arm relaxed at side", "composer_text": "arm relaxed at side"},
    ]
    out, bindings, warnings = mod._bind_configuration_laterality(configuration, _asymmetric_arm_points())
    assert warnings == []
    assert out[0]["composer_text"] == "right hand resting on hip"
    assert out[0]["promotion_status"] == "accepted_specialist_lateralized_candidate"
    assert out[0]["laterality_binding"]["anatomical_side"] == "right"
    assert out[0]["laterality_binding"]["source_side_label_trusted"] is False
    assert out[1]["composer_text"] == "left arm relaxed at side"
    assert out[1]["laterality_binding"]["anatomical_side"] == "left"
    assert out[1]["laterality_binding"]["source_side_label"] == "left"
    assert out[1]["laterality_binding"]["source_side_label_trusted"] is False
    assert [x["anatomical_side"] for x in bindings] == ["right", "left"]


def test_qwen_relaxed_arm_side_label_is_ignored_and_rebound_from_geometry():
    configuration = [
        {"text": "left hand on hip", "composer_text": "hand on hip"},
        {"text": "right arm relaxed at side", "composer_text": "arm relaxed at side"},
    ]
    out, bindings, warnings = mod._bind_configuration_laterality(configuration, _asymmetric_arm_points())
    assert warnings == []
    # Geometry, not Qwen's side words, establishes right-hand-on-hip / left-arm-relaxed.
    assert out[0]["composer_text"] == "right hand resting on hip"
    assert out[1]["composer_text"] == "left arm relaxed at side"
    assert out[1]["laterality_binding"]["source_side_label"] == "right"
    assert out[1]["laterality_binding"]["source_side_label_trusted"] is False
    assert [x["anatomical_side"] for x in bindings] == ["right", "left"]


def test_other_arm_wording_still_binds_to_opposite_observed_chain():
    configuration = [
        {"text": "one hand on hip", "composer_text": "one hand on hip"},
        {"text": "other arm relaxed at side", "composer_text": "other arm relaxed at side"},
    ]
    out, bindings, warnings = mod._bind_configuration_laterality(configuration, _asymmetric_arm_points())
    assert warnings == []
    assert out[0]["composer_text"] == "right hand resting on hip"
    assert out[1]["composer_text"] == "left arm relaxed at side"
    assert [x["anatomical_side"] for x in bindings] == ["right", "left"]


def test_multiple_relaxed_arm_relations_are_not_all_forced_to_same_opposite_side():
    configuration = [
        {"text": "one hand on hip", "composer_text": "one hand on hip"},
        {"text": "left arm relaxed at side", "composer_text": "arm relaxed at side"},
        {"text": "other arm hanging at side", "composer_text": "arm hanging at side"},
    ]
    out, bindings, warnings = mod._bind_configuration_laterality(configuration, _asymmetric_arm_points())
    assert out[0]["composer_text"] == "right hand resting on hip"
    assert out[1]["composer_text"] == "arm relaxed at side"
    assert out[2]["composer_text"] == "arm hanging at side"
    assert [x["anatomical_side"] for x in bindings] == ["right"]
    assert "relaxed_arm_laterality_unresolved_multiple_relations" in warnings


def test_ambiguous_hand_on_hip_geometry_stays_unlateralized():
    points = {
        "left_shoulder": (20.0, 0.0),
        "left_elbow": (20.0, 50.0),
        "left_wrist": (25.0, 95.0),
        "left_hip": (30.0, 100.0),
        "right_shoulder": (80.0, 0.0),
        "right_elbow": (80.0, 50.0),
        "right_wrist": (75.0, 95.0),
        "right_hip": (70.0, 100.0),
    }
    configuration = [{"text": "one hand on hip", "composer_text": "one hand on hip"}]
    out, bindings, warnings = mod._bind_configuration_laterality(configuration, points)
    assert out[0]["composer_text"] == "one hand on hip"
    assert bindings == []
    assert "hand_on_hip_anatomical_laterality_unresolved" in warnings
