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


def test_trigger_projection_adds_feminine_pronoun_contract_from_subject_class():
    sheet = {
        "facts": {"body": {}, "visual": {}},
        "context_only": {},
        "audit": {},
    }
    projection, projection_audit = mod._projection(
        sheet,
        trigger_token="sH1VX",
        subject_class="woman",
    )
    subject = projection["subject"]
    assert subject["trigger_token"] == "sH1VX"
    assert subject["subject_class"] == "woman"
    assert subject["grammar_profile"] == "feminine"
    assert subject["subject_pronoun"] == "she"
    assert subject["object_pronoun"] == "her"
    assert subject["possessive_pronoun"] == "her"
    assert subject["reflexive_pronoun"] == "herself"
    assert "repeat the exact trigger" in subject["trigger_remention_policy"]
    assert projection_audit["trigger_subject_binding_projected"] is True
    assert projection_audit["subject_grammar_profile"] == "feminine"


def test_trigger_remention_and_possessive_are_allowed_when_first_binding_is_valid():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "grammar_profile": "feminine",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {},
    }
    caption = "sH1VX wears a dark jacket. sH1VX's sleeve is partly hidden while she smiles."
    audit = mod._caption_audit(caption, projection)
    assert not any(v.startswith("primary_trigger_") for v in audit["violations"])
    assert audit["trigger_binding"]["exact_occurrences"] == 2
    assert audit["trigger_binding"]["starts_with_trigger"] is True
    assert "frequent_trigger_remention:2" not in audit["warnings"]


def test_missing_or_noninitial_trigger_fails_binding_audit():
    projection = {
        "subject": {"trigger_token": "sH1VX", "grammar_profile": "feminine"},
        "authoritative_facts": {},
    }
    audit = mod._caption_audit("A woman wears a dark jacket and smiles.", projection)
    assert "primary_trigger_missing" in audit["violations"]
    assert "primary_trigger_not_first" in audit["violations"]


def test_generic_primary_subject_immediately_after_trigger_fails_binding_audit():
    projection = {
        "subject": {"trigger_token": "sH1VX", "grammar_profile": "feminine"},
        "authoritative_facts": {},
    }
    audit = mod._caption_audit("sH1VX, a woman, wears a dark jacket and smiles.", projection)
    assert "generic_primary_subject_after_trigger" in audit["violations"]
