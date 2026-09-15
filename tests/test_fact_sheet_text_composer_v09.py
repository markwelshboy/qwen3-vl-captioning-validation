from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v09 as mod


def test_phase58_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.9"


def test_support_contact_language_without_authority_is_violation():
    projection = {
        "authoritative_facts": {
            "body": {
                "broad_pose": "standing",
                "configuration": ["both hands holding visible fabric"],
            }
        },
        "subject": {},
        "omitted_review_conflict_domains": [],
    }
    caption = "She stands in a kitchen without any visible support or contact with the floor."
    audit = mod._caption_audit(caption, projection)
    assert "support_contact_language_without_authority" in audit["violations"]


def test_explicit_planted_feet_authorize_support_contact_language():
    projection = {
        "authoritative_facts": {
            "body": {
                "broad_pose": "standing",
                "configuration": ["feet planted on rug", "legs straight and apart"],
            }
        },
        "subject": {},
        "omitted_review_conflict_domains": [],
    }
    caption = "She stands with her feet planted firmly on the rug."
    audit = mod._caption_audit(caption, projection)
    assert "support_contact_language_without_authority" not in audit["violations"]


def test_counter_rotation_for_same_direction_closer_to_frontal_is_violation():
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
    caption = (
        "She crouches with her torso about 35 degrees toward frame left and her upper torso about 20 degrees toward frame left, "
        "creating a subtle counter-rotation while the upper torso remains closer to frontal."
    )
    audit = mod._caption_audit(caption, projection)
    assert "counter_rotation_language_without_counter_rotated_geometry" in audit["violations"]


def test_counter_rotation_is_allowed_when_relationship_is_counter_rotated():
    projection = {
        "authoritative_facts": {
            "body": {
                "broad_pose": "standing",
                "torso_orientation": {
                    "mode": "articulated",
                    "articulated_relative_orientation": {
                        "relationship": "counter_rotated",
                    },
                },
            }
        },
        "subject": {},
        "omitted_review_conflict_domains": [],
    }
    caption = "She stands with a subtle counter-rotation between her body and upper torso."
    audit = mod._caption_audit(caption, projection)
    assert "counter_rotation_language_without_counter_rotated_geometry" not in audit["violations"]


def test_retry_prompt_forbids_new_support_claims_and_negative_disclaimers():
    prompt = mod._retry_prompt(
        "BASE",
        "She stands, though her gaze direction is not specified.",
        ["gaze_language_without_publishable_gaze"],
    )
    assert "Do not introduce any new support/contact" in prompt
    assert "support/contact is absent or not visible" in prompt


def test_retry_prompt_repairs_false_counter_rotation_literally():
    prompt = mod._retry_prompt(
        "BASE",
        "Her upper torso creates a subtle counter-rotation.",
        ["counter_rotation_language_without_counter_rotated_geometry"],
    )
    assert "Remove 'counter-rotation' wording" in prompt
    assert "same-direction segments" in prompt
