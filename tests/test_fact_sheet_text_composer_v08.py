from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v08 as mod


def test_phase57_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.8"
    assert mod.DEFAULT_PROMPT.name == "fact_sheet_text_composer_v06.txt"


def _articulated_sheet():
    return {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {
            "framing": {"extent": "full_length"},
            "body": {
                "pose_candidate": {
                    "text": "crouching",
                    "composer_text": "crouching",
                    "promotion_status": "accepted_specialist_adjudicated_candidate",
                },
                "broad_pose_adjudication": {
                    "composer_authoritative": True,
                    "canonical_pose_text": "crouching",
                },
                "configuration": [
                    {"composer_text": "upper body bent forward from the hips"},
                    {"composer_text": "both knees bent"},
                ],
                "torso_geometry": {
                    "available": True,
                    "composer_eligible": True,
                    "caption_orientation": {
                        "mode": "articulated",
                        "relative_twist_magnitude_deg": 13.5,
                    },
                    "body_root_orientation": {
                        "orientation_band": "oblique",
                        "yaw_magnitude_deg": 34.9,
                        "approx_yaw_deg": 35,
                        "turn_direction": "frame_left",
                        "turn_direction_publishable": True,
                    },
                    "upper_torso_orientation": {
                        "orientation_band": "slightly_angled",
                        "yaw_magnitude_deg": 21.4,
                        "approx_yaw_deg": 20,
                        "turn_direction": "frame_left",
                        "turn_direction_publishable": True,
                    },
                },
            },
            "visual": {},
        },
        "context_only": {},
        "audit": {},
    }


def test_direct_projection_restores_validated_articulated_torso_stack():
    # v07's custom main bypassed the chained runtime assignment that normally
    # installs the v03 torso/gaze helpers. v08 must restore them explicitly.
    projection, audit = mod._projection(_articulated_sheet())
    torso = projection["authoritative_facts"]["body"]["torso_orientation"]
    assert torso["mode"] == "articulated"
    assert torso["body_orientation"]["turn_direction"] == "frame_left"
    assert torso["upper_torso_orientation"]["turn_direction"] == "frame_left"
    relation = torso["articulated_relative_orientation"]
    assert relation["relationship"] == "upper_torso_closer_to_frontal"
    assert relation["body_yaw_deg"] == 35.0
    assert relation["upper_torso_yaw_deg"] == 20.0
    assert relation["relative_twist_magnitude_deg"] == 13.5
    assert audit["validated_torso_projection_helpers_restored"] is True
    assert audit["articulated_relative_orientation_projected"] is True


def test_head_direction_in_same_sentence_does_not_count_as_torso_direction():
    projection = {
        "authoritative_facts": {
            "body": {
                "broad_pose": "seated",
                "torso_orientation": {
                    "camera_orientation": "oblique",
                    "yaw_magnitude_deg": 29.1,
                    "approx_yaw_deg": 30,
                },
            },
            "head": {"horizontal": "frame_left", "vertical": "up"},
        },
        "subject": {},
        "omitted_review_conflict_domains": [],
    }
    caption = (
        "She sits with her torso angled at an oblique orientation to the camera, "
        "and her head is turned toward frame left and tilted upward."
    )
    audit = mod._caption_audit(caption, projection)
    assert not any(v.startswith("unauthorized_torso_turn_direction") for v in audit["violations"])


def test_true_unsigned_torso_direction_still_fails_audit():
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
    caption = "She stands with her torso subtly turned toward frame left."
    audit = mod._caption_audit(caption, projection)
    assert "unauthorized_torso_turn_direction:frame_left" in audit["violations"]


def test_articulated_relationship_reversal_remains_audit_violation():
    projection, _ = mod._projection(_articulated_sheet())
    caption = (
        "She crouches with her torso about 35 degrees toward frame left, "
        "while her upper torso turns more toward frame left."
    )
    audit = mod._caption_audit(caption, projection)
    assert "articulated_torso_relationship_reversed" in audit["violations"]
