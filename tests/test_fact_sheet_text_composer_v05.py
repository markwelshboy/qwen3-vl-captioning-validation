from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v05 as mod


def test_phase54_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.5"


def test_support_geometry_stays_out_of_projection():
    sheet = {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {
            "body": {
                "pose_candidate": {"text": "standing", "promotion_status": "candidate"},
                "support_geometry": {
                    "available": True,
                    "composer_eligible": True,
                    "overall_shape": "mostly_upright_over_support_leg",
                },
            },
            "visual": {},
        },
        "context_only": {},
        "audit": {},
    }
    projection, audit = mod._projection(sheet)
    projected = projection["authoritative_facts"]["body"]
    assert projected["broad_pose"] == "standing"
    assert "global_support_shape" not in projected
    assert audit["support_geometry_withheld_from_composer"] is True


def test_authoritative_adjudicated_pose_projects_canonical_text():
    sheet = {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {
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
                ],
            },
            "visual": {},
        },
        "context_only": {},
        "audit": {},
    }
    projection, audit = mod._projection(sheet)
    body = projection["authoritative_facts"]["body"]
    assert body["broad_pose"] == "crouching"
    assert body["configuration"] == ["upper body bent forward from the hips"]
    assert audit["canonical_broad_pose_projected"] == "crouching"


def test_holistic_context_with_residual_support_mechanics_is_withheld():
    sheet = {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {
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
                ],
            },
            "visual": {
                "scene": [
                    {"composer_text": "bright kitchen with white cabinetry"},
                ]
            },
        },
        "context_only": {
            "gestalt": {
                "text": "A woman standing mostly upright over her left leg in a bright kitchen."
            }
        },
        "audit": {},
    }
    projection, audit = mod._projection(sheet)
    assert "holistic_context_non_authoritative" not in projection
    assert audit["holistic_context_withheld_for_body_mechanics"] is True
    assert projection["authoritative_facts"]["scene"] == ["bright kitchen with white cabinetry"]
    assert projection["authoritative_facts"]["body"]["broad_pose"] == "crouching"


def test_scene_only_holistic_context_can_survive():
    sheet = {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {"body": {}, "visual": {}},
        "context_only": {
            "gestalt": {
                "text": "A bright domestic kitchen with soft natural window light."
            }
        },
        "audit": {},
    }
    projection, audit = mod._projection(sheet)
    assert projection["holistic_context_non_authoritative"] == "A bright domestic kitchen with soft natural window light"
    assert audit["holistic_context_withheld_for_body_mechanics"] is False
