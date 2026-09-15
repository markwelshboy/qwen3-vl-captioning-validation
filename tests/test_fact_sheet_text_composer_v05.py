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
