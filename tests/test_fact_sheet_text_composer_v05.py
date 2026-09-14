from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v05 as mod


def test_phase54_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.5"
    assert mod.DEFAULT_PROMPT.name == "fact_sheet_text_composer_v04.txt"
    assert mod.SCHEMA_VERSION == "fact-sheet-text-composer-0.5"


def test_global_support_shape_projects_compact_authoritative_fact():
    sheet = {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {
            "body": {
                "configuration": [
                    {"composer_text": "upper body bent forward from the hips while overall stance remains mostly upright"},
                    {"composer_text": "left knee raised high"},
                    {"composer_text": "right foot planted"},
                ],
                "support_geometry": {
                    "available": True,
                    "composer_eligible": True,
                    "overall_shape": "mostly_upright_over_support_leg",
                    "support_side": "right",
                    "elevated_side": "left",
                    "elevated_leg_height": "high",
                    "support_axis_angle_from_vertical_deg": 2.3,
                    "elevated_ankle_height_gap_norm": 2.45,
                    "authority": "dwpose_bound_leg_relations_plus_hip_to_support_ankle_axis",
                },
            },
            "visual": {},
        },
        "context_only": {},
        "audit": {},
    }
    projection, audit = mod._projection(sheet, trigger_token="sH1VX", subject_class="woman")
    support = projection["authoritative_facts"]["body"]["global_support_shape"]
    assert support == {
        "overall_shape": "mostly_upright_over_support_leg",
        "support_side": "right",
        "elevated_side": "left",
        "elevated_leg_height": "high",
        "support_axis_angle_from_vertical_deg": 2.3,
    }
    assert audit["global_support_shape_projected"] is True
    assert projection["subject"]["trigger_token"] == "sH1VX"
    assert projection["subject"]["subject_pronoun"] == "she"


def test_noneligible_support_geometry_is_not_projected():
    sheet = {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {
            "body": {
                "support_geometry": {
                    "available": True,
                    "composer_eligible": False,
                    "overall_shape": "unresolved",
                }
            },
            "visual": {},
        },
        "context_only": {},
        "audit": {},
    }
    projection, audit = mod._projection(sheet)
    body = projection.get("authoritative_facts", {}).get("body", {})
    assert "global_support_shape" not in body
    assert audit["global_support_shape_projected"] is False
