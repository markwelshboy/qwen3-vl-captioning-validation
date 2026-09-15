from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v06 as mod


def test_phase55_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.6"
    assert mod.DEFAULT_PROMPT.name == "fact_sheet_text_composer_v05.txt"
    assert mod.SCHEMA_VERSION == "fact-sheet-text-composer-0.6"


def test_prompt_removes_obsolete_support_leg_priming():
    text = mod.DEFAULT_PROMPT.read_text(encoding="utf-8")
    assert "stands mostly upright over her right leg" not in text
    assert "mostly_upright_over_support_leg" not in text
    assert "Do not infer a support side" in text
    assert "A raised leg or raised knee does NOT authorize any claim" in text


def test_prompt_preserves_authoritative_raised_leg_specificity():
    text = mod.DEFAULT_PROMPT.read_text(encoding="utf-8")
    assert "left knee raised high with thigh nearly horizontal" in text
    assert "It does not authorize any claim about the other leg" in text


def test_projection_preserves_specialist_bound_unilateral_relation_without_support_shape():
    sheet = {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {
            "body": {
                "pose_candidate": {
                    "text": "standing with one leg raised",
                    "composer_text": "standing with one leg raised",
                    "promotion_status": "accepted_specialist_unilateral_raised_leg_candidate",
                },
                "configuration": [
                    {"composer_text": "upper body bent forward from the hips"},
                    {"composer_text": "both hands holding visible fabric"},
                    {
                        "composer_text": "left knee raised high with thigh nearly horizontal",
                        "specialist_owner": "sam3d_v16_unilateral_raised_leg_specialist",
                    },
                ],
                "support_geometry": {
                    "available": True,
                    "composer_eligible": True,
                    "overall_shape": "mostly_upright_over_support_leg",
                    "support_side": "right",
                },
            },
            "visual": {},
        },
        "context_only": {},
        "audit": {},
    }
    projection, audit = mod._projection(sheet)
    body = projection["authoritative_facts"]["body"]
    assert body["broad_pose"] == "standing with one leg raised"
    assert "left knee raised high with thigh nearly horizontal" in body["configuration"]
    assert "global_support_shape" not in body
    assert audit["support_geometry_withheld_from_composer"] is True


def test_caption_audit_allows_authorized_left_knee_but_rejects_invented_right_leg():
    sheet = {
        "schema_version": "caption-fact-sheet-0.3",
        "facts": {
            "body": {
                "pose_candidate": {
                    "text": "standing with one leg raised",
                    "composer_text": "standing with one leg raised",
                    "promotion_status": "candidate",
                },
                "configuration": [
                    {"composer_text": "left knee raised high with thigh nearly horizontal"},
                ],
            },
            "visual": {},
        },
        "context_only": {},
        "audit": {},
    }
    projection, _ = mod._projection(sheet, trigger_token="sH1VX", subject_class="woman")

    good = mod._caption_audit(
        "sH1VX stands with her left knee raised high and her thigh nearly horizontal.",
        projection,
    )
    assert "unauthorized_anatomical_laterality" not in good.get("violations", [])

    bad = mod._caption_audit(
        "sH1VX stands with her left knee raised high while her right leg supports her weight.",
        projection,
    )
    assert "unauthorized_anatomical_laterality" in bad.get("violations", [])
