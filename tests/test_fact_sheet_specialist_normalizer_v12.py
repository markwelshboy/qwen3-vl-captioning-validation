from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v12 as mod


def _item(text, composer, relation=None, authority=None):
    out = {"text": text, "composer_text": composer}
    if relation and authority:
        out["laterality_binding"] = {
            "semantic_relation": relation,
            "authority": authority,
            "anatomical_side": "left" if "left" in composer else "right",
        }
    return out


def test_defaults():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.11"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.11"


def test_00064_actual_weak_support_seed_is_de_lateralized():
    sheet = {
        "facts": {
            "body": {
                "pose_candidate": {"text": "standing with one leg raised", "composer_text": "standing with one leg raised"},
                "configuration": [
                    _item("torso bent forward", "upper body bent forward from the hips"),
                    _item("both hands holding visible fabric", "both hands holding visible fabric"),
                    _item(
                        "one knee raised",
                        "left knee raised high",
                        "knee_raised",
                        mod.DERIVED_KNEE_FROM_WEAK_PLANTED_AUTHORITY,
                    ),
                    _item(
                        "one foot planted",
                        "right foot planted",
                        "foot_planted",
                        mod.ANKLE_HEIGHT_ONLY_AUTHORITY,
                    ),
                ],
                "support_geometry": {
                    "available": True,
                    "composer_eligible": True,
                    "elevated_relation": "knee_raised",
                },
            }
        }
    }
    out = mod._apply_support_seed_truth_gate(sheet)
    body = out["facts"]["body"]
    assert [x["composer_text"] for x in body["configuration"]] == [
        "upper body bent forward from the hips",
        "both hands holding visible fabric",
        "one knee raised",
    ]
    knee = body["configuration"][2]
    assert "laterality_binding" not in knee
    assert knee["withheld_laterality_binding"]["authority"] == mod.DERIVED_KNEE_FROM_WEAK_PLANTED_AUTHORITY
    assert body["support_topology_adjudication"]["status"] == "withheld_weak_support_seed"
    assert body["support_topology_adjudication"]["applied"] is True
    assert len(body["support_topology_adjudication"]["removed_configuration_relations"]) == 1
    assert body["support_geometry"]["composer_eligible"] is False


def test_direct_raised_knee_and_opposite_support_foot_are_preserved():
    sheet = {
        "facts": {
            "body": {
                "configuration": [
                    _item(
                        "one knee raised",
                        "left knee raised",
                        "knee_raised",
                        "dwpose_bilateral_thigh_angle_from_vertical",
                    ),
                    _item(
                        "one foot planted",
                        "right foot planted",
                        "foot_planted",
                        mod.STRONG_PLANTED_FROM_KNEE_AUTHORITY,
                    ),
                ]
            }
        }
    }
    out = mod._apply_support_seed_truth_gate(sheet)
    body = out["facts"]["body"]
    assert [x["composer_text"] for x in body["configuration"]] == ["left knee raised", "right foot planted"]
    assert body["support_topology_adjudication"]["status"] == "not_applicable"


def test_unrelated_configuration_is_unchanged():
    sheet = {
        "facts": {
            "body": {
                "configuration": [
                    _item("hands on hips", "hands on hips"),
                    _item("feet planted on rug", "feet planted on rug"),
                ]
            }
        }
    }
    out = mod._apply_support_seed_truth_gate(sheet)
    body = out["facts"]["body"]
    assert [x["composer_text"] for x in body["configuration"]] == ["hands on hips", "feet planted on rug"]
    assert body["support_topology_adjudication"]["status"] == "not_applicable"
