from __future__ import annotations

import json
from pathlib import Path

from qwen_caption_validate import crop_consistency_diagnostic_v01 as mod


def _record(
    key: str,
    *,
    framing: str = "medium shot",
    shot: str = "medium",
    broad: str | None = None,
    torso_direction: str | None = None,
    head_horizontal: str | None = None,
) -> dict:
    body = {}
    if broad is not None:
        body["broad_pose"] = broad
    if torso_direction is not None:
        body["torso_orientation"] = {
            "camera_orientation": "three_quarter",
            "turn_direction": torso_direction,
            "approx_yaw_deg": 45,
        }
    auth = {
        "framing": {
            "composer_text": framing,
            "surface_source": "standard_shot_scale",
            "shot_scale_label": shot,
        },
        "body": body,
    }
    if head_horizontal is not None:
        auth["head"] = {
            "composer_text": f"head turned toward {head_horizontal}",
            "horizontal": head_horizontal,
        }
    return {
        "image_key": key,
        "status": "ok",
        "evidence_projection": {"authoritative_facts": auth},
        "caption": "caption",
        "caption_audit": {"violations": []},
    }


def test_snapshot_groups_specific_standing_phrase_under_standing():
    snap = mod.snapshot_record(_record("wide", broad="standing with one leg raised"))
    assert snap["body"]["broad_pose_group"] == "standing"
    assert snap["framing"]["coverage_score"] == 3.0


def test_crop_comparison_treats_pose_loss_as_collapse_not_contradiction():
    wide = mod.snapshot_record(
        _record("wide", framing="medium-wide shot", shot="medium_wide", broad="standing")
    )
    tight = mod.snapshot_record(_record("tight", framing="close-up", shot="close_up"))
    result = mod.compare_snapshots(wide, tight)
    broad = next(
        x for x in result["semantic_transitions"]
        if x["domain"] == "broad_pose_group"
    )
    assert broad["state"] == "collapsed"
    assert broad["contradiction"] is False
    assert result["contradictions"] == []


def test_crop_comparison_flags_surviving_direction_flip():
    wide = mod.snapshot_record(_record("wide", torso_direction="frame_left"))
    tight = mod.snapshot_record(_record("tight", torso_direction="frame_right"))
    result = mod.compare_snapshots(wide, tight)
    assert any(
        x["domain"] == "torso.turn_direction"
        for x in result["contradictions"]
    )


def test_family_uses_declared_source_crop_order_and_duplicate_metadata():
    records = {
        "a": _record("a", broad="standing"),
        "b": _record("b", broad="standing"),
        "c": _record("c"),
    }
    family = {
        "family_id": "f",
        "kind": "crop_family",
        "members": ["a", "b", "c"],
        "wide_to_tight": ["b", "a", "c"],
        "relative_fov_area": {"b": 2.0, "a": 2.0, "c": 1.0},
        "exact_duplicate_groups": [["b", "a"]],
    }
    out = mod.audit_family(family, records)
    assert out["ordered_wide_to_tight"] == ["b", "a", "c"]
    assert out["pairwise_crop_transitions"][0]["source_crop_relation"] == "exact_duplicate"
    assert out["pairwise_crop_transitions"][1]["source_crop_relation"] == "wider_to_tighter"
    assert out["pairwise_crop_transitions"][1]["source_fov_ratio"] == 2.0
    broad = next(
        x for x in out["pairwise_crop_transitions"][1]["semantic_transitions"]
        if x["domain"] == "broad_pose_group"
    )
    assert broad["state"] == "collapsed"


def test_exact_duplicate_projection_mismatch_requires_review():
    records = {
        "a": _record("a", head_horizontal="frame_left"),
        "b": _record("b", head_horizontal="frame_right"),
    }
    family = {
        "family_id": "f",
        "kind": "crop_family",
        "members": ["a", "b"],
        "wide_to_tight": ["a", "b"],
        "relative_fov_area": {"a": 1.0, "b": 1.0},
        "exact_duplicate_groups": [["a", "b"]],
    }
    out = mod.audit_family(family, records)
    assert out["status"] == "review"
    assert out["exact_duplicate_projection_mismatches"] == [{"a": "a", "b": "b"}]


def test_registry_covers_all_87_records_in_56_source_families():
    registry = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "notes"
            / "semantic_v3_blind_crop_family_registry_v01.json"
        ).read_text()
    )
    families = registry["families"]
    assert registry["record_count"] == 87
    assert registry["source_family_count"] == 56
    assert registry["multi_member_family_count"] == 26
    assert len(families) == 56
    assert sum(len(x["members"]) for x in families) == 87
    family_2 = next(
        x for x in families
        if x["family_id"] == "source-image-00002"
    )
    assert family_2["members"] == [
        "imageblind-01_00002",
        "imageblind-01_00046",
        "imageblind-01_00050",
    ]
    assert family_2["exact_duplicate_groups"] == [
        ["imageblind-01_00002", "imageblind-01_00046"]
    ]
