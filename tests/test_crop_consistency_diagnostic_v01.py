from __future__ import annotations

import json
from pathlib import Path

from qwen_caption_validate import crop_consistency_diagnostic_v01 as mod


def _projection_record(
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


def _fact_sheet(
    key: str,
    *,
    broad: str | None = None,
    broad_supported: bool = False,
    torso_direction: str | None = None,
) -> dict:
    pose_candidate = None
    adjudication = {
        "status": "abstain",
        "composer_authoritative": False,
    }
    if broad is not None:
        pose_candidate = {"text": broad}
        adjudication = {
            "status": "confirmed",
            "composer_authoritative": True,
            "canonical_pose_text": broad,
        }
    body = {
        "pose_candidate": pose_candidate,
        "configuration": [],
        "broad_pose_adjudication": adjudication,
        "torso_geometry": {
            "composer_eligible": torso_direction is not None,
            "caption_orientation": {
                "preferred_orientation_band": "three_quarter",
                "preferred_approx_yaw_deg": 45,
                "preferred_turn_direction": torso_direction,
                "turn_direction_publishable": torso_direction is not None,
            },
        },
    }
    return {
        "image_key": key,
        "status": "ok",
        "facts": {
            "framing": {
                "broad_pose_supported": broad_supported,
                "anatomical_span": {
                    "status": "available",
                    "upper_anchor": "head",
                    "lower_anchor": "ankles",
                },
                "standard_shot_scale": {
                    "status": "withheld",
                    "label": None,
                },
                "composer_framing": {
                    "source": "anatomical_span",
                    "composer_text": "framed from the head through the ankles",
                },
            },
            "body": body,
            "visual": {
                "appearance": [],
                "objects": [],
                "scene": [],
            },
            "head_pose": {},
            "gaze": {},
        },
    }


def test_projection_snapshot_groups_specific_standing_phrase_under_standing():
    snap = mod.snapshot_projection(
        _projection_record(
            "wide",
            broad="standing with one leg raised",
        )
    )
    assert snap["body"]["broad_pose_group"] == "standing"
    assert snap["framing"]["coverage_score"] == 3.0


def test_fact_sheet_snapshot_uses_pre_composer_pose_authority():
    snap = mod.snapshot_fact_sheet(
        _fact_sheet(
            "wide",
            broad="standing with one leg raised",
            broad_supported=True,
            torso_direction="frame_left",
        )
    )
    assert snap["body"]["broad_pose_group"] == "standing"
    assert snap["body"]["torso"]["turn_direction"] == "frame_left"
    assert snap["framing"]["composer_text"] == "framed from the head through the ankles"


def test_crop_comparison_treats_pose_loss_as_collapse_not_contradiction():
    wide = mod.snapshot_projection(
        _projection_record(
            "wide",
            framing="medium-wide shot",
            shot="medium_wide",
            broad="standing",
        )
    )
    tight = mod.snapshot_projection(
        _projection_record(
            "tight",
            framing="close-up",
            shot="close_up",
        )
    )
    result = mod.compare_snapshots(wide, tight)
    broad = next(
        x
        for x in result["semantic_transitions"]
        if x["domain"] == "broad_pose_group"
    )
    assert broad["state"] == "collapsed"
    assert broad["contradiction"] is False
    assert result["contradictions"] == []


def test_crop_comparison_flags_surviving_direction_flip():
    wide = mod.snapshot_projection(
        _projection_record(
            "wide",
            torso_direction="frame_left",
        )
    )
    tight = mod.snapshot_projection(
        _projection_record(
            "tight",
            torso_direction="frame_right",
        )
    )
    result = mod.compare_snapshots(wide, tight)
    assert any(
        x["domain"] == "torso.turn_direction"
        for x in result["contradictions"]
    )


def test_family_compares_fact_sheet_first_then_projection():
    family = {
        "family_id": "f",
        "kind": "crop_family",
        "members": ["wide", "tight"],
        "wide_to_tight": ["wide", "tight"],
        "relative_fov_area": {
            "wide": 2.0,
            "tight": 1.0,
        },
    }
    fact_sheets = {
        "wide": _fact_sheet(
            "wide",
            broad="standing",
            broad_supported=True,
        ),
        "tight": _fact_sheet(
            "tight",
            broad=None,
            broad_supported=False,
        ),
    }
    records = {
        "wide": _projection_record(
            "wide",
            broad="standing",
        ),
        "tight": _projection_record(
            "tight",
        ),
    }

    out = mod.audit_family(
        family,
        fact_sheets,
        records,
    )

    assert out["status"] == "consistent"
    assert out["fact_sheet"]["ordered_wide_to_tight"] == [
        "wide",
        "tight",
    ]
    assert out["composer_projection"]["ordered_wide_to_tight"] == [
        "wide",
        "tight",
    ]
    fact_broad = next(
        x
        for x in out["fact_sheet"]["pairwise_crop_transitions"][0][
            "semantic_transitions"
        ]
        if x["domain"] == "broad_pose_group"
    )
    projection_broad = next(
        x
        for x in out["composer_projection"]["pairwise_crop_transitions"][0][
            "semantic_transitions"
        ]
        if x["domain"] == "broad_pose_group"
    )
    assert fact_broad["state"] == "collapsed"
    assert projection_broad["state"] == "collapsed"


def test_exact_duplicate_semantic_mismatch_requires_review():
    family = {
        "family_id": "f",
        "kind": "crop_family",
        "members": ["a", "b"],
        "wide_to_tight": ["a", "b"],
        "relative_fov_area": {
            "a": 1.0,
            "b": 1.0,
        },
        "exact_duplicate_groups": [["a", "b"]],
    }
    fact_sheets = {
        "a": _fact_sheet("a"),
        "b": _fact_sheet("b"),
    }
    records = {
        "a": _projection_record(
            "a",
            head_horizontal="frame_left",
        ),
        "b": _projection_record(
            "b",
            head_horizontal="frame_right",
        ),
    }

    out = mod.audit_family(
        family,
        fact_sheets,
        records,
    )

    assert out["status"] == "review"
    assert out["fact_sheet"]["exact_duplicate_semantic_mismatches"] == []
    assert out["composer_projection"][
        "exact_duplicate_semantic_mismatches"
    ] == [{"a": "a", "b": "b"}]


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
        x
        for x in families
        if x["family_id"] == "source-image-00002"
    )
    assert family_2["members"] == [
        "imageblind-01_00002",
        "imageblind-01_00046",
        "imageblind-01_00050",
    ]
    assert family_2["exact_duplicate_groups"] == [
        [
            "imageblind-01_00002",
            "imageblind-01_00046",
        ]
    ]
