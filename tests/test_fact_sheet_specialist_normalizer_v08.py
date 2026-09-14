from __future__ import annotations

import copy
from pathlib import Path

import pytest

from qwen_caption_validate import fact_sheet_specialist_normalizer_v08 as mod


def _sheet(
    *,
    mode: str = "pose_guided",
    pose: str | None = "standing",
    shape: str | None = "mostly_upright_over_support_leg",
    support_side: str = "right",
    configuration: list[dict] | None = None,
) -> dict:
    support = None
    if shape is not None:
        support = {
            "available": True,
            "composer_eligible": shape == "mostly_upright_over_support_leg",
            "overall_shape": shape,
            "support_side": support_side,
            "support_axis_angle_from_vertical_deg": 2.0,
        }
    return {
        "schema_version": "caption-fact-sheet-0.2.6",
        "policy": {"mode": mode},
        "facts": {
            "body": {
                "pose_candidate": {"text": pose, "composer_text": pose} if pose is not None else None,
                "configuration": copy.deepcopy(configuration or []),
                "support_geometry": support,
            }
        },
        "audit": {"warnings": [], "invariants": {}},
    }


def _deep_right_support_points() -> dict:
    # Hip->ankle is vertical, but the knee is displaced laterally: the support
    # chain is deeply flexed even though v07's global support axis looks upright.
    return {
        "right_hip": (0.0, 0.0),
        "right_knee": (80.0, 80.0),
        "right_ankle": (0.0, 160.0),
    }


def _straight_right_support_points() -> dict:
    return {
        "right_hip": (0.0, 0.0),
        "right_knee": (0.0, 80.0),
        "right_ankle": (0.0, 160.0),
    }


def test_phase4b7_defaults():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.7"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.7"


def test_deep_support_knee_vetoes_upright_shape_and_proposes_crouched_without_mutating_v07_facts():
    source = _sheet()
    original_support = copy.deepcopy(source["facts"]["body"]["support_geometry"])
    original_pose = copy.deepcopy(source["facts"]["body"]["pose_candidate"])
    original_configuration = copy.deepcopy(source["facts"]["body"]["configuration"])

    out = mod._apply_shadow_adjudication(source, points=_deep_right_support_points())
    body = out["facts"]["body"]

    support_adj = body["support_shape_adjudication"]
    assert support_adj["status"] == "adjudicated"
    assert support_adj["would_change"] is True
    assert support_adj["original_shape"] == "mostly_upright_over_support_leg"
    assert support_adj["proposed_shape"] == "deeply_flexed_over_support_leg"
    assert support_adj["evidence"]["support_knee_angle_deg"] < mod.DEEP_SUPPORT_KNEE_ANGLE_DEG
    assert support_adj["shadow_only"] is True
    assert support_adj["composer_authoritative"] is False

    pose_adj = body["pose_adjudication"]
    assert pose_adj["status"] == "adjudicated"
    assert pose_adj["would_change"] is True
    assert pose_adj["proposed_pose"] == "crouched"

    assert body["support_geometry"] == original_support
    assert body["pose_candidate"] == original_pose
    assert body["configuration"] == original_configuration
    assert source["facts"]["body"]["support_geometry"] == original_support


def test_straight_support_knee_preserves_upright_shape():
    out = mod._apply_shadow_adjudication(_sheet(), points=_straight_right_support_points())
    body = out["facts"]["body"]

    assert body["support_shape_adjudication"]["status"] == "preserved"
    assert body["support_shape_adjudication"]["would_change"] is False
    assert body["support_shape_adjudication"]["proposed_shape"] is None
    assert body["support_shape_adjudication"]["evidence"]["support_knee_angle_deg"] == 180.0
    assert body["pose_adjudication"]["status"] == "preserved"
    assert body["pose_adjudication"]["would_change"] is False


def test_explicit_seated_object_supported_pose_is_protected_even_with_deep_knee_flexion():
    source = _sheet(
        pose="seated",
        configuration=[
            {"text": "back against tree trunk", "composer_text": "back against tree trunk"},
            {"text": "knees bent", "composer_text": "knees bent"},
        ],
    )
    out = mod._apply_shadow_adjudication(source, points=_deep_right_support_points())
    body = out["facts"]["body"]

    assert body["support_shape_adjudication"]["status"] == "protected"
    assert body["support_shape_adjudication"]["would_change"] is False
    assert body["support_shape_adjudication"]["reason"] == "explicit_seated_object_supported_context"
    assert body["pose_adjudication"]["status"] == "protected"
    assert body["pose_adjudication"]["proposed_pose"] is None
    assert body["pose_adjudication"]["would_change"] is False


@pytest.mark.parametrize("mode", ["framing_only", "configuration"])
def test_crop_policy_modes_abstain_before_pose_adjudication(mode: str):
    out = mod._apply_shadow_adjudication(
        _sheet(mode=mode),
        points=_deep_right_support_points(),
    )
    body = out["facts"]["body"]

    assert body["support_shape_adjudication"]["status"] == "abstain"
    assert body["support_shape_adjudication"]["would_change"] is False
    assert body["pose_adjudication"]["status"] == "abstain"
    assert body["pose_adjudication"]["would_change"] is False


def test_missing_support_knee_chain_is_insufficient_not_a_pose_guess():
    out = mod._apply_shadow_adjudication(
        _sheet(),
        points={"right_hip": (0.0, 0.0), "right_ankle": (0.0, 160.0)},
    )
    body = out["facts"]["body"]

    assert body["support_shape_adjudication"]["status"] == "insufficient_evidence"
    assert body["support_shape_adjudication"]["would_change"] is False
    assert body["pose_adjudication"]["status"] == "insufficient_evidence"
    assert body["pose_adjudication"]["proposed_pose"] is None


def test_non_upright_v07_support_shape_is_outside_adjudication_target():
    out = mod._apply_shadow_adjudication(
        _sheet(shape="unresolved"),
        points=_deep_right_support_points(),
    )
    body = out["facts"]["body"]

    assert body["support_shape_adjudication"]["status"] == "preserved"
    assert body["support_shape_adjudication"]["would_change"] is False
    assert body["pose_adjudication"]["status"] == "preserved"


def test_explicit_nonstanding_pose_is_not_reclassified_by_deep_flexion():
    out = mod._apply_shadow_adjudication(
        _sheet(pose="kneeling"),
        points=_deep_right_support_points(),
    )
    body = out["facts"]["body"]

    assert body["support_shape_adjudication"]["status"] == "adjudicated"
    assert body["support_shape_adjudication"]["would_change"] is True
    assert body["pose_adjudication"]["status"] == "preserved"
    assert body["pose_adjudication"]["would_change"] is False
    assert body["pose_adjudication"]["proposed_pose"] is None


def test_missing_v07_support_geometry_is_insufficient_even_if_pose_is_noncommittal():
    out = mod._apply_shadow_adjudication(
        _sheet(pose=None, shape=None),
        points=_deep_right_support_points(),
    )
    body = out["facts"]["body"]

    assert body["support_shape_adjudication"]["status"] == "insufficient_evidence"
    assert body["support_shape_adjudication"]["reason"] == "v07_support_geometry_unavailable"
    assert body["pose_adjudication"]["status"] == "insufficient_evidence"
    assert body["pose_adjudication"]["proposed_pose"] is None
