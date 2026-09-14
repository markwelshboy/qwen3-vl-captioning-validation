from __future__ import annotations

import copy
from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v09 as mod


def _sheet(*, mode="pose_guided", pose="standing", configuration=None):
    return {
        "schema_version": "caption-fact-sheet-0.2.6",
        "policy": {"mode": mode},
        "facts": {
            "body": {
                "pose_candidate": {"text": pose, "composer_text": pose} if pose is not None else None,
                "configuration": copy.deepcopy(configuration or []),
            }
        },
        "audit": {"warnings": [], "invariants": {}},
    }


def _projected(*, pose, best=None, path="posture_region_weights", lean="upright", direction="direction_indeterminate"):
    return {
        "pose": pose,
        "best_candidate_pose": best or pose,
        "posture_score_percent": {
            "standing": 84 if pose == "standing" else 0,
            "crouching": 92 if pose == "crouching" else 0,
        },
        "crop_support_percent": 93,
        "assertion_authority": {
            "selected_path": path,
            "selected_path_authority_percent": 93,
            "withheld_reason": None,
        },
        "posture_modifier_diagnostic": {
            "lean_severity": lean,
            "lean_direction": direction,
            "torso_inclination_from_vertical_deg": 10.0 if lean == "upright" else 30.0,
            "shoulder_line_declination_deg": 13.0,
        },
    }


def test_defaults():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.8"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.8"


def test_00014_like_standing_is_confirmed_and_forward_relation_is_contradicted():
    sheet = _sheet(
        pose="standing",
        configuration=[{"text": "torso bent forward", "composer_text": "torso bent forward"}],
    )
    out = mod._apply_shadow_adjudication(
        sheet,
        projected=_projected(pose="standing", lean="upright", direction="direction_indeterminate"),
    )
    body = out["facts"]["body"]
    assert body["broad_pose_adjudication"]["status"] == "confirmed"
    assert body["broad_pose_adjudication"]["would_change"] is False
    assert body["torso_relation_adjudication"]["status"] == "contradicted"
    assert body["torso_relation_adjudication"]["would_change"] is True


def test_00064_like_standing_is_confirmed_and_forward_relation_is_supported():
    sheet = _sheet(
        pose="standing with one leg raised",
        configuration=[{"text": "torso bent forward", "composer_text": "torso bent forward"}],
    )
    out = mod._apply_shadow_adjudication(
        sheet,
        projected=_projected(pose="standing", lean="slight", direction="forward"),
    )
    body = out["facts"]["body"]
    assert body["broad_pose_adjudication"]["status"] == "confirmed"
    assert body["broad_pose_adjudication"]["qwen_pose_family"] == "standing"
    assert body["torso_relation_adjudication"]["status"] == "supported"
    assert body["torso_relation_adjudication"]["would_change"] is False


def test_00066_like_observed_crouch_path_can_challenge_qwen_standing():
    sheet = _sheet(pose="standing")
    out = mod._apply_shadow_adjudication(
        sheet,
        projected=_projected(
            pose="crouching",
            path="observed_crouch_hip_knee_chain",
            lean="heavy",
            direction="forward",
        ),
    )
    adj = out["facts"]["body"]["broad_pose_adjudication"]
    assert adj["status"] == "adjudicated"
    assert adj["would_change"] is True
    assert adj["proposed_pose_family"] == "crouching"


def test_observed_crouch_path_can_supply_noncommittal_pose_candidate():
    sheet = _sheet(pose=None)
    out = mod._apply_shadow_adjudication(
        sheet,
        projected=_projected(pose="crouching", path="observed_crouch_hip_knee_chain"),
    )
    adj = out["facts"]["body"]["broad_pose_adjudication"]
    assert adj["status"] == "adjudicated"
    assert adj["would_change"] is True
    assert adj["proposed_pose_family"] == "crouching"


def test_generic_sam3d_disagreement_does_not_get_replacement_authority():
    sheet = _sheet(pose="crouching")
    out = mod._apply_shadow_adjudication(sheet, projected=_projected(pose="standing"))
    adj = out["facts"]["body"]["broad_pose_adjudication"]
    assert adj["status"] == "conflict_no_authority_to_replace"
    assert adj["would_change"] is False


def test_explicit_seated_pose_is_protected_before_specialist():
    sheet = _sheet(pose="seated on a branch")
    out = mod._apply_shadow_adjudication(
        sheet,
        projected=_projected(pose="crouching", path="observed_crouch_hip_knee_chain"),
    )
    body = out["facts"]["body"]
    assert body["broad_pose_adjudication"]["status"] == "protected"
    assert body["broad_pose_adjudication"]["would_change"] is False
    assert body["torso_relation_adjudication"]["status"] == "protected"


def test_framing_and_configuration_routes_abstain():
    for mode in ("framing_only", "configuration"):
        sheet = _sheet(mode=mode, pose="standing")
        out = mod._apply_shadow_adjudication(
            sheet,
            projected=_projected(pose="crouching", path="observed_crouch_hip_knee_chain"),
        )
        assert out["facts"]["body"]["broad_pose_adjudication"]["status"] == "abstain"


def test_shadow_records_do_not_rewrite_existing_pose_or_configuration():
    configuration = [{"text": "torso bent forward", "composer_text": "torso bent forward"}]
    sheet = _sheet(pose="standing", configuration=configuration)
    original = copy.deepcopy(sheet)
    out = mod._apply_shadow_adjudication(
        sheet,
        projected=_projected(pose="crouching", path="observed_crouch_hip_knee_chain"),
    )
    assert out["facts"]["body"]["pose_candidate"] == original["facts"]["body"]["pose_candidate"]
    assert out["facts"]["body"]["configuration"] == original["facts"]["body"]["configuration"]
