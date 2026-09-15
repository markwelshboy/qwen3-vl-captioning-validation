from __future__ import annotations

import copy
from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v10 as mod


def _sheet(*, mode="pose_guided", pose="standing", configuration=None):
    return {
        "schema_version": "caption-fact-sheet-0.2.8",
        "policy": {"mode": mode},
        "facts": {
            "body": {
                "pose_candidate": {
                    "text": pose,
                    "composer_text": pose,
                    "promotion_status": "accepted_candidate",
                } if pose is not None else None,
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
            "squatting": 84 if pose == "squatting" else 0,
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
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.9"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.9"


def test_00014_like_forward_relation_is_removed_but_other_configuration_survives():
    sheet = _sheet(
        pose="standing",
        configuration=[
            {
                "text": "torso bent forward",
                "composer_text": "upper body bent forward from the hips while overall stance remains mostly upright",
            },
            {"text": "both hands holding visible fabric", "composer_text": "both hands holding visible fabric"},
            {"text": "one foot planted", "composer_text": "left foot planted"},
            {"text": "other foot slightly lifted", "composer_text": "right foot slightly lifted"},
        ],
    )
    out = mod._apply_authoritative_adjudication(
        sheet,
        projected=_projected(pose="standing", lean="upright"),
    )
    body = out["facts"]["body"]
    torso = body["torso_relation_adjudication"]
    assert torso["status"] == "contradicted"
    assert torso["shadow_only"] is False
    assert torso["composer_authoritative"] is True
    assert torso["applied"] is True
    assert [x["composer_text"] for x in body["configuration"]] == [
        "both hands holding visible fabric",
        "left foot planted",
        "right foot slightly lifted",
    ]
    assert len(torso["removed_configuration_relations"]) == 1


def test_00039_like_slight_forward_relation_is_removed_when_specialist_is_upright():
    sheet = _sheet(
        pose="standing",
        configuration=[
            {"text": "torso slightly bent forward", "composer_text": "torso slightly bent forward"},
            {"text": "hands on hips", "composer_text": "hands on hips"},
        ],
    )
    out = mod._apply_authoritative_adjudication(
        sheet,
        projected=_projected(pose="standing", lean="upright"),
    )
    body = out["facts"]["body"]
    assert body["torso_relation_adjudication"]["status"] == "contradicted"
    assert [x["composer_text"] for x in body["configuration"]] == ["hands on hips"]


def test_00064_like_supported_torso_relation_is_kept_local():
    sheet = _sheet(
        pose="standing with one leg raised",
        configuration=[
            {
                "text": "torso bent forward",
                "composer_text": "upper body bent forward from the hips while overall stance remains mostly upright",
            },
            {"text": "one knee raised", "composer_text": "left knee raised high"},
            {"text": "one foot planted", "composer_text": "right foot planted"},
        ],
    )
    out = mod._apply_authoritative_adjudication(
        sheet,
        projected=_projected(pose="standing", lean="slight", direction="forward"),
    )
    body = out["facts"]["body"]
    broad = body["broad_pose_adjudication"]
    torso = body["torso_relation_adjudication"]
    assert broad["status"] == "confirmed"
    assert broad["composer_authoritative"] is True
    assert torso["status"] == "supported"
    assert torso["composer_authoritative"] is True
    assert body["configuration"][0]["composer_text"] == "upper body bent forward from the hips"
    assert body["pose_candidate"]["composer_text"] == "standing with one leg raised"


def test_00066_like_crouch_replaces_composer_pose_and_retains_local_forward_torso():
    sheet = _sheet(
        pose="standing with torso bent forward",
        configuration=[
            {
                "text": "torso bent forward",
                "composer_text": "upper body bent forward from the hips while overall stance remains mostly upright",
            },
            {"text": "both hands holding visible fabric", "composer_text": "both hands holding visible fabric"},
            {"text": "one foot planted", "composer_text": "left foot planted"},
            {"text": "other foot slightly lifted", "composer_text": "right foot slightly lifted"},
        ],
    )
    out = mod._apply_authoritative_adjudication(
        sheet,
        projected=_projected(
            pose="crouching",
            path="observed_crouch_hip_knee_chain",
            lean="heavy",
            direction="forward",
        ),
    )
    body = out["facts"]["body"]
    broad = body["broad_pose_adjudication"]
    torso = body["torso_relation_adjudication"]
    assert broad["status"] == "adjudicated"
    assert broad["shadow_only"] is False
    assert broad["composer_authoritative"] is True
    assert broad["applied"] is True
    assert broad["canonical_pose_text"] == "crouching"
    assert body["pose_candidate"]["text"] == "standing with torso bent forward"
    assert body["pose_candidate"]["composer_text"] == "crouching"
    assert body["pose_candidate"]["normalized_text"] == "crouching"
    assert body["pose_candidate"]["specialist_owner"] == "sam3d_v16_broad_pose_adjudicator"
    assert torso["status"] == "supported"
    assert body["configuration"][0]["composer_text"] == "upper body bent forward from the hips"
    # Support/elevation semantics are deliberately not repaired by this phase.
    assert body["configuration"][2]["composer_text"] == "left foot planted"
    assert body["configuration"][3]["composer_text"] == "right foot slightly lifted"


def test_generic_pose_disagreement_remains_non_authoritative():
    sheet = _sheet(pose="crouching")
    out = mod._apply_authoritative_adjudication(
        sheet,
        projected=_projected(pose="standing"),
    )
    body = out["facts"]["body"]
    adj = body["broad_pose_adjudication"]
    assert adj["status"] == "conflict_no_authority_to_replace"
    assert adj["composer_authoritative"] is False
    assert adj["applied"] is False
    assert body["pose_candidate"]["composer_text"] == "crouching"


def test_explicit_seated_pose_remains_protected_and_unmutated():
    sheet = _sheet(pose="seated on a branch")
    original = copy.deepcopy(sheet)
    out = mod._apply_authoritative_adjudication(
        sheet,
        projected=_projected(pose="crouching", path="observed_crouch_hip_knee_chain"),
    )
    body = out["facts"]["body"]
    assert body["broad_pose_adjudication"]["status"] == "protected"
    assert body["broad_pose_adjudication"]["composer_authoritative"] is False
    assert body["pose_candidate"] == original["facts"]["body"]["pose_candidate"]


def test_non_pose_routes_still_abstain_without_mutation():
    for mode in ("framing_only", "configuration"):
        sheet = _sheet(mode=mode, pose=None, configuration=[])
        out = mod._apply_authoritative_adjudication(
            sheet,
            projected=_projected(pose="crouching", path="observed_crouch_hip_knee_chain"),
        )
        body = out["facts"]["body"]
        assert body["broad_pose_adjudication"]["status"] == "abstain"
        assert body["broad_pose_adjudication"]["composer_authoritative"] is False
        assert body["torso_relation_adjudication"]["status"] == "abstain"
        assert body["configuration"] == []
