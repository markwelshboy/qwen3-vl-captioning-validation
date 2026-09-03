from __future__ import annotations

"""v0.14 targeted physical-governance refinement.

This pass intentionally leaves the broad posture families, v0.12 foot-contact
hull, v0.13 retreat logic, single-leg topology, and crop-authority model alone.
It addresses three narrow review failures:

* a strongly reclined whole-body plane must suppress the v0.13 ordinary-sitting
  promotion even when flexed legs resemble a seated posture;
* an open hand near the face is not head support unless palm/wrist proximity
  tracks the distal hand-to-face proximity (peace-sign / fingertip guard);
* a deep squat needs enough torso compensation relative to pelvis->foot
  displacement, not merely some absolute forward shoulder motion.

As before, reconstructed geometry and observed crop authority remain separate.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_13 as v13


v12 = v13.v12
v09 = v13.v09

# Whole-body recline vs ordinary sitting.  Review controls place obvious bed
# recline around 0.58-0.60 while ordinary/reclined sitting controls are much
# lower (~0.1-0.2), leaving a useful separation band without broad retuning.
WHOLE_BODY_RECLINE_START = 0.45
WHOLE_BODY_RECLINE_FULL = 0.62
WHOLE_BODY_RECLINE_MIN_RETREAT = 0.55
WHOLE_BODY_RECLINE_MIN_EXTERNAL = 0.35
WHOLE_BODY_SITTING_MAX_SUPPRESSION = 0.78

# Relative torso compensation for a squat.  A compensation fraction near 1.0
# means the shoulders have moved toward the feet by roughly the amount demanded
# by pelvis displacement.  Values around 0.6 are insufficient for a deep squat;
# ~0.95+ is intentionally left untouched.
SQUAT_COMP_MIN_FLEX = 0.70
SQUAT_COMP_MIN_NEED = 0.30
SQUAT_COMP_MIN_ADVANCE = 0.25
SQUAT_COMP_LOW = 0.62
SQUAT_COMP_FULL = 0.95
SQUAT_COMP_MAX_SUPPRESSION = 0.60

# Open-hand support requires the proximal chain to arrive with the distal hand.
# These are *gaps* relative to the existing hand-to-face distance, not new
# absolute proximity thresholds.  Closed-fist support retains the v0.12 guard.
OPEN_HAND_MAX_PALM_EXCESS_SW = 0.18
OPEN_HAND_MAX_WRIST_EXCESS_SW = 0.30


def _round(value: float | None, digits: int = 3) -> float | None:
    return v13._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v13._ramp(value, low, high)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _apply_factor(row: dict[str, Any], factor: float, label: str) -> None:
    if row.get("hard_rejected"):
        return
    factor = _clamp(factor)
    current = float(row.get("governed_score") or 0.0)
    row[label] = _round(factor, 4)
    row["governed_score"] = _round(current * factor, 4)
    row["governed_score_percent"] = int(round(100.0 * float(row["governed_score"])))


def _whole_body_recline_override(profile: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    rows = governance.get("per_pose") or {}
    if not rows:
        return

    recline_diag = projected.get("recline_diagnostic") or {}
    directional = projected.get("directional_recline_diagnostic") or {}
    support = projected.get("independent_support_diagnostic") or {}

    lower_recline = float(recline_diag.get("score") or 0.0)
    flatness = float(recline_diag.get("body_flatness_ratio") or 0.0)
    retreat = float(directional.get("retreat_from_support_score") or 0.0)
    external = float(support.get("external_support_requirement") or 0.0)

    direction_gate = _ramp(retreat, 0.35, 0.85)
    external_gate = _ramp(external, 0.25, 0.75)
    whole_body_recline = _clamp(
        lower_recline * (0.72 + 0.18 * direction_gate + 0.10 * external_gate)
    )
    applicable = bool(
        lower_recline >= WHOLE_BODY_RECLINE_START
        and retreat >= WHOLE_BODY_RECLINE_MIN_RETREAT
        and external >= WHOLE_BODY_RECLINE_MIN_EXTERNAL
    )

    sitting_factor = 1.0
    if applicable:
        strength = _ramp(
            whole_body_recline,
            WHOLE_BODY_RECLINE_START,
            WHOLE_BODY_RECLINE_FULL,
        )
        sitting_factor = 1.0 - WHOLE_BODY_SITTING_MAX_SUPPRESSION * strength
        sitting = rows.get("sitting") or {}
        _apply_factor(sitting, sitting_factor, "v14_whole_body_recline_sitting_factor")
        rows["sitting"] = sitting

        # Do not fabricate a new recline score; merely ensure the already-derived
        # whole-body diagnostic is not lost beneath later family competition.
        reclined = rows.get("reclined") or {}
        if not reclined.get("hard_rejected"):
            current = float(reclined.get("governed_score") or 0.0)
            reclined["v14_whole_body_recline_candidate"] = _round(whole_body_recline, 4)
            reclined["governed_score"] = _round(max(current, whole_body_recline), 4)
            reclined["governed_score_percent"] = int(round(100.0 * float(reclined["governed_score"])))
        rows["reclined"] = reclined

    governance["per_pose"] = rows
    governance["v14_whole_body_recline_override"] = {
        "lower_body_recline_score": _round(lower_recline, 4),
        "body_flatness_ratio": _round(flatness),
        "retreat_from_support": _round(retreat, 4),
        "external_support_requirement": _round(external, 4),
        "direction_gate": _round(direction_gate, 4),
        "external_gate": _round(external_gate, 4),
        "whole_body_recline_score": _round(whole_body_recline, 4),
        "whole_body_recline_score_percent": int(round(100.0 * whole_body_recline)),
        "applied": applicable,
        "sitting_factor": _round(sitting_factor, 4),
        "interpretation": (
            "A strong reclined body plane plus torso retreat and external-support geometry "
            "suppresses ordinary sitting. Flexed legs alone cannot re-promote sitting over "
            "a clearly reclined whole-body topology."
        ),
    }
    projected["physical_governance"] = governance
    projected["whole_body_recline_override"] = governance["v14_whole_body_recline_override"]
    profile["sam3d_projected_pose"] = projected


def _relative_squat_compensation(profile: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    rows = governance.get("per_pose") or {}
    support = projected.get("independent_support_diagnostic") or {}
    geometry = support.get("geometry") or {}
    leg = projected.get("leg_state_diagnostic") or {}
    directional = projected.get("directional_recline_diagnostic") or {}
    if not rows or not geometry.get("available"):
        return

    flex = float(leg.get("bilateral_flexion_score") or 0.0)
    advance = float(directional.get("advance_toward_support_score") or 0.0)
    retreat = float(directional.get("retreat_from_support_score") or 0.0)
    comp_value = geometry.get("shoulder_compensation_fraction")
    need_value = geometry.get("compensation_need_score")
    compensation = float(comp_value) if comp_value is not None else None
    need = float(need_value) if need_value is not None else 0.0

    squat = rows.get("squatting") or {}
    applicable = bool(
        compensation is not None
        and not squat.get("hard_rejected")
        and flex >= SQUAT_COMP_MIN_FLEX
        and need >= SQUAT_COMP_MIN_NEED
        and advance >= SQUAT_COMP_MIN_ADVANCE
        and retreat < 0.20
    )
    factor = 1.0
    sufficiency = None
    if applicable and compensation is not None:
        sufficiency = _ramp(compensation, SQUAT_COMP_LOW, SQUAT_COMP_FULL)
        factor = 1.0 - SQUAT_COMP_MAX_SUPPRESSION * (1.0 - sufficiency)
        _apply_factor(squat, factor, "v14_relative_torso_compensation_factor")
        rows["squatting"] = squat

    governance["per_pose"] = rows
    governance["v14_relative_squat_compensation"] = {
        "bilateral_flexion": _round(flex, 4),
        "advance_toward_support": _round(advance, 4),
        "retreat_from_support": _round(retreat, 4),
        "compensation_need": _round(need, 4),
        "shoulder_compensation_fraction": _round(compensation, 4) if compensation is not None else None,
        "compensation_sufficiency": _round(sufficiency, 4) if sufficiency is not None else None,
        "applied": applicable,
        "squat_factor": _round(factor, 4),
        "interpretation": (
            "For a flexed squat with displaced pelvis geometry, absolute forward shoulder "
            "motion is insufficient. The torso must compensate by a substantial fraction of "
            "the pelvis-to-foot displacement; near-unity compensation leaves squat untouched."
        ),
    }
    projected["physical_governance"] = governance
    projected["relative_squat_compensation"] = governance["v14_relative_squat_compensation"]
    profile["sam3d_projected_pose"] = projected


def _open_hand_proximal_support_guard(profile: dict[str, Any]) -> None:
    relations = profile.get("relations") or {}
    broad = relations.get("head_supported_by_hand") or {}
    if not broad.get("geometry_match"):
        return

    shape = str(broad.get("hand_shape_label") or "")
    open_score = float(broad.get("open_hand_score") or 0.0)
    if shape != "open_hand" and open_score < 0.55:
        return

    topology = broad.get("support_topology_guard") or {}
    hand_face_value = broad.get("hand_to_face_shoulder_widths")
    palm_value = topology.get("palm_root_to_head_shoulder_widths")
    wrist_value = topology.get("wrist_to_head_shoulder_widths")
    if hand_face_value is None or palm_value is None or wrist_value is None:
        broad["v14_proximal_chain_guard"] = {"available": False}
        relations["head_supported_by_hand"] = broad
        profile["relations"] = relations
        return

    hand_face = float(hand_face_value)
    palm = float(palm_value)
    wrist = float(wrist_value)
    palm_excess = palm - hand_face
    wrist_excess = wrist - hand_face
    match = bool(
        palm_excess <= OPEN_HAND_MAX_PALM_EXCESS_SW
        and wrist_excess <= OPEN_HAND_MAX_WRIST_EXCESS_SW
    )
    guard = {
        "available": True,
        "geometry_match": match,
        "hand_to_face_shoulder_widths": _round(hand_face),
        "palm_root_to_head_shoulder_widths": _round(palm),
        "wrist_to_head_shoulder_widths": _round(wrist),
        "palm_excess_over_distal_hand_shoulder_widths": _round(palm_excess),
        "wrist_excess_over_distal_hand_shoulder_widths": _round(wrist_excess),
        "thresholds": {
            "max_open_hand_palm_excess_shoulder_widths": OPEN_HAND_MAX_PALM_EXCESS_SW,
            "max_open_hand_wrist_excess_shoulder_widths": OPEN_HAND_MAX_WRIST_EXCESS_SW,
        },
        "interpretation": (
            "Open-hand head support requires the palm root and wrist to arrive with the "
            "distal hand. Fingertip/peace-sign proximity without proximal-chain proximity "
            "is not a support topology."
        ),
    }
    broad["v14_proximal_chain_guard"] = guard

    if not match:
        broad["geometry_match_before_v14_proximal_chain_guard"] = True
        broad["geometry_match"] = False
        broad["crop_support_before_v14_proximal_chain_guard"] = broad.get("crop_support")
        broad["crop_support"] = 0.0
        broad["crop_support_percent"] = 0
        broad["support_class"] = "not_matched"
        broad["rejection_reason"] = "open_hand_distal_proximity_without_proximal_palm_wrist_support"

        fist = relations.get("head_supported_by_fist") or {}
        if fist.get("geometry_match"):
            fist["geometry_match_before_v14_proximal_chain_guard"] = True
            fist["geometry_match"] = False
            fist["crop_support"] = 0.0
            fist["crop_support_percent"] = 0
            fist["support_class"] = "not_matched"
            fist["rejection_reason"] = "parent_head_hand_support_topology_rejected_v14"
            relations["head_supported_by_fist"] = fist

    relations["head_supported_by_hand"] = broad
    profile["relations"] = relations


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v13.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.14"

    _whole_body_recline_override(profile)
    _relative_squat_compensation(profile)
    _open_hand_proximal_support_guard(profile)
    v12._recompute_public(profile)

    policy = profile.get("policy") or {}
    policy.update({
        "v14_whole_body_recline_suppresses_ordinary_sitting": True,
        "v14_open_hand_head_support_requires_proximal_chain": True,
        "v14_squat_requires_relative_torso_compensation": True,
        "v14_preserves_v12_support_hull_and_v13_retreat_governance": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-14",
        description=(
            "Build v0.14 governed pose profiles with whole-body recline precedence, "
            "open-hand proximal head-support topology, and relative squat compensation."
        ),
    )
    parser.add_argument("sam3d_dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")
    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else sam3d_dir.parent / "dwpose"
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else sam3d_dir.parent / "images"
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.14")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    helpers = v09.v08.v07.v06.v05.v04.v03
    rows: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        dwpose = helpers._read_json(dwpose_path)
        width = int(dwpose.get("image_width") or 0)
        height = int(dwpose.get("image_height") or 0)
        if width <= 0 or height <= 0:
            image_matches = [p for p in images_dir.rglob(f"{key}.*") if p.is_file()] if images_dir.is_dir() else []
            if not image_matches:
                raise SystemExit(f"Cannot determine image size for {key}")
            with Image.open(image_matches[0]) as im:
                width, height = im.size

        profile = build_profile(arrays, dwpose or None, width, height)
        record = {
            "image_key": key,
            "sam3d_arrays": str(path),
            "dwpose": str(dwpose_path) if dwpose_path.is_file() else None,
            "image_width": width,
            "image_height": height,
            "profile": profile,
        }
        out_path = output / f"{key}.sam3d_relational_pose.json"
        helpers._write_json(out_path, record)
        rows.append(record)

        projected = profile["sam3d_projected_pose"]
        scores = projected.get("posture_score_percent") or {}
        whole = projected.get("whole_body_recline_override") or {}
        comp = projected.get("relative_squat_compensation") or {}
        head = (profile.get("relations") or {}).get("head_supported_by_hand") or {}
        governance = projected.get("physical_governance") or {}
        authority = governance.get("authority") or {}
        print(
            f"{key}: pose={projected.get('pose')} best={projected.get('best_candidate_pose')} "
            f"scores=stand:{scores.get('standing',0)} crouch:{scores.get('crouching',0)} "
            f"squat:{scores.get('squatting',0)} sit:{scores.get('sitting',0)} recl:{scores.get('reclined',0)} "
            f"whole_recl:{whole.get('whole_body_recline_score_percent',0)} "
            f"recl_override:{'Y' if whole.get('applied') else '-'} "
            f"comp:{comp.get('shoulder_compensation_fraction')} "
            f"squat_factor:{comp.get('squat_factor',1.0)} "
            f"head_support:{'Y' if head.get('geometry_match') else '-'} "
            f"authority:{authority.get('crop_support_percent',0)}%"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.14",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    helpers._write_json(output / "sam3d_relational_pose.index.json", index)
    print(f"Index: {output / 'sam3d_relational_pose.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
