from __future__ import annotations

"""v0.13 seated-vs-low-stance topology refinement.

v0.12 fixed the foot support model, but its low-support-authority fallback was
still too permissive: a support-based hard veto could be removed and the old
crouch/squat score restored almost intact.  This layer keeps v0.12 and changes
that fallback into a soft, directional feasibility decision.

Key rules:
* unseen feet cannot hard-prove a low stance impossible, but they also cannot
  restore crouch/squat to the raw additive score;
* a flexed body whose shoulders retreat away from the reconstructed support
  direction is strongly incompatible with an unsupported crouch/squat;
* flexed legs + horizontal-ish thighs + upright/backward torso form a positive
  seated topology and a negative squat topology;
* strong forward compensation may rescue a genuinely weight-bearing squat from
  an over-pessimistic support-area veto;
* public crop authority remains separate from reconstructed pose ranking.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_12 as v12


v11 = v12.v11
v09 = v12.v09
v08 = v12.v08
MHR = v12.MHR

DEFERRED_BASE_FACTOR = 0.42
DEFERRED_FORWARD_BONUS = 0.46
RETREAT_CROUCH_MAX_SUPPRESSION = 0.90
RETREAT_SQUAT_MAX_SUPPRESSION = 0.95
SEATED_TOPOLOGY_MIN_FLEX = 0.45
SEATED_TOPOLOGY_MIN_RETREAT = 0.20
SQUAT_COUNTEREVIDENCE_MAX_SUPPRESSION = 0.78
FORWARD_SUPPORT_RESCUE_MIN_ADVANCE = 0.65
FORWARD_SUPPORT_RESCUE_MIN_FLEX = 0.70
FORWARD_SUPPORT_RESCUE_MIN_RAW_SQUAT = 0.55
FORWARD_SUPPORT_RESCUE_MIN_SUPPORT_CROP = 0.50


def _round(value: float | None, digits: int = 3) -> float | None:
    return v12._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v12._ramp(value, low, high)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _apply_current_factor(row: dict[str, Any], factor: float, label: str) -> None:
    if row.get("hard_rejected"):
        return
    factor = _clamp(factor)
    current = float(row.get("governed_score") or 0.0)
    row[label] = _round(factor, 4)
    row["governed_score"] = _round(current * factor, 4)
    row["governed_score_percent"] = int(round(100.0 * float(row["governed_score"])))


def _unreject_for_forward_supported_squat(
    row: dict[str, Any],
    *,
    flex: float,
    advance: float,
    support_crop: float,
) -> bool:
    if not row.get("hard_rejected"):
        return False
    reasons = set(row.get("hard_rejection_reasons") or [])
    support_only = {
        "bilateral_flexion_with_near_zero_foot_support_feasibility",
        "strong_external_support_requirement",
    }
    raw = float(row.get("raw_score") or 0.0)
    if (
        not reasons
        or not reasons.issubset(support_only)
        or flex < FORWARD_SUPPORT_RESCUE_MIN_FLEX
        or advance < FORWARD_SUPPORT_RESCUE_MIN_ADVANCE
        or support_crop < FORWARD_SUPPORT_RESCUE_MIN_SUPPORT_CROP
        or raw < FORWARD_SUPPORT_RESCUE_MIN_RAW_SQUAT
    ):
        return False

    factor = _clamp(0.72 + 0.28 * advance)
    row["hard_rejected"] = False
    row["hard_rejection_reasons"] = []
    row["rescued_hard_rejection_reasons"] = sorted(reasons)
    row["forward_supported_squat_rescue"] = True
    row["soft_feasibility_factor"] = _round(factor, 4)
    row["governed_score"] = _round(raw * factor, 4)
    row["governed_score_percent"] = int(round(100.0 * float(row["governed_score"])))
    return True


def _seated_low_stance_refine(profile: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    rows = governance.get("per_pose") or {}
    if not rows:
        return

    leg = projected.get("leg_state_diagnostic") or {}
    support = projected.get("independent_support_diagnostic") or {}
    directional = projected.get("directional_recline_diagnostic") or {}
    geometry = projected.get("geometry") or {}

    flex = float(leg.get("bilateral_flexion_score") or 0.0)
    feasibility = float(support.get("support_feasibility_score") or 0.0)
    external = float(support.get("external_support_requirement") or 0.0)
    support_crop = float(support.get("crop_support") or 0.0)
    retreat = float(directional.get("retreat_from_support_score") or 0.0)
    advance = float(directional.get("advance_toward_support_score") or 0.0)
    raw_upper_inclination = float(directional.get("raw_upper_body_inclination_score") or 0.0)
    directional_recline = float(directional.get("directional_upper_recline_score") or 0.0)
    shift = directional.get("shoulder_shift_toward_feet_shoulder_widths")

    thigh_angle_value = geometry.get("mean_thigh_axis_from_image_down_deg")
    thigh_angle = float(thigh_angle_value) if thigh_angle_value is not None else None
    thigh_horizontal = _ramp(thigh_angle, 35.0, 80.0) if thigh_angle is not None else 0.35
    torso_upright = _clamp(1.0 - raw_upper_inclination)

    # v0.12 restored low-stance scores to ~raw when the foot support itself was
    # poorly observed.  Replace that with a soft fallback: neutral unseen support
    # gets only moderate credit; forward compensation restores confidence; retreat
    # never does.
    deferred_rows: list[str] = []
    for name in ("crouching", "squatting"):
        row = rows.get(name) or {}
        if not row.get("support_veto_deferred"):
            continue
        deferred_rows.append(name)
        raw = float(row.get("raw_score") or 0.0)
        forward_rescue = _ramp(advance, 0.15, 0.75)
        neutral_feasibility = 0.10 * feasibility
        factor = _clamp(DEFERRED_BASE_FACTOR + DEFERRED_FORWARD_BONUS * forward_rescue + neutral_feasibility)
        row["v13_deferred_support_soft_factor"] = _round(factor, 4)
        row["governed_score"] = _round(raw * factor, 4)
        row["governed_score_percent"] = int(round(100.0 * float(row["governed_score"])))
        rows[name] = row

    # A clearly retreating torso is counter-evidence for a weight-bearing low
    # stance.  This is intentionally independent of the foot-crop hard-veto:
    # when support geometry is uncertain it becomes a soft physical penalty,
    # not a restoration to raw similarity.
    retreat_effect = _clamp(retreat * flex)
    crouching = rows.get("crouching") or {}
    if not crouching.get("hard_rejected") and retreat_effect > 0.0:
        factor = 1.0 - RETREAT_CROUCH_MAX_SUPPRESSION * retreat_effect
        _apply_current_factor(crouching, factor, "v13_torso_retreat_factor")
        rows["crouching"] = crouching

    squatting = rows.get("squatting") or {}
    if not squatting.get("hard_rejected") and retreat_effect > 0.0:
        factor = 1.0 - RETREAT_SQUAT_MAX_SUPPRESSION * retreat_effect
        _apply_current_factor(squatting, factor, "v13_torso_retreat_factor")
        rows["squatting"] = squatting

    # Flexed legs, horizontal-ish thighs and an upright/backward torso are the
    # characteristic geometry that made sitting and squatting falsely compete.
    # Require some retreat so a normal forward-balanced squat does not receive
    # the sitting boost.
    seated_topology = 0.0
    seated_topology_match = bool(
        flex >= SEATED_TOPOLOGY_MIN_FLEX
        and retreat >= SEATED_TOPOLOGY_MIN_RETREAT
        and advance < 0.35
    )
    if seated_topology_match:
        seated_topology = _clamp(
            flex
            * (0.55 + 0.45 * retreat)
            * (0.65 + 0.35 * torso_upright)
            * (0.70 + 0.30 * thigh_horizontal)
        )

        sitting = rows.get("sitting") or {}
        if not sitting.get("hard_rejected"):
            current = float(sitting.get("governed_score") or 0.0)
            sitting["v13_seated_flexion_topology_candidate"] = _round(seated_topology, 4)
            sitting["governed_score"] = _round(max(current, seated_topology), 4)
            sitting["governed_score_percent"] = int(round(100.0 * float(sitting["governed_score"])))
            rows["sitting"] = sitting

        squatting = rows.get("squatting") or {}
        if not squatting.get("hard_rejected"):
            counter = _clamp(flex * retreat * torso_upright * thigh_horizontal)
            factor = 1.0 - SQUAT_COUNTEREVIDENCE_MAX_SUPPRESSION * counter
            _apply_current_factor(squatting, factor, "v13_seated_topology_squat_factor")
            rows["squatting"] = squatting

    # Safety valve for the deep-squat control: if the feet are actually visible,
    # the raw squat geometry is strong, and the torso compensates decisively
    # toward support, an over-pessimistic support-area estimate may not hard-zero
    # the squat.  This does not apply to crouching or to reconstruction-only feet.
    squatting = rows.get("squatting") or {}
    squat_rescued = _unreject_for_forward_supported_squat(
        squatting,
        flex=flex,
        advance=advance,
        support_crop=support_crop,
    )
    rows["squatting"] = squatting

    leaning_back_candidate = _clamp(seated_topology * max(retreat, directional_recline))
    governance["per_pose"] = rows
    governance["v13_seated_low_stance_refine"] = {
        "bilateral_flexion": _round(flex, 4),
        "support_feasibility": _round(feasibility, 4),
        "external_support_requirement": _round(external, 4),
        "support_crop_authority": _round(support_crop, 4),
        "shoulder_shift_toward_feet_shoulder_widths": _round(float(shift) if shift is not None else None),
        "retreat_from_support": _round(retreat, 4),
        "advance_toward_support": _round(advance, 4),
        "raw_upper_body_inclination": _round(raw_upper_inclination, 4),
        "directional_recline": _round(directional_recline, 4),
        "thigh_axis_from_image_down_deg": _round(thigh_angle),
        "thigh_horizontal_score": _round(thigh_horizontal, 4),
        "torso_upright_score": _round(torso_upright, 4),
        "retreat_low_stance_counterevidence": _round(retreat_effect, 4),
        "seated_flexion_topology_match": seated_topology_match,
        "seated_flexion_topology_score": _round(seated_topology, 4),
        "seated_flexion_topology_score_percent": int(round(100.0 * seated_topology)),
        "seated_leaning_back_candidate": _round(leaning_back_candidate, 4),
        "seated_leaning_back_candidate_percent": int(round(100.0 * leaning_back_candidate)),
        "deferred_support_rows_reweighted": deferred_rows,
        "forward_supported_squat_rescue": squat_rescued,
        "interpretation": (
            "Low foot-support authority defers a hard impossibility claim but no longer restores "
            "the raw crouch/squat score. Torso retreat is negative low-stance evidence; flexed "
            "legs plus horizontal thighs and an upright/backward torso support sitting."
        ),
    }
    projected["physical_governance"] = governance
    projected["seated_low_stance_diagnostic"] = governance["v13_seated_low_stance_refine"]
    profile["sam3d_projected_pose"] = projected


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v12.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.13"

    _seated_low_stance_refine(profile)
    v12._recompute_public(profile)

    policy = profile.get("policy") or {}
    policy.update({
        "v13_low_support_authority_defers_veto_but_does_not_restore_raw_score": True,
        "v13_torso_retreat_is_low_stance_counterevidence": True,
        "v13_seated_flexion_topology_competes_against_squat": True,
        "v13_forward_supported_visible_squat_can_rescue_support_veto": True,
        "v13_seated_leaning_back_is_modifier_candidate_not_primary_pose": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-13",
        description=(
            "Build v0.13 governed pose profiles with soft deferred support vetoes, "
            "torso-direction low-stance gating, and seated-vs-squat topology."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.13")).expanduser().resolve()
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
        diagnostic = projected.get("seated_low_stance_diagnostic") or {}
        governance = projected.get("physical_governance") or {}
        authority = governance.get("authority") or {}
        print(
            f"{key}: pose={projected.get('pose')} best={projected.get('best_candidate_pose')} "
            f"scores=stand:{scores.get('standing',0)} crouch:{scores.get('crouching',0)} "
            f"squat:{scores.get('squatting',0)} sit:{scores.get('sitting',0)} recl:{scores.get('reclined',0)} "
            f"retreat:{int(round(100*float(diagnostic.get('retreat_from_support') or 0.0)))} "
            f"advance:{int(round(100*float(diagnostic.get('advance_toward_support') or 0.0)))} "
            f"seat_top:{diagnostic.get('seated_flexion_topology_score_percent',0)} "
            f"deferred:{','.join(diagnostic.get('deferred_support_rows_reweighted') or []) or '-'} "
            f"authority:{authority.get('crop_support_percent',0)}%"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.13",
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
