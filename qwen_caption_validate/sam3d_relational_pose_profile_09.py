from __future__ import annotations

"""v0.9 governed posture scoring.

This layer keeps the v0.8 reconstruction diagnostics, but changes the posture
architecture from purely additive evidence to:

    raw reconstruction similarity -> physical feasibility gates -> crop authority

The original scores are retained for audit.  Hard physical exclusions may zero
an otherwise high raw posture score.  The governed pose is then withheld when
its required anatomy is reconstruction-dominant.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_08 as v08


MIN_POSE_AUTHORITY = 0.10
HARD_BILATERAL_FLEXION = 0.65
LOW_STANCE_FLEXION = 0.45
HARD_LOW_STANCE_FEASIBILITY = 0.15
HARD_EXTERNAL_SUPPORT = 0.65
HARD_PELVIS_DISPLACEMENT = 0.40
HARD_SHOULDER_RETREAT = -0.08


def _round(value: float | None, digits: int = 3) -> float | None:
    return v08._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v08._ramp(value, low, high)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _pose_row(raw: float) -> dict[str, Any]:
    return {
        "raw_score": _round(raw, 4),
        "raw_score_percent": int(round(100.0 * raw)),
        "hard_rejected": False,
        "hard_rejection_reasons": [],
        "soft_feasibility_factor": 1.0,
        "governed_score": _round(raw, 4),
        "governed_score_percent": int(round(100.0 * raw)),
    }


def _reject(row: dict[str, Any], reason: str) -> None:
    row["hard_rejected"] = True
    row.setdefault("hard_rejection_reasons", []).append(reason)
    row["soft_feasibility_factor"] = 0.0
    row["governed_score"] = 0.0
    row["governed_score_percent"] = 0


def _apply_factor(row: dict[str, Any], factor: float) -> None:
    if row.get("hard_rejected"):
        return
    factor = _clamp(factor)
    current = float(row.get("soft_feasibility_factor") or 0.0)
    factor *= current
    raw = float(row.get("raw_score") or 0.0)
    score = _clamp(raw * factor)
    row["soft_feasibility_factor"] = _round(factor, 4)
    row["governed_score"] = _round(score, 4)
    row["governed_score_percent"] = int(round(100.0 * score))


def _govern_postures(profile: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    raw_scores = {
        str(name): float(value or 0.0)
        for name, value in (projected.get("posture_scores") or {}).items()
    }
    for name in ("standing", "crouching", "squatting", "sitting", "reclined"):
        raw_scores.setdefault(name, 0.0)

    raw_pose = str(projected.get("pose") or "uncertain")
    raw_best = str(projected.get("best_candidate_pose") or "uncertain")
    raw_margin = float(projected.get("winner_margin") or 0.0)
    raw_reconstruction = float(projected.get("reconstruction_match") or 0.0)

    leg = projected.get("leg_state_diagnostic") or {}
    support = projected.get("independent_support_diagnostic") or {}
    support_state = projected.get("support_state") or {}
    recline_diag = projected.get("recline_diagnostic") or {}

    flex = float(leg.get("bilateral_flexion_score") or 0.0)
    straight = float(leg.get("bilateral_straight_score") or 0.0)
    asymmetry = float(leg.get("asymmetry_score") or 0.0)
    feasibility = float(support.get("support_feasibility_score") or 0.0)
    external = float(support.get("external_support_requirement") or 0.0)
    recline = float(recline_diag.get("score") or 0.0)
    geometry = support.get("geometry") or {}
    pelvis_segment = float(geometry.get("pelvis_to_support_segment_shoulder_widths") or 0.0)
    shoulder_shift = float(geometry.get("shoulder_shift_toward_feet_shoulder_widths") or 0.0)
    single_leg = bool(support_state.get("geometry_match"))

    rows = {name: _pose_row(raw_scores[name]) for name in raw_scores}

    # Standing: strong bilateral flexion is physically incompatible with an
    # ordinary standing support topology.  A supported single-leg topology is
    # the explicit exception (e.g. standing while raising the other leg).
    standing = rows["standing"]
    if flex >= HARD_BILATERAL_FLEXION and not single_leg:
        _reject(standing, "strong_bilateral_leg_flexion_without_single_leg_support")
    elif not single_leg:
        # Do not let upright torso/body-axis terms hide moderately folded legs.
        standing_factor = 1.0 - 0.75 * _ramp(flex, 0.25, HARD_BILATERAL_FLEXION)
        _apply_factor(standing, standing_factor)
    else:
        standing["single_leg_support_exception"] = True

    # Crouch/squat are weight-bearing low stances.  If both legs are flexed but
    # the reconstructed feet cannot support the pelvis/torso, or the shoulders
    # retreat farther away from the foot support while the pelvis is displaced,
    # the stance is physically impossible without some external support.
    hard_low_support = bool(
        flex >= LOW_STANCE_FLEXION
        and feasibility <= HARD_LOW_STANCE_FEASIBILITY
    )
    hard_external = bool(
        flex >= LOW_STANCE_FLEXION
        and external >= HARD_EXTERNAL_SUPPORT
        and feasibility <= 0.30
    )
    hard_retreat = bool(
        flex >= LOW_STANCE_FLEXION
        and pelvis_segment >= HARD_PELVIS_DISPLACEMENT
        and shoulder_shift <= HARD_SHOULDER_RETREAT
    )

    for name in ("crouching", "squatting"):
        row = rows[name]
        if single_leg:
            _reject(row, "single_leg_free_leg_topology_not_bilateral_low_stance")
            continue
        if hard_low_support:
            _reject(row, "bilateral_flexion_with_near_zero_foot_support_feasibility")
            continue
        if hard_external:
            _reject(row, "strong_external_support_requirement")
            continue
        if hard_retreat:
            _reject(row, "pelvis_displaced_and_shoulders_retreat_from_foot_support")
            continue

        flex_gate = _ramp(flex, 0.18, 0.58)
        if name == "crouching":
            factor = (0.45 + 0.55 * flex_gate) * (0.35 + 0.65 * feasibility)
        else:
            factor = (0.25 + 0.75 * flex_gate) * (0.20 + 0.80 * feasibility)
        _apply_factor(row, factor)

    # External support does not by itself prove sitting versus reclining, but it
    # is positive evidence for that family.  Torso/whole-body recline geometry
    # decides how much of the external-support evidence goes to each branch.
    sitting_support_candidate = _clamp(external * (1.0 - 0.70 * recline))
    sitting_governed = max(
        float(rows["sitting"].get("governed_score") or 0.0),
        0.85 * sitting_support_candidate,
    )
    rows["sitting"]["external_support_candidate"] = _round(sitting_support_candidate, 4)
    rows["sitting"]["governed_score"] = _round(sitting_governed, 4)
    rows["sitting"]["governed_score_percent"] = int(round(100.0 * sitting_governed))

    recline_support_candidate = _clamp(0.70 * recline + 0.30 * external)
    reclined_governed = max(
        float(rows["reclined"].get("governed_score") or 0.0),
        recline_support_candidate,
    )
    rows["reclined"]["recline_support_candidate"] = _round(recline_support_candidate, 4)
    rows["reclined"]["governed_score"] = _round(reclined_governed, 4)
    rows["reclined"]["governed_score_percent"] = int(round(100.0 * reclined_governed))

    governed_scores = {
        name: float(row.get("governed_score") or 0.0)
        for name, row in rows.items()
    }
    candidate_pose, best_candidate, best_score, margin = v08.v07.v06.v05._choose_posture(governed_scores)

    # Crop authority is a separate gate.  Physical governance may improve the
    # reconstructed candidate even when the relevant lower body is invisible,
    # but reconstruction-dominant anatomy must not become an observed pose fact.
    region_support = {
        str(name): float(value or 0.0)
        for name, value in (projected.get("region_support") or {}).items()
    }
    coverage, authority_support, coverage_regions, support_regions = v08.v07.v06.v05._crop_support(
        region_support, candidate_pose, best_candidate
    )
    support_class = v08.v07.v06.v05._projected_support_class(authority_support)
    reconstruction_dominant = authority_support < MIN_POSE_AUTHORITY
    usable = bool(candidate_pose != "uncertain" and not reconstruction_dominant)
    public_pose = candidate_pose if usable else "uncertain"

    projected["reconstruction_pose_before_governance"] = raw_pose
    projected["reconstruction_best_candidate_before_governance"] = raw_best
    projected["posture_scores_before_physical_governance"] = {
        name: _round(value, 4) for name, value in raw_scores.items()
    }
    projected["posture_score_percent_before_physical_governance"] = {
        name: int(round(100.0 * value)) for name, value in raw_scores.items()
    }
    projected["winner_margin_before_physical_governance"] = _round(raw_margin, 4)
    projected["reconstruction_match_before_physical_governance"] = _round(raw_reconstruction, 4)

    governance = {
        "architecture": "raw_similarity_then_physical_exclusion_then_crop_authority",
        "per_pose": rows,
        "inputs": {
            "bilateral_flexion_score": _round(flex, 4),
            "bilateral_straight_score": _round(straight, 4),
            "leg_asymmetry_score": _round(asymmetry, 4),
            "single_leg_support": single_leg,
            "foot_support_feasibility": _round(feasibility, 4),
            "external_support_requirement": _round(external, 4),
            "recline_geometry_score": _round(recline, 4),
            "pelvis_to_support_segment_shoulder_widths": _round(pelvis_segment),
            "shoulder_shift_toward_feet_shoulder_widths": _round(shoulder_shift),
        },
        "governed_pose_before_authority": candidate_pose,
        "governed_best_candidate_pose": best_candidate,
        "governed_best_score": _round(best_score, 4),
        "governed_best_score_percent": int(round(100.0 * best_score)),
        "governed_winner_margin": _round(margin, 4),
        "governed_winner_margin_percent": int(round(100.0 * margin)),
        "authority": {
            "minimum_pose_authority": MIN_POSE_AUTHORITY,
            "crop_support": _round(authority_support, 4),
            "crop_support_percent": int(round(100.0 * authority_support)),
            "support_class": support_class,
            "reconstruction_dominant": reconstruction_dominant,
            "usable_as_projected_pose": usable,
            "withheld_reason": (
                "insufficient_observed_support" if reconstruction_dominant
                else ("insufficient_governed_score_or_margin" if candidate_pose == "uncertain" else None)
            ),
        },
    }
    projected["physical_governance"] = governance

    projected["pose"] = public_pose
    projected["best_candidate_pose"] = best_candidate
    projected["posture_scores"] = {
        name: _round(value, 4) for name, value in governed_scores.items()
    }
    projected["posture_score_percent"] = {
        name: int(round(100.0 * value)) for name, value in governed_scores.items()
    }
    projected["winner_margin"] = _round(margin, 4)
    projected["winner_margin_percent"] = int(round(100.0 * margin))
    projected["reconstruction_match"] = _round(best_score, 4)
    projected["reconstruction_match_percent"] = int(round(100.0 * best_score))
    projected["crop_coverage"] = _round(coverage, 4)
    projected["crop_coverage_percent"] = int(round(100.0 * coverage))
    projected["crop_support"] = _round(authority_support, 4)
    projected["crop_support_percent"] = int(round(100.0 * authority_support))
    projected["pose_support"] = _round(authority_support, 4)
    projected["pose_support_percent"] = int(round(100.0 * authority_support))
    projected["crop_supported_regions"] = coverage_regions
    projected["pose_support_regions"] = support_regions
    projected["support_class"] = support_class
    profile["sam3d_projected_pose"] = projected


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v08.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.9"
    _govern_postures(profile)

    policy = profile.get("policy") or {}
    policy.update({
        "v09_raw_posture_similarity_is_not_final_pose": True,
        "v09_hard_physical_exclusions_precede_ranking": True,
        "v09_low_stances_require_plausible_foot_support": True,
        "v09_shoulder_retreat_can_invalidate_displaced_low_stance": True,
        "v09_crop_authority_can_withhold_governed_reconstruction": True,
        "v09_reconstruction_dominant_pose_is_not_observed_fact": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-09",
        description=(
            "Build v0.9 governed pose profiles: raw reconstruction similarity -> "
            "hard/soft physical feasibility -> crop-authority withholding."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.9")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    helpers = v08.v07.v06.v05.v04.v03
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
        raw = projected.get("posture_score_percent_before_physical_governance") or {}
        governed = projected.get("posture_score_percent") or {}
        governance = projected.get("physical_governance") or {}
        authority = governance.get("authority") or {}
        rejected = [
            name for name, row in (governance.get("per_pose") or {}).items()
            if isinstance(row, dict) and row.get("hard_rejected")
        ]
        print(
            f"{key}: raw={projected.get('reconstruction_pose_before_governance')} "
            f"raw_best={projected.get('reconstruction_best_candidate_before_governance')} "
            f"raw_scores=stand:{raw.get('standing', 0)} crouch:{raw.get('crouching', 0)} "
            f"squat:{raw.get('squatting', 0)} sit:{raw.get('sitting', 0)} recl:{raw.get('reclined', 0)} "
            f"=> governed={projected.get('pose')} best={projected.get('best_candidate_pose')} "
            f"scores=stand:{governed.get('standing', 0)} crouch:{governed.get('crouching', 0)} "
            f"squat:{governed.get('squatting', 0)} sit:{governed.get('sitting', 0)} recl:{governed.get('reclined', 0)} "
            f"authority={authority.get('crop_support_percent', 0)}%[{authority.get('support_class')}] "
            f"rejected={','.join(rejected) or '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.9",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    index_path = output / "sam3d_relational_pose.index.json"
    helpers._write_json(index_path, index)
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
