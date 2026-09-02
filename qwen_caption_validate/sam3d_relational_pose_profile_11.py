from __future__ import annotations

"""v0.11 direction-aware sitting/recline governance.

v0.10 correctly added upper-body recline evidence, but its inclination terms were
unsigned: a torso bending *forward* could look identical to one reclining
*backward*.  v0.11 keeps the v0.9 physical-governance baseline and uses the v0.10
upper-body measurements only after resolving the direction against the
reconstructed foot-support vector.

The key distinction is:

* shoulders/head moving TOWARD the foot-support region -> forward compensation
  (standing/crouching-like), not recline;
* shoulders/head moving AWAY from the foot-support region -> backward support
  geometry compatible with sitting/reclining.

The support vector is still reconstructed geometry.  It is used to disambiguate
inclination direction, while crop authority remains a separate observed-evidence
question.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_10 as v10


v09 = v10.v09
MHR = v10.MHR

FORWARD_BEND_HARD_SHIFT = 0.30
FORWARD_BEND_HARD_ADVANCE = 0.65
FORWARD_BEND_MAX_LOWER_RECLINE = 0.45
RECLINE_RETREAT_AUTHORITY_MIN = 0.35
UPPER_RECLINE_AUTHORITY_MIN_SCORE = 0.55


def _round(value: float | None, digits: int = 3) -> float | None:
    return v10._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v10._ramp(value, low, high)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _directional_upper_recline(profile: dict[str, Any], upper: dict[str, Any]) -> dict[str, Any]:
    projected = profile.get("sam3d_projected_pose") or {}
    support = projected.get("independent_support_diagnostic") or {}
    geometry = support.get("geometry") or {}
    recline_diag = projected.get("recline_diagnostic") or {}

    raw_upper = float(upper.get("score") or 0.0)
    lower_recline = float(recline_diag.get("score") or 0.0)
    shift_value = geometry.get("shoulder_shift_toward_feet_shoulder_widths")
    shift = float(shift_value) if shift_value is not None else None

    if shift is None:
        retreat = 0.0
        advance = 0.0
        direction_factor = 0.35
        direction = "unresolved"
    else:
        retreat = _ramp(-shift, 0.05, 0.65)
        advance = _ramp(shift, 0.05, 0.65)
        # A clear retreat preserves the upper-body inclination signal.  A clear
        # advance suppresses it almost completely: that is forward compensation,
        # not reclining.  A small neutral residual is retained for noisy geometry.
        direction_factor = _clamp(0.10 + 0.90 * retreat - 0.08 * advance)
        direction = (
            "retreats_from_support" if retreat >= 0.35
            else "advances_toward_support" if advance >= 0.35
            else "near_neutral"
        )

    directional = _clamp(raw_upper * direction_factor)
    hard_forward_bend = bool(
        shift is not None
        and shift >= FORWARD_BEND_HARD_SHIFT
        and advance >= FORWARD_BEND_HARD_ADVANCE
        and lower_recline <= FORWARD_BEND_MAX_LOWER_RECLINE
    )

    return {
        "available": bool(upper.get("available")),
        "raw_upper_body_inclination_score": _round(raw_upper, 4),
        "raw_upper_body_inclination_score_percent": int(round(100.0 * raw_upper)),
        "lower_body_recline_score": _round(lower_recline, 4),
        "lower_body_recline_score_percent": int(round(100.0 * lower_recline)),
        "shoulder_shift_toward_feet_shoulder_widths": _round(shift),
        "retreat_from_support_score": _round(retreat, 4),
        "retreat_from_support_score_percent": int(round(100.0 * retreat)),
        "advance_toward_support_score": _round(advance, 4),
        "advance_toward_support_score_percent": int(round(100.0 * advance)),
        "direction_factor": _round(direction_factor, 4),
        "direction": direction,
        "directional_upper_recline_score": _round(directional, 4),
        "directional_upper_recline_score_percent": int(round(100.0 * directional)),
        "hard_forward_bend_recline_rejection": hard_forward_bend,
        "interpretation": (
            "Unsigned upper-body inclination is not recline by itself.  Recline requires the "
            "upper body to retreat from the reconstructed support direction; movement toward "
            "support is forward compensation/bending and suppresses the recline claim."
        ),
    }


def _refine_directional_sitting_recline(
    profile: dict[str, Any],
    upper: dict[str, Any],
    directional: dict[str, Any],
) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    per_pose = governance.get("per_pose") or {}
    if not per_pose:
        return

    projected["v09_pose_before_directional_recline_refine"] = projected.get("pose")
    projected["v09_best_candidate_before_directional_recline_refine"] = projected.get("best_candidate_pose")
    projected["v09_posture_scores_before_directional_recline_refine"] = dict(projected.get("posture_scores") or {})
    projected["v09_posture_score_percent_before_directional_recline_refine"] = dict(projected.get("posture_score_percent") or {})

    support = projected.get("independent_support_diagnostic") or {}
    recline_diag = projected.get("recline_diagnostic") or {}
    external = float(support.get("external_support_requirement") or 0.0)
    lower_recline = float(recline_diag.get("score") or 0.0)
    directional_upper = float(directional.get("directional_upper_recline_score") or 0.0)
    combined_recline = max(lower_recline, directional_upper)
    flatness = float(recline_diag.get("body_flatness_ratio") or 0.0)
    knee_drop_value = upper.get("mean_hip_to_knee_vertical_drop_shoulder_widths")
    knee_drop = float(knee_drop_value) if knee_drop_value is not None else None

    # Sitting now has the same architecture as the weight-bearing families:
    # positive resemblance cannot survive a geometry that is incompatible with
    # ordinary sitting.
    sitting = per_pose.get("sitting") or {}
    if not sitting.get("hard_rejected"):
        hard_flat_recline = bool(
            combined_recline >= v10.HARD_SITTING_RECLINE
            and flatness >= v10.HARD_SITTING_FLATNESS
            and external >= v10.HARD_SITTING_EXTERNAL
        )
        hard_raised_knees = bool(
            combined_recline >= v10.HARD_SITTING_KNEE_RAISE_RECLINE
            and knee_drop is not None
            and knee_drop < -0.05
            and external >= 0.45
        )
        if hard_flat_recline:
            v09._reject(sitting, "strong_directional_recline_flat_body_incompatible_with_ordinary_sitting")
        elif hard_raised_knees:
            v09._reject(sitting, "raised_knees_plus_strong_directional_recline_is_not_ordinary_sitting")
        else:
            factor = 1.0 - 0.82 * _ramp(
                combined_recline,
                v10.SOFT_SITTING_RECLINE_START,
                v10.SOFT_SITTING_RECLINE_FULL,
            )
            current = float(sitting.get("governed_score") or 0.0)
            current_factor = float(sitting.get("soft_feasibility_factor") or 1.0)
            sitting["sitting_recline_feasibility_factor"] = _round(factor, 4)
            sitting["soft_feasibility_factor"] = _round(_clamp(current_factor * factor), 4)
            sitting["governed_score"] = _round(_clamp(current * factor), 4)
            sitting["governed_score_percent"] = int(round(100.0 * float(sitting["governed_score"])))
    per_pose["sitting"] = sitting

    reclined = per_pose.get("reclined") or {}
    if directional.get("hard_forward_bend_recline_rejection"):
        v09._reject(reclined, "upper_body_advances_toward_support_forward_bend_not_recline")
        reclined["directional_recline_veto"] = True
    elif not reclined.get("hard_rejected"):
        current_reclined = float(reclined.get("governed_score") or 0.0)
        upper_candidate = _clamp(directional_upper * (0.68 + 0.32 * external))
        fused_candidate = _clamp(0.55 * lower_recline + 0.45 * directional_upper)
        reclined_score = max(current_reclined, upper_candidate, fused_candidate)
        reclined["directional_upper_body_recline_candidate"] = _round(upper_candidate, 4)
        reclined["directional_fused_recline_candidate"] = _round(fused_candidate, 4)
        reclined["governed_score"] = _round(reclined_score, 4)
        reclined["governed_score_percent"] = int(round(100.0 * reclined_score))
    per_pose["reclined"] = reclined

    governed_scores = {
        name: float((row or {}).get("governed_score") or 0.0)
        for name, row in per_pose.items()
    }
    candidate_pose, best_candidate, best_score, margin = v09.v08.v07.v06.v05._choose_posture(governed_scores)

    region_support = {
        str(name): float(value or 0.0)
        for name, value in (projected.get("region_support") or {}).items()
    }
    coverage, authority_support, coverage_regions, support_regions = v09.v08.v07.v06.v05._crop_support(
        region_support, candidate_pose, best_candidate
    )

    upper_authority_used = False
    retreat = float(directional.get("retreat_from_support_score") or 0.0)
    if (
        best_candidate == "reclined"
        and directional_upper >= UPPER_RECLINE_AUTHORITY_MIN_SCORE
        and retreat >= RECLINE_RETREAT_AUTHORITY_MIN
    ):
        upper_authority = float(upper.get("path_authority") or 0.0)
        if upper_authority > authority_support:
            authority_support = upper_authority
            upper_authority_used = True
            support_regions = [
                name for name in ("head", "shoulders", "hips")
                if float(region_support.get(name, 0.0)) >= v09.v08.v07.v06.v05.v04.v03.v02.REGION_SUPPORT_THRESHOLD
            ]

    support_class = v09.v08.v07.v06.v05._projected_support_class(authority_support)
    reconstruction_dominant = authority_support < v09.MIN_POSE_AUTHORITY
    usable = bool(candidate_pose != "uncertain" and not reconstruction_dominant)
    public_pose = candidate_pose if usable else "uncertain"

    governance["architecture"] = (
        "raw_similarity_then_physical_exclusion_then_directional_sitting_recline_refine_then_path_authority"
    )
    governance["per_pose"] = per_pose
    governance["upper_body_recline"] = upper
    governance["directional_recline"] = directional
    governance["sitting_recline_inputs"] = {
        "lower_body_recline_score": _round(lower_recline, 4),
        "raw_upper_body_inclination_score": upper.get("score"),
        "directional_upper_recline_score": _round(directional_upper, 4),
        "combined_directional_recline_score": _round(combined_recline, 4),
        "body_flatness_ratio": _round(flatness),
        "external_support_requirement": _round(external, 4),
        "mean_hip_to_knee_vertical_drop_shoulder_widths": _round(knee_drop),
    }
    governance["governed_pose_before_authority"] = candidate_pose
    governance["governed_best_candidate_pose"] = best_candidate
    governance["governed_best_score"] = _round(best_score, 4)
    governance["governed_best_score_percent"] = int(round(100.0 * best_score))
    governance["governed_winner_margin"] = _round(margin, 4)
    governance["governed_winner_margin_percent"] = int(round(100.0 * margin))
    governance["authority"] = {
        "minimum_pose_authority": v09.MIN_POSE_AUTHORITY,
        "crop_support": _round(authority_support, 4),
        "crop_support_percent": int(round(100.0 * authority_support)),
        "support_class": support_class,
        "reconstruction_dominant": reconstruction_dominant,
        "usable_as_projected_pose": usable,
        "authority_path": "directional_upper_body_recline" if upper_authority_used else "posture_region_weights",
        "upper_body_path_available_percent": upper.get("path_authority_percent"),
        "withheld_reason": (
            "insufficient_observed_support" if reconstruction_dominant
            else ("insufficient_governed_score_or_margin" if candidate_pose == "uncertain" else None)
        ),
    }

    projected["physical_governance"] = governance
    projected["upper_body_recline_diagnostic"] = upper
    projected["directional_recline_diagnostic"] = directional
    projected["pose"] = public_pose
    projected["best_candidate_pose"] = best_candidate
    projected["posture_scores"] = {name: _round(value, 4) for name, value in governed_scores.items()}
    projected["posture_score_percent"] = {name: int(round(100.0 * value)) for name, value in governed_scores.items()}
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
    # Start from v0.9 rather than v0.10 so the unsigned v0.10 recline boost never
    # contaminates the candidate that v0.11 is trying to correct.
    profile = v09.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.11"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]

    upper = v10._upper_body_recline(profile, keypoints)
    directional = _directional_upper_recline(profile, upper)
    _refine_directional_sitting_recline(profile, upper, directional)

    policy = profile.get("policy") or {}
    policy.update({
        "v11_upper_body_inclination_is_not_recline_without_direction": True,
        "v11_recline_requires_retreat_from_support_or_strong_lower_recline": True,
        "v11_forward_compensation_can_hard_reject_recline": True,
        "v11_sitting_recline_governance_uses_directional_recline": True,
        "v11_recline_authority_requires_directionally_supported_upper_body_path": True,
        "v11_support_direction_is_reconstructed_geometry_not_observed_contact": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-11",
        description=(
            "Build v0.11 governed pose profiles with direction-aware upper-body recline: "
            "forward compensation is separated from backward/reclined support geometry."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.11")).expanduser().resolve()
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
        governance = projected.get("physical_governance") or {}
        directional = projected.get("directional_recline_diagnostic") or {}
        authority = governance.get("authority") or {}
        rejected = [
            name for name, row in (governance.get("per_pose") or {}).items()
            if (row or {}).get("hard_rejected")
        ]
        print(
            f"{key}: raw={projected.get('reconstruction_pose_before_governance')} "
            f"=> governed={projected.get('pose')} best={projected.get('best_candidate_pose')} "
            f"scores=stand:{scores.get('standing',0)} crouch:{scores.get('crouching',0)} "
            f"squat:{scores.get('squatting',0)} sit:{scores.get('sitting',0)} recl:{scores.get('reclined',0)} "
            f"direction={directional.get('direction','-')} "
            f"upper_raw:{directional.get('raw_upper_body_inclination_score_percent',0)} "
            f"upper_dir:{directional.get('directional_upper_recline_score_percent',0)} "
            f"shift:{directional.get('shoulder_shift_toward_feet_shoulder_widths')} "
            f"authority={authority.get('crop_support_percent',0)}%[{authority.get('authority_path','-')}] "
            f"rejected={','.join(rejected) if rejected else '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.11",
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
