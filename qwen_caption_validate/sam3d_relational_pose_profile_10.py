from __future__ import annotations

"""v0.10 posture governance.

Extends v0.9 with two missing pieces exposed by review:

* ordinary sitting can be physically invalidated by strong crop-supported
  recline geometry instead of being protected by its raw additive score;
* recline can obtain authority through an upper-body evidence path
  (head/shoulders and/or shoulders/hips), rather than requiring visible legs.

The v0.9 result is retained for audit before this refinement is applied.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_09 as v09


MHR = v09.v08.v07.MHR
HARD_SITTING_RECLINE = 0.82
HARD_SITTING_FLATNESS = 1.35
HARD_SITTING_EXTERNAL = 0.55
HARD_SITTING_KNEE_RAISE_RECLINE = 0.75
SOFT_SITTING_RECLINE_START = 0.45
SOFT_SITTING_RECLINE_FULL = 0.80
UPPER_RECLINE_AUTHORITY_MIN_SCORE = 0.55


def _round(value: float | None, digits: int = 3) -> float | None:
    return v09._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v09._ramp(value, low, high)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _point(keypoints: np.ndarray, name: str) -> np.ndarray | None:
    idx = MHR.get(name)
    if idx is None or idx >= len(keypoints):
        return None
    p = np.asarray(keypoints[idx, :3], dtype=np.float64)
    if p.size < 3 or not np.all(np.isfinite(p)):
        return None
    return p


def _mean_points(points: list[np.ndarray | None]) -> np.ndarray | None:
    good = [np.asarray(p, dtype=np.float64) for p in points if p is not None]
    if not good:
        return None
    return np.mean(np.stack(good, axis=0), axis=0)


def _angle_from_vertical(vector: np.ndarray) -> float:
    v = np.asarray(vector, dtype=np.float64)
    vertical = abs(float(v[1]))
    horizontal = float(np.linalg.norm(v[[0, 2]]))
    return float(np.degrees(np.arctan2(horizontal, max(1e-9, vertical))))


def _horizontal_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    ah = np.asarray(a, dtype=np.float64)[[0, 2]]
    bh = np.asarray(b, dtype=np.float64)[[0, 2]]
    na = float(np.linalg.norm(ah))
    nb = float(np.linalg.norm(bh))
    if na <= 1e-9 or nb <= 1e-9:
        return None
    return float(np.dot(ah, bh) / (na * nb))


def _region_support(profile: dict[str, Any], name: str) -> float:
    projected = profile.get("sam3d_projected_pose") or {}
    return float((projected.get("region_support") or {}).get(name) or 0.0)


def _upper_body_recline(profile: dict[str, Any], keypoints: np.ndarray) -> dict[str, Any]:
    head = _mean_points([
        _point(keypoints, "nose"),
        _point(keypoints, "left_eye"), _point(keypoints, "right_eye"),
        _point(keypoints, "left_ear"), _point(keypoints, "right_ear"),
    ])
    ls, rs = _point(keypoints, "left_shoulder"), _point(keypoints, "right_shoulder")
    lh, rh = _point(keypoints, "left_hip"), _point(keypoints, "right_hip")
    lk, rk = _point(keypoints, "left_knee"), _point(keypoints, "right_knee")
    if head is None or ls is None or rs is None:
        return {"report_only": False, "available": False}

    shoulder_mid = (ls + rs) / 2.0
    shoulder_width = float(np.linalg.norm(ls - rs))
    if shoulder_width <= 1e-9:
        return {"report_only": False, "available": False}

    hip_mid = (lh + rh) / 2.0 if lh is not None and rh is not None else None
    knee_mid = (lk + rk) / 2.0 if lk is not None and rk is not None else None

    head_shoulder_angle = _angle_from_vertical(shoulder_mid - head)
    head_shoulder_score = _ramp(head_shoulder_angle, 22.0, 68.0)

    torso_angle = None
    torso_score = 0.0
    head_hip_angle = None
    head_hip_score = 0.0
    continuation_cosine = None
    continuation_score = 0.5
    hip_to_knee_drop = None
    if hip_mid is not None:
        torso_angle = _angle_from_vertical(hip_mid - shoulder_mid)
        torso_score = _ramp(torso_angle, 25.0, 72.0)
        head_hip_angle = _angle_from_vertical(hip_mid - head)
        head_hip_score = _ramp(head_hip_angle, 25.0, 72.0)
        continuation_cosine = _horizontal_cosine(shoulder_mid - hip_mid, head - shoulder_mid)
        if continuation_cosine is not None:
            continuation_score = _ramp(continuation_cosine, 0.10, 0.90)
        if knee_mid is not None:
            hip_to_knee_drop = float((knee_mid[1] - hip_mid[1]) / shoulder_width)

    # The head->shoulder segment is intentionally a substantial term: it gives
    # tight upper-body crops a real recline pathway. Shoulder->hip and full-chain
    # geometry strengthen the claim when those landmarks are also supported.
    score = _clamp(
        0.38 * head_shoulder_score
        + 0.32 * torso_score
        + 0.20 * head_hip_score
        + 0.10 * continuation_score
    )

    head_support = _region_support(profile, "head")
    shoulder_support = _region_support(profile, "shoulders")
    hip_support = _region_support(profile, "hips")
    head_shoulders_path = min(head_support, shoulder_support)
    shoulder_hips_path = min(shoulder_support, hip_support)
    full_chain_path = min(head_support, shoulder_support, hip_support)
    path_authority = max(head_shoulders_path, shoulder_hips_path, full_chain_path)

    return {
        "report_only": False,
        "available": True,
        "score": _round(score, 4),
        "score_percent": int(round(100.0 * score)),
        "head_to_shoulders_axis_from_vertical_deg": _round(head_shoulder_angle),
        "shoulder_to_hips_axis_from_vertical_deg": _round(torso_angle),
        "head_to_hips_axis_from_vertical_deg": _round(head_hip_angle),
        "horizontal_chain_continuation_cosine": _round(continuation_cosine, 4),
        "mean_hip_to_knee_vertical_drop_shoulder_widths": _round(hip_to_knee_drop),
        "knee_position_relative_to_hips": (
            "raised_above_hips" if hip_to_knee_drop is not None and hip_to_knee_drop < -0.05
            else "near_hip_height" if hip_to_knee_drop is not None and hip_to_knee_drop <= 0.15
            else "below_hips" if hip_to_knee_drop is not None
            else "unavailable"
        ),
        "authority_paths": {
            "head_shoulders": _round(head_shoulders_path, 4),
            "shoulder_hips": _round(shoulder_hips_path, 4),
            "full_head_shoulder_hip_chain": _round(full_chain_path, 4),
        },
        "path_authority": _round(path_authority, 4),
        "path_authority_percent": int(round(100.0 * path_authority)),
        "interpretation": (
            "Upper-body recline uses head->shoulder, shoulder->hip and full head->hip "
            "inclination plus chain continuation. Its authority can come from visible "
            "head/shoulders even when lower-body landmarks are outside the crop."
        ),
    }


def _refine_sitting_recline(profile: dict[str, Any], upper: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    per_pose = governance.get("per_pose") or {}
    if not per_pose:
        return

    # Snapshot the complete v0.9 public/governed result before changing it.
    projected["v09_pose_before_sitting_recline_refine"] = projected.get("pose")
    projected["v09_best_candidate_before_sitting_recline_refine"] = projected.get("best_candidate_pose")
    projected["v09_posture_scores_before_sitting_recline_refine"] = dict(projected.get("posture_scores") or {})
    projected["v09_posture_score_percent_before_sitting_recline_refine"] = dict(projected.get("posture_score_percent") or {})

    support = projected.get("independent_support_diagnostic") or {}
    recline_diag = projected.get("recline_diagnostic") or {}
    external = float(support.get("external_support_requirement") or 0.0)
    lower_recline = float(recline_diag.get("score") or 0.0)
    upper_recline = float(upper.get("score") or 0.0)
    combined_recline = max(lower_recline, upper_recline)
    flatness = float(recline_diag.get("body_flatness_ratio") or 0.0)
    knee_drop = upper.get("mean_hip_to_knee_vertical_drop_shoulder_widths")
    knee_drop = float(knee_drop) if knee_drop is not None else None

    sitting = per_pose.get("sitting") or {}
    if not sitting.get("hard_rejected"):
        hard_flat_recline = bool(
            combined_recline >= HARD_SITTING_RECLINE
            and flatness >= HARD_SITTING_FLATNESS
            and external >= HARD_SITTING_EXTERNAL
        )
        hard_raised_knees = bool(
            combined_recline >= HARD_SITTING_KNEE_RAISE_RECLINE
            and knee_drop is not None
            and knee_drop < -0.05
            and external >= 0.45
        )
        if hard_flat_recline:
            v09._reject(sitting, "strong_recline_flat_body_incompatible_with_ordinary_sitting")
        elif hard_raised_knees:
            v09._reject(sitting, "raised_knees_plus_strong_recline_is_reclined_not_ordinary_sitting")
        else:
            factor = 1.0 - 0.82 * _ramp(
                combined_recline,
                SOFT_SITTING_RECLINE_START,
                SOFT_SITTING_RECLINE_FULL,
            )
            current = float(sitting.get("governed_score") or 0.0)
            current_factor = float(sitting.get("soft_feasibility_factor") or 1.0)
            new_factor = _clamp(current_factor * factor)
            new_score = _clamp(current * factor)
            sitting["soft_feasibility_factor"] = _round(new_factor, 4)
            sitting["sitting_recline_feasibility_factor"] = _round(factor, 4)
            sitting["governed_score"] = _round(new_score, 4)
            sitting["governed_score_percent"] = int(round(100.0 * new_score))
    per_pose["sitting"] = sitting

    reclined = per_pose.get("reclined") or {}
    current_reclined = float(reclined.get("governed_score") or 0.0)
    upper_candidate = _clamp(upper_recline * (0.68 + 0.32 * external))
    fused_candidate = _clamp(0.55 * lower_recline + 0.45 * upper_recline)
    reclined_score = max(current_reclined, upper_candidate, fused_candidate)
    reclined["upper_body_recline_candidate"] = _round(upper_candidate, 4)
    reclined["fused_recline_candidate"] = _round(fused_candidate, 4)
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
    if best_candidate == "reclined" and upper_recline >= UPPER_RECLINE_AUTHORITY_MIN_SCORE:
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
        "raw_similarity_then_physical_exclusion_then_sitting_recline_refine_then_path_authority"
    )
    governance["per_pose"] = per_pose
    governance["upper_body_recline"] = upper
    governance["sitting_recline_inputs"] = {
        "lower_body_recline_score": _round(lower_recline, 4),
        "upper_body_recline_score": _round(upper_recline, 4),
        "combined_recline_score": _round(combined_recline, 4),
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
        "authority_path": "upper_body_recline" if upper_authority_used else "posture_region_weights",
        "upper_body_path_available_percent": upper.get("path_authority_percent"),
        "withheld_reason": (
            "insufficient_observed_support" if reconstruction_dominant
            else ("insufficient_governed_score_or_margin" if candidate_pose == "uncertain" else None)
        ),
    }
    projected["physical_governance"] = governance
    projected["upper_body_recline_diagnostic"] = upper
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
    profile = v09.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.10"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]
    upper = _upper_body_recline(profile, keypoints)
    _refine_sitting_recline(profile, upper)

    policy = profile.get("policy") or {}
    policy.update({
        "v10_sitting_has_recline_impossibility_gates": True,
        "v10_raw_sitting_score_can_be_reduced_or_hard_rejected": True,
        "v10_raised_knees_only_reject_sitting_when_combined_with_strong_recline": True,
        "v10_upper_body_recline_is_a_pose_evidence_path": True,
        "v10_recline_authority_can_use_observed_head_shoulders_without_visible_legs": True,
        "v10_recline_authority_is_path_specific": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-10",
        description=(
            "Build v0.10 governed pose profiles with sitting impossibility gates and "
            "upper-body recline evidence/authority pathways."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.10")).expanduser().resolve()
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
        gov = projected.get("physical_governance") or {}
        per = gov.get("per_pose") or {}
        upper = projected.get("upper_body_recline_diagnostic") or {}
        auth = gov.get("authority") or {}
        raw = projected.get("posture_score_percent_before_physical_governance") or {}
        scores = projected.get("posture_score_percent") or {}
        rejected = [name for name, row in per.items() if (row or {}).get("hard_rejected")]
        print(
            f"{key}: raw_best={projected.get('reconstruction_best_candidate_before_governance')} "
            f"raw=stand:{raw.get('standing',0)} crouch:{raw.get('crouching',0)} squat:{raw.get('squatting',0)} "
            f"sit:{raw.get('sitting',0)} recl:{raw.get('reclined',0)} => pose={projected.get('pose')} "
            f"best={projected.get('best_candidate_pose')} scores=stand:{scores.get('standing',0)} "
            f"crouch:{scores.get('crouching',0)} squat:{scores.get('squatting',0)} sit:{scores.get('sitting',0)} "
            f"recl:{scores.get('reclined',0)} upper_recl:{upper.get('score_percent',0)} "
            f"authority:{auth.get('crop_support_percent',0)}%[{auth.get('authority_path','-')}] "
            f"rejected={','.join(rejected) or '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.10",
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
