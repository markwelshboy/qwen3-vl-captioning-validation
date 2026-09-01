from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_07 as v07


MHR = v07.MHR
EXTERNAL_SUPPORT_REVIEW_THRESHOLD = 0.35
STANDING_JOINT_CONFLICT_REVIEW_THRESHOLD = 0.45
LOW_STANCE_FEASIBILITY_REVIEW_THRESHOLD = 0.40


def _round(value: float | None, digits: int = 3) -> float | None:
    return v07._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v07._ramp(value, low, high)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _point(keypoints: np.ndarray, name: str) -> np.ndarray | None:
    idx = MHR.get(name)
    if idx is None or idx >= len(keypoints):
        return None
    value = np.asarray(keypoints[idx, :3], dtype=np.float64)
    if value.size < 3 or not np.all(np.isfinite(value)):
        return None
    return value


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0


def _leg_state(profile: dict[str, Any]) -> dict[str, Any]:
    projected = profile.get("sam3d_projected_pose") or {}
    lower = ((projected.get("geometry") or {}).get("asymmetric_lower_body") or {})
    per_side = lower.get("per_side") or {}
    if not per_side.get("left") or not per_side.get("right"):
        return {"report_only": True, "available": False}

    side_rows: dict[str, dict[str, Any]] = {}
    flex_values: dict[str, float] = {}
    straight_values: dict[str, float] = {}
    for side in ("left", "right"):
        row = per_side[side]
        knee = float(row.get("knee_flexion_deg") or 0.0)
        hip = float(row.get("hip_flexion_deg") or 0.0)

        # Angles here are joint interior angles: ~180 degrees is straight.
        # The diagnostic intentionally uses stricter straightness than the old
        # additive standing score so a visibly folded bilateral leg state is
        # not hidden by torso/body-axis terms.
        knee_flexed = 1.0 - _ramp(knee, 120.0, 165.0)
        hip_flexed = 1.0 - _ramp(hip, 115.0, 165.0)
        knee_straight = _ramp(knee, 145.0, 172.0)
        hip_open = _ramp(hip, 145.0, 172.0)
        flexed = _clamp(0.75 * knee_flexed + 0.25 * hip_flexed)
        straight = _clamp(0.75 * knee_straight + 0.25 * hip_open)

        flex_values[side] = flexed
        straight_values[side] = straight
        side_rows[side] = {
            "knee_angle_deg": _round(knee),
            "hip_angle_deg": _round(hip),
            "thigh_axis_from_image_down_deg": row.get("thigh_axis_from_image_down_deg"),
            "shin_axis_from_image_down_deg": row.get("shin_axis_from_image_down_deg"),
            "leg_extension_ratio": row.get("leg_extension_ratio"),
            "flexed_score": _round(flexed, 4),
            "flexed_score_percent": int(round(100.0 * flexed)),
            "straight_score": _round(straight, 4),
            "straight_score_percent": int(round(100.0 * straight)),
            "crop_support": _round(v07.v06._joint_side_support(profile, side), 4),
            "crop_support_percent": int(round(100.0 * v07.v06._joint_side_support(profile, side))),
        }

    bilateral_flexion = min(flex_values.values())
    bilateral_straight = min(straight_values.values())
    flex_difference = abs(flex_values["left"] - flex_values["right"])
    raw_asym = lower.get("asymmetry") or {}
    knee_difference = float(raw_asym.get("knee_flexion_difference_deg") or 0.0)
    ankle_height_difference = float(raw_asym.get("ankle_height_difference_shoulder_widths") or 0.0)
    asymmetry = _clamp(
        0.50 * _ramp(flex_difference, 0.15, 0.65)
        + 0.30 * _ramp(knee_difference, 18.0, 75.0)
        + 0.20 * _ramp(ankle_height_difference, 0.10, 0.70)
    )

    if bilateral_straight >= 0.55:
        state = "bilateral_straight"
    elif bilateral_flexion >= 0.45:
        state = "bilateral_flexed"
    elif asymmetry >= 0.45:
        state = "asymmetric"
    else:
        state = "mixed_or_indeterminate"

    crop_support = min(
        float(side_rows["left"]["crop_support"] or 0.0),
        float(side_rows["right"]["crop_support"] or 0.0),
    )
    return {
        "report_only": True,
        "available": True,
        "state": state,
        "per_side": side_rows,
        "bilateral_flexion_score": _round(bilateral_flexion, 4),
        "bilateral_flexion_score_percent": int(round(100.0 * bilateral_flexion)),
        "bilateral_straight_score": _round(bilateral_straight, 4),
        "bilateral_straight_score_percent": int(round(100.0 * bilateral_straight)),
        "asymmetry_score": _round(asymmetry, 4),
        "asymmetry_score_percent": int(round(100.0 * asymmetry)),
        "crop_support": _round(crop_support, 4),
        "crop_support_percent": int(round(100.0 * crop_support)),
        "support_class": v07.v06.v05._projected_support_class(crop_support),
        "interpretation": (
            "Joint-state diagnostic only. It is independent of the existing posture scores; "
            "it asks whether both reconstructed legs are straight, both are flexed, or the "
            "lower body is strongly asymmetric."
        ),
    }


def _support_geometry(keypoints: np.ndarray) -> dict[str, Any]:
    names = (
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_ankle", "right_ankle",
    )
    points = {name: _point(keypoints, name) for name in names}
    if any(value is None for value in points.values()):
        return {"available": False}

    ls = points["left_shoulder"]
    rs = points["right_shoulder"]
    lh = points["left_hip"]
    rh = points["right_hip"]
    la = points["left_ankle"]
    ra = points["right_ankle"]
    assert all(value is not None for value in (ls, rs, lh, rh, la, ra))

    shoulder_width = float(np.linalg.norm(ls - rs))
    if shoulder_width <= 1e-9:
        return {"available": False}

    shoulder_mid = _midpoint(ls, rs)
    hip_mid = _midpoint(lh, rh)
    left_foot_candidate = v07._foot_point(keypoints, "left")
    right_foot_candidate = v07._foot_point(keypoints, "right")
    left_foot = left_foot_candidate if left_foot_candidate is not None else la
    right_foot = right_foot_candidate if right_foot_candidate is not None else ra
    foot_mid = _midpoint(left_foot, right_foot)
    torso_proxy = 0.42 * shoulder_mid + 0.58 * hip_mid

    pelvis_segment = v07._distance_to_support_segment(hip_mid, left_foot, right_foot, shoulder_width)
    shoulder_segment = v07._distance_to_support_segment(shoulder_mid, left_foot, right_foot, shoulder_width)
    torso_segment = v07._distance_to_support_segment(torso_proxy, left_foot, right_foot, shoulder_width)
    pelvis_centroid = v07._horizontal_distance(hip_mid, foot_mid, shoulder_width)
    shoulder_centroid = v07._horizontal_distance(shoulder_mid, foot_mid, shoulder_width)

    hip_to_feet = np.asarray(foot_mid[[0, 2]] - hip_mid[[0, 2]], dtype=np.float64)
    shoulder_from_hip = np.asarray(shoulder_mid[[0, 2]] - hip_mid[[0, 2]], dtype=np.float64)
    foot_distance = float(np.linalg.norm(hip_to_feet) / shoulder_width)
    if float(np.linalg.norm(hip_to_feet)) > 1e-9:
        unit_to_feet = hip_to_feet / float(np.linalg.norm(hip_to_feet))
        shoulder_toward_feet = float(np.dot(shoulder_from_hip, unit_to_feet) / shoulder_width)
        compensation_fraction = shoulder_toward_feet / max(0.05, foot_distance)
    else:
        shoulder_toward_feet = 0.0
        compensation_fraction = 1.0

    pelvis_support = 1.0 - _ramp(pelvis_segment, 0.08, 0.55)
    torso_support = 1.0 - _ramp(torso_segment, 0.08, 0.60)
    centroid_support = 1.0 - _ramp(pelvis_centroid, 0.15, 0.85)
    distance_feasibility = _clamp(
        0.40 * pelvis_support
        + 0.35 * torso_support
        + 0.25 * centroid_support
    )

    compensation_need = _ramp(pelvis_segment, 0.06, 0.35)
    if compensation_need <= 0.05:
        compensation_score = 1.0
    else:
        compensation_score = _ramp(compensation_fraction, 0.05, 0.65)

    # If the pelvis is displaced from the reconstructed foot support, a flexed
    # weight-bearing stance normally needs the upper body to move toward that
    # support. This multiplicative term prevents a near-ish pelvis distance from
    # masking a complete lack of compensating torso shift.
    compensation_factor = 1.0 - 0.75 * compensation_need * (1.0 - compensation_score)
    support_feasibility = _clamp(distance_feasibility * compensation_factor)

    return {
        "available": True,
        "shoulder_width_3d": _round(shoulder_width),
        "pelvis_to_support_segment_shoulder_widths": _round(pelvis_segment),
        "shoulder_to_support_segment_shoulder_widths": _round(shoulder_segment),
        "torso_proxy_to_support_segment_shoulder_widths": _round(torso_segment),
        "pelvis_to_foot_centroid_shoulder_widths": _round(pelvis_centroid),
        "shoulder_to_foot_centroid_shoulder_widths": _round(shoulder_centroid),
        "hip_to_foot_centroid_horizontal_distance_shoulder_widths": _round(foot_distance),
        "shoulder_shift_toward_feet_shoulder_widths": _round(shoulder_toward_feet),
        "shoulder_compensation_fraction": _round(compensation_fraction, 4),
        "pelvis_support_score": _round(pelvis_support, 4),
        "torso_support_score": _round(torso_support, 4),
        "foot_centroid_support_score": _round(centroid_support, 4),
        "distance_feasibility_score": _round(distance_feasibility, 4),
        "compensation_need_score": _round(compensation_need, 4),
        "compensation_score": _round(compensation_score, 4),
        "support_feasibility_score": _round(support_feasibility, 4),
        "support_feasibility_score_percent": int(round(100.0 * support_feasibility)),
    }


def _independent_support_diagnostic(
    profile: dict[str, Any],
    keypoints: np.ndarray,
    leg_state: dict[str, Any],
) -> dict[str, Any]:
    geometry = _support_geometry(keypoints)
    if not leg_state.get("available") or not geometry.get("available"):
        return {"report_only": True, "available": False}

    bilateral_flexion = float(leg_state.get("bilateral_flexion_score") or 0.0)
    bilateral_straight = float(leg_state.get("bilateral_straight_score") or 0.0)
    asymmetry = float(leg_state.get("asymmetry_score") or 0.0)
    feasibility = float(geometry.get("support_feasibility_score") or 0.0)

    # Crucially, none of these values use standing/crouching/squatting/sitting
    # posture scores. This is an independent physical-consistency diagnostic.
    foot_supported_flexed = _clamp(bilateral_flexion * feasibility)
    external_support_required = _clamp(bilateral_flexion * (1.0 - feasibility))
    asymmetric_rescue = _clamp(asymmetry * (1.0 - bilateral_flexion))
    standing_joint_conflict = _clamp(bilateral_flexion * (1.0 - asymmetric_rescue))

    candidates = {
        "bilateral_straight_stance": bilateral_straight,
        "foot_supported_flexed_stance": foot_supported_flexed,
        "externally_supported_flexed_posture": external_support_required,
        "asymmetric_leg_state": asymmetry,
    }
    best_name, best_value = max(candidates.items(), key=lambda item: item[1])
    candidate = best_name if best_value >= 0.35 else "indeterminate"

    balance_support = v07._balance_support(profile)
    leg_support = float(leg_state.get("crop_support") or 0.0)
    crop_support = min(balance_support, leg_support)
    return {
        "report_only": True,
        "available": True,
        "independent_of_existing_posture_scores": True,
        "candidate": candidate,
        "candidate_scores": {name: _round(value, 4) for name, value in candidates.items()},
        "candidate_score_percent": {name: int(round(100.0 * value)) for name, value in candidates.items()},
        "foot_supported_flexed_stance_feasibility": _round(foot_supported_flexed, 4),
        "foot_supported_flexed_stance_feasibility_percent": int(round(100.0 * foot_supported_flexed)),
        "external_support_requirement": _round(external_support_required, 4),
        "external_support_requirement_percent": int(round(100.0 * external_support_required)),
        "standing_joint_conflict": _round(standing_joint_conflict, 4),
        "standing_joint_conflict_percent": int(round(100.0 * standing_joint_conflict)),
        "support_feasibility_score": geometry.get("support_feasibility_score"),
        "support_feasibility_score_percent": geometry.get("support_feasibility_score_percent"),
        "geometry": geometry,
        "crop_support": _round(crop_support, 4),
        "crop_support_percent": int(round(100.0 * crop_support)),
        "support_class": v07.v06.v05._projected_support_class(crop_support),
        "external_support_review_match": bool(
            external_support_required >= EXTERNAL_SUPPORT_REVIEW_THRESHOLD
        ),
        "standing_joint_conflict_review_match": bool(
            standing_joint_conflict >= STANDING_JOINT_CONFLICT_REVIEW_THRESHOLD
        ),
        "low_stance_feasibility_review_match": bool(
            bilateral_flexion >= 0.40
            and feasibility <= LOW_STANCE_FEASIBILITY_REVIEW_THRESHOLD
        ),
        "interpretation": (
            "First determines reconstructed leg state, then asks whether a bilateral flexed "
            "body could plausibly be foot-supported. A flexed posture with poor foot-support "
            "feasibility raises an external-support requirement. This remains reconstructed "
            "geometry, not an observed chair/bed/contact assertion."
        ),
    }


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v07.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.8"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]

    leg_state = _leg_state(profile)
    independent = _independent_support_diagnostic(profile, keypoints, leg_state)
    projected = profile.get("sam3d_projected_pose") or {}
    projected["leg_state_diagnostic"] = leg_state
    projected["independent_support_diagnostic"] = independent
    profile["sam3d_projected_pose"] = projected

    policy = profile.get("policy") or {}
    policy.update({
        "v08_posture_scores_are_frozen_from_v06": True,
        "v08_leg_state_is_independent_of_posture_scores": True,
        "v08_support_feasibility_is_independent_of_posture_scores": True,
        "external_support_requirement_is_report_only": True,
        "external_support_requirement_is_not_a_contact_claim": True,
        "crop_support_remains_separate_from_reconstruction_geometry": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-08",
        description=(
            "Build v0.8 report-only leg-state -> support-feasibility -> external-support "
            "diagnostics on top of the frozen v0.6 posture classifier."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.8")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    helpers = v07.v06.v05.v04.v03
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
        leg = projected.get("leg_state_diagnostic") or {}
        support = projected.get("independent_support_diagnostic") or {}
        state = projected.get("support_state") or {}
        state_label = "-"
        if state.get("geometry_match"):
            state_label = (
                f"single_leg:{state.get('candidate_support_side')} "
                f"free:{state.get('candidate_free_leg')}@{state.get('crop_support_percent', 0)}%"
            )
        print(
            f"{key}: projected={projected['pose']} best={projected['best_candidate_pose']} "
            f"scores=stand:{scores.get('standing', 0)} crouch:{scores.get('crouching', 0)} "
            f"squat:{scores.get('squatting', 0)} sit:{scores.get('sitting', 0)} recl:{scores.get('reclined', 0)} "
            f"crop={projected['crop_support_percent']}%[{projected.get('support_class')}] "
            f"support={state_label} leg={leg.get('state', '-')} "
            f"flex:{leg.get('bilateral_flexion_score_percent', 0)} straight:{leg.get('bilateral_straight_score_percent', 0)} "
            f"foot_feas:{support.get('support_feasibility_score_percent', 0)} "
            f"external:{support.get('external_support_requirement_percent', 0)} "
            f"stand_conflict:{support.get('standing_joint_conflict_percent', 0)} "
            f"candidate={support.get('candidate', '-')}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.8",
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
