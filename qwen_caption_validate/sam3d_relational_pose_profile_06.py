from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_05 as v05


MHR = v05.MHR
MIN_SINGLE_LEG_SCORE = 0.62
MIN_SUPPORT_LEG_SCORE = 0.62
MIN_FREE_LEG_SCORE = 0.50
MIN_ASYMMETRY_SCORE = 0.40
KNEELING_REVIEW_THRESHOLD = 0.60


def _round(value: float | None, digits: int = 3) -> float | None:
    return v05._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v05._ramp(value, low, high)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _leg_geometry(keypoints: np.ndarray) -> dict[str, Any]:
    needed = [
        MHR["left_shoulder"], MHR["right_shoulder"],
        MHR["left_hip"], MHR["right_hip"],
        MHR["left_knee"], MHR["right_knee"],
        MHR["left_ankle"], MHR["right_ankle"],
    ]
    if not len(keypoints) or max(needed) >= len(keypoints):
        return {"per_side": {}, "asymmetry": {}}

    ls = keypoints[MHR["left_shoulder"], :3]
    rs = keypoints[MHR["right_shoulder"], :3]
    shoulder_width = float(np.linalg.norm(ls - rs))
    if shoulder_width <= 1e-9:
        return {"per_side": {}, "asymmetry": {}}

    side_points: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        side_points[side] = {
            "shoulder": keypoints[MHR[f"{side}_shoulder"], :3],
            "hip": keypoints[MHR[f"{side}_hip"], :3],
            "knee": keypoints[MHR[f"{side}_knee"], :3],
            "ankle": keypoints[MHR[f"{side}_ankle"], :3],
        }
    if not all(np.all(np.isfinite(point)) for values in side_points.values() for point in values.values()):
        return {"per_side": {}, "asymmetry": {}}

    per_side: dict[str, dict[str, float]] = {}
    for side, points in side_points.items():
        shoulder = points["shoulder"]
        hip = points["hip"]
        knee = points["knee"]
        ankle = points["ankle"]
        thigh_len = float(np.linalg.norm(knee - hip))
        shin_len = float(np.linalg.norm(ankle - knee))
        chain_len = thigh_len + shin_len
        direct_len = float(np.linalg.norm(ankle - hip))
        per_side[side] = {
            "knee_flexion_deg": float(v05._angle(hip, knee, ankle) or 0.0),
            "hip_flexion_deg": float(v05._angle(shoulder, hip, knee) or 0.0),
            "thigh_axis_from_image_down_deg": float(v05.v04._angle_from_image_down(knee - hip) or 0.0),
            "shin_axis_from_image_down_deg": float(v05.v04._angle_from_image_down(ankle - knee) or 0.0),
            "hip_to_knee_vertical_drop_shoulder_widths": float((knee[1] - hip[1]) / shoulder_width),
            "knee_to_ankle_vertical_drop_shoulder_widths": float((ankle[1] - knee[1]) / shoulder_width),
            "hip_to_ankle_vertical_drop_shoulder_widths": float((ankle[1] - hip[1]) / shoulder_width),
            "leg_extension_ratio": float(direct_len / chain_len) if chain_len > 1e-9 else 0.0,
            "knee_image_y_shoulder_widths": float(knee[1] / shoulder_width),
            "ankle_image_y_shoulder_widths": float(ankle[1] / shoulder_width),
        }

    left = per_side["left"]
    right = per_side["right"]
    asymmetry = {
        "knee_flexion_difference_deg": abs(left["knee_flexion_deg"] - right["knee_flexion_deg"]),
        "hip_flexion_difference_deg": abs(left["hip_flexion_deg"] - right["hip_flexion_deg"]),
        "hip_to_ankle_drop_difference_shoulder_widths": abs(
            left["hip_to_ankle_vertical_drop_shoulder_widths"]
            - right["hip_to_ankle_vertical_drop_shoulder_widths"]
        ),
        "knee_height_difference_shoulder_widths": abs(
            left["knee_image_y_shoulder_widths"] - right["knee_image_y_shoulder_widths"]
        ),
        "ankle_height_difference_shoulder_widths": abs(
            left["ankle_image_y_shoulder_widths"] - right["ankle_image_y_shoulder_widths"]
        ),
    }
    return {
        "per_side": {
            side: {name: _round(value) for name, value in values.items()}
            for side, values in per_side.items()
        },
        "asymmetry": {name: _round(value) for name, value in asymmetry.items()},
    }


def _standing_leg_score(leg: dict[str, Any]) -> float:
    if not leg:
        return 0.0
    knee = float(leg.get("knee_flexion_deg") or 0.0)
    hip = float(leg.get("hip_flexion_deg") or 0.0)
    drop = float(leg.get("hip_to_ankle_vertical_drop_shoulder_widths") or 0.0)
    shin = float(leg.get("shin_axis_from_image_down_deg") or 0.0)
    extension = float(leg.get("leg_extension_ratio") or 0.0)
    return float(
        0.30 * _ramp(knee, 120.0, 165.0)
        + 0.20 * _ramp(hip, 120.0, 165.0)
        + 0.20 * _ramp(drop, 0.65, 1.65)
        + 0.15 * (1.0 - _ramp(shin, 15.0, 60.0))
        + 0.15 * _ramp(extension, 0.65, 0.95)
    )


def _free_leg_score(free: dict[str, Any], support: dict[str, Any]) -> float:
    if not free or not support:
        return 0.0
    knee = float(free.get("knee_flexion_deg") or 0.0)
    hip = float(free.get("hip_flexion_deg") or 0.0)
    free_drop = float(free.get("hip_to_ankle_vertical_drop_shoulder_widths") or 0.0)
    ankle_raise = (
        float(support.get("ankle_image_y_shoulder_widths") or 0.0)
        - float(free.get("ankle_image_y_shoulder_widths") or 0.0)
    )
    knee_raise = (
        float(support.get("knee_image_y_shoulder_widths") or 0.0)
        - float(free.get("knee_image_y_shoulder_widths") or 0.0)
    )
    return float(
        0.25 * (1.0 - _ramp(knee, 85.0, 150.0))
        + 0.20 * (1.0 - _ramp(hip, 95.0, 155.0))
        + 0.25 * _ramp(ankle_raise, 0.12, 0.85)
        + 0.15 * _ramp(knee_raise, 0.05, 0.65)
        + 0.15 * (1.0 - _ramp(free_drop, 0.55, 1.45))
    )


def _asymmetry_score(geometry: dict[str, Any]) -> float:
    a = geometry.get("asymmetry") or {}
    if not a:
        return 0.0
    return float(
        0.40 * _ramp(float(a.get("knee_flexion_difference_deg") or 0.0), 20.0, 80.0)
        + 0.30 * _ramp(float(a.get("ankle_height_difference_shoulder_widths") or 0.0), 0.12, 0.80)
        + 0.30 * _ramp(float(a.get("hip_to_ankle_drop_difference_shoulder_widths") or 0.0), 0.18, 0.90)
    )


def _joint_side_support(profile: dict[str, Any], side: str) -> float:
    joint_support = ((profile.get("evidence_support") or {}).get("joint_crop_support") or {})
    return _mean([
        float(joint_support.get(f"{side}_hip") or 0.0),
        float(joint_support.get(f"{side}_knee") or 0.0),
        float(joint_support.get(f"{side}_ankle") or 0.0),
    ])


def _single_leg_support(profile: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    sides = geometry.get("per_side") or {}
    if not sides.get("left") or not sides.get("right"):
        return {"type": "undetermined", "geometry_match": False, "geometry_score": 0.0}

    standing_scores = {side: _standing_leg_score(sides[side]) for side in ("left", "right")}
    support_side = max(standing_scores, key=standing_scores.get)
    free_side = "right" if support_side == "left" else "left"
    support_leg = standing_scores[support_side]
    free_leg = _free_leg_score(sides[free_side], sides[support_side])
    asymmetry = _asymmetry_score(geometry)
    combined = 0.45 * support_leg + 0.35 * free_leg + 0.20 * asymmetry
    match = bool(
        support_leg >= MIN_SUPPORT_LEG_SCORE
        and free_leg >= MIN_FREE_LEG_SCORE
        and asymmetry >= MIN_ASYMMETRY_SCORE
        and combined >= MIN_SINGLE_LEG_SCORE
    )

    support_crop = _joint_side_support(profile, support_side)
    free_crop = _joint_side_support(profile, free_side)
    crop_support = min(support_crop, free_crop)
    return {
        "type": "single_leg_support" if match else "undetermined",
        "geometry_match": match,
        "geometry_score": _round(combined, 4),
        "geometry_score_percent": int(round(100.0 * combined)),
        "candidate_support_side": support_side,
        "candidate_free_leg": free_side,
        "support_leg_standing_score": _round(support_leg, 4),
        "free_leg_raised_score": _round(free_leg, 4),
        "asymmetry_score": _round(asymmetry, 4),
        "crop_support": _round(crop_support, 4),
        "crop_support_percent": int(round(100.0 * crop_support)),
        "support_class": v05._projected_support_class(crop_support),
        "support_components": {
            "candidate_support_leg_crop_support": _round(support_crop, 4),
            "candidate_free_leg_crop_support": _round(free_crop, 4),
            "aggregation": "minimum_required_side",
        },
    }


def _kneeling_candidate(profile: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    sides = geometry.get("per_side") or {}
    if not sides.get("left") or not sides.get("right"):
        return {"report_only": True, "geometry_match": False, "score": 0.0}

    def kneel_side_score(side: str) -> float:
        leg = sides[side]
        knee = float(leg.get("knee_flexion_deg") or 0.0)
        hip = float(leg.get("hip_flexion_deg") or 0.0)
        knee_ankle_drop = abs(float(leg.get("knee_to_ankle_vertical_drop_shoulder_widths") or 0.0))
        return float(
            0.40 * (1.0 - _ramp(knee, 75.0, 135.0))
            + 0.25 * (1.0 - _ramp(knee_ankle_drop, 0.12, 0.80))
            + 0.20 * (1.0 - _ramp(hip, 90.0, 155.0))
            + 0.15 * _ramp(
                float((geometry.get("asymmetry") or {}).get("knee_height_difference_shoulder_widths") or 0.0),
                0.10,
                0.70,
            )
        )

    side_scores = {side: kneel_side_score(side) for side in ("left", "right")}
    knee_side = max(side_scores, key=side_scores.get)
    other_side = "right" if knee_side == "left" else "left"
    other_knee = float(sides[other_side].get("knee_flexion_deg") or 0.0)
    other_leg_participation = v05._band_score(other_knee, 55.0, 75.0, 125.0, 155.0)
    score = 0.80 * side_scores[knee_side] + 0.20 * other_leg_participation
    crop_support = min(_joint_side_support(profile, knee_side), _joint_side_support(profile, other_side))
    return {
        "report_only": True,
        "geometry_match": bool(score >= KNEELING_REVIEW_THRESHOLD),
        "score": _round(score, 4),
        "score_percent": int(round(100.0 * score)),
        "candidate_kneeling_side": knee_side,
        "candidate_other_leg": other_side,
        "crop_support": _round(crop_support, 4),
        "crop_support_percent": int(round(100.0 * crop_support)),
        "support_class": v05._projected_support_class(crop_support),
        "threshold": KNEELING_REVIEW_THRESHOLD,
    }


def _apply_single_leg_topology(profile: dict[str, Any], support_state: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    raw_scores = {
        str(name): float(value or 0.0)
        for name, value in (projected.get("posture_scores") or {}).items()
    }
    projected["posture_scores_before_support_topology"] = {
        name: _round(value, 4) for name, value in raw_scores.items()
    }
    projected["posture_score_percent_before_support_topology"] = {
        name: int(round(100.0 * value)) for name, value in raw_scores.items()
    }

    scores = dict(raw_scores)
    if support_state.get("geometry_match"):
        topology = float(support_state.get("geometry_score") or 0.0)
        support_leg = float(support_state.get("support_leg_standing_score") or 0.0)
        topology_standing = 0.50 * support_leg + 0.50 * topology
        scores["standing"] = max(float(scores.get("standing") or 0.0), topology_standing)
        scores["crouching"] = float(scores.get("crouching") or 0.0) * (1.0 - 0.45 * topology)
        scores["squatting"] = float(scores.get("squatting") or 0.0) * (1.0 - 0.55 * topology)
        scores["sitting"] = float(scores.get("sitting") or 0.0) * (1.0 - 0.45 * topology)

    pose, best_candidate, best_score, margin = v05._choose_posture(scores)
    region_support = {
        str(name): float(value or 0.0)
        for name, value in (projected.get("region_support") or {}).items()
    }
    coverage, support, coverage_regions, support_regions = v05._crop_support(
        region_support, pose, best_candidate
    )
    projected.update({
        "pose": pose,
        "best_candidate_pose": best_candidate,
        "posture_scores": {name: _round(value, 4) for name, value in scores.items()},
        "posture_score_percent": {name: int(round(100.0 * value)) for name, value in scores.items()},
        "winner_margin": _round(margin, 4),
        "winner_margin_percent": int(round(100.0 * margin)),
        "reconstruction_match": _round(best_score, 4),
        "reconstruction_match_percent": int(round(100.0 * best_score)),
        "crop_coverage": _round(coverage, 4),
        "crop_coverage_percent": int(round(100.0 * coverage)),
        "crop_support": _round(support, 4),
        "crop_support_percent": int(round(100.0 * support)),
        "pose_support": _round(support, 4),
        "pose_support_percent": int(round(100.0 * support)),
        "crop_supported_regions": coverage_regions,
        "pose_support_regions": support_regions,
        "support_class": v05._projected_support_class(support),
        "support_state": support_state,
    })
    if support_state.get("geometry_match"):
        projected.setdefault("modifiers", {})["one_leg_raised"] = {
            "geometry_match": True,
            "side": support_state.get("candidate_free_leg"),
            "geometry_score": support_state.get("geometry_score"),
            "geometry_score_percent": support_state.get("geometry_score_percent"),
            "crop_support": support_state.get("crop_support"),
            "crop_support_percent": support_state.get("crop_support_percent"),
            "support_class": support_state.get("support_class"),
        }
    profile["sam3d_projected_pose"] = projected


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v05.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.6"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]

    geometry = _leg_geometry(keypoints)
    projected = profile.get("sam3d_projected_pose") or {}
    projected_geometry = dict(projected.get("geometry") or {})
    projected_geometry["asymmetric_lower_body"] = geometry
    projected["geometry"] = projected_geometry
    profile["sam3d_projected_pose"] = projected

    support_state = _single_leg_support(profile, geometry)
    _apply_single_leg_topology(profile, support_state)
    profile["sam3d_projected_pose"]["kneeling_candidate"] = _kneeling_candidate(profile, geometry)

    policy = profile.get("policy") or {}
    policy.update({
        "bilateral_posture_means_do_not_override_supported_leg_topology": True,
        "single_leg_support_is_geometry_not_action_semantics": True,
        "one_leg_raised_is_geometry_modifier": True,
        "kneeling_candidate_is_report_only_until_calibrated": True,
        "kneeling_not_yet_a_projected_posture_family": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-06",
        description=(
            "Build report-only SAM3D/DWPose pose profiles with asymmetric lower-body "
            "support topology, one-leg-raised modifiers, and report-only kneeling candidates."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.6")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    helpers = v05.v04.v03
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
        state = projected.get("support_state") or {}
        state_label = "-"
        if state.get("geometry_match"):
            state_label = (
                f"single_leg:{state.get('candidate_support_side')} "
                f"free:{state.get('candidate_free_leg')}@{state.get('crop_support_percent', 0)}%"
            )
        kneel = projected.get("kneeling_candidate") or {}
        kneel_label = f"{kneel.get('score_percent', 0)}%" + ("*" if kneel.get("geometry_match") else "")
        relations = profile.get("relations") or {}
        flags = []
        for name in ("hands_on_hips", "head_supported_by_hand", "head_supported_by_fist"):
            value = relations.get(name) or {}
            if value.get("geometry_match"):
                label = name
                if value.get("side"):
                    label += f":{value['side']}"
                label += f"@{value.get('crop_support_percent') or 0}%[{value.get('support_class')}]"
                flags.append(label)
        hand_labels = ",".join(helpers._hand_console_label(profile, side) for side in ("left", "right"))
        print(
            f"{key}: projected={projected['pose']} best={projected['best_candidate_pose']} "
            f"scores=stand:{scores.get('standing', 0)} crouch:{scores.get('crouching', 0)} "
            f"squat:{scores.get('squatting', 0)} sit:{scores.get('sitting', 0)} recl:{scores.get('reclined', 0)} "
            f"crop={projected['crop_support_percent']}%[{projected.get('support_class')}] "
            f"support={state_label} kneel_candidate={kneel_label} "
            f"hands={hand_labels} relations={','.join(flags) or '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.6",
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
