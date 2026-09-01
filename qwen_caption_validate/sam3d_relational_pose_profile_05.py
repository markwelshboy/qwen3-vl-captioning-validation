from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from . import sam3d_relational_pose_profile_04 as v04


MHR = v04.MHR
MIN_POSTURE_SCORE = 0.52
MIN_POSTURE_MARGIN = 0.08


def _round(value: float | None, digits: int = 3) -> float | None:
    return v04._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v04._ramp(value, low, high)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return None
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def _mean(values: list[float | None]) -> float | None:
    good = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(good)) if good else None


def _lower_body_geometry(keypoints: np.ndarray) -> dict[str, float | None]:
    needed = [
        MHR["left_shoulder"], MHR["right_shoulder"],
        MHR["left_hip"], MHR["right_hip"],
        MHR["left_knee"], MHR["right_knee"],
        MHR["left_ankle"], MHR["right_ankle"],
    ]
    empty = {
        "left_hip_flexion_deg": None,
        "right_hip_flexion_deg": None,
        "mean_hip_flexion_deg": None,
        "mean_thigh_axis_from_image_down_deg": None,
        "mean_shin_axis_from_image_down_deg": None,
        "mean_hip_to_knee_vertical_drop_shoulder_widths": None,
        "mean_knee_to_ankle_vertical_drop_shoulder_widths": None,
        "hip_to_ankle_horizontal_offset_shoulder_widths": None,
        "shoulder_to_ankle_horizontal_offset_shoulder_widths": None,
    }
    if not len(keypoints) or max(needed) >= len(keypoints):
        return empty

    ls = keypoints[MHR["left_shoulder"], :3]
    rs = keypoints[MHR["right_shoulder"], :3]
    lh = keypoints[MHR["left_hip"], :3]
    rh = keypoints[MHR["right_hip"], :3]
    lk = keypoints[MHR["left_knee"], :3]
    rk = keypoints[MHR["right_knee"], :3]
    la = keypoints[MHR["left_ankle"], :3]
    ra = keypoints[MHR["right_ankle"], :3]
    points = np.stack([ls, rs, lh, rh, lk, rk, la, ra], axis=0)
    if not np.all(np.isfinite(points)):
        return empty

    shoulder_width = float(np.linalg.norm(ls - rs))
    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    ankle_mid = (la + ra) / 2.0

    left_hip_angle = _angle(ls, lh, lk)
    right_hip_angle = _angle(rs, rh, rk)

    thigh_angles = [
        v04._angle_from_image_down(lk - lh),
        v04._angle_from_image_down(rk - rh),
    ]
    shin_angles = [
        v04._angle_from_image_down(la - lk),
        v04._angle_from_image_down(ra - rk),
    ]

    hip_knee_drops: list[float] = []
    knee_ankle_drops: list[float] = []
    if shoulder_width > 1e-9:
        hip_knee_drops = [
            float((lk[1] - lh[1]) / shoulder_width),
            float((rk[1] - rh[1]) / shoulder_width),
        ]
        knee_ankle_drops = [
            float((la[1] - lk[1]) / shoulder_width),
            float((ra[1] - rk[1]) / shoulder_width),
        ]

    hip_ankle_horizontal = None
    shoulder_ankle_horizontal = None
    if shoulder_width > 1e-9:
        # Image-horizontal + camera-depth displacement. This is useful for
        # distinguishing a balanced low stance from a seated-like geometry
        # where the pelvis is substantially displaced from the foot support.
        hip_delta = hip_mid - ankle_mid
        shoulder_delta = shoulder_mid - ankle_mid
        hip_ankle_horizontal = float(np.linalg.norm(hip_delta[[0, 2]]) / shoulder_width)
        shoulder_ankle_horizontal = float(np.linalg.norm(shoulder_delta[[0, 2]]) / shoulder_width)

    return {
        "left_hip_flexion_deg": _round(left_hip_angle),
        "right_hip_flexion_deg": _round(right_hip_angle),
        "mean_hip_flexion_deg": _round(_mean([left_hip_angle, right_hip_angle])),
        "mean_thigh_axis_from_image_down_deg": _round(_mean(thigh_angles)),
        "mean_shin_axis_from_image_down_deg": _round(_mean(shin_angles)),
        "mean_hip_to_knee_vertical_drop_shoulder_widths": _round(_mean(hip_knee_drops)),
        "mean_knee_to_ankle_vertical_drop_shoulder_widths": _round(_mean(knee_ankle_drops)),
        "hip_to_ankle_horizontal_offset_shoulder_widths": _round(hip_ankle_horizontal),
        "shoulder_to_ankle_horizontal_offset_shoulder_widths": _round(shoulder_ankle_horizontal),
    }


def _band_score(value: float, low: float, peak_low: float, peak_high: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if peak_low <= value <= peak_high:
        return 1.0
    if value < peak_low:
        return _ramp(value, low, peak_low)
    return 1.0 - _ramp(value, peak_high, high)


def _posture_scores(geometry: dict[str, Any]) -> dict[str, float]:
    knee = geometry.get("mean_knee_angle_deg")
    hip = geometry.get("mean_hip_flexion_deg")
    leg_angle = geometry.get("leg_axis_from_image_down_deg")
    drop = geometry.get("mean_hip_to_ankle_vertical_drop_shoulder_widths")
    torso_angle = geometry.get("torso_axis_from_image_down_deg")
    body_angle = geometry.get("body_axis_from_image_down_deg")
    thigh_angle = geometry.get("mean_thigh_axis_from_image_down_deg")
    shin_angle = geometry.get("mean_shin_axis_from_image_down_deg")
    hip_knee_drop = geometry.get("mean_hip_to_knee_vertical_drop_shoulder_widths")
    hip_ankle_offset = geometry.get("hip_to_ankle_horizontal_offset_shoulder_widths")

    required = (knee, hip, leg_angle, drop, torso_angle, body_angle, thigh_angle, shin_angle, hip_knee_drop, hip_ankle_offset)
    if any(value is None for value in required):
        return {"standing": 0.0, "crouching": 0.0, "squatting": 0.0, "sitting": 0.0, "reclined": 0.0}

    knee = float(knee)
    hip = float(hip)
    leg_angle = float(leg_angle)
    drop = float(drop)
    torso_angle = float(torso_angle)
    body_angle = float(body_angle)
    thigh_angle = float(thigh_angle)
    shin_angle = float(shin_angle)
    hip_knee_drop = float(hip_knee_drop)
    hip_ankle_offset = float(hip_ankle_offset)

    straight_knee = _ramp(knee, 115.0, 165.0)
    straight_hip = _ramp(hip, 125.0, 172.0)
    bent_knee = 1.0 - _ramp(knee, 80.0, 150.0)
    deep_knee = 1.0 - _ramp(knee, 75.0, 125.0)
    flexed_hip = 1.0 - _ramp(hip, 90.0, 160.0)
    deep_hip = 1.0 - _ramp(hip, 80.0, 135.0)
    torso_upright = 1.0 - _ramp(torso_angle, 20.0, 65.0)
    torso_bent = _ramp(torso_angle, 18.0, 60.0)
    body_vertical = 1.0 - _ramp(body_angle, 20.0, 65.0)
    thigh_horizontal = _ramp(thigh_angle, 35.0, 80.0)
    shin_vertical = 1.0 - _ramp(shin_angle, 15.0, 55.0)
    pelvis_low = 1.0 - _ramp(hip_knee_drop, 0.15, 0.85)
    pelvis_above_knee = _ramp(hip_knee_drop, 0.15, 0.85)
    balanced_over_feet = 1.0 - _ramp(hip_ankle_offset, 0.35, 1.35)
    pelvis_displaced_from_feet = _ramp(hip_ankle_offset, 0.45, 1.45)

    standing = (
        0.25 * straight_knee
        + 0.20 * straight_hip
        + 0.15 * (1.0 - _ramp(leg_angle, 12.0, 50.0))
        + 0.15 * _ramp(drop, 0.80, 1.80)
        + 0.15 * torso_upright
        + 0.10 * body_vertical
    )

    # Crouching is a lowered, flexed stance with the pelvis still clearly above
    # the knees and a commonly forward-bent torso. It intentionally tolerates
    # asymmetry and is broader than a deep squat.
    crouching = (
        0.25 * _band_score(knee, 70.0, 90.0, 130.0, 155.0)
        + 0.20 * flexed_hip
        + 0.20 * torso_bent
        + 0.15 * pelvis_above_knee
        + 0.10 * _ramp(body_angle, 15.0, 55.0)
        + 0.10 * _ramp(drop, 0.65, 1.55)
    )

    # Squatting is the deeper low stance: strong bilateral knee/hip flexion,
    # pelvis close to knee height, thighs approaching horizontal, and the pelvis
    # remaining comparatively balanced over the feet. Scene semantics are not
    # used, so seated-vs-squat can remain competitive when geometry alone is
    # genuinely ambiguous.
    squatting = (
        0.25 * deep_knee
        + 0.20 * deep_hip
        + 0.15 * thigh_horizontal
        + 0.15 * pelvis_low
        + 0.10 * shin_vertical
        + 0.15 * balanced_over_feet
    )

    sitting = (
        0.30 * bent_knee
        + 0.20 * flexed_hip
        + 0.15 * thigh_horizontal
        + 0.15 * torso_upright
        + 0.20 * pelvis_displaced_from_feet
    )

    reclined = (
        0.45 * _ramp(torso_angle, 35.0, 75.0)
        + 0.30 * _ramp(body_angle, 35.0, 75.0)
        + 0.15 * _ramp(leg_angle, 25.0, 70.0)
        + 0.10 * (1.0 - _ramp(drop, 0.60, 1.60))
    )

    return {
        "standing": max(0.0, min(1.0, float(standing))),
        "crouching": max(0.0, min(1.0, float(crouching))),
        "squatting": max(0.0, min(1.0, float(squatting))),
        "sitting": max(0.0, min(1.0, float(sitting))),
        "reclined": max(0.0, min(1.0, float(reclined))),
    }


def _choose_posture(scores: dict[str, float]) -> tuple[str, str, float, float]:
    ranked = sorted(scores.items(), key=lambda item: (float(item[1]), item[0]), reverse=True)
    best_name, best_score = ranked[0]
    second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    margin = float(best_score) - second_score
    pose = best_name if float(best_score) >= MIN_POSTURE_SCORE and margin >= MIN_POSTURE_MARGIN else "uncertain"
    return pose, best_name, float(best_score), margin


def _crop_support(region_support: dict[str, float], pose: str, best_candidate: str) -> tuple[float, float, list[str], list[str]]:
    ordered = ["head", "shoulders", "hips", "thighs", "knees", "lower_legs", "feet"]
    coverage = float(np.mean([float(region_support.get(name, 0.0)) for name in ordered]))
    coverage_regions = [name for name in ordered if float(region_support.get(name, 0.0)) >= v04.v03.v02.REGION_SUPPORT_THRESHOLD]

    family = pose if pose != "uncertain" else best_candidate
    if family == "standing":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.20, "lower_legs": 0.20, "feet": 0.10}
    elif family == "crouching":
        weights = {"head": 0.00, "shoulders": 0.15, "hips": 0.25, "thighs": 0.20, "knees": 0.20, "lower_legs": 0.15, "feet": 0.05}
    elif family == "squatting":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.20, "lower_legs": 0.15, "feet": 0.15}
    elif family == "sitting":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.25, "lower_legs": 0.15, "feet": 0.10}
    elif family == "reclined":
        weights = {"head": 0.00, "shoulders": 0.15, "hips": 0.25, "thighs": 0.20, "knees": 0.15, "lower_legs": 0.15, "feet": 0.10}
    else:
        weights = {name: 1.0 / len(ordered) for name in ordered}

    support = float(sum(float(region_support.get(name, 0.0)) * weight for name, weight in weights.items()))
    support_regions = [
        name for name in ordered
        if weights.get(name, 0.0) > 0.0 and float(region_support.get(name, 0.0)) >= v04.v03.v02.REGION_SUPPORT_THRESHOLD
    ]
    return coverage, support, coverage_regions, support_regions


def _projected_support_class(support: float) -> str:
    if support >= 0.50:
        return "strongly_crop_supported"
    if support >= 0.20:
        return "moderately_crop_supported"
    if support >= 0.10:
        return "weakly_crop_supported"
    return "reconstruction_dominant"


def _cap_head_by_observed_hand(profile: dict[str, Any]) -> None:
    relations = profile.get("relations") or {}
    head = relations.get("head_supported_by_hand") or {}
    if head.get("geometry_match"):
        side = str(head.get("side") or "")
        hand_evidence = ((profile.get("hand_geometry") or {}).get(side) or {}).get("dwpose_hand") or {}
        hand_support = float(hand_evidence.get("crop_support") or 0.0)
        raw_support = float(head.get("crop_support") or 0.0)
        support = min(raw_support, hand_support)
        components = dict(head.get("support_components") or {})
        components["head_relation_support_before_hand_cap"] = _round(raw_support, 4)
        components["observed_hand_crop_support"] = _round(hand_support, 4)
        components["aggregation"] = "minimum_required_component"
        head["crop_support"] = _round(support, 4)
        head["crop_support_percent"] = int(round(100.0 * support))
        head["support_components"] = components
        head["support_class"] = v04._support_class(head)
    relations["head_supported_by_hand"] = head

    fist = relations.get("head_supported_by_fist") or {}
    if fist.get("geometry_match"):
        side = str(fist.get("side") or "")
        hand_evidence = ((profile.get("hand_geometry") or {}).get(side) or {}).get("dwpose_hand") or {}
        hand_support = float(hand_evidence.get("crop_support") or 0.0)
        support = min(float(head.get("crop_support") or 0.0), hand_support)
        fist["crop_support"] = _round(support, 4)
        fist["crop_support_percent"] = int(round(100.0 * support))
        fist["support_class"] = v04._support_class(fist)
    relations["head_supported_by_fist"] = fist
    profile["relations"] = relations


def build_profile(arrays: dict[str, np.ndarray], dwpose: dict[str, Any] | None, width: int, height: int) -> dict[str, Any]:
    profile = v04.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.5"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]

    projected = profile.get("sam3d_projected_pose") or {}
    geometry = dict(projected.get("geometry") or {})
    geometry.update(_lower_body_geometry(keypoints))
    scores = _posture_scores(geometry)
    pose, best_candidate, best_score, margin = _choose_posture(scores)

    region_support = {str(name): float(value or 0.0) for name, value in (projected.get("region_support") or {}).items()}
    coverage, support, coverage_regions, support_regions = _crop_support(region_support, pose, best_candidate)

    projected.update(
        {
            "pose": pose,
            "best_candidate_pose": best_candidate,
            "posture_scores": {name: _round(value, 4) for name, value in scores.items()},
            "posture_score_percent": {name: int(round(100.0 * value)) for name, value in scores.items()},
            "winner_margin": _round(margin, 4),
            "winner_margin_percent": int(round(100.0 * margin)),
            "decision_thresholds": {
                "minimum_winner_score": MIN_POSTURE_SCORE,
                "minimum_winner_margin": MIN_POSTURE_MARGIN,
            },
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
            "support_class": _projected_support_class(support),
            "geometry": geometry,
        }
    )
    profile["sam3d_projected_pose"] = projected

    _cap_head_by_observed_hand(profile)

    policy = profile.get("policy") or {}
    policy.update(
        {
            "projected_posture_families": ["standing", "crouching", "squatting", "sitting", "reclined"],
            "crouching_is_flexed_lowered_stance_geometry": True,
            "squatting_is_deep_low_stance_geometry": True,
            "sitting_vs_squatting_may_require_scene_support_in_fusion": True,
            "projected_pose_support_class_is_descriptive_not_a_hard_fusion_gate": True,
            "head_supported_relation_requires_observed_hand_support": True,
        }
    )
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-05",
        description=(
            "Build report-only SAM3D/DWPose pose profiles with competing standing, "
            "crouching, squatting, sitting and reclined geometry scores."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.5")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    rows: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        dwpose = v04.v03._read_json(dwpose_path)
        width = int(dwpose.get("image_width") or 0)
        height = int(dwpose.get("image_height") or 0)
        if width <= 0 or height <= 0:
            image_matches = [p for p in images_dir.rglob(f"{key}.*") if p.is_file()] if images_dir.is_dir() else []
            if not image_matches:
                raise SystemExit(f"Cannot determine image size for {key}")
            from PIL import Image
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
        v04.v03._write_json(out_path, record)
        rows.append(record)

        projected = profile["sam3d_projected_pose"]
        scores = projected.get("posture_score_percent") or {}
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

        hand_labels = ",".join(v04.v03._hand_console_label(profile, side) for side in ("left", "right"))
        print(
            f"{key}: projected={projected['pose']} best={projected['best_candidate_pose']} "
            f"scores=stand:{scores.get('standing', 0)} crouch:{scores.get('crouching', 0)} "
            f"squat:{scores.get('squatting', 0)} sit:{scores.get('sitting', 0)} recl:{scores.get('reclined', 0)} "
            f"crop={projected['crop_support_percent']}%[{projected['support_class']}] "
            f"coverage={projected['crop_coverage_percent']}% hands={hand_labels} relations={','.join(flags) or '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.5",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    index_path = output / "sam3d_relational_pose.index.json"
    v04.v03._write_json(index_path, index)
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
