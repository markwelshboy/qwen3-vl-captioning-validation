from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from . import pose_atlas_v3_03 as atlas
from . import sam3d_relational_pose_profile_03 as v03


MHR = atlas.MHR_BODY
MIN_POSTURE_SCORE = 0.52
MIN_POSTURE_MARGIN = 0.08
NAMED_RELATION_SUPPORT_THRESHOLD = 0.50


def _round(value: float | None, digits: int = 3) -> float | None:
    return v03._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _angle_from_image_down(vector: np.ndarray) -> float | None:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape[0] < 3 or not np.all(np.isfinite(vector[:3])):
        return None
    norm = float(np.linalg.norm(vector[:3]))
    if norm <= 1e-9:
        return None
    cosine = float(vector[1] / norm)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def _screen_angle_from_image_down(vector: np.ndarray) -> float | None:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape[0] < 2 or not np.all(np.isfinite(vector[:2])):
        return None
    norm = float(np.linalg.norm(vector[:2]))
    if norm <= 1e-9:
        return None
    cosine = float(vector[1] / norm)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def _axis_geometry(keypoints: np.ndarray) -> dict[str, float | None]:
    needed = [
        MHR["left_shoulder"], MHR["right_shoulder"],
        MHR["left_hip"], MHR["right_hip"],
        MHR["left_ankle"], MHR["right_ankle"],
    ]
    if not len(keypoints) or max(needed) >= len(keypoints):
        return {
            "torso_axis_from_image_down_deg": None,
            "torso_screen_axis_from_image_down_deg": None,
            "body_axis_from_image_down_deg": None,
            "torso_vertical_drop_shoulder_widths": None,
            "torso_depth_fraction": None,
        }

    ls = keypoints[MHR["left_shoulder"], :3]
    rs = keypoints[MHR["right_shoulder"], :3]
    lh = keypoints[MHR["left_hip"], :3]
    rh = keypoints[MHR["right_hip"], :3]
    la = keypoints[MHR["left_ankle"], :3]
    ra = keypoints[MHR["right_ankle"], :3]
    points = np.stack([ls, rs, lh, rh, la, ra], axis=0)
    if not np.all(np.isfinite(points)):
        return {
            "torso_axis_from_image_down_deg": None,
            "torso_screen_axis_from_image_down_deg": None,
            "body_axis_from_image_down_deg": None,
            "torso_vertical_drop_shoulder_widths": None,
            "torso_depth_fraction": None,
        }

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    ankle_mid = (la + ra) / 2.0
    torso = hip_mid - shoulder_mid
    body = ankle_mid - shoulder_mid
    shoulder_width = float(np.linalg.norm(ls - rs))
    torso_norm = float(np.linalg.norm(torso))

    return {
        "torso_axis_from_image_down_deg": _round(_angle_from_image_down(torso)),
        "torso_screen_axis_from_image_down_deg": _round(_screen_angle_from_image_down(torso)),
        "body_axis_from_image_down_deg": _round(_angle_from_image_down(body)),
        "torso_vertical_drop_shoulder_widths": _round(
            float(torso[1] / shoulder_width) if shoulder_width > 1e-9 else None
        ),
        "torso_depth_fraction": _round(
            float(abs(torso[2]) / torso_norm) if torso_norm > 1e-9 else None
        ),
    }


def _posture_scores(geometry: dict[str, Any]) -> dict[str, float]:
    knee = geometry.get("mean_knee_angle_deg")
    leg_angle = geometry.get("leg_axis_from_image_down_deg")
    drop = geometry.get("mean_hip_to_ankle_vertical_drop_shoulder_widths")
    torso_angle = geometry.get("torso_axis_from_image_down_deg")
    body_angle = geometry.get("body_axis_from_image_down_deg")
    if any(value is None for value in (knee, leg_angle, drop, torso_angle, body_angle)):
        return {"standing": 0.0, "sitting": 0.0, "reclined": 0.0}

    knee = float(knee)
    leg_angle = float(leg_angle)
    drop = float(drop)
    torso_angle = float(torso_angle)
    body_angle = float(body_angle)

    # Standing: increasingly straight knees, vertically descending legs/body,
    # and substantial hip-to-ankle vertical extent. These are soft scores, not
    # hard gates, so a 107-degree mean knee no longer falls off a 110-degree cliff.
    standing = (
        0.30 * _ramp(knee, 95.0, 160.0)
        + 0.20 * (1.0 - _ramp(leg_angle, 12.0, 50.0))
        + 0.20 * _ramp(drop, 0.75, 1.80)
        + 0.15 * (1.0 - _ramp(torso_angle, 20.0, 65.0))
        + 0.15 * (1.0 - _ramp(body_angle, 20.0, 65.0))
    )

    # Sitting: bent knees plus a compact/non-vertical leg chain, while the
    # torso itself is usually substantially more upright than in a reclined pose.
    sitting = (
        0.40 * (1.0 - _ramp(knee, 75.0, 140.0))
        + 0.20 * _ramp(leg_angle, 15.0, 60.0)
        + 0.20 * (1.0 - _ramp(drop, 0.75, 1.75))
        + 0.20 * (1.0 - _ramp(torso_angle, 25.0, 70.0))
    )

    # Reclined is deliberately a geometric family rather than the semantic
    # distinction between "lying" and "reclining". A strongly non-vertical
    # torso/body axis dominates; leg bend is intentionally not required because
    # a reclined person may have straight or bent legs.
    reclined = (
        0.45 * _ramp(torso_angle, 35.0, 75.0)
        + 0.30 * _ramp(body_angle, 35.0, 75.0)
        + 0.15 * _ramp(leg_angle, 25.0, 70.0)
        + 0.10 * (1.0 - _ramp(drop, 0.60, 1.60))
    )

    return {
        "standing": max(0.0, min(1.0, float(standing))),
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


def _crop_support_v04(
    region_support: dict[str, float],
    pose: str,
    best_candidate: str,
) -> tuple[float, float, list[str], list[str]]:
    ordered = ["head", "shoulders", "hips", "thighs", "knees", "lower_legs", "feet"]
    coverage = float(np.mean([float(region_support.get(name, 0.0)) for name in ordered]))
    coverage_regions = [
        name for name in ordered
        if float(region_support.get(name, 0.0)) >= v03.v02.REGION_SUPPORT_THRESHOLD
    ]

    family = pose if pose != "uncertain" else best_candidate
    if family == "standing":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.20, "lower_legs": 0.20, "feet": 0.10}
    elif family == "sitting":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.25, "lower_legs": 0.15, "feet": 0.10}
    elif family == "reclined":
        # Shoulders are the coarse BODY18 proxy for upper-torso observation.
        weights = {"head": 0.00, "shoulders": 0.15, "hips": 0.25, "thighs": 0.20, "knees": 0.15, "lower_legs": 0.15, "feet": 0.10}
    else:
        weights = {name: 1.0 / len(ordered) for name in ordered}

    support = float(sum(float(region_support.get(name, 0.0)) * weight for name, weight in weights.items()))
    support_regions = [
        name for name in ordered
        if weights.get(name, 0.0) > 0.0
        and float(region_support.get(name, 0.0)) >= v03.v02.REGION_SUPPORT_THRESHOLD
    ]
    return coverage, support, coverage_regions, support_regions


def _support_class(relation: dict[str, Any]) -> str:
    if not relation.get("geometry_match"):
        return "not_matched"
    support = float(relation.get("crop_support") or 0.0)
    if support >= NAMED_RELATION_SUPPORT_THRESHOLD:
        return "crop_supported"
    if support > 0.0:
        return "weakly_crop_supported"
    return "reconstruction_only"


def _cap_head_relations(profile: dict[str, Any]) -> None:
    relations = profile.get("relations") or {}
    projected = profile.get("sam3d_projected_pose") or {}
    head_region_support = float((projected.get("region_support") or {}).get("head") or 0.0)

    head = relations.get("head_supported_by_hand") or {}
    if head.get("geometry_match"):
        raw_support = float(head.get("crop_support") or 0.0)
        support = min(raw_support, head_region_support)
        head["crop_support_before_head_observation_cap"] = _round(raw_support, 4)
        head["crop_support"] = _round(support, 4)
        head["crop_support_percent"] = int(round(100.0 * support))
        head["support_components"] = {
            "arm_hand_relation_crop_support": _round(raw_support, 4),
            "observed_head_region_support": _round(head_region_support, 4),
            "aggregation": "minimum_required_component",
        }
    head["support_class"] = _support_class(head)
    relations["head_supported_by_hand"] = head

    fist = relations.get("head_supported_by_fist") or {}
    if fist.get("geometry_match"):
        side = str(fist.get("side") or "")
        observed_hand = ((profile.get("hand_geometry") or {}).get(side) or {}).get("dwpose_hand") or {}
        fist_support = float(observed_hand.get("crop_support") or 0.0)
        head_support = float(head.get("crop_support") or 0.0)
        support = min(head_support, fist_support)
        fist["crop_support"] = _round(support, 4)
        fist["crop_support_percent"] = int(round(100.0 * support))
        fist["support_components"] = {
            "head_supported_by_hand_crop_support": _round(head_support, 4),
            "observed_fist_crop_support": _round(fist_support, 4),
            "aggregation": "minimum_required_component",
        }
    fist["support_class"] = _support_class(fist)
    relations["head_supported_by_fist"] = fist

    hips = relations.get("hands_on_hips") or {}
    hips["support_class"] = _support_class(hips)
    relations["hands_on_hips"] = hips
    profile["relations"] = relations


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v03.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.4"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]

    projected = profile.get("sam3d_projected_pose") or {}
    geometry = dict(projected.get("geometry") or {})
    geometry.update(_axis_geometry(keypoints))
    scores = _posture_scores(geometry)
    pose, best_candidate, best_score, margin = _choose_posture(scores)

    region_support = {
        str(name): float(value or 0.0)
        for name, value in (projected.get("region_support") or {}).items()
    }
    coverage, support, coverage_regions, support_regions = _crop_support_v04(
        region_support, pose, best_candidate
    )

    projected.update(
        {
            "pose": pose,
            "source": "sam3d_reconstruction",
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
            "region_support": {name: _round(value, 4) for name, value in region_support.items()},
            "geometry": geometry,
        }
    )
    profile["sam3d_projected_pose"] = projected

    _cap_head_relations(profile)

    policy = profile.get("policy") or {}
    policy.update(
        {
            "projected_posture_uses_competing_continuous_scores": True,
            "projected_posture_families": ["standing", "sitting", "reclined"],
            "reclined_is_geometry_not_surface_semantics": True,
            "lying_vs_reclining_reserved_for_fusion_caption": True,
            "named_relation_support_threshold": NAMED_RELATION_SUPPORT_THRESHOLD,
            "head_supported_relation_requires_observed_head_support": True,
        }
    )
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-04",
        description=(
            "Build report-only SAM3D/DWPose pose profiles with continuous "
            "standing/sitting/reclined scores and crop-governed relations."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.4")).expanduser().resolve()
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
        dwpose = v03._read_json(dwpose_path)
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
        v03._write_json(out_path, record)
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

        hand_labels = ",".join(v03._hand_console_label(profile, side) for side in ("left", "right"))
        print(
            f"{key}: projected={projected['pose']} best={projected['best_candidate_pose']} "
            f"scores=stand:{scores.get('standing', 0)} sit:{scores.get('sitting', 0)} recl:{scores.get('reclined', 0)} "
            f"crop={projected['crop_support_percent']}% coverage={projected['crop_coverage_percent']}% "
            f"hands={hand_labels} relations={','.join(flags) or '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.4",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    index_path = output / "sam3d_relational_pose.index.json"
    v03._write_json(index_path, index)
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
