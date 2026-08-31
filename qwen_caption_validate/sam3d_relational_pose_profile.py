from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import pose_atlas_v3 as base
from . import pose_atlas_v3_03 as atlas


MHR = atlas.MHR_BODY
LEFT_HAND = list(range(42, 62))
RIGHT_HAND = list(range(21, 41))
REGION_SUPPORT_THRESHOLD = 0.35
EDGE_MARGIN_FRACTION = 0.08


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return None
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _mean_point(points: np.ndarray, indices: list[int]) -> np.ndarray:
    valid = [idx for idx in indices if idx < len(points) and np.all(np.isfinite(points[idx, :3]))]
    if not valid:
        return np.full(3, np.nan, dtype=np.float64)
    return np.mean(points[valid, :3], axis=0)


def _agreement_factor(residual_px: float, shoulder_span_px: float) -> float:
    if shoulder_span_px <= 1e-9:
        return 0.0
    x = float(residual_px) / (0.35 * shoulder_span_px)
    return 1.0 / (1.0 + x * x)


def _edge_factor(point: np.ndarray, width: int, height: int) -> float:
    if not atlas._finite_xy(point):
        return 0.0
    x = float(point[0]) / float(width)
    y = float(point[1]) / float(height)
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        return 0.0
    margin = min(x, 1.0 - x, y, 1.0 - y)
    return max(0.0, min(1.0, margin / EDGE_MARGIN_FRACTION))


def _joint_support(
    dw_points: np.ndarray,
    in_frame_names: set[str],
    residual: dict[str, Any],
    sam2d: np.ndarray,
    width: int,
    height: int,
) -> dict[str, float]:
    if len(sam2d) <= max(MHR["left_shoulder"], MHR["right_shoulder"]):
        return {name: 0.0 for name in base.BODY18}
    shoulder_span = _distance(
        sam2d[MHR["left_shoulder"], :2],
        sam2d[MHR["right_shoulder"], :2],
    )
    per_joint = residual.get("per_joint_px") or {}
    out: dict[str, float] = {}
    for name in base.BODY18:
        idx = base.IDX[name]
        if name not in in_frame_names or idx >= len(dw_points) or name not in per_joint:
            out[name] = 0.0
            continue
        agreement = _agreement_factor(float(per_joint[name]), shoulder_span)
        edge = _edge_factor(dw_points[idx], width, height)
        out[name] = float(agreement * edge)
    return out


def _mean_support(joint_support: dict[str, float], names: list[str]) -> float:
    return float(np.mean([joint_support.get(name, 0.0) for name in names])) if names else 0.0


def _region_support(joint_support: dict[str, float]) -> dict[str, float]:
    # These are deliberately coarse vertical body regions for projected-pose
    # support. Arms are described separately by the relational profile.
    regions = {
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck"],
        "shoulders": ["left_shoulder", "right_shoulder"],
        "hips": ["left_hip", "right_hip"],
        "thighs": ["left_hip", "right_hip", "left_knee", "right_knee"],
        "knees": ["left_knee", "right_knee"],
        "lower_legs": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
        # BODY18 has no toe/heel observation. Ankles are used only as a weak
        # proxy for whether the crop reaches the feet region.
        "feet": ["left_ankle", "right_ankle"],
    }
    return {name: _mean_support(joint_support, joints) for name, joints in regions.items()}


def _projected_pose(keypoints: np.ndarray, shoulder_width: float) -> tuple[str, dict[str, float | None]]:
    left_knee = _angle(keypoints[MHR["left_hip"]], keypoints[MHR["left_knee"]], keypoints[MHR["left_ankle"]])
    right_knee = _angle(keypoints[MHR["right_hip"]], keypoints[MHR["right_knee"]], keypoints[MHR["right_ankle"]])
    knee_values = [v for v in (left_knee, right_knee) if v is not None]
    mean_knee = float(np.mean(knee_values)) if knee_values else None

    hip_mid = (keypoints[MHR["left_hip"]] + keypoints[MHR["right_hip"]]) / 2.0
    ankle_mid = (keypoints[MHR["left_ankle"]] + keypoints[MHR["right_ankle"]]) / 2.0
    leg_axis = ankle_mid - hip_mid
    leg_norm = float(np.linalg.norm(leg_axis))
    leg_down_angle = None
    if leg_norm > 1e-9:
        cosine = float(leg_axis[1] / leg_norm)
        leg_down_angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    drops = []
    if shoulder_width > 1e-9:
        for side in ("left", "right"):
            drops.append(
                float(
                    (keypoints[MHR[f"{side}_ankle"], 1] - keypoints[MHR[f"{side}_hip"], 1])
                    / shoulder_width
                )
            )
    mean_vertical_drop = float(np.mean(drops)) if drops else None

    pose = "uncertain"
    if (
        mean_knee is not None
        and leg_down_angle is not None
        and mean_vertical_drop is not None
        and mean_knee >= 110.0
        and leg_down_angle <= 25.0
        and mean_vertical_drop >= 1.25
    ):
        pose = "standing"
    elif (
        mean_knee is not None
        and leg_down_angle is not None
        and mean_vertical_drop is not None
        and mean_knee <= 105.0
        and (leg_down_angle >= 30.0 or mean_vertical_drop <= 1.20)
    ):
        pose = "sitting"

    return pose, {
        "left_knee_angle_deg": _round(left_knee),
        "right_knee_angle_deg": _round(right_knee),
        "mean_knee_angle_deg": _round(mean_knee),
        "leg_axis_from_image_down_deg": _round(leg_down_angle),
        "mean_hip_to_ankle_vertical_drop_shoulder_widths": _round(mean_vertical_drop),
    }


def _crop_support(region_support: dict[str, float], pose: str) -> tuple[float, float, list[str], list[str]]:
    ordered = ["head", "shoulders", "hips", "thighs", "knees", "lower_legs", "feet"]
    crop = float(np.mean([region_support[name] for name in ordered]))
    crop_regions = [name for name in ordered if region_support[name] >= REGION_SUPPORT_THRESHOLD]

    if pose == "standing":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.20, "lower_legs": 0.20, "feet": 0.10}
    elif pose == "sitting":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.25, "lower_legs": 0.15, "feet": 0.10}
    else:
        weights = {name: 1.0 / len(ordered) for name in ordered}
    pose_support = float(sum(region_support[name] * weight for name, weight in weights.items()))
    pose_regions = [name for name in ordered if weights.get(name, 0.0) > 0.0 and region_support[name] >= REGION_SUPPORT_THRESHOLD]
    return crop, pose_support, crop_regions, pose_regions


def _flexion_band(angle_deg: float | None) -> str | None:
    if angle_deg is None:
        return None
    if angle_deg < 60.0:
        return "tightly_flexed"
    if angle_deg < 100.0:
        return "flexed"
    if angle_deg < 145.0:
        return "moderately_flexed"
    return "near_straight"


def _relation_support(joint_support: dict[str, float], anchors: list[str]) -> float:
    return _mean_support(joint_support, anchors)


def _arm_profile(
    keypoints: np.ndarray,
    sam2d: np.ndarray,
    side: str,
    shoulder_width: float,
    joint_support: dict[str, float],
) -> dict[str, Any]:
    hand_indices = LEFT_HAND if side == "left" else RIGHT_HAND
    shoulder = keypoints[MHR[f"{side}_shoulder"]]
    elbow = keypoints[MHR[f"{side}_elbow"]]
    wrist = keypoints[MHR[f"{side}_wrist"]]
    hip = keypoints[MHR[f"{side}_hip"]]
    knee = keypoints[MHR[f"{side}_knee"]]
    hand = _mean_point(keypoints, hand_indices)
    neck = keypoints[MHR["neck"]]
    nose = keypoints[MHR["nose"]]
    elbow_angle = _angle(shoulder, elbow, wrist)

    def normdist(a: np.ndarray, b: np.ndarray) -> float | None:
        return _round(_distance(a, b) / shoulder_width) if shoulder_width > 1e-9 else None

    forearm_angle = None
    if len(sam2d) > MHR[f"{side}_wrist"] and len(sam2d) > MHR[f"{side}_elbow"]:
        e2 = sam2d[MHR[f"{side}_elbow"], :2]
        w2 = sam2d[MHR[f"{side}_wrist"], :2]
        if atlas._finite_xy(e2) and atlas._finite_xy(w2):
            forearm_angle = math.degrees(math.atan2(float(w2[1] - e2[1]), float(w2[0] - e2[0])))

    hand_hip = normdist(hand, hip)
    hand_neck = normdist(hand, neck)
    hand_nose = normdist(hand, nose)
    hand_knee = normdist(hand, knee)
    anchors = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]

    return {
        "elbow_flexion_deg": _round(elbow_angle),
        "elbow_flexion_band": _flexion_band(elbow_angle),
        "hand_to_hip_shoulder_widths": hand_hip,
        "hand_to_neck_shoulder_widths": hand_neck,
        "hand_to_nose_shoulder_widths": hand_nose,
        "hand_to_knee_shoulder_widths": hand_knee,
        "forearm_screen_angle_deg": _round(forearm_angle),
        "anchor_crop_support": _round(_relation_support(joint_support, anchors), 4),
        "geometry_flags": {
            "hand_near_hip": bool(hand_hip is not None and hand_hip <= 0.65),
            "hand_near_face": bool(min(v for v in (hand_neck, hand_nose) if v is not None) <= 0.65) if hand_neck is not None and hand_nose is not None else False,
            "hand_near_knee": bool(hand_knee is not None and hand_knee <= 0.70),
        },
    }


def _relations(arms: dict[str, dict[str, Any]], joint_support: dict[str, float]) -> dict[str, Any]:
    left, right = arms["left"], arms["right"]

    left_hip = bool(left["geometry_flags"]["hand_near_hip"] and (left.get("elbow_flexion_deg") or 180.0) <= 145.0)
    right_hip = bool(right["geometry_flags"]["hand_near_hip"] and (right.get("elbow_flexion_deg") or 180.0) <= 145.0)
    hands_on_hips = left_hip and right_hip
    hands_on_hips_support = _relation_support(
        joint_support,
        ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "right_shoulder", "right_elbow", "right_wrist", "right_hip"],
    )

    head_candidates = []
    for side in ("left", "right"):
        arm = arms[side]
        face_distance = min(
            v for v in (arm.get("hand_to_neck_shoulder_widths"), arm.get("hand_to_nose_shoulder_widths"))
            if v is not None
        )
        angle = arm.get("elbow_flexion_deg")
        matched = face_distance <= 0.45 and angle is not None and float(angle) <= 85.0
        support = _relation_support(
            joint_support,
            [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", "neck", "nose"],
        )
        head_candidates.append((matched, support, side, face_distance))
    head_candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    head_match, head_support, head_side, head_distance = head_candidates[0]

    return {
        "hands_on_hips": {
            "geometry_match": hands_on_hips,
            "crop_support": _round(hands_on_hips_support, 4),
            "crop_support_percent": int(round(100.0 * hands_on_hips_support)),
            "evidence": {
                "left_hand_near_hip": left_hip,
                "right_hand_near_hip": right_hip,
            },
        },
        "head_supported_by_hand": {
            "geometry_match": bool(head_match),
            "side": head_side if head_match else None,
            "crop_support": _round(head_support, 4),
            "crop_support_percent": int(round(100.0 * head_support)),
            "hand_to_face_shoulder_widths": _round(head_distance),
        },
    }


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    if keypoints.ndim != 2 or keypoints.shape[0] < 70 or keypoints.shape[1] < 3:
        raise ValueError("pred_keypoints_3d must contain at least 70 MHR keypoints")
    keypoints = keypoints[:, :3]

    sam2d = atlas._sam2d_points(arrays)
    if dwpose:
        dw_points, accepted_names, in_frame_names = atlas._dwpose_target_points(dwpose, width, height)
        residual = atlas._reprojection_residual(
            dw_points,
            sam2d,
            accepted_names,
            in_frame_names,
            width=width,
            height=height,
            dwpose=dwpose,
        ) if len(dw_points) and len(sam2d) else atlas._empty_residual()
    else:
        dw_points = np.empty((0, 2), dtype=np.float64)
        accepted_names, in_frame_names = set(), set()
        residual = atlas._empty_residual()

    joint_support = _joint_support(dw_points, in_frame_names, residual, sam2d, width, height)
    region_support = _region_support(joint_support)
    shoulder_width = _distance(keypoints[MHR["left_shoulder"]], keypoints[MHR["right_shoulder"]])
    pose, pose_geometry = _projected_pose(keypoints, shoulder_width)
    crop_support, pose_support, crop_regions, pose_regions = _crop_support(region_support, pose)

    arms = {
        side: _arm_profile(keypoints, sam2d, side, shoulder_width, joint_support)
        for side in ("left", "right")
    }
    relations = _relations(arms, joint_support)

    return {
        "schema_version": "sam3d-relational-pose-profile-0.1",
        "sam3d_projected_pose": {
            "pose": pose,
            "source": "sam3d_reconstruction",
            "crop_support": _round(crop_support, 4),
            "crop_support_percent": int(round(100.0 * crop_support)),
            "pose_support": _round(pose_support, 4),
            "pose_support_percent": int(round(100.0 * pose_support)),
            "crop_supported_regions": crop_regions,
            "pose_support_regions": pose_regions,
            "region_support": {name: _round(value, 4) for name, value in region_support.items()},
            "geometry": pose_geometry,
        },
        "arm_geometry": arms,
        "relations": relations,
        "evidence_support": {
            "dwpose_accepted_joint_names": sorted(accepted_names),
            "dwpose_in_frame_accepted_joint_names": sorted(in_frame_names),
            "joint_crop_support": {name: _round(value, 4) for name, value in joint_support.items()},
            "projected_fit_residual": residual,
        },
        "policy": {
            "report_only": True,
            "projected_pose_is_reconstruction_not_direct_observation": True,
            "crop_support_is_not_model_confidence": True,
            "crop_support_formula": "DWPose in-frame acceptance × DWPose↔SAM3D agreement × distance from crop edge",
            "relation_geometry_may_extend_beyond_observed_crop": True,
            "scene_contact_requires_later_cross_modal_fusion": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile",
        description="Build report-only SAM3D projected-pose and body-relation profiles from cached SAM3D/DWPose evidence.",
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
    output = (args.output or (sam3d_dir / "relational-pose-profile")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    rows = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        dwpose = _read_json(dwpose_path)

        width = int(dwpose.get("image_width") or 0)
        height = int(dwpose.get("image_height") or 0)
        if width <= 0 or height <= 0:
            image_matches = []
            if images_dir.is_dir():
                image_matches = [p for p in images_dir.rglob(f"{key}.*") if p.is_file()]
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
        _write_json(out_path, record)
        rows.append(record)

        projected = profile["sam3d_projected_pose"]
        relations = profile["relations"]
        flags = []
        if relations["hands_on_hips"]["geometry_match"]:
            flags.append("hands_on_hips")
        if relations["head_supported_by_hand"]["geometry_match"]:
            flags.append(f"head_supported_by_{relations['head_supported_by_hand']['side']}_hand")
        print(
            f"{key}: projected={projected['pose']} crop={projected['crop_support_percent']}% "
            f"pose_support={projected['pose_support_percent']}% regions={','.join(projected['crop_supported_regions']) or '-'} "
            f"relations={','.join(flags) or '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.1",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    _write_json(output / "sam3d_relational_pose.index.json", index)
    print(f"Index: {output / 'sam3d_relational_pose.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
