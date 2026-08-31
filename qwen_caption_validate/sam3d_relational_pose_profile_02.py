from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import pose_atlas_v3 as base
from . import pose_atlas_v3_03 as atlas
from . import sam3d_relational_pose_profile as v01
from . import sam3d_hand_geometry as hand


MHR = atlas.MHR_BODY
REGION_SUPPORT_THRESHOLD = 0.35
EDGE_FULL_SUPPORT_FRACTION = 0.05
EDGE_MIN_IN_FRAME_SUPPORT = 0.65
HAND_CONFIDENCE_THRESHOLD = 0.30


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


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _edge_factor(point: np.ndarray, width: int, height: int) -> float:
    """Apply only a mild uncertainty penalty to accepted in-frame landmarks.

    v0.1 drove support to zero exactly at the crop boundary.  That made a very
    well-aligned knee at y~=0.98 almost worthless.  Here an accepted point that
    remains inside the source frame keeps at least 65% edge authority and ramps
    to full support by 5% inside the image.  Outside-frame extrapolations still
    receive zero direct crop support.
    """
    if not atlas._finite_xy(point):
        return 0.0
    x = float(point[0]) / float(width)
    y = float(point[1]) / float(height)
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        return 0.0
    margin = min(x, 1.0 - x, y, 1.0 - y)
    ramp = max(0.0, min(1.0, margin / EDGE_FULL_SUPPORT_FRACTION))
    return EDGE_MIN_IN_FRAME_SUPPORT + (1.0 - EDGE_MIN_IN_FRAME_SUPPORT) * ramp


def _agreement_factor(residual_px: float, scale_px: float, fraction: float = 0.35) -> float:
    if scale_px <= 1e-9:
        return 0.0
    x = float(residual_px) / (fraction * scale_px)
    return 1.0 / (1.0 + x * x)


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
        agreement = _agreement_factor(float(per_joint[name]), shoulder_span, 0.35)
        edge = _edge_factor(dw_points[idx], width, height)
        out[name] = float(agreement * edge)
    return out


def _mean_support(joint_support: dict[str, float], names: list[str]) -> float:
    return float(np.mean([joint_support.get(name, 0.0) for name in names])) if names else 0.0


def _region_support(joint_support: dict[str, float]) -> dict[str, float]:
    regions = {
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck"],
        "shoulders": ["left_shoulder", "right_shoulder"],
        "hips": ["left_hip", "right_hip"],
        "thighs": ["left_hip", "right_hip", "left_knee", "right_knee"],
        "knees": ["left_knee", "right_knee"],
        "lower_legs": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
        "feet": ["left_ankle", "right_ankle"],
    }
    return {name: _mean_support(joint_support, joints) for name, joints in regions.items()}


def _crop_support(region_support: dict[str, float], pose: str) -> tuple[float, float, list[str], list[str]]:
    ordered = ["head", "shoulders", "hips", "thighs", "knees", "lower_legs", "feet"]
    coverage = float(np.mean([region_support[name] for name in ordered]))
    coverage_regions = [name for name in ordered if region_support[name] >= REGION_SUPPORT_THRESHOLD]

    if pose == "standing":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.20, "lower_legs": 0.20, "feet": 0.10}
    elif pose == "sitting":
        weights = {"head": 0.00, "shoulders": 0.05, "hips": 0.20, "thighs": 0.25, "knees": 0.25, "lower_legs": 0.15, "feet": 0.10}
    else:
        weights = {name: 1.0 / len(ordered) for name in ordered}

    pose_support = float(sum(region_support[name] * weight for name, weight in weights.items()))
    pose_regions = [
        name for name in ordered
        if weights.get(name, 0.0) > 0.0 and region_support[name] >= REGION_SUPPORT_THRESHOLD
    ]
    return coverage, pose_support, coverage_regions, pose_regions


def _reconstruction_match(pose: str, geometry: dict[str, Any]) -> float:
    knee = geometry.get("mean_knee_angle_deg")
    leg_angle = geometry.get("leg_axis_from_image_down_deg")
    drop = geometry.get("mean_hip_to_ankle_vertical_drop_shoulder_widths")
    if knee is None or leg_angle is None or drop is None:
        return 0.0

    knee = float(knee)
    leg_angle = float(leg_angle)
    drop = float(drop)
    if pose == "standing":
        knee_score = _ramp(knee, 105.0, 160.0)
        vertical_score = 1.0 - _ramp(leg_angle, 8.0, 40.0)
        drop_score = _ramp(drop, 0.90, 1.80)
        return 0.45 * knee_score + 0.25 * vertical_score + 0.30 * drop_score
    if pose == "sitting":
        knee_score = 1.0 - _ramp(knee, 80.0, 120.0)
        nonvertical_score = _ramp(leg_angle, 20.0, 50.0)
        compact_drop_score = 1.0 - _ramp(drop, 0.70, 1.40)
        return 0.45 * knee_score + 0.25 * nonvertical_score + 0.30 * compact_drop_score
    return 0.0


def _hand_shape_label(shape: dict[str, Any]) -> str:
    closed = shape.get("closed_fist_score")
    opened = shape.get("open_hand_score")
    if closed is not None and float(closed) >= 0.70:
        return "closed_fist"
    if opened is not None and float(opened) >= 0.70:
        return "open_hand"
    if closed is None and opened is None:
        return "unavailable"
    return "mixed_or_uncertain"


def _hand_evidence(
    side: str,
    decoded: dict[str, np.ndarray],
    sam2d: np.ndarray,
    keypoints3d: np.ndarray,
    shoulder_span_px: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    points = np.asarray(decoded.get("points", np.empty((0, 2))), dtype=np.float64)
    scores = np.asarray(decoded.get("scores", np.empty((0,))), dtype=np.float64)
    accepted = np.asarray(decoded.get("accepted_mask", np.empty((0,), dtype=bool)), dtype=bool)
    mapping = hand.mhr_hand_order(side)

    per_point_support: list[float] = []
    residuals: list[float] = []
    in_frame_count = 0
    for index in range(21):
        support = 0.0
        if index < len(points) and index < len(scores) and index < len(accepted) and bool(accepted[index]):
            p = points[index]
            in_frame = atlas._finite_xy(p) and 0.0 <= float(p[0]) <= width - 1 and 0.0 <= float(p[1]) <= height - 1
            mi = mapping[index]
            if in_frame and mi < len(sam2d) and atlas._finite_xy(sam2d[mi]):
                in_frame_count += 1
                residual_px = _distance(p[:2], sam2d[mi, :2])
                residuals.append(residual_px)
                agreement = _agreement_factor(residual_px, shoulder_span_px, 0.25)
                confidence = max(0.0, min(1.0, float(scores[index])))
                support = confidence * agreement * _edge_factor(p, width, height)
        per_point_support.append(float(support))

    accepted_count = int(np.count_nonzero(accepted[:21])) if len(accepted) else 0
    mean_confidence = (
        float(np.mean(scores[:21][accepted[:21]]))
        if len(scores) >= 21 and len(accepted) >= 21 and accepted_count
        else None
    )
    crop_support = float(np.mean(per_point_support)) if per_point_support else 0.0

    dw_shape = hand.summarize_finger_shape(hand.dwpose_finger_extension_ratios(points)) if len(points) >= 21 else hand.summarize_finger_shape({finger: None for finger in hand.FINGERS})
    sam_shape = hand.summarize_finger_shape(hand.sam3d_finger_extension_ratios(keypoints3d, side))

    dw_ratio = dw_shape.get("mean_non_thumb_extension_ratio")
    sam_ratio = sam_shape.get("mean_non_thumb_extension_ratio")
    shape_agreement = None
    if dw_ratio is not None and sam_ratio is not None:
        shape_agreement = max(0.0, 1.0 - min(1.0, abs(float(dw_ratio) - float(sam_ratio)) / 0.35))

    use_observed = accepted_count >= 12 and crop_support >= 0.25
    preferred = dw_shape if use_observed else sam_shape
    preferred_source = "dwpose_observed" if use_observed else "sam3d_reconstruction"

    return {
        "dwpose_hand": {
            "accepted_landmark_count": accepted_count,
            "in_frame_accepted_landmark_count": in_frame_count,
            "mean_confidence": _round(mean_confidence, 4),
            "median_sam3d_residual_px": _round(float(np.median(residuals)) if residuals else None, 2),
            "crop_support": _round(crop_support, 4),
            "crop_support_percent": int(round(100.0 * crop_support)),
            "shape": dw_shape,
            "shape_label": _hand_shape_label(dw_shape),
        },
        "sam3d_hand": {
            "shape": sam_shape,
            "shape_label": _hand_shape_label(sam_shape),
        },
        "cross_model_shape_agreement": _round(shape_agreement, 4),
        "preferred_shape_source": preferred_source,
        "preferred_shape": preferred,
        "preferred_shape_label": _hand_shape_label(preferred),
        "per_landmark_crop_support": [round(float(v), 4) for v in per_point_support],
    }


def _relation_support(joint_support: dict[str, float], anchors: list[str]) -> float:
    return _mean_support(joint_support, anchors)


def _enhance_relations(
    relations: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    hand_geometry: dict[str, dict[str, Any]],
    joint_support: dict[str, float],
    sam2d: np.ndarray,
    shoulder_span_px: float,
) -> dict[str, Any]:
    out = json.loads(json.dumps(relations))

    head = out.get("head_supported_by_hand") or {}
    head_side = head.get("side") if head.get("geometry_match") else None
    head_fist_match = False
    if head_side in hand_geometry:
        h = hand_geometry[head_side]
        preferred = h.get("preferred_shape") or {}
        closed = preferred.get("closed_fist_score")
        observed = h.get("preferred_shape_source") == "dwpose_observed"
        head_fist_match = bool(observed and closed is not None and float(closed) >= 0.70)
        head["hand_shape_source"] = h.get("preferred_shape_source")
        head["hand_shape_label"] = h.get("preferred_shape_label")
        head["closed_fist_score"] = closed
        head["open_hand_score"] = preferred.get("open_hand_score")
    out["head_supported_by_hand"] = head
    out["head_supported_by_fist"] = {
        "geometry_match": bool(head.get("geometry_match") and head_fist_match),
        "side": head_side if head.get("geometry_match") and head_fist_match else None,
        "crop_support": head.get("crop_support") if head.get("geometry_match") and head_fist_match else 0.0,
        "crop_support_percent": head.get("crop_support_percent") if head.get("geometry_match") and head_fist_match else 0,
        "requires_scene_semantics": False,
    }

    raised: dict[str, Any] = {}
    for side in ("left", "right"):
        shoulder_idx = MHR[f"{side}_shoulder"]
        wrist_idx = MHR[f"{side}_wrist"]
        above = None
        if shoulder_idx < len(sam2d) and wrist_idx < len(sam2d) and shoulder_span_px > 1e-9:
            above = float((sam2d[shoulder_idx, 1] - sam2d[wrist_idx, 1]) / shoulder_span_px)
        h = hand_geometry[side]
        preferred = h.get("preferred_shape") or {}
        open_score = preferred.get("open_hand_score")
        observed_shape = h.get("preferred_shape_source") == "dwpose_observed"
        geometry_match = bool(
            observed_shape
            and above is not None
            and above >= 0.15
            and open_score is not None
            and float(open_score) >= 0.65
        )
        body_support = _relation_support(joint_support, [f"{side}_shoulder", f"{side}_wrist"])
        hand_support = float((h.get("dwpose_hand") or {}).get("crop_support") or 0.0)
        support = 0.5 * body_support + 0.5 * hand_support
        raised[side] = {
            "geometry_match": geometry_match,
            "wrist_above_shoulder_shoulder_widths": _round(above),
            "open_hand_score": open_score,
            "hand_shape_source": h.get("preferred_shape_source"),
            "crop_support": _round(support, 4),
            "crop_support_percent": int(round(100.0 * support)),
        }

    matches = [side for side, value in raised.items() if value["geometry_match"]]
    best_side = max(matches, key=lambda side: float(raised[side]["crop_support"] or 0.0)) if matches else None
    out["raised_open_hand"] = {
        "geometry_match": bool(best_side),
        "side": best_side,
        "per_side": raised,
    }
    out["waving_candidate"] = {
        "geometry_match": bool(best_side),
        "side": best_side,
        "crop_support": raised[best_side]["crop_support"] if best_side else 0.0,
        "crop_support_percent": raised[best_side]["crop_support_percent"] if best_side else 0,
        "requires_vlm_confirmation": True,
        "interpretation": "Raised open-hand geometry is compatible with waving, but a still-image geometry source does not establish the action by itself.",
    }
    return out


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
    shoulder_width_3d = _distance(keypoints[MHR["left_shoulder"]], keypoints[MHR["right_shoulder"]])
    shoulder_span_px = _distance(sam2d[MHR["left_shoulder"], :2], sam2d[MHR["right_shoulder"], :2]) if len(sam2d) > max(MHR["left_shoulder"], MHR["right_shoulder"]) else 0.0

    pose, pose_geometry = v01._projected_pose(keypoints, shoulder_width_3d)
    reconstruction_match = _reconstruction_match(pose, pose_geometry)
    coverage, pose_crop_support, coverage_regions, pose_regions = _crop_support(region_support, pose)

    arms = {
        side: v01._arm_profile(keypoints, sam2d, side, shoulder_width_3d, joint_support)
        for side in ("left", "right")
    }
    base_relations = v01._relations(arms, joint_support)

    decoded_hands = hand.decode_dwpose_target_hands(dwpose, width, height)
    hand_geometry = {
        side: _hand_evidence(
            side,
            decoded_hands[side],
            sam2d,
            keypoints,
            shoulder_span_px,
            width,
            height,
        )
        for side in ("left", "right")
    }
    relations = _enhance_relations(
        base_relations,
        arms,
        hand_geometry,
        joint_support,
        sam2d,
        shoulder_span_px,
    )

    return {
        "schema_version": "sam3d-relational-pose-profile-0.2",
        "sam3d_projected_pose": {
            "pose": pose,
            "source": "sam3d_reconstruction",
            "reconstruction_match": _round(reconstruction_match, 4),
            "reconstruction_match_percent": int(round(100.0 * reconstruction_match)),
            "crop_coverage": _round(coverage, 4),
            "crop_coverage_percent": int(round(100.0 * coverage)),
            "crop_support": _round(pose_crop_support, 4),
            "crop_support_percent": int(round(100.0 * pose_crop_support)),
            # Compatibility aliases from v0.1; both now explicitly mean the
            # pose-specific crop support rather than reconstruction confidence.
            "pose_support": _round(pose_crop_support, 4),
            "pose_support_percent": int(round(100.0 * pose_crop_support)),
            "crop_supported_regions": coverage_regions,
            "pose_support_regions": pose_regions,
            "region_support": {name: _round(value, 4) for name, value in region_support.items()},
            "geometry": pose_geometry,
        },
        "arm_geometry": arms,
        "hand_geometry": hand_geometry,
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
            "reconstruction_match_is_geometry_template_match_not_crop_support": True,
            "crop_support_is_not_model_confidence": True,
            "crop_support_formula": "DWPose in-frame acceptance × DWPose↔SAM3D agreement × mild crop-edge penalty",
            "accepted_in_frame_landmarks_keep_at_least_65_percent_edge_authority": True,
            "dwpose_hand_shape_preferred_when_adequately_observed": True,
            "sam3d_hand_shape_retained_as_reconstruction_even_when_it_disagrees": True,
            "relation_geometry_may_extend_beyond_observed_crop": True,
            "waving_requires_vlm_confirmation": True,
            "scene_contact_requires_later_cross_modal_fusion": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-02",
        description="Build report-only SAM3D relational pose profiles with explicit reconstruction-vs-crop support and hand/finger evidence.",
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.2")).expanduser().resolve()
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
        for name in ("hands_on_hips", "head_supported_by_hand", "head_supported_by_fist", "raised_open_hand", "waving_candidate"):
            value = relations.get(name) or {}
            if value.get("geometry_match"):
                label = name
                if value.get("side"):
                    label += f":{value['side']}"
                flags.append(label)
        hand_labels = ",".join(
            f"{side}:{profile['hand_geometry'][side]['preferred_shape_label']}@{profile['hand_geometry'][side]['dwpose_hand']['crop_support_percent']}%"
            for side in ("left", "right")
        )
        print(
            f"{key}: projected={projected['pose']} recon={projected['reconstruction_match_percent']}% "
            f"crop={projected['crop_support_percent']}% coverage={projected['crop_coverage_percent']}% "
            f"regions={','.join(projected['crop_supported_regions']) or '-'} "
            f"hands={hand_labels} relations={','.join(flags) or '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.2",
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
