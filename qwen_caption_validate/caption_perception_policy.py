from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "caption-perception-policy-0.1"
MODES = ("framing_only", "configuration", "pose_allowed", "pose_guided")
BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
MHR70 = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_hip": 9, "right_hip": 10,
    "left_knee": 11, "right_knee": 12, "left_ankle": 13, "right_ankle": 14,
    "neck": 69,
}


def _point(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return None
    return float(arr[0]), float(arr[1])


def _inside(p: tuple[float, float] | None, width: int, height: int) -> bool:
    return bool(p and 0 <= p[0] <= width - 1 and 0 <= p[1] <= height - 1)


def _dwpose_points(record: dict[str, Any], width: int, height: int) -> dict[str, tuple[float, float] | None]:
    from .dwpose_compat import target_points_from_profile_record

    raw = target_points_from_profile_record(record, width, height)
    out = {name: None for name in BODY18}
    for i, name in enumerate(BODY18):
        p = _point(raw[i]) if i < len(raw) else None
        if _inside(p, width, height):
            out[name] = p
    return out


def _to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)[..., :2].copy()
    valid = np.isfinite(arr).all(axis=-1) if arr.size else np.asarray([], dtype=bool)
    if not np.any(valid):
        return arr
    observed = arr[valid]
    lo, hi = float(observed.min()), float(observed.max())
    if lo >= -0.05 and hi <= 1.25:
        arr[..., 0] *= width
        arr[..., 1] *= height
    elif lo >= -1.25 and hi <= 1.25:
        arr[..., 0] = (arr[..., 0] + 1) * 0.5 * width
        arr[..., 1] = (arr[..., 1] + 1) * 0.5 * height
    return arr


def _sam3d_summary(arrays: Mapping[str, np.ndarray] | None, width: int, height: int) -> dict[str, Any]:
    out = {
        "available": False,
        "projected_selected_joint_count": 0,
        "projected_inside_frame_count": 0,
        "projected_outside_frame_count": 0,
        "projected_outside_frame_fraction": None,
        "reported_confidence": None,
        "reconstruction_is_observation": False,
        "can_promote_observability": False,
    }
    if not arrays:
        return out
    raw = np.asarray(arrays.get("pred_keypoints_2d", np.empty((0, 2))), dtype=np.float64)
    if raw.size:
        if raw.ndim > 2:
            raw = raw.reshape(-1, raw.shape[-1])
        projected = _to_pixels(raw[:, :2], width, height)
        selected = [_point(projected[i]) for i in MHR70.values() if i < len(projected)]
        selected = [p for p in selected if p is not None]
        inside = sum(_inside(p, width, height) for p in selected)
        outside = len(selected) - inside
        out.update(
            available=True,
            projected_selected_joint_count=len(selected),
            projected_inside_frame_count=inside,
            projected_outside_frame_count=outside,
            projected_outside_frame_fraction=round(outside / len(selected), 4) if selected else None,
        )
    for key in ("confidence", "pred_confidence", "score", "pred_score"):
        if key in arrays:
            values = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size:
                out["reported_confidence"] = round(float(values.mean()), 6)
                out["available"] = True
                break
    return out


def _mid(a: tuple[float, float] | None, b: tuple[float, float] | None):
    return None if a is None or b is None else ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _line_angle(a, b):
    if a is None or b is None:
        return None
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _vertical_angle(a, b):
    if a is None or b is None:
        return None
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))


def _joint_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    u = np.asarray((a[0] - b[0], a[1] - b[1]))
    v = np.asarray((c[0] - b[0], c[1] - b[1]))
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 1e-8:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / denom, -1, 1))))


def _bbox(points, width, height):
    values = [p for p in points.values() if p is not None]
    if not values:
        return None
    xs, ys = [p[0] for p in values], [p[1] for p in values]
    return {
        "width_fraction": round((max(xs) - min(xs)) / width, 4),
        "height_fraction": round((max(ys) - min(ys)) / height, 4),
    }


def _visibility(points, width, height):
    visible = {name for name, p in points.items() if p is not None}
    count = lambda names: sum(name in visible for name in names)
    heads = count(("nose", "neck", "left_eye", "right_eye", "left_ear", "right_ear"))
    shoulders = count(("left_shoulder", "right_shoulder"))
    hips = count(("left_hip", "right_hip"))
    knees = count(("left_knee", "right_knee"))
    ankles = count(("left_ankle", "right_ankle"))
    state = lambda n, strong=2: "strong" if n >= strong else ("partial" if n else "absent")
    broad = (hips == 2 and knees >= 1) or any(
        all(points.get(f"{side}_{part}") is not None for part in ("hip", "knee", "ankle"))
        for side in ("left", "right")
    )
    extent = "full_length" if ankles else "three_quarter_or_long" if knees else "waist_or_upper_body" if hips else "close_or_medium_close" if shoulders else "face_or_partial_body"
    return {
        "head": state(heads), "shoulders": state(shoulders),
        "torso": "strong" if shoulders == hips == 2 else ("partial" if shoulders and (hips or shoulders == 2) else "absent"),
        "hips": state(hips), "knees": state(knees), "feet": state(ankles),
        "observed_landmarks": [name for name in BODY18 if name in visible],
        "observed_bbox": _bbox(points, width, height), "extent_hint": extent,
        "broad_pose_supported": broad,
    }


def _geometry(points, width, height):
    ls, rs = points.get("left_shoulder"), points.get("right_shoulder")
    lh, rh = points.get("left_hip"), points.get("right_hip")
    shoulder_angle = _line_angle(rs, ls)
    torso_tilt = _vertical_angle(points.get("neck"), _mid(lh, rh))
    shoulder_mid, head_offset = _mid(ls, rs), None
    if ls and rs and shoulder_mid:
        sw = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
        head = points.get("nose") or points.get("neck")
        if head and sw > 1e-6:
            head_offset = abs(head[0] - shoulder_mid[0]) / sw

    elbow, knee = {}, {}
    for side in ("left", "right"):
        elbow[side] = _joint_angle(points.get(f"{side}_shoulder"), points.get(f"{side}_elbow"), points.get(f"{side}_wrist"))
        knee[side] = _joint_angle(points.get(f"{side}_hip"), points.get(f"{side}_knee"), points.get(f"{side}_ankle"))

    config, config_cues = 0, []
    if shoulder_angle is not None and abs(shoulder_angle) >= 12:
        config += 1; config_cues.append("shoulder_line_tilt")
    if head_offset is not None and head_offset >= 0.28:
        config += 1; config_cues.append("head_offset_from_shoulders")
    if any(sum(points.get(f"{side}_{j}") is not None for j in ("shoulder", "elbow", "wrist")) >= 2 for side in ("left", "right")):
        config += 1; config_cues.append("visible_arm_relationship")
    if any(points.get(f"{s}_hip") is not None for s in ("left", "right")) and not any(points.get(f"{s}_knee") is not None for s in ("left", "right")):
        config += 1; config_cues.append("visible_torso_without_leg_support")

    complexity, complexity_cues = 0, []
    if torso_tilt is not None and abs(torso_tilt) >= 25:
        complexity += 2; complexity_cues.append("strong_torso_tilt")
    elif torso_tilt is not None and abs(torso_tilt) >= 15:
        complexity += 1; complexity_cues.append("moderate_torso_tilt")
    if shoulder_angle is not None and abs(shoulder_angle) >= 18:
        complexity += 1; complexity_cues.append("strong_shoulder_tilt")
    for side, angle in knee.items():
        if angle is not None and angle <= 135:
            complexity += 2; complexity_cues.append(f"deep_{side}_knee_flexion")
        elif angle is not None and angle <= 155:
            complexity += 1; complexity_cues.append(f"moderate_{side}_knee_flexion")
    if any(angle is not None and angle <= 110 for angle in elbow.values()):
        complexity += 1; complexity_cues.append("bent_arm")
    box = _bbox(points, width, height)
    aspect = box["width_fraction"] / box["height_fraction"] if box and box["height_fraction"] > 1e-6 else None
    if aspect is not None and aspect >= 0.8:
        complexity += 2; complexity_cues.append("strongly_nonvertical_body_extent")
    elif aspect is not None and aspect >= 0.6:
        complexity += 1; complexity_cues.append("broad_body_extent")

    return {
        "shoulder_line_angle_deg": round(shoulder_angle, 2) if shoulder_angle is not None else None,
        "torso_axis_tilt_from_vertical_deg": round(torso_tilt, 2) if torso_tilt is not None else None,
        "head_offset_shoulder_widths": round(head_offset, 3) if head_offset is not None else None,
        "configuration_score": config, "configuration_cues": config_cues,
        "pose_complexity_score": complexity, "pose_complexity_cues": complexity_cues,
    }


def route_policy(points, *, width: int, height: int, sam3d: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """DWPose crop visibility owns eligibility; SAM3D can only guide an already-supported pose."""
    visibility, geometry = _visibility(points, width, height), _geometry(points, width, height)
    sam = dict(sam3d or {})
    sam.setdefault("available", False)
    sam["reconstruction_is_observation"] = False
    sam["can_promote_observability"] = False
    broad = bool(visibility["broad_pose_supported"])
    guidance_usable = bool(sam["available"] and int(sam.get("projected_selected_joint_count") or 0) >= 8)
    reasons = []
    if broad:
        reasons.append("Observed DWPose crop evidence includes hips plus leg landmarks, so broad-pose wording is supportable.")
        if geometry["pose_complexity_score"] >= 2 and guidance_usable:
            mode, relevance = "pose_guided", "high"
            reasons.append("Observed 2D geometry is non-routine enough to benefit from SAM3D relational guidance.")
        else:
            mode, relevance = "pose_allowed", "medium"
            reasons.append("Broad pose is visible, but geometry-heavy SAM3D guidance is not required or not sufficiently supported.")
    else:
        reasons.append("Observed crop evidence does not expose enough lower-body structure to support a broad posture claim.")
        if geometry["configuration_score"] >= 2:
            mode, relevance = "configuration", "low"
            reasons.append("Visible local body relationships are informative; describe configuration without naming a broad posture.")
        else:
            mode, relevance = "framing_only", "negligible"
            reasons.append("Local body geometry is not distinctive enough to justify pose language; framing/head semantics should dominate.")
        if sam["available"]:
            reasons.append("SAM3D reconstruction stayed diagnostic and did not promote hidden/out-of-crop anatomy into observed evidence.")
    return {"visibility": visibility, "geometry": geometry, "sam3d": sam, "pose_relevance": relevance, "policy": {"mode": mode}, "reasons": reasons}


def policy_from_artifacts(dwpose_record: dict[str, Any], *, width: int, height: int, sam3d_arrays=None):
    return route_policy(
        _dwpose_points(dwpose_record, width, height), width=width, height=height,
        sam3d=_sam3d_summary(sam3d_arrays, width, height),
    )


def _find(directory: Path | None, key: str, suffix: str) -> Path | None:
    if directory is None:
        return None
    direct = directory / f"{key}{suffix}"
    if direct.is_file():
        return direct
    matches = sorted(directory.rglob(f"{key}{suffix}"))
    return matches[0] if matches else None


def main() -> int:
    from PIL import Image
    from . import pose_atlas_v3 as atlas

    parser = argparse.ArgumentParser(description="Phase-1 deterministic caption perception router.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--images-dir", type=Path); parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--sam3d-dir", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--only", action="append", default=[]); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else run_dir / "images"
    dwpose_dir = atlas._resolve_dir(run_dir, args.dwpose_dir, ["dwpose", "dwpose-v1"], "dwpose")
    sam3d_dir = atlas._resolve_dir(run_dir, args.sam3d_dir, ["sam3d", "sam3d-probe"], "sam3d")
    if not images_dir.is_dir() or dwpose_dir is None:
        raise SystemExit("Need a valid images directory and cached DWPose directory.")
    output = args.output.expanduser().resolve() if args.output else run_dir / "semantic-v3" / "caption-perception-policy-v0.1"
    output.mkdir(parents=True, exist_ok=True)
    images = [p for p in atlas._discover_images(images_dir) if atlas._matches_only(p.stem, p, args.only)]
    records, missing = [], []
    for image_path in images:
        key, out_path = image_path.stem, output / f"{image_path.stem}.perception_policy.json"
        if out_path.is_file() and not args.overwrite:
            records.append(atlas._read_json(out_path)); continue
        dw_path, sam_path = _find(dwpose_dir, key, ".dwpose.json"), _find(sam3d_dir, key, ".sam3d_arrays.npz")
        if dw_path is None:
            missing.append({"image_key": key, "reason": "missing_dwpose"}); continue
        with Image.open(image_path) as image:
            width, height = image.size
        arrays = None
        if sam_path:
            with np.load(sam_path, allow_pickle=False) as loaded:
                arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        record = {
            "schema_version": SCHEMA_VERSION, "image_key": key, "image": str(image_path), "image_size": [width, height],
            "sources": {"dwpose": str(dw_path), "sam3d_arrays": str(sam_path) if sam_path else None},
            **policy_from_artifacts(atlas._read_json(dw_path), width=width, height=height, sam3d_arrays=arrays),
        }
        atlas._write_json(out_path, record); records.append(record)
        print(f"{key}: {record['policy']['mode']} ({record['pose_relevance']})")
    counts = {mode: sum((r.get("policy") or {}).get("mode") == mode for r in records) for mode in MODES}
    index = {
        "schema_version": SCHEMA_VERSION + "-run", "record_count": len(records), "mode_counts": counts, "missing": missing,
        "invariants": {"sam3d_reconstruction_is_observation": False, "sam3d_can_promote_observability": False, "broad_pose_requires_dwpose_crop_support": True},
        "records": [{"image_key": r.get("image_key"), "mode": (r.get("policy") or {}).get("mode"), "pose_relevance": r.get("pose_relevance"), "visibility": r.get("visibility"), "reasons": r.get("reasons")} for r in records],
    }
    atlas._write_json(output / "caption_perception_policy.index.json", index)
    print(f"Index: {output / 'caption_perception_policy.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
