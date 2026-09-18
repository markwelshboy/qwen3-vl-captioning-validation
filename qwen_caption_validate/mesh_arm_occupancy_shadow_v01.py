from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCHEMA_VERSION = "mesh-arm-occupancy-shadow-0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "mesh-arm-occupancy-shadow-v0.1"

MHR = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "right_wrist": 41,
    "left_wrist": 62,
    "neck": 69,
}

BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]

OCCUPANCY_LARGE = 0.08
OCCUPANCY_MEDIUM = 0.04
OCCUPANCY_SMALL = 0.015
RASTER_LONG_EDGE = 512


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            if len(parts) < 3:
                continue
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            for i in range(1, len(idx) - 1):
                faces.append([idx[0], idx[i], idx[i + 1]])
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )


def _to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)[..., :2].copy()
    if arr.size == 0:
        return arr
    valid = np.isfinite(arr).all(axis=-1)
    if not np.any(valid):
        return arr
    observed = arr[valid]
    lo, hi = float(observed.min()), float(observed.max())
    if lo >= -0.05 and hi <= 1.25:
        arr[..., 0] *= width
        arr[..., 1] *= height
    elif lo >= -1.25 and hi <= 1.25:
        arr[..., 0] = (arr[..., 0] + 1.0) * 0.5 * width
        arr[..., 1] = (arr[..., 1] + 1.0) * 0.5 * height
    return arr


def _recover_focal_length(
    keypoints3d: np.ndarray,
    keypoints2d: np.ndarray,
    cam_t: np.ndarray,
    width: int,
    height: int,
) -> tuple[float | None, dict[str, Any]]:
    k3 = np.asarray(keypoints3d, dtype=np.float64)
    k2 = _to_pixels(np.asarray(keypoints2d, dtype=np.float64), width, height)
    t = np.asarray(cam_t, dtype=np.float64).reshape(-1)
    if k3.ndim != 2 or k3.shape[1] < 3 or k2.ndim != 2 or k2.shape[1] < 2 or t.size < 3:
        return None, {"status": "unavailable", "reason": "camera_projection_inputs_unavailable"}

    cam = k3[:, :3] + t[:3]
    estimates: list[float] = []
    for i in range(min(len(cam), len(k2))):
        x, y, z = (float(v) for v in cam[i])
        u, v = (float(q) for q in k2[i, :2])
        if not all(math.isfinite(q) for q in (x, y, z, u, v)) or z <= 1e-6:
            continue
        if abs(x) > 1e-6:
            f = (u - width / 2.0) * z / x
            if math.isfinite(f) and f > 0:
                estimates.append(f)
        if abs(y) > 1e-6:
            f = (v - height / 2.0) * z / y
            if math.isfinite(f) and f > 0:
                estimates.append(f)

    if not estimates:
        return None, {"status": "unavailable", "reason": "could_not_recover_positive_focal_length"}

    values = np.asarray(estimates, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, {
        "status": "available",
        "estimate_count": int(len(values)),
        "median_px": round(median, 4),
        "mad_px": round(mad, 4),
        "authority": "recovered_from_cached_sam3d_3d_to_2d_projection",
    }


def _project(points3d: np.ndarray, cam_t: np.ndarray, focal: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points3d, dtype=np.float64)
    t = np.asarray(cam_t, dtype=np.float64).reshape(-1)
    cam = pts[:, :3] + t[:3]
    z = cam[:, 2]
    uv = np.full((len(cam), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(cam).all(axis=1) & (z > 1e-6)
    uv[valid, 0] = cam[valid, 0] * focal / z[valid] + width / 2.0
    uv[valid, 1] = cam[valid, 1] * focal / z[valid] + height / 2.0
    return uv, z


def _projection_residual(
    keypoints3d: np.ndarray,
    keypoints2d: np.ndarray,
    cam_t: np.ndarray,
    focal: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    projected, _ = _project(keypoints3d, cam_t, focal, width, height)
    cached = _to_pixels(keypoints2d, width, height)
    n = min(len(projected), len(cached))
    errors: list[float] = []
    for i in range(n):
        a, b = projected[i], cached[i]
        if np.isfinite(a).all() and np.isfinite(b).all():
            errors.append(float(np.linalg.norm(a - b)))
    if not errors:
        return {"count": 0, "median_px": None, "median_fraction_of_image_diagonal": None}
    median = float(np.median(errors))
    diag = math.hypot(width, height)
    return {
        "count": len(errors),
        "median_px": round(median, 4),
        "median_fraction_of_image_diagonal": round(median / diag, 6) if diag > 0 else None,
    }


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)[:3]
    b = np.asarray(b, dtype=np.float64)[:3]
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return np.linalg.norm(p - a[None, :], axis=1)
    t = np.clip(((p - a[None, :]) @ ab) / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(p - closest, axis=1)


def _segment_definitions(kp: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    def p(name: str) -> np.ndarray:
        return np.asarray(kp[MHR[name]], dtype=np.float64)[:3]

    return [
        ("left_arm", p("left_shoulder"), p("left_elbow")),
        ("left_arm", p("left_elbow"), p("left_wrist")),
        ("right_arm", p("right_shoulder"), p("right_elbow")),
        ("right_arm", p("right_elbow"), p("right_wrist")),
        ("other", p("neck"), p("left_shoulder")),
        ("other", p("neck"), p("right_shoulder")),
        ("other", p("left_shoulder"), p("left_hip")),
        ("other", p("right_shoulder"), p("right_hip")),
        ("other", p("left_hip"), p("right_hip")),
        ("other", p("left_hip"), p("left_knee")),
        ("other", p("right_hip"), p("right_knee")),
        ("other", p("left_knee"), p("left_ankle")),
        ("other", p("right_knee"), p("right_ankle")),
        ("other", p("nose"), p("neck")),
    ]


def _classify_face_parts(vertices: np.ndarray, faces: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    centroids = vertices[faces].mean(axis=1)
    segments = _segment_definitions(keypoints)
    distances = np.stack(
        [_point_segment_distance(centroids, a, b) for _, a, b in segments],
        axis=1,
    )
    winner = np.argmin(distances, axis=1)
    labels = np.zeros(len(faces), dtype=np.uint8)
    for i, seg_idx in enumerate(winner):
        name = segments[int(seg_idx)][0]
        labels[i] = 1 if name == "other" else (2 if name == "left_arm" else 3)
    return labels


def _raster_size(width: int, height: int) -> tuple[int, int, float, float]:
    scale = RASTER_LONG_EDGE / max(width, height)
    rw = max(32, int(round(width * scale)))
    rh = max(32, int(round(height * scale)))
    return rw, rh, rw / width, rh / height


def _render_part_mask(
    uv: np.ndarray,
    z: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    rw, rh, sx, sy = _raster_size(width, height)
    canvas = Image.new("L", (rw, rh), 0)
    draw = ImageDraw.Draw(canvas)

    order: list[tuple[float, int]] = []
    for i, face in enumerate(faces):
        pts = uv[face]
        depths = z[face]
        if not np.isfinite(pts).all() or not np.isfinite(depths).all() or np.any(depths <= 1e-6):
            continue
        minx, maxx = float(pts[:, 0].min()), float(pts[:, 0].max())
        miny, maxy = float(pts[:, 1].min()), float(pts[:, 1].max())
        if maxx < 0 or maxy < 0 or minx >= width or miny >= height:
            continue
        order.append((float(np.mean(depths)), i))

    # Painter's algorithm: far triangles first, near triangles overwrite.
    order.sort(reverse=True)
    for _, i in order:
        face = faces[i]
        pts = uv[face]
        polygon = [(float(p[0]) * sx, float(p[1]) * sy) for p in pts]
        draw.polygon(polygon, fill=int(labels[i]))

    return np.asarray(canvas, dtype=np.uint8)


def _save_debug_overlay(image_path: Path, raster: np.ndarray, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGBA")
    labels = Image.fromarray(raster, mode="L").resize(image.size, resample=Image.Resampling.NEAREST)
    arr = np.asarray(labels, dtype=np.uint8)

    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    # Left/right are intentionally different only for debugging. Caption-facing
    # foreground language remains side-neutral unless another specialist binds anatomy.
    overlay[arr == 2] = [255, 80, 80, 115]
    overlay[arr == 3] = [80, 180, 255, 115]
    rgba = Image.fromarray(overlay, mode="RGBA")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, rgba).convert("RGB").save(out_path, quality=92)


def _mask_bbox(mask: np.ndarray) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "x0": round(x0 / w, 4),
        "y0": round(y0 / h, 4),
        "x1": round((x1 + 1) / w, 4),
        "y1": round((y1 + 1) / h, 4),
        "center_x": round(((x0 + x1 + 1) / 2) / w, 4),
        "center_y": round(((y0 + y1 + 1) / 2) / h, 4),
        "touches": {
            "left": x0 <= 1,
            "top": y0 <= 1,
            "right": x1 >= w - 2,
            "bottom": y1 >= h - 2,
        },
    }


def _frame_region(bbox: dict[str, Any] | None) -> str | None:
    if not bbox:
        return None
    x = float(bbox["center_x"])
    y = float(bbox["center_y"])
    horizontal = "left" if x < 0.43 else ("right" if x > 0.57 else "center")
    if y >= 0.55:
        if horizontal == "left":
            return "lower_frame_left"
        if horizontal == "right":
            return "lower_frame_right"
        return "lower_center"
    if horizontal == "left":
        return "frame_left"
    if horizontal == "right":
        return "frame_right"
    return "center"


def _occupancy_band(value: float) -> str:
    if value >= OCCUPANCY_LARGE:
        return "large"
    if value >= OCCUPANCY_MEDIUM:
        return "medium"
    if value >= OCCUPANCY_SMALL:
        return "small"
    if value > 0:
        return "trace"
    return "absent"


def _dwpose_visible_names(dwpose: dict[str, Any]) -> set[str]:
    target = ((dwpose.get("derived") or {}).get("target") or {})
    names = target.get("visible_body_landmarks")
    if isinstance(names, list):
        return {str(v) for v in names if isinstance(v, str)}
    return set()


def _arm_observation_support(dwpose: dict[str, Any], side: str) -> dict[str, Any]:
    visible = _dwpose_visible_names(dwpose)
    states = {
        "shoulder": f"{side}_shoulder" in visible,
        "elbow": f"{side}_elbow" in visible,
        "wrist": f"{side}_wrist" in visible,
    }
    count = sum(states.values())
    adjacent = (states["shoulder"] and states["elbow"]) or (states["elbow"] and states["wrist"])
    if count == 3:
        grade = "strong"
    elif adjacent:
        grade = "moderate"
    elif count:
        grade = "weak"
    else:
        grade = "none"
    return {
        "grade": grade,
        "observed": states,
        "observed_count": count,
        "adjacent_pair_observed": adjacent,
        "authority": "dwpose_observed_joint_visibility",
    }


def _evidence_grade(occupancy: float, observation: dict[str, Any], residual: dict[str, Any]) -> str:
    band = _occupancy_band(occupancy)
    support = str(observation.get("grade") or "none")
    residual_frac = residual.get("median_fraction_of_image_diagonal")
    projection_ok = isinstance(residual_frac, (int, float)) and residual_frac <= 0.01

    if band == "large" and support in {"strong", "moderate"} and projection_ok:
        return "strong"
    if band in {"large", "medium"} and support in {"strong", "moderate"} and projection_ok:
        return "moderate"
    if band == "large" and support == "weak" and projection_ok:
        return "weak"
    if band in {"medium", "small"} and support in {"strong", "moderate"}:
        return "weak"
    return "insufficient"


def evaluate(
    policy: dict[str, Any],
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any],
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    image_path: Path | None = None,
    debug_overlay_path: Path | None = None,
) -> dict[str, Any]:
    key = str(policy.get("image_key") or "")
    size = policy.get("image_size") or []
    if len(size) < 2:
        raise ValueError("policy image_size missing")
    width, height = int(size[0]), int(size[1])

    keypoints3d = np.asarray(arrays.get("pred_keypoints_3d"), dtype=np.float64)
    keypoints2d = np.asarray(arrays.get("pred_keypoints_2d"), dtype=np.float64)
    cam_t = np.asarray(arrays.get("pred_cam_t"), dtype=np.float64)
    focal, focal_meta = _recover_focal_length(
        keypoints3d, keypoints2d, cam_t, width, height
    )
    if focal is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "image_key": key,
            "reason": "focal_length_recovery_failed",
            "focal_length_recovery": focal_meta,
        }

    residual = _projection_residual(
        keypoints3d, keypoints2d, cam_t, focal, width, height
    )
    uv, z = _project(vertices, cam_t, focal, width, height)
    face_labels = _classify_face_parts(vertices, faces, keypoints3d)
    raster = _render_part_mask(uv, z, faces, face_labels, width, height)
    if image_path is not None and debug_overlay_path is not None and image_path.is_file():
        _save_debug_overlay(image_path, raster, debug_overlay_path)

    total = float(raster.size)
    arms: dict[str, Any] = {}
    for side, label in (("left", 2), ("right", 3)):
        mask = raster == label
        occupancy = float(mask.sum()) / total if total else 0.0
        bbox = _mask_bbox(mask)
        observation = _arm_observation_support(dwpose, side)
        region = _frame_region(bbox)
        grade = _evidence_grade(occupancy, observation, residual)
        composer_text = None
        if grade in {"strong", "moderate"} and region:
            if _occupancy_band(occupancy) == "large":
                composer_text = f"an outstretched arm fills much of the {region.replace('_', '-')} foreground"
            else:
                composer_text = f"an arm occupies the {region.replace('_', '-')} foreground"
        arms[side] = {
            "anatomical_side": side,
            "visible_mesh_area_fraction": round(occupancy, 6),
            "occupancy_band": _occupancy_band(occupancy),
            "frame_region": region,
            "visible_bbox": bbox,
            "dwpose_observation_support": observation,
            "evidence_grade": grade,
            "composer_text_side_neutral": composer_text,
            "mesh_part_authority": "reconstructed_mesh_partition_shadow_not_image_segmentation",
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": key,
        "image_size": [width, height],
        "focal_length_recovery": focal_meta,
        "projection_residual": residual,
        "mesh": {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "raster_size": [int(raster.shape[1]), int(raster.shape[0])],
            "raster_method": "depth_sorted_projected_mesh_part_labels",
            "part_assignment": "nearest_major_skeleton_segment_in_reconstructed_3d",
            "debug_overlay": str(debug_overlay_path) if debug_overlay_path is not None else None,
            "debug_overlay_legend": {
                "left_arm_internal": "red",
                "right_arm_internal": "blue"
            },
        },
        "arms": arms,
        "invariants": {
            "mesh_is_reconstruction_not_observation": True,
            "mesh_area_cannot_be_strong_without_dwpose_arm_observation_support": True,
            "foreground_composer_text_is_side_neutral": True,
            "frame_left_right_is_image_relative": True,
            "anatomical_side_is_retained_only_for_internal_cross_checks": True,
            "projected_mesh_is_self_occlusion_approximated_by_depth_sorted_rasterization": True,
        },
        "calibration": {
            "large_area_fraction_min": OCCUPANCY_LARGE,
            "medium_area_fraction_min": OCCUPANCY_MEDIUM,
            "small_area_fraction_min": OCCUPANCY_SMALL,
        },
    }


def _find_sam_artifact(arrays_path: Path, suffix: str) -> Path | None:
    key = arrays_path.name.removesuffix(".sam3d_arrays.npz")
    direct = arrays_path.parent / f"{key}{suffix}"
    if direct.is_file():
        return direct
    matches = sorted(arrays_path.parent.rglob(f"{key}{suffix}"))
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate visible projected arm occupancy from cached SAM3D OBJ meshes, gated by DWPose observation."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

    if not run_dir.is_dir() or not policy_dir.is_dir():
        print(f"Required directory missing: run={run_dir} policy={policy_dir}", file=sys.stderr)
        return 2

    requested = set(args.only)
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in requested]
    if not paths:
        print("No matching policy records.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for policy_path in paths:
        policy = _read_json(policy_path)
        key = str(policy.get("image_key") or policy_path.name.removesuffix(".perception_policy.json"))
        out_path = output_dir / f"{key}.mesh_arm_occupancy.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
            records.append(record)
            continue

        sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
        arrays_path = Path(str(sources.get("sam3d_arrays") or "")).expanduser()
        dwpose_path = Path(str(sources.get("dwpose") or "")).expanduser()
        obj_path = _find_sam_artifact(arrays_path, ".sam3d.obj") if arrays_path.is_file() else None

        if not arrays_path.is_file() or not dwpose_path.is_file() or obj_path is None:
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "unavailable",
                "image_key": key,
                "reason": "missing_cached_sam3d_arrays_dwpose_or_obj_mesh",
                "sources": {
                    "policy": str(policy_path),
                    "sam3d_arrays": str(arrays_path),
                    "dwpose": str(dwpose_path),
                    "sam3d_obj": str(obj_path) if obj_path else None,
                },
            }
        else:
            vertices, faces = _load_obj(obj_path)
            image_path = Path(str(policy.get("image") or "")).expanduser()
            overlay_path = output_dir / f"{key}.mesh_arm_overlay.jpg"
            record = evaluate(
                policy,
                _load_arrays(arrays_path),
                _read_json(dwpose_path),
                vertices,
                faces,
                image_path=image_path if image_path.is_file() else None,
                debug_overlay_path=overlay_path if image_path.is_file() else None,
            )
            record["sources"] = {
                "policy": str(policy_path),
                "sam3d_arrays": str(arrays_path),
                "dwpose": str(dwpose_path),
                "sam3d_obj": str(obj_path),
            }

        _write_json(out_path, record)
        records.append(record)
        if record.get("status") == "ok":
            la = ((record.get("arms") or {}).get("left") or {})
            ra = ((record.get("arms") or {}).get("right") or {})
            print(
                f"{key}: "
                f"L={la.get('visible_mesh_area_fraction')}:{la.get('evidence_grade')}:{la.get('frame_region')} "
                f"R={ra.get('visible_mesh_area_fraction')}:{ra.get('evidence_grade')}:{ra.get('frame_region')}"
            )
        else:
            print(f"{key}: {record.get('status')} ({record.get('reason')})")

    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    grade_counts = Counter()
    for record in records:
        for arm in ((record.get("arms") or {}).values()):
            if isinstance(arm, dict):
                grade_counts[str(arm.get("evidence_grade") or "unknown")] += 1

    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "arm_evidence_grade_counts": dict(sorted(grade_counts.items())),
        "records": records,
    }
    _write_json(output_dir / "mesh_arm_occupancy.index.json", index)
    print(f"Index: {output_dir / 'mesh_arm_occupancy.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
