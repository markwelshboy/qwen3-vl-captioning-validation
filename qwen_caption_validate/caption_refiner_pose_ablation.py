from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import caption_refiner_pose_vlm as base
from . import pose_atlas_v3 as atlas


CARD_MODES = {"front-only", "camera-only", "crop-mesh-only"}
_ACTIVE_CARD_MODE = "front-only"


def _extract_card_mode(argv: list[str]) -> tuple[str, list[str]]:
    mode = None
    cleaned = [argv[0]]
    index = 1
    while index < len(argv):
        value = argv[index]
        if value == "--card-mode":
            if index + 1 >= len(argv):
                raise SystemExit("--card-mode requires front-only, camera-only, or crop-mesh-only")
            mode = argv[index + 1]
            index += 2
            continue
        if value.startswith("--card-mode="):
            mode = value.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(value)
        index += 1
    if mode not in CARD_MODES:
        raise SystemExit("--card-mode must be one of: front-only, camera-only, crop-mesh-only")
    return str(mode), cleaned


def _reshape_points(value: np.ndarray, dimensions: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, dimensions), dtype=np.float64)
    if arr.ndim == 1:
        if arr.size < dimensions:
            return np.empty((0, dimensions), dtype=np.float64)
        arr = arr.reshape((-1, dimensions))
    elif arr.ndim > 2:
        arr = arr.reshape((-1, arr.shape[-1]))
    if arr.shape[-1] < dimensions:
        return np.empty((0, dimensions), dtype=np.float64)
    return arr[:, :dimensions]


def _fit_mesh_to_image(
    vertices_camera: np.ndarray,
    arrays: dict[str, np.ndarray],
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate the SAM3D camera-space mesh projection from cached keypoint pairs.

    The cache contains both pred_keypoints_3d and their corresponding pred_keypoints_2d,
    but not a ready-made 2-D projection for every mesh vertex. Fit a small linear camera
    map [x,y,z,1] -> [u,v] from those paired keypoints and apply it to every mesh vertex.
    The residual is retained so this experimental visualization can be sanity-checked.
    """
    key3d = _reshape_points(arrays.get("pred_keypoints_3d", np.empty((0, 3))), 3)
    key2d = _reshape_points(arrays.get("pred_keypoints_2d", np.empty((0, 2))), 2)
    count = min(len(key3d), len(key2d))
    key3d = key3d[:count]
    key2d = key2d[:count]
    if count:
        key2d = atlas._normalized_to_pixels(key2d, width, height)

    valid = (
        np.isfinite(key3d).all(axis=1)
        & np.isfinite(key2d).all(axis=1)
    ) if count else np.zeros((0,), dtype=bool)
    key3d = key3d[valid]
    key2d = key2d[valid]
    if len(key3d) < 6:
        raise ValueError(f"Need at least 6 finite 3D/2D keypoint pairs; found {len(key3d)}")

    features = np.column_stack([key3d, np.ones(len(key3d), dtype=np.float64)])
    coefficients, _, rank, _ = np.linalg.lstsq(features, key2d, rcond=None)
    if rank < 4:
        # Perspective/depth may not be independently identifiable in a nearly planar pose.
        # Fall back to a stable camera-plane affine fit in x/y.
        features = np.column_stack([key3d[:, :2], np.ones(len(key3d), dtype=np.float64)])
        coefficients, _, rank, _ = np.linalg.lstsq(features, key2d, rcond=None)
        mesh_features = np.column_stack([
            np.asarray(vertices_camera, dtype=np.float64)[:, :2],
            np.ones(len(vertices_camera), dtype=np.float64),
        ])
        fit_kind = "affine_xy"
    else:
        mesh_features = np.column_stack([
            np.asarray(vertices_camera, dtype=np.float64)[:, :3],
            np.ones(len(vertices_camera), dtype=np.float64),
        ])
        fit_kind = "linear_xyz"

    predicted_keypoints = features @ coefficients
    residuals = np.linalg.norm(predicted_keypoints - key2d, axis=1)
    projected = mesh_features @ coefficients
    diagonal = max(1.0, float(np.hypot(width, height)))
    return projected, {
        "projection_fit_kind": fit_kind,
        "projection_fit_pairs": int(len(key3d)),
        "projection_fit_rms_px": round(float(np.sqrt(np.mean(residuals ** 2))), 3),
        "projection_fit_median_px": round(float(np.median(residuals)), 3),
        "projection_fit_rms_fraction_of_image_diagonal": round(float(np.sqrt(np.mean(residuals ** 2))) / diagonal, 6),
    }


def _crop_mesh_panel(
    projected_mesh: np.ndarray,
    width: int,
    height: int,
) -> tuple[Image.Image, dict[str, Any]]:
    panel = base._panel("Body mesh inside photograph frame only")
    draw = ImageDraw.Draw(panel)

    top_area = base.TITLE_H + 24
    bottom_area = base.PANEL_H - 26
    left_area = 26
    right_area = base.PANEL_W - 26
    available_w = right_area - left_area
    available_h = bottom_area - top_area
    scale = min(available_w / max(1.0, float(width)), available_h / max(1.0, float(height)))
    frame_w = float(width) * scale
    frame_h = float(height) * scale
    x0 = left_area + (available_w - frame_w) / 2.0
    y0 = top_area + (available_h - frame_h) / 2.0
    x1 = x0 + frame_w
    y1 = y0 + frame_h
    draw.rectangle((x0, y0, x1, y1), fill=base.FRAME_FILL, outline=base.FRAME_OUTLINE, width=3)

    points = np.asarray(projected_mesh, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1) if points.size else np.zeros((0,), dtype=bool)
    inside = finite.copy()
    if len(points):
        inside &= points[:, 0] >= 0.0
        inside &= points[:, 0] <= float(width)
        inside &= points[:, 1] >= 0.0
        inside &= points[:, 1] <= float(height)
    visible = points[inside]
    if not len(visible):
        draw.text((18, 72), "No projected mesh vertices fall inside frame", fill=base.MUTED, font=base._font(16))
    else:
        step = max(1, len(visible) // 12000)
        for u, v in visible[::step, :2]:
            sx = x0 + float(u) * scale
            sy = y0 + float(v) * scale
            draw.point((sx, sy), fill=base.MESH_POINT)

    return panel, {
        "mesh_vertices_total": int(len(points)),
        "mesh_vertices_inside_frame": int(len(visible)),
        "mesh_inside_fraction": round(float(len(visible)) / max(1, len(points)), 6),
    }


def make_pose_card_ablation(
    image_path: Path,
    sam_npz: Path,
    sam_obj: Path | None,
    output: Path,
) -> dict[str, Any]:
    with Image.open(image_path) as source:
        width, height = source.size
    arrays = base._load_arrays(sam_npz)
    vertices_camera = atlas._load_obj_vertices(sam_obj)
    if vertices_camera.size == 0:
        raise ValueError(f"Ablation mode {_ACTIVE_CARD_MODE} requires a cached SAM3D OBJ mesh: {sam_obj}")

    metadata: dict[str, Any] = {
        "source_image_size": [width, height],
        "sam3d_arrays": str(sam_npz),
        "sam3d_mesh": str(sam_obj) if sam_obj else None,
        "pose_card": str(output),
        "card_mode": _ACTIVE_CARD_MODE,
    }

    if _ACTIVE_CARD_MODE == "front-only":
        # Deliberately normalize away camera-relative body orientation. This is the
        # earlier ablation retained for comparison with camera-only.
        vertices_body = atlas._body_frame_vertices(vertices_camera, arrays)
        card = base._mesh_panel(vertices_body, (0, 1), "Full body — body-frame frontal mesh")
        metadata["visual_contract"] = "single_full_body_body_frame_frontal_mesh_no_crop_no_side_view"
    elif _ACTIVE_CARD_MODE == "camera-only":
        # pred_vertices is written to OBJ exactly as returned by SAM3D. Do not apply
        # _body_frame_vertices here: x/y therefore retain the pose orientation seen
        # by the reconstruction camera. This is the clean single-view ablation.
        card = base._mesh_panel(vertices_camera, (0, 1), "Full body — camera-space mesh view")
        metadata["visual_contract"] = "single_full_body_camera_space_mesh_no_rotation_no_crop_no_side_view"
    elif _ACTIVE_CARD_MODE == "crop-mesh-only":
        projected_mesh, fit_meta = _fit_mesh_to_image(vertices_camera, arrays, width, height)
        card, crop_meta = _crop_mesh_panel(projected_mesh, width, height)
        metadata.update(fit_meta)
        metadata.update(crop_meta)
        metadata["visual_contract"] = "camera_aligned_mesh_clipped_to_photograph_frame_no_out_of_frame_geometry"
    else:
        raise AssertionError(_ACTIVE_CARD_MODE)

    output.parent.mkdir(parents=True, exist_ok=True)
    card.save(output, format="WEBP", quality=92, method=6)
    return metadata


def main() -> int:
    global _ACTIVE_CARD_MODE
    mode, cleaned_argv = _extract_card_mode(sys.argv)
    _ACTIVE_CARD_MODE = mode
    sys.argv = cleaned_argv
    base.make_pose_card = make_pose_card_ablation
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
