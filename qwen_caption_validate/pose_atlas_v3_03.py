from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import pose_atlas_v3 as base
from . import pose_atlas_v3_02 as v02
from .sam3d_subject_geometry_diagnostic_02 import build_subject_geometry


# Meta's MHR70 display topology for the body landmarks we retain in the atlas.
# In particular, the face connects through eyes/ears to the shoulders rather
# than using the v0.2 atlas' synthetic nose->neck->shoulder display links.
MHR_BODY = v02.MHR_BODY
MHR_BODY_EDGES = [
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_ankle", "left_big_toe"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"),
    ("right_ankle", "right_heel"),
]

DW_TO_MHR = v02.DW_TO_MHR
DISPLAY_GUTTER_PX = 8
DWPOSE_COLOR = "#66d9ef"
SAM3D_COLOR = "#ffcc66"
OUT_OF_FRAME_FILL = "#ff9f43"
FRAME_COLOR = "#69707d"
PAD_COLOR = "#080a0d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-atlas-v3-03",
        description=(
            "Build Pose Atlas v0.3 with explicit DWPose/SAM3D coordinate contracts, "
            "historical multi-person DWPose decoding, and padded 2D displays that "
            "show accepted/reconstructed joints outside the source image."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--sam3d-dir", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _finite_xy(point: np.ndarray) -> bool:
    p = np.asarray(point, dtype=np.float64)
    return bool(p.size >= 2 and np.all(np.isfinite(p[:2])))


def _in_frame(point: np.ndarray, width: int, height: int) -> bool:
    if not _finite_xy(point):
        return False
    x, y = float(point[0]), float(point[1])
    return 0.0 <= x <= float(width - 1) and 0.0 <= y <= float(height - 1)


def _target_person_index(record: dict[str, Any], person_count: int) -> int:
    value = (record.get("derived") or {}).get("target_person_index")
    try:
        index = int(value) if value is not None else 0
    except (TypeError, ValueError):
        index = 0
    if index < 0 or index >= max(1, person_count):
        index = 0
    return index


def _raw_dwpose_people(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Decode historical easy-dwpose body arrays without coordinate guessing.

    easy-dwpose serializes body coordinates as a flattened ``(people*18, 2)``
    array while ``body_scores`` remains ``(people, 18)``.  Some newer wrappers
    may retain a ``candidate`` dictionary or an already-unflattened 3-D array.
    This decoder normalizes all of those shapes to ``(people, 18, 2)`` and
    returns the score/index array alongside it.
    """
    raw = record.get("raw_pose") or {}
    bodies: Any = raw.get("bodies") if isinstance(raw, dict) else raw
    if isinstance(bodies, dict):
        for key in ("candidate", "bodies", "body", "people", "persons", "poses"):
            if key in bodies:
                bodies = bodies.get(key)
                break

    try:
        arr = np.asarray(bodies if bodies is not None else [], dtype=np.float64)
    except (TypeError, ValueError):
        arr = np.empty((0, 2), dtype=np.float64)

    try:
        scores = np.asarray(
            raw.get("body_scores", []) if isinstance(raw, dict) else [],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        scores = np.empty((0, 18), dtype=np.float64)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if scores.ndim == 1 and scores.size >= 18:
        scores = scores[None, :18]

    if arr.ndim == 3 and arr.shape[1] >= 18 and arr.shape[2] >= 2:
        return arr[:, :18, :2], scores

    if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 18:
        if scores.ndim == 2 and scores.shape[0] > 0 and scores.shape[1] >= 18:
            people = int(scores.shape[0])
        elif arr.shape[0] % 18 == 0:
            people = max(1, int(arr.shape[0] // 18))
        else:
            people = 1
        needed = people * 18
        if arr.shape[0] >= needed:
            return arr[:needed, :2].reshape(people, 18, 2), scores

    return np.empty((0, 18, 2), dtype=np.float64), scores


def _accepted_mask(
    record: dict[str, Any],
    scores: np.ndarray,
    target_index: int,
) -> np.ndarray:
    """Return the DWPose acceptance gate, not a claim of visual observation.

    Historical easy-dwpose replaces confidence values with joint indices for
    detections above 0.3 and ``-1`` otherwise.  If a future cache contains raw
    confidence values in [0,1], retain the original >0.3 threshold instead.
    """
    if scores.ndim == 2 and target_index < scores.shape[0] and scores.shape[1] >= 18:
        row = np.asarray(scores[target_index, :18], dtype=np.float64)
        finite = np.isfinite(row)
        good = row[finite]
        if good.size:
            index_encoded = bool(np.any(good < 0.0) or np.nanmax(good) > 1.5)
            return finite & ((row >= 0.0) if index_encoded else (row > 0.3))

    target = ((record.get("derived") or {}).get("target") or {})
    names = {
        str(name)
        for name in (target.get("visible_body_landmarks") or [])
        if isinstance(name, str)
    }
    return np.asarray([name in names for name in base.BODY18], dtype=bool)


def _dwpose_target_points(
    record: dict[str, Any], width: int, height: int
) -> tuple[np.ndarray, set[str], set[str]]:
    people, scores = _raw_dwpose_people(record)
    if people.size == 0:
        return np.empty((0, 2), dtype=np.float64), set(), set()

    target_index = _target_person_index(record, len(people))
    normalized = np.asarray(people[target_index, :18, :2], dtype=np.float64).copy()

    # Contract: easy-dwpose coordinates are normalized by image width/height.
    # Values outside [0,1] are legitimate extrapolated model coordinates and
    # must not cause the whole array to be reinterpreted as [-1,+1] NDC.
    pixels = normalized.copy()
    pixels[:, 0] *= float(width)
    pixels[:, 1] *= float(height)

    accepted_mask = _accepted_mask(record, scores, target_index)
    accepted_names = {
        base.BODY18[i]
        for i in range(min(18, len(accepted_mask), len(pixels)))
        if bool(accepted_mask[i]) and _finite_xy(pixels[i])
    }
    in_frame_names = {
        name
        for name in accepted_names
        if _in_frame(pixels[base.IDX[name]], width, height)
    }
    return pixels, accepted_names, in_frame_names


def _sam2d_points(arrays: dict[str, np.ndarray]) -> np.ndarray:
    value = np.asarray(
        arrays.get("pred_keypoints_2d", np.empty((0, 2))), dtype=np.float64
    )
    if value.ndim > 2:
        value = value.reshape((-1, value.shape[-1]))
    if value.size == 0 or value.ndim != 2 or value.shape[-1] < 2:
        return np.empty((0, 2), dtype=np.float64)

    # Contract: SAM 3D Body's perspective head returns original-image pixels.
    return value[:, :2].copy()


def _sam_body_points(sam2d: np.ndarray) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for idx in MHR_BODY.values():
        if idx < len(sam2d) and _finite_xy(sam2d[idx]):
            points.append(np.asarray(sam2d[idx], dtype=np.float64))
    return points


def _display_padding(
    width: int,
    height: int,
    dw_points: np.ndarray,
    accepted_dwpose: set[str],
    sam2d: np.ndarray,
) -> dict[str, Any]:
    """Pad to the maximum projected body extent from either model.

    Rejected DWPose coordinates are deliberately excluded because their original
    confidence was below the easy-dwpose threshold and historical caches no
    longer retain that confidence.  Every accepted DWPose joint and every
    finite SAM3D MHR body joint participates, including coordinates outside the
    source image.
    """
    points: list[np.ndarray] = []
    for name in accepted_dwpose:
        idx = base.IDX.get(name)
        if idx is not None and idx < len(dw_points) and _finite_xy(dw_points[idx]):
            points.append(np.asarray(dw_points[idx], dtype=np.float64))
    points.extend(_sam_body_points(sam2d))

    if points:
        cloud = np.stack(points, axis=0)
        min_x, min_y = np.min(cloud[:, :2], axis=0)
        max_x, max_y = np.max(cloud[:, :2], axis=0)
    else:
        min_x = min_y = 0.0
        max_x, max_y = float(width - 1), float(height - 1)

    data = {
        "left": max(0, int(math.ceil(-float(min_x)))),
        "top": max(0, int(math.ceil(-float(min_y)))),
        "right": max(0, int(math.ceil(float(max_x) - float(width - 1)))),
        "bottom": max(0, int(math.ceil(float(max_y) - float(height - 1)))),
    }
    display = {
        side: int(amount + DISPLAY_GUTTER_PX) if amount > 0 else 0
        for side, amount in data.items()
    }
    return {
        "source": "union_of_dwpose_accepted_and_sam3d_mhr_body",
        "data_padding_px": data,
        "display_padding_px": display,
        "drawing_gutter_px": DISPLAY_GUTTER_PX,
        "projected_extent_xyxy": [
            round(float(min_x), 2),
            round(float(min_y), 2),
            round(float(max_x), 2),
            round(float(max_y), 2),
        ],
    }


def _padded_canvas(image: Image.Image, padding: dict[str, Any]) -> tuple[Image.Image, tuple[int, int]]:
    pad = padding.get("display_padding_px") or {}
    left = int(pad.get("left") or 0)
    top = int(pad.get("top") or 0)
    right = int(pad.get("right") or 0)
    bottom = int(pad.get("bottom") or 0)
    src = image.convert("RGB")
    canvas = Image.new(
        "RGB",
        (src.width + left + right, src.height + top + bottom),
        PAD_COLOR,
    )
    canvas.paste(src, (left, top))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (left, top, left + src.width - 1, top + src.height - 1),
        outline=FRAME_COLOR,
        width=2,
    )
    return canvas, (left, top)


def _shift(point: np.ndarray, offset: tuple[int, int]) -> tuple[float, float]:
    return float(point[0]) + offset[0], float(point[1]) + offset[1]


def _draw_dwpose_padded(
    image: Image.Image,
    points: np.ndarray,
    accepted_names: set[str],
    padding: dict[str, Any],
) -> Image.Image:
    out, offset = _padded_canvas(image, padding)
    draw = ImageDraw.Draw(out)

    for a, b in base.DWPOSE_EDGES:
        if a not in accepted_names or b not in accepted_names:
            continue
        ia, ib = base.IDX[a], base.IDX[b]
        if ia >= len(points) or ib >= len(points):
            continue
        pa, pb = points[ia], points[ib]
        if _finite_xy(pa) and _finite_xy(pb):
            draw.line((*_shift(pa, offset), *_shift(pb, offset)), fill=DWPOSE_COLOR, width=4)

    for name in base.BODY18:
        if name not in accepted_names:
            continue
        idx = base.IDX[name]
        if idx >= len(points) or not _finite_xy(points[idx]):
            continue
        x, y = _shift(points[idx], offset)
        r = 4
        fill = "#ffffff" if _in_frame(points[idx], image.width, image.height) else OUT_OF_FRAME_FILL
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline="#111111")
    return out


def _draw_sam3d_padded(
    image: Image.Image,
    sam2d: np.ndarray,
    padding: dict[str, Any],
) -> Image.Image:
    out, offset = _padded_canvas(image, padding)
    draw = ImageDraw.Draw(out)

    def point(name: str) -> np.ndarray | None:
        idx = MHR_BODY.get(name)
        if idx is None or idx >= len(sam2d):
            return None
        p = np.asarray(sam2d[idx], dtype=np.float64)
        return p if _finite_xy(p) else None

    for a, b in MHR_BODY_EDGES:
        pa, pb = point(a), point(b)
        if pa is not None and pb is not None:
            draw.line((*_shift(pa, offset), *_shift(pb, offset)), fill=SAM3D_COLOR, width=4)

    for name in MHR_BODY:
        p = point(name)
        if p is None:
            continue
        x, y = _shift(p, offset)
        r = 4 if "toe" not in name and "heel" not in name else 3
        fill = "#ffffff" if _in_frame(p, image.width, image.height) else OUT_OF_FRAME_FILL
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline="#111111")
    return out


def _bbox_diag_pixels(dwpose: dict[str, Any], width: int, height: int) -> float | None:
    return v02._bbox_diag_pixels(dwpose, width, height)


def _reprojection_residual(
    dw_points: np.ndarray,
    sam2d: np.ndarray,
    accepted_dwpose: set[str],
    in_frame_dwpose: set[str],
    *,
    width: int,
    height: int,
    dwpose: dict[str, Any],
) -> dict[str, Any]:
    accepted_errors: list[float] = []
    in_frame_errors: list[float] = []
    per_joint: dict[str, float] = {}
    extrapolated: list[str] = []

    for dw_name, mhr_name in DW_TO_MHR.items():
        if dw_name not in accepted_dwpose:
            continue
        di = base.IDX.get(dw_name)
        si = MHR_BODY.get(mhr_name)
        if di is None or si is None or di >= len(dw_points) or si >= len(sam2d):
            continue
        dp = np.asarray(dw_points[di], dtype=np.float64)
        sp = np.asarray(sam2d[si], dtype=np.float64)
        if not _finite_xy(dp) or not _finite_xy(sp):
            continue
        distance = float(np.linalg.norm(dp[:2] - sp[:2]))
        if not math.isfinite(distance):
            continue
        accepted_errors.append(distance)
        per_joint[dw_name] = round(distance, 2)
        if dw_name in in_frame_dwpose:
            in_frame_errors.append(distance)
        else:
            extrapolated.append(dw_name)

    median = float(np.median(in_frame_errors)) if in_frame_errors else None
    mean = float(np.mean(in_frame_errors)) if in_frame_errors else None
    accepted_median = float(np.median(accepted_errors)) if accepted_errors else None
    accepted_mean = float(np.mean(accepted_errors)) if accepted_errors else None
    diag = _bbox_diag_pixels(dwpose, width, height)
    fraction = median / diag if median is not None and diag else None

    return {
        "reference_gate": "dwpose_accepted_and_in_original_frame",
        "common_joint_count": len(in_frame_errors),
        "median_px": round(median, 2) if median is not None else None,
        "mean_px": round(mean, 2) if mean is not None else None,
        "median_fraction_of_dwpose_bbox_diagonal": (
            round(float(fraction), 4) if fraction is not None else None
        ),
        "accepted_common_joint_count": len(accepted_errors),
        "accepted_median_px": round(accepted_median, 2) if accepted_median is not None else None,
        "accepted_mean_px": round(accepted_mean, 2) if accepted_mean is not None else None,
        "extrapolated_dwpose_joint_names": sorted(extrapolated),
        "per_joint_px": per_joint,
        "interpretation": (
            "Cross-model agreement only. median_px is restricted to DWPose-accepted joints "
            "whose DWPose projection lies inside the source frame; accepted_median_px also "
            "includes DWPose extrapolations outside the image."
        ),
    }


def _camera_keypoints_3d(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return v02._camera_keypoints_3d(arrays)


def _draw_3d_pose_panel(
    keypoints: np.ndarray,
    mesh_vertices: np.ndarray,
    axes: tuple[int, int],
    title: str,
) -> Image.Image:
    panel = base._panel(title)
    draw = ImageDraw.Draw(panel)
    top = base.TITLE_H + 28
    bottom = base.PANEL_H - 28
    left = 28
    right = base.PANEL_W - 28

    usable_indices = [
        idx for idx in MHR_BODY.values()
        if idx < len(keypoints) and np.all(np.isfinite(keypoints[idx]))
    ]
    if not usable_indices:
        draw.text((16, 64), "No SAM3D 3D keypoints available", fill="#c7cbd1", font=base._font(15))
        return panel

    body = keypoints[usable_indices]
    clouds = [body]
    finite_mesh = np.empty((0, 3), dtype=np.float64)
    if mesh_vertices.size:
        finite_mesh = mesh_vertices[np.isfinite(mesh_vertices).all(axis=1)]
        if len(finite_mesh):
            clouds.append(finite_mesh)

    extent = np.concatenate(clouds, axis=0)
    x = extent[:, axes[0]]
    y = extent[:, axes[1]]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(1e-9, xmax-xmin), max(1e-9, ymax-ymin)
    scale = min((right-left)/dx, (bottom-top)/dy)
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    ox = (left+right)/2.0
    oy = (top+bottom)/2.0

    def project(p: np.ndarray) -> tuple[float, float]:
        return (
            ox + (float(p[axes[0]]) - cx) * scale,
            oy - (float(p[axes[1]]) - cy) * scale,
        )

    if len(finite_mesh):
        step = max(1, len(finite_mesh) // 4500)
        for p in finite_mesh[::step]:
            draw.point(project(p), fill="#59616d")

    for a, b in MHR_BODY_EDGES:
        ia, ib = MHR_BODY[a], MHR_BODY[b]
        if ia >= len(keypoints) or ib >= len(keypoints):
            continue
        pa, pb = keypoints[ia], keypoints[ib]
        if not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)):
            continue
        draw.line((*project(pa), *project(pb)), fill=SAM3D_COLOR, width=4)

    for name, idx in MHR_BODY.items():
        if idx >= len(keypoints) or not np.all(np.isfinite(keypoints[idx])):
            continue
        x2, y2 = project(keypoints[idx])
        r = 4 if "toe" not in name and "heel" not in name else 3
        draw.ellipse((x2-r, y2-r, x2+r, y2+r), fill="#ffffff", outline="#111111")

    if not len(finite_mesh):
        draw.text(
            (14, base.PANEL_H - 24),
            "3D skeleton fallback (OBJ mesh not retained)",
            fill="#9aa1ab",
            font=base._font(13),
        )
    return panel


def _format_padding(padding: dict[str, Any]) -> str:
    pad = padding.get("data_padding_px") or {}
    return "/".join(str(int(pad.get(side) or 0)) for side in ("left", "top", "right", "bottom"))


def _worst_residuals(residual: dict[str, Any], limit: int = 3) -> str:
    values = residual.get("per_joint_px") or {}
    if not values:
        return "none"
    ordered = sorted(values.items(), key=lambda item: float(item[1]), reverse=True)[:limit]
    return ", ".join(f"{name}={value}px" for name, value in ordered)


def _summary_panel(
    key: str,
    diagnostic: dict[str, Any],
    dwpose: dict[str, Any],
    annotation: dict[str, Any],
    residual: dict[str, Any],
    padding: dict[str, Any],
    accepted_names: set[str],
    in_frame_names: set[str],
    mesh_available: bool,
) -> Image.Image:
    panel = base._panel("Semantic / calibration summary")
    draw = ImageDraw.Draw(panel)
    y = 52
    draw.text((14, y), key, fill="#ffffff", font=base._font(18, bold=True))
    y += 30
    body = diagnostic.get("body_camera_relation") or {}
    face = diagnostic.get("face_camera_relation") or {}
    target = ((dwpose.get("derived") or {}).get("target") or {})
    residual_fraction = residual.get("median_fraction_of_dwpose_bbox_diagonal")
    extrapolated = sorted(accepted_names - in_frame_names)

    rows = [
        f"SAM3D body: {body.get('orientation_band') or 'unknown'}  yaw={body.get('yaw_deg')}  frame={body.get('faces_frame') or '-'}",
        f"SAM3D face: {face.get('orientation_band') or 'unknown'}  yaw={face.get('yaw_deg')}  turn→cam={face.get('head_turn_toward_camera_deg')}",
        f"DWPose extent: {target.get('pose_extent_hint') or 'unknown'}",
        f"DWPose accepted: {len(accepted_names)}/18; in-frame accepted: {len(in_frame_names)}/18",
        f"DWPose extrapolated outside frame: {', '.join(extrapolated) if extrapolated else 'none'}",
        f"Display data pad L/T/R/B: {_format_padding(padding)} px",
        (
            f"2D fit in-frame: median={residual.get('median_px')} px  "
            f"bbox-frac={residual_fraction}  n={residual.get('common_joint_count')}"
        ),
        (
            f"2D fit all accepted: median={residual.get('accepted_median_px')} px  "
            f"n={residual.get('accepted_common_joint_count')}"
        ),
        f"Largest residuals: {_worst_residuals(residual)}",
        f"3D display: {'OBJ mesh + skeleton' if mesh_available else 'cached 3D skeleton (mesh unavailable)'}",
    ]
    for row in rows:
        y = base._draw_text_wrapped(draw, (14, y), row)
        y += 3

    if annotation and y < base.PANEL_H - 72:
        y += 6
        draw.text((14, y), "Human target", fill="#ffffff", font=base._font(15, bold=True))
        y += 23
        pose = annotation.get("pose_family") or "unlabeled"
        mods = annotation.get("modifiers") or []
        gestalt = annotation.get("human_gestalt") or annotation.get("gestalt")
        y = base._draw_text_wrapped(draw, (14, y), f"Pose: {pose}")
        if mods and y < base.PANEL_H - 45:
            y = base._draw_text_wrapped(draw, (14, y+2), "Modifiers: " + ", ".join(str(v) for v in mods))
        if gestalt and y < base.PANEL_H - 30:
            base._draw_text_wrapped(draw, (14, y+4), "Gestalt: " + str(gestalt), width_chars=54)
    return panel


def _empty_residual() -> dict[str, Any]:
    return {
        "reference_gate": "dwpose_accepted_and_in_original_frame",
        "common_joint_count": 0,
        "median_px": None,
        "mean_px": None,
        "median_fraction_of_dwpose_bbox_diagonal": None,
        "accepted_common_joint_count": 0,
        "accepted_median_px": None,
        "accepted_mean_px": None,
        "extrapolated_dwpose_joint_names": [],
        "per_joint_px": {},
        "interpretation": "unavailable",
    }


def _make_card(
    image_path: Path,
    dwpose_path: Path | None,
    sam_npz_path: Path,
    sam_obj_path: Path | None,
    annotation: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    dwpose = base._read_json(dwpose_path)
    with np.load(sam_npz_path) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}

    diagnostic = build_subject_geometry(arrays, dwpose or None)
    if dwpose:
        dw_points, accepted_names, in_frame_names = _dwpose_target_points(dwpose, width, height)
    else:
        dw_points, accepted_names, in_frame_names = np.empty((0, 2)), set(), set()
    sam2d = _sam2d_points(arrays)
    padding = _display_padding(width, height, dw_points, accepted_names, sam2d)

    residual = (
        _reprojection_residual(
            dw_points,
            sam2d,
            accepted_names,
            in_frame_names,
            width=width,
            height=height,
            dwpose=dwpose,
        )
        if dwpose and len(dw_points) and len(sam2d)
        else _empty_residual()
    )

    original = base._panel("Original", image)
    dw_overlay_img = (
        _draw_dwpose_padded(image, dw_points, accepted_names, padding)
        if len(dw_points) else _padded_canvas(image, padding)[0]
    )
    dw_overlay = base._panel("DWPose accepted 2D — padded", dw_overlay_img)
    sam_overlay_img = (
        _draw_sam3d_padded(image, sam2d, padding)
        if len(sam2d) else _padded_canvas(image, padding)[0]
    )
    sam_overlay = base._panel("SAM3D full projected 2D — padded", sam_overlay_img)

    keypoints3d = _camera_keypoints_3d(arrays)
    mesh = base._load_obj_vertices(sam_obj_path)
    camera_front = _draw_3d_pose_panel(
        keypoints3d, mesh, (0, 1), "SAM3D reconstructed 3D — camera view"
    )
    side_depth = _draw_3d_pose_panel(
        keypoints3d, mesh, (2, 1), "SAM3D reconstructed 3D — side/depth"
    )
    summary = _summary_panel(
        base._image_key(image_path),
        diagnostic,
        dwpose,
        annotation,
        residual,
        padding,
        accepted_names,
        in_frame_names,
        bool(len(mesh)),
    )

    card = Image.new(
        "RGB",
        (base.PANEL_W * base.GRID_COLS, base.PANEL_H * base.GRID_ROWS),
        "#0b0d10",
    )
    for index, panel in enumerate(
        [original, dw_overlay, sam_overlay, camera_front, side_depth, summary]
    ):
        x = (index % base.GRID_COLS) * base.PANEL_W
        y = (index // base.GRID_COLS) * base.PANEL_H
        card.paste(panel, (x, y))

    record = {
        "schema_version": "pose-atlas-v3-record-0.3",
        "image_key": base._image_key(image_path),
        "image": str(image_path),
        "dwpose": str(dwpose_path) if dwpose_path else None,
        "sam3d_arrays": str(sam_npz_path),
        "sam3d_mesh": str(sam_obj_path) if sam_obj_path else None,
        "sam3d_mesh_available": bool(len(mesh)),
        "sam3d_diagnostic": diagnostic,
        "dwpose_derived": dwpose.get("derived") if dwpose else None,
        "dwpose_accepted_joint_names": sorted(accepted_names),
        "dwpose_in_frame_accepted_joint_names": sorted(in_frame_names),
        "display_padding": padding,
        "projected_fit_residual": residual,
        "human_annotation": annotation or None,
        "interpretation_policy": {
            "purpose": "visual calibration of broad pose/crop semantics, not proof of invisible anatomy",
            "dwpose_coordinate_contract": "normalized_xy_times_original_image_width_height_no_ndc_guessing",
            "sam3d_coordinate_contract": "pred_keypoints_2d_are_original_image_pixels",
            "dwpose_accepted_is_not_synonymous_with_observed": True,
            "reconstruction_is_not_observation": True,
            "missing_out_of_crop_anatomy_is_not_counterevidence": True,
            "padded_2d_display_uses_union_of_dwpose_accepted_and_full_sam3d_body_extents": True,
            "projected_fit_residual_uses_dwpose_as_reference_not_ground_truth": True,
        },
    }
    return card, record


def _html_index(records: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for record in records:
        key = html.escape(str(record["image_key"]))
        webp = html.escape(str(record["card_webp"]))
        body = (record.get("sam3d_diagnostic") or {}).get("body_camera_relation") or {}
        ann = record.get("human_annotation") or {}
        human = ann.get("pose_family") or "unlabeled"
        residual = record.get("projected_fit_residual") or {}
        padding = record.get("display_padding") or {}
        cards.append(
            f'<article class="card"><h2>{key}</h2>'
            f'<img src="{webp}" loading="lazy" alt="Pose atlas card for {key}">'
            f'<p><b>Human:</b> {html.escape(str(human))} &nbsp; '
            f'<b>SAM3D:</b> {html.escape(str(body.get("orientation_band") or "unknown"))} '
            f'({html.escape(str(body.get("yaw_deg")))}°) &nbsp; '
            f'<b>2D in-frame median:</b> {html.escape(str(residual.get("median_px")))} px &nbsp; '
            f'<b>pad L/T/R/B:</b> {html.escape(_format_padding(padding))} px</p>'
            f'</article>'
        )
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Semantic Fusion V3 Pose Atlas v0.3</title>
<style>
body{font-family:system-ui,sans-serif;background:#0d0f12;color:#edf0f3;margin:24px}
header{max-width:1100px;margin:auto auto 28px}.card{max-width:1560px;margin:0 auto 34px;background:#171a1f;padding:18px;border-radius:12px}
.card img{display:block;width:100%;height:auto;border-radius:8px;background:#08090b}.card h2{margin:0 0 12px}.card p{color:#c9ced6}
code{background:#222831;padding:2px 5px;border-radius:4px}
</style></head><body><header><h1>Semantic Fusion V3 — Pose Atlas v0.3</h1>
<p>DWPose coordinates are treated explicitly as normalized image coordinates and SAM3D projected keypoints explicitly as original-image pixels; no coordinate-space autodetection is used.</p>
<p>The two 2D pose panels share padding derived from the union of DWPose-accepted and full SAM3D body projections, so joints reconstructed outside the source frame remain visible. Orange points lie outside the original image boundary. DWPose “accepted” means it passed that model's threshold, not that the joint was directly visible.</p>
<p>The in-frame residual is a cross-model agreement diagnostic, not ground truth.</p></header>
""" + "\n".join(cards) + "\n</body></html>\n"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else run_dir / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    dwpose_dir = base._resolve_dir(run_dir, args.dwpose_dir, ["dwpose", "dwpose-v1"], "dwpose")
    sam3d_dir = base._resolve_dir(run_dir, args.sam3d_dir, ["sam3d", "sam3d-probe"], "sam3d")
    if sam3d_dir is None:
        raise SystemExit("Could not find SAM3D cache directory; pass --sam3d-dir explicitly.")

    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "semantic-v3" / "pose-atlas-v0.3"
    )
    output.mkdir(parents=True, exist_ok=True)
    annotations = base._load_annotations(
        args.annotations.expanduser().resolve() if args.annotations else None
    )

    images = [
        p for p in base._discover_images(images_dir)
        if base._matches_only(base._image_key(p), p, args.only)
    ]
    if not images:
        raise SystemExit("No matching images found.")

    records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for image_path in images:
        key = base._image_key(image_path)
        sam_npz = sam3d_dir / f"{key}.sam3d_arrays.npz"
        if not sam_npz.exists():
            found = list(sam3d_dir.rglob(f"{key}.sam3d_arrays.npz"))
            sam_npz = found[0] if found else sam_npz
        if not sam_npz.exists():
            missing.append({"image_key": key, "reason": "missing_sam3d_arrays"})
            continue

        dwpose_path = (dwpose_dir / f"{key}.dwpose.json") if dwpose_dir else None
        if dwpose_path is not None and not dwpose_path.exists():
            found = list(dwpose_dir.rglob(f"{key}.dwpose.json")) if dwpose_dir else []
            dwpose_path = found[0] if found else None

        sam_obj = v02._resolve_mesh(sam3d_dir, key)

        out_webp = output / f"{key}.pose_atlas.webp"
        out_json = output / f"{key}.pose_atlas.json"
        if out_webp.exists() and out_json.exists() and not args.overwrite:
            record = base._read_json(out_json)
        else:
            card, record = _make_card(
                image_path,
                dwpose_path,
                sam_npz,
                sam_obj,
                annotations.get(key, {}),
            )
            card.save(
                out_webp,
                format="WEBP",
                quality=max(1, min(100, int(args.quality))),
                method=6,
            )
            record["card_webp"] = out_webp.name
            base._write_json(out_json, record)

        if "card_webp" not in record:
            record["card_webp"] = out_webp.name
        records.append(record)
        print(
            f"{key}: {out_webp} "
            f"mesh={'yes' if record.get('sam3d_mesh_available') else 'no'} "
            f"median2d={(record.get('projected_fit_residual') or {}).get('median_px')} "
            f"pad={_format_padding(record.get('display_padding') or {})}"
        )

    index = {
        "schema_version": "pose-atlas-v3-run-0.3",
        "run_dir": str(run_dir),
        "images_dir": str(images_dir),
        "dwpose_dir": str(dwpose_dir) if dwpose_dir else None,
        "sam3d_dir": str(sam3d_dir),
        "record_count": len(records),
        "missing": missing,
        "records": records,
    }
    base._write_json(output / "pose_atlas.index.json", index)
    (output / "index.html").write_text(_html_index(records), encoding="utf-8")
    print(f"Atlas: {output / 'index.html'}")
    if missing:
        print(f"Skipped {len(missing)} image(s) without cached SAM3D arrays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
