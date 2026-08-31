from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import pose_atlas_v3 as base
from .pose_atlas_v3_compat import _dwpose_target_points_compat
from .sam3d_subject_geometry_diagnostic_02 import build_subject_geometry


MHR_BODY = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
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
    "left_big_toe": 15,
    "left_heel": 17,
    "right_big_toe": 18,
    "right_heel": 20,
    "right_wrist": 41,
    "left_wrist": 62,
    "neck": 69,
}

MHR_BODY_EDGES = [
    ("nose", "neck"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_big_toe"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_big_toe"),
]

DW_TO_MHR = {
    "nose": "nose",
    "neck": "neck",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",
    "right_eye": "right_eye",
    "left_eye": "left_eye",
    "right_ear": "right_ear",
    "left_ear": "left_ear",
}
MHR_TO_DW = {mhr_name: dw_name for dw_name, mhr_name in DW_TO_MHR.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-atlas-v3-02",
        description=(
            "Build Pose Atlas v0.2 from cached image, DWPose and SAM3D evidence. "
            "Adds historical-cache compatibility, observed-joint filtering, "
            "2D reprojection residuals and a 3D skeleton fallback when OBJ meshes "
            "were not retained."
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


def _visible_dwpose_names(record: dict[str, Any]) -> set[str]:
    target = ((record.get("derived") or {}).get("target") or {})
    return {
        str(name)
        for name in (target.get("visible_body_landmarks") or [])
        if isinstance(name, str)
    }


def _masked_dwpose_points(
    record: dict[str, Any], width: int, height: int
) -> tuple[np.ndarray, set[str]]:
    points = _dwpose_target_points_compat(record, width, height)
    visible = _visible_dwpose_names(record)
    if not len(points) or not visible:
        return points, visible
    masked = np.asarray(points, dtype=np.float64).copy()
    for name, idx in base.IDX.items():
        if idx < len(masked) and name not in visible:
            masked[idx, :2] = -1.0
    return masked, visible


def _sam2d_points(arrays: dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    value = np.asarray(arrays.get("pred_keypoints_2d", np.empty((0, 2))), dtype=np.float64)
    if value.ndim > 2:
        value = value.reshape((-1, value.shape[-1]))
    if value.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return base._normalized_to_pixels(value[:, :2], width, height)


def _draw_dwpose(image: Image.Image, points: np.ndarray) -> Image.Image:
    return base._draw_skeleton_on_image(image, points, source="dwpose")


def _draw_sam3d_observed_fit(
    image: Image.Image,
    sam2d: np.ndarray,
    visible_dwpose: set[str],
) -> Image.Image:
    """Draw only the SAM3D joints whose body regions DWPose actually observed.

    The full SAM3D reconstruction remains available in the lower 3D panels.  On
    the source image, hiding reconstruction-only knees/ankles/hips avoids making
    latent joints look like observed reference points in tight crops.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    def point(name: str) -> np.ndarray | None:
        idx = MHR_BODY.get(name)
        if idx is None or idx >= len(sam2d):
            return None
        p = np.asarray(sam2d[idx], dtype=np.float64)
        if not base._point_visible(p):
            return None
        if visible_dwpose:
            dw_name = MHR_TO_DW.get(name)
            if dw_name is None or dw_name not in visible_dwpose:
                return None
        return p

    for a, b in MHR_BODY_EDGES:
        pa, pb = point(a), point(b)
        if pa is not None and pb is not None:
            draw.line(
                (float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1])),
                fill="#ffcc66",
                width=4,
            )
    for name in MHR_BODY:
        p = point(name)
        if p is None:
            continue
        x, y = float(p[0]), float(p[1])
        r = 4
        draw.ellipse((x-r, y-r, x+r, y+r), fill="#ffffff", outline="#111111")
    return out


def _bbox_diag_pixels(dwpose: dict[str, Any], width: int, height: int) -> float | None:
    target = ((dwpose.get("derived") or {}).get("target") or {})
    bbox = target.get("keypoint_bbox") or {}
    try:
        bw = float(bbox["width_fraction"]) * width
        bh = float(bbox["height_fraction"]) * height
    except (KeyError, TypeError, ValueError):
        return None
    diag = math.hypot(bw, bh)
    return diag if diag > 1e-9 else None


def _reprojection_residual(
    dw_points: np.ndarray,
    sam2d: np.ndarray,
    visible_dwpose: set[str],
    *,
    width: int,
    height: int,
    dwpose: dict[str, Any],
) -> dict[str, Any]:
    """Measure SAM3D projected-keypoint agreement with DWPose-observed joints.

    This is deliberately *not* called an error to ground truth: DWPose is itself
    a model observation.  The number is useful for spotting coordinate-space
    mistakes and unusually poor cross-model fits in the calibration atlas.
    """
    errors: list[float] = []
    per_joint: dict[str, float] = {}
    for dw_name, mhr_name in DW_TO_MHR.items():
        if visible_dwpose and dw_name not in visible_dwpose:
            continue
        di = base.IDX.get(dw_name)
        si = MHR_BODY.get(mhr_name)
        if di is None or si is None or di >= len(dw_points) or si >= len(sam2d):
            continue
        dp = np.asarray(dw_points[di], dtype=np.float64)
        sp = np.asarray(sam2d[si], dtype=np.float64)
        if not base._point_visible(dp) or not base._point_visible(sp):
            continue
        distance = float(np.linalg.norm(dp[:2] - sp[:2]))
        if not math.isfinite(distance):
            continue
        errors.append(distance)
        per_joint[dw_name] = round(distance, 2)

    if not errors:
        return {
            "common_joint_count": 0,
            "median_px": None,
            "mean_px": None,
            "median_fraction_of_dwpose_bbox_diagonal": None,
            "per_joint_px": {},
            "interpretation": "unavailable",
        }

    median = float(np.median(errors))
    mean = float(np.mean(errors))
    diag = _bbox_diag_pixels(dwpose, width, height)
    fraction = median / diag if diag else None
    return {
        "common_joint_count": len(errors),
        "median_px": round(median, 2),
        "mean_px": round(mean, 2),
        "median_fraction_of_dwpose_bbox_diagonal": (
            round(float(fraction), 4) if fraction is not None else None
        ),
        "per_joint_px": per_joint,
        "interpretation": (
            "DWPose↔SAM3D projected agreement only; DWPose is not ground-truth annotation."
        ),
    }


def _camera_keypoints_3d(arrays: dict[str, np.ndarray]) -> np.ndarray:
    value = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if value.ndim > 2:
        value = value.reshape((-1, value.shape[-1]))
    return value[:, :3] if value.ndim == 2 and value.shape[-1] >= 3 else np.empty((0, 3))


def _draw_3d_pose_panel(
    keypoints: np.ndarray,
    mesh_vertices: np.ndarray,
    axes: tuple[int, int],
    title: str,
) -> Image.Image:
    """Render cached SAM3D 3D keypoints, with the OBJ as optional context.

    The skeleton is always available from the cached NPZ.  Historical runs that
    did not retain ``pred_vertices``/OBJ files therefore remain useful for pose
    calibration without another SAM3D inference pass.
    """
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
        draw.line((*project(pa), *project(pb)), fill="#ffcc66", width=4)

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


def _summary_panel(
    key: str,
    diagnostic: dict[str, Any],
    dwpose: dict[str, Any],
    annotation: dict[str, Any],
    residual: dict[str, Any],
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
    rows = [
        f"SAM3D body: {body.get('orientation_band') or 'unknown'}  yaw={body.get('yaw_deg')}  frame={body.get('faces_frame') or '-'}",
        f"SAM3D face: {face.get('orientation_band') or 'unknown'}  yaw={face.get('yaw_deg')}  turn→cam={face.get('head_turn_toward_camera_deg')}",
        f"DWPose extent: {target.get('pose_extent_hint') or 'unknown'}",
        f"DWPose visible: {target.get('visible_body_landmark_count') or 0}/18 body landmarks",
        (
            f"2D fit residual: median={residual.get('median_px')} px  "
            f"bbox-frac={residual_fraction}  n={residual.get('common_joint_count')}"
        ),
        f"3D display: {'OBJ mesh + skeleton' if mesh_available else 'cached 3D skeleton (mesh unavailable)'}",
    ]
    for row in rows:
        y = base._draw_text_wrapped(draw, (14, y), row)
        y += 3

    if annotation:
        y += 8
        draw.text((14, y), "Human target", fill="#ffffff", font=base._font(16, bold=True))
        y += 26
        pose = annotation.get("pose_family") or "unlabeled"
        mods = annotation.get("modifiers") or []
        gestalt = annotation.get("human_gestalt") or annotation.get("gestalt")
        y = base._draw_text_wrapped(draw, (14, y), f"Pose: {pose}")
        if mods:
            y = base._draw_text_wrapped(draw, (14, y+2), "Modifiers: " + ", ".join(str(v) for v in mods))
        if gestalt:
            base._draw_text_wrapped(draw, (14, y+4), "Gestalt: " + str(gestalt), width_chars=54)
    else:
        y += 8
        draw.text((14, y), "Human target: not annotated", fill="#9aa1ab", font=base._font(15))
    return panel


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
    dw_points, visible_names = (
        _masked_dwpose_points(dwpose, width, height)
        if dwpose else (np.empty((0, 2)), set())
    )
    sam2d = _sam2d_points(arrays, width, height)
    residual = _reprojection_residual(
        dw_points,
        sam2d,
        visible_names,
        width=width,
        height=height,
        dwpose=dwpose,
    ) if dwpose and len(dw_points) and len(sam2d) else {
        "common_joint_count": 0,
        "median_px": None,
        "mean_px": None,
        "median_fraction_of_dwpose_bbox_diagonal": None,
        "per_joint_px": {},
        "interpretation": "unavailable",
    }

    original = base._panel("Original", image)
    dw_overlay_img = _draw_dwpose(image, dw_points) if len(dw_points) else image
    dw_overlay = base._panel("DWPose observed 2D", dw_overlay_img)
    sam_overlay_img = (
        _draw_sam3d_observed_fit(image, sam2d, visible_names)
        if len(sam2d) else image
    )
    sam_overlay = base._panel("SAM3D fit on DWPose-observed joints", sam_overlay_img)

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
        "schema_version": "pose-atlas-v3-record-0.2",
        "image_key": base._image_key(image_path),
        "image": str(image_path),
        "dwpose": str(dwpose_path) if dwpose_path else None,
        "sam3d_arrays": str(sam_npz_path),
        "sam3d_mesh": str(sam_obj_path) if sam_obj_path else None,
        "sam3d_mesh_available": bool(len(mesh)),
        "sam3d_diagnostic": diagnostic,
        "dwpose_derived": dwpose.get("derived") if dwpose else None,
        "projected_fit_residual": residual,
        "human_annotation": annotation or None,
        "interpretation_policy": {
            "purpose": "visual calibration of broad pose/crop semantics, not proof of invisible anatomy",
            "reconstruction_is_not_observation": True,
            "missing_out_of_crop_anatomy_is_not_counterevidence": True,
            "sam3d_image_overlay_is_restricted_to_dwpose_observed_joint_names": True,
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
        cards.append(
            f'<article class="card"><h2>{key}</h2>'
            f'<img src="{webp}" loading="lazy" alt="Pose atlas card for {key}">'
            f'<p><b>Human:</b> {html.escape(str(human))} &nbsp; '
            f'<b>SAM3D:</b> {html.escape(str(body.get("orientation_band") or "unknown"))} '
            f'({html.escape(str(body.get("yaw_deg")))}°) &nbsp; '
            f'<b>2D median residual:</b> {html.escape(str(residual.get("median_px")))} px</p>'
            f'</article>'
        )
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Semantic Fusion V3 Pose Atlas v0.2</title>
<style>
body{font-family:system-ui,sans-serif;background:#0d0f12;color:#edf0f3;margin:24px}
header{max-width:1100px;margin:auto auto 28px}.card{max-width:1560px;margin:0 auto 34px;background:#171a1f;padding:18px;border-radius:12px}
.card img{display:block;width:100%;height:auto;border-radius:8px;background:#08090b}.card h2{margin:0 0 12px}.card p{color:#c9ced6}
code{background:#222831;padding:2px 5px;border-radius:4px}
</style></head><body><header><h1>Semantic Fusion V3 — Pose Atlas v0.2</h1>
<p>Cached calibration only. The image overlay shows DWPose-observed joints separately from SAM3D reconstruction. Full cached SAM3D 3D keypoints are always shown, even if historical OBJ meshes were not retained.</p>
<p>The 2D residual is a cross-model agreement diagnostic, not ground truth. Use the 3D views to judge broad pose family and crop compatibility rather than invisible-joint perfection.</p></header>
""" + "\n".join(cards) + "\n</body></html>\n"


def _resolve_mesh(sam3d_dir: Path, key: str) -> Path | None:
    direct = sam3d_dir / f"{key}.sam3d.obj"
    if direct.is_file():
        return direct
    matches = list(sam3d_dir.rglob(f"{key}.sam3d.obj"))
    if matches:
        return matches[0]

    record_path = sam3d_dir / f"{key}.sam3d.json"
    if not record_path.is_file():
        found = list(sam3d_dir.rglob(f"{key}.sam3d.json"))
        record_path = found[0] if found else record_path
    record = base._read_json(record_path)
    mesh_obj = record.get("mesh_obj") if record else None
    if mesh_obj:
        candidate = Path(str(mesh_obj)).expanduser()
        if candidate.is_file():
            return candidate
        basename = candidate.name
        found = list(sam3d_dir.rglob(basename))
        if found:
            return found[0]
    return None


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
        else run_dir / "semantic-v3" / "pose-atlas-v0.2"
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

        sam_obj = _resolve_mesh(sam3d_dir, key)

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
            f"median2d={(record.get('projected_fit_residual') or {}).get('median_px')}"
        )

    index = {
        "schema_version": "pose-atlas-v3-run-0.2",
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
