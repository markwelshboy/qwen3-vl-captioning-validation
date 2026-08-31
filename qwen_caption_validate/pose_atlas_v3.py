from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dwpose_profile import BODY18, IDX
from .sam3d_subject_geometry_diagnostic import CAMERA_SYSTEM_FLIP, MHR70, euler_zyx_to_rotmat
from .sam3d_subject_geometry_diagnostic_02 import build_subject_geometry
from .runner import IMAGE_EXTENSIONS


DWPOSE_EDGES = [
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_hip", "left_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
]

MHR_KNOWN_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "neck"),
]

PANEL_W = 520
PANEL_H = 520
TITLE_H = 34
GRID_COLS = 3
GRID_ROWS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-atlas-v3",
        description=(
            "Build a cached, no-new-VLM-call pose calibration atlas from original images, "
            "DWPose JSON, and SAM3D arrays/OBJ meshes."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory, e.g. runs/Caption02-02.")
    parser.add_argument("--images-dir", type=Path, help="Image directory. Defaults to <run_dir>/images.")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose cache directory. Auto-discovered when omitted.")
    parser.add_argument("--sam3d-dir", type=Path, help="SAM3D cache directory. Auto-discovered when omitted.")
    parser.add_argument("--annotations", type=Path, help="Optional human pose annotation JSON.")
    parser.add_argument("--output", type=Path, help="Output directory. Defaults to <run_dir>/semantic-v3/pose-atlas-v0.1.")
    parser.add_argument("--only", action="append", default=[], help="Image key/basename filter; repeatable.")
    parser.add_argument("--quality", type=int, default=88, help="WebP quality, 1-100 (default 88).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing atlas cards.")
    return parser.parse_args()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _font(size: int = 16, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _resolve_dir(run_dir: Path, supplied: Path | None, candidates: list[str], contains: str) -> Path | None:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Directory not found: {value}")
        return value
    for name in candidates:
        value = run_dir / name
        if value.is_dir():
            return value
    matches = sorted(p for p in run_dir.iterdir() if p.is_dir() and contains in p.name.lower())
    return matches[-1] if matches else None


def _discover_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _image_key(path: Path) -> str:
    return path.stem


def _matches_only(key: str, path: Path, only: list[str]) -> bool:
    if not only:
        return True
    values = {key.lower(), path.name.lower(), path.stem.lower()}
    wanted = {str(v).lower() for v in only}
    return bool(values & wanted)


def _fit_image(image: Image.Image, width: int, height: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    src = image.convert("RGB")
    scale = min(width / src.width, height / src.height)
    target = (max(1, int(round(src.width * scale))), max(1, int(round(src.height * scale))))
    resized = src.resize(target, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#101215")
    x = (width - target[0]) // 2
    y = (height - target[1]) // 2
    canvas.paste(resized, (x, y))
    return canvas, (x, y, target[0], target[1])


def _panel(title: str, body: Image.Image | None = None) -> Image.Image:
    out = Image.new("RGB", (PANEL_W, PANEL_H), "#101215")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, PANEL_W, TITLE_H), fill="#1c2026")
    draw.text((12, 8), title, fill="#f3f4f6", font=_font(16, bold=True))
    if body is not None:
        fitted, _ = _fit_image(body, PANEL_W, PANEL_H - TITLE_H)
        out.paste(fitted, (0, TITLE_H))
    return out


def _normalized_to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)[..., :2].copy()
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr).all(axis=-1)]
    if not finite.size:
        return arr
    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    if minimum >= -0.05 and maximum <= 1.25:
        arr[..., 0] *= width
        arr[..., 1] *= height
    elif minimum >= -1.25 and maximum <= 1.25:
        arr[..., 0] = (arr[..., 0] + 1.0) * 0.5 * width
        arr[..., 1] = (arr[..., 1] + 1.0) * 0.5 * height
    return arr


def _dwpose_target_points(record: dict[str, Any], width: int, height: int) -> np.ndarray:
    raw = record.get("raw_pose") or {}
    bodies = raw.get("bodies") or {}
    candidate = np.asarray(bodies.get("candidate", []), dtype=np.float64)
    if candidate.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if candidate.ndim == 2:
        candidate = candidate[None, ...]
    target_index = int(((record.get("derived") or {}).get("target_person_index") or 0))
    if target_index < 0 or target_index >= candidate.shape[0]:
        target_index = 0
    points = candidate[target_index, :18, :2]
    return _normalized_to_pixels(points, width, height)


def _point_visible(point: np.ndarray) -> bool:
    return bool(
        point.size >= 2
        and np.all(np.isfinite(point[:2]))
        and float(point[0]) >= 0.0
        and float(point[1]) >= 0.0
    )


def _draw_skeleton_on_image(image: Image.Image, points: np.ndarray, *, source: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    if source == "dwpose":
        edges = [(IDX[a], IDX[b]) for a, b in DWPOSE_EDGES]
    else:
        edges = [(MHR70[a], MHR70[b]) for a, b in MHR_KNOWN_EDGES]
    for a, b in edges:
        if a >= len(points) or b >= len(points):
            continue
        pa, pb = points[a], points[b]
        if _point_visible(pa) and _point_visible(pb):
            draw.line((float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1])), fill="#ffcc66", width=4)
    for point in points:
        if not _point_visible(point):
            continue
        x, y = float(point[0]), float(point[1])
        r = 4
        draw.ellipse((x-r, y-r, x+r, y+r), fill="#ffffff", outline="#111111")
    return out


def _load_obj_vertices(path: Path | None) -> np.ndarray:
    if path is None or not path.is_file():
        return np.empty((0, 3), dtype=np.float64)
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
    return np.asarray(vertices, dtype=np.float64)


def _body_frame_vertices(vertices: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    if vertices.size == 0:
        return vertices
    global_rot = np.asarray(arrays.get("global_rot"), dtype=np.float64).reshape(3)
    body_to_camera = CAMERA_SYSTEM_FLIP @ euler_zyx_to_rotmat(global_rot)
    return (body_to_camera.T @ np.asarray(vertices, dtype=np.float64).T).T


def _project_cloud(points: np.ndarray, axes: tuple[int, int], title: str) -> Image.Image:
    panel = _panel(title)
    draw = ImageDraw.Draw(panel)
    if points.size == 0:
        draw.text((16, 64), "No mesh available", fill="#c7cbd1", font=_font(15))
        return panel
    finite = points[np.isfinite(points).all(axis=1)]
    if not len(finite):
        return panel
    # Downsample dense meshes for compact diagnostics while retaining silhouette.
    step = max(1, len(finite) // 5000)
    cloud = finite[::step]
    x = cloud[:, axes[0]]
    y = cloud[:, axes[1]]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(1e-9, xmax-xmin), max(1e-9, ymax-ymin)
    margin = 28
    top = TITLE_H + margin
    bottom = PANEL_H - margin
    left = margin
    right = PANEL_W - margin
    scale = min((right-left)/dx, (bottom-top)/dy)
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    ox = (left+right)/2.0
    oy = (top+bottom)/2.0
    for px, py in zip(x, y):
        sx = ox + (float(px)-cx)*scale
        sy = oy - (float(py)-cy)*scale
        draw.point((sx, sy), fill="#d8dde5")
    return panel


def _draw_text_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, width_chars: int = 58, line_h: int = 22) -> int:
    x, y = xy
    words = str(text).split()
    line = ""
    lines: list[str] = []
    for word in words:
        trial = f"{line} {word}".strip()
        if len(trial) > width_chars and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    for value in lines:
        draw.text((x, y), value, fill="#e3e6ea", font=_font(15))
        y += line_h
    return y


def _summary_panel(key: str, diagnostic: dict[str, Any], dwpose: dict[str, Any], annotation: dict[str, Any]) -> Image.Image:
    panel = _panel("Semantic / calibration summary")
    draw = ImageDraw.Draw(panel)
    y = 52
    draw.text((14, y), key, fill="#ffffff", font=_font(18, bold=True)); y += 30
    body = diagnostic.get("body_camera_relation") or {}
    face = diagnostic.get("face_camera_relation") or {}
    target = ((dwpose.get("derived") or {}).get("target") or {})
    rows = [
        f"SAM3D body: {body.get('orientation_band') or 'unknown'}  yaw={body.get('yaw_deg')}  frame={body.get('faces_frame') or '-'}",
        f"SAM3D face: {face.get('orientation_band') or 'unknown'}  yaw={face.get('yaw_deg')}  turn→cam={face.get('head_turn_toward_camera_deg')}",
        f"DWPose extent: {target.get('pose_extent_hint') or 'unknown'}",
        f"DWPose visible: {target.get('visible_body_landmark_count') or 0}/18 body landmarks",
    ]
    for row in rows:
        y = _draw_text_wrapped(draw, (14, y), row); y += 3
    if annotation:
        y += 8
        draw.text((14, y), "Human target", fill="#ffffff", font=_font(16, bold=True)); y += 26
        pose = annotation.get("pose_family") or "unlabeled"
        mods = annotation.get("modifiers") or []
        gestalt = annotation.get("human_gestalt") or annotation.get("gestalt")
        y = _draw_text_wrapped(draw, (14, y), f"Pose: {pose}")
        if mods:
            y = _draw_text_wrapped(draw, (14, y+2), "Modifiers: " + ", ".join(str(v) for v in mods))
        if gestalt:
            y = _draw_text_wrapped(draw, (14, y+4), "Gestalt: " + str(gestalt), width_chars=54)
    else:
        y += 8
        draw.text((14, y), "Human target: not annotated", fill="#9aa1ab", font=_font(15))
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
    dwpose = _read_json(dwpose_path)
    with np.load(sam_npz_path) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    diagnostic = build_subject_geometry(arrays, dwpose or None)

    original = _panel("Original", image)

    dw_points = _dwpose_target_points(dwpose, width, height) if dwpose else np.empty((0, 2))
    dw_overlay = _panel("DWPose observed 2D", _draw_skeleton_on_image(image, dw_points, source="dwpose") if len(dw_points) else image)

    sam2d = np.asarray(arrays.get("pred_keypoints_2d", np.empty((0, 2))), dtype=np.float64)
    if sam2d.ndim > 2:
        sam2d = sam2d.reshape((-1, sam2d.shape[-1]))
    sam2d = _normalized_to_pixels(sam2d[:, :2], width, height) if sam2d.size else sam2d
    sam_overlay_image = _draw_skeleton_on_image(image, sam2d, source="sam3d") if len(sam2d) else image
    sam_overlay = _panel("SAM3D projected fit", sam_overlay_image)

    vertices_camera = _load_obj_vertices(sam_obj_path)
    vertices_body = _body_frame_vertices(vertices_camera, arrays) if len(vertices_camera) else np.empty((0, 3))
    mesh_front = _project_cloud(vertices_body, (0, 1), "SAM3D body-frame front")
    mesh_side = _project_cloud(vertices_body, (2, 1), "SAM3D body-frame side/depth")
    summary = _summary_panel(_image_key(image_path), diagnostic, dwpose, annotation)

    card = Image.new("RGB", (PANEL_W*GRID_COLS, PANEL_H*GRID_ROWS), "#0b0d10")
    for index, panel in enumerate([original, dw_overlay, sam_overlay, mesh_front, mesh_side, summary]):
        x = (index % GRID_COLS) * PANEL_W
        y = (index // GRID_COLS) * PANEL_H
        card.paste(panel, (x, y))

    record = {
        "schema_version": "pose-atlas-v3-record-0.1",
        "image_key": _image_key(image_path),
        "image": str(image_path),
        "dwpose": str(dwpose_path) if dwpose_path else None,
        "sam3d_arrays": str(sam_npz_path),
        "sam3d_mesh": str(sam_obj_path) if sam_obj_path else None,
        "sam3d_diagnostic": diagnostic,
        "dwpose_derived": dwpose.get("derived") if dwpose else None,
        "human_annotation": annotation or None,
        "interpretation_policy": {
            "purpose": "visual calibration of broad pose/crop semantics, not proof of invisible anatomy",
            "reconstruction_is_not_observation": True,
            "missing_out_of_crop_anatomy_is_not_counterevidence": True,
        },
    }
    return card, record


def _load_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        return {}
    return {str(k): v for k, v in records.items() if isinstance(v, dict)}


def _html_index(records: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for record in records:
        key = html.escape(str(record["image_key"]))
        webp = html.escape(str(record["card_webp"]))
        body = (record.get("sam3d_diagnostic") or {}).get("body_camera_relation") or {}
        ann = record.get("human_annotation") or {}
        human = ann.get("pose_family") or "unlabeled"
        cards.append(
            f'<article class="card"><h2>{key}</h2>'
            f'<img src="{webp}" loading="lazy" alt="Pose atlas card for {key}">'
            f'<p><b>Human:</b> {html.escape(str(human))} &nbsp; '
            f'<b>SAM3D:</b> {html.escape(str(body.get("orientation_band") or "unknown"))} '
            f'({html.escape(str(body.get("yaw_deg"))) }°)</p></article>'
        )
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Semantic Fusion V3 Pose Atlas</title>
<style>
body{font-family:system-ui,sans-serif;background:#0d0f12;color:#edf0f3;margin:24px}
header{max-width:1100px;margin:auto auto 28px}.card{max-width:1560px;margin:0 auto 34px;background:#171a1f;padding:18px;border-radius:12px}
.card img{display:block;width:100%;height:auto;border-radius:8px;background:#08090b}.card h2{margin:0 0 12px}.card p{color:#c9ced6}
code{background:#222831;padding:2px 5px;border-radius:4px}
</style></head><body><header><h1>Semantic Fusion V3 — Pose Atlas v0.1</h1>
<p>Cached visual calibration only: original image, DWPose observation, SAM3D projected fit, and body-frame mesh views. No new VLM call is made.</p>
<p>Use this atlas to judge <em>semantic pose family</em> and crop compatibility rather than invisible-joint perfection.</p></header>
""" + "\n".join(cards) + "\n</body></html>\n"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    images_dir = (args.images_dir.expanduser().resolve() if args.images_dir else run_dir / "images")
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    dwpose_dir = _resolve_dir(run_dir, args.dwpose_dir, ["dwpose", "dwpose-v1"], "dwpose")
    sam3d_dir = _resolve_dir(run_dir, args.sam3d_dir, ["sam3d", "sam3d-probe"], "sam3d")
    if sam3d_dir is None:
        raise SystemExit("Could not find SAM3D cache directory; pass --sam3d-dir explicitly.")

    output = (args.output.expanduser().resolve() if args.output else run_dir / "semantic-v3" / "pose-atlas-v0.1")
    output.mkdir(parents=True, exist_ok=True)
    annotations = _load_annotations(args.annotations.expanduser().resolve() if args.annotations else None)

    images = [p for p in _discover_images(images_dir) if _matches_only(_image_key(p), p, args.only)]
    if not images:
        raise SystemExit("No matching images found.")

    records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for image_path in images:
        key = _image_key(image_path)
        sam_npz = sam3d_dir / f"{key}.sam3d_arrays.npz"
        if not sam_npz.exists():
            matches = list(sam3d_dir.rglob(f"{key}.sam3d_arrays.npz"))
            sam_npz = matches[0] if matches else sam_npz
        if not sam_npz.exists():
            missing.append({"image_key": key, "reason": "missing_sam3d_arrays"})
            continue
        dwpose_path = (dwpose_dir / f"{key}.dwpose.json") if dwpose_dir else None
        if dwpose_path is not None and not dwpose_path.exists():
            matches = list(dwpose_dir.rglob(f"{key}.dwpose.json")) if dwpose_dir else []
            dwpose_path = matches[0] if matches else None
        sam_obj = sam3d_dir / f"{key}.sam3d.obj"
        if not sam_obj.exists():
            matches = list(sam3d_dir.rglob(f"{key}.sam3d.obj"))
            sam_obj = matches[0] if matches else None

        out_webp = output / f"{key}.pose_atlas.webp"
        out_json = output / f"{key}.pose_atlas.json"
        if out_webp.exists() and out_json.exists() and not args.overwrite:
            record = _read_json(out_json)
        else:
            card, record = _make_card(image_path, dwpose_path, sam_npz, sam_obj, annotations.get(key, {}))
            card.save(out_webp, format="WEBP", quality=max(1, min(100, int(args.quality))), method=6)
            record["card_webp"] = out_webp.name
            _write_json(out_json, record)
        if "card_webp" not in record:
            record["card_webp"] = out_webp.name
        records.append(record)
        print(f"{key}: {out_webp}")

    index = {
        "schema_version": "pose-atlas-v3-run-0.1",
        "run_dir": str(run_dir),
        "images_dir": str(images_dir),
        "dwpose_dir": str(dwpose_dir) if dwpose_dir else None,
        "sam3d_dir": str(sam3d_dir),
        "record_count": len(records),
        "missing": missing,
        "records": records,
    }
    _write_json(output / "pose_atlas.index.json", index)
    (output / "index.html").write_text(_html_index(records), encoding="utf-8")
    print(f"Atlas: {output / 'index.html'}")
    if missing:
        print(f"Skipped {len(missing)} image(s) without cached SAM3D arrays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
