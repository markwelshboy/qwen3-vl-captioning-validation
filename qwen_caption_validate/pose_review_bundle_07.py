from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import pose_review_bundle as base
from . import pose_review_bundle_06 as _v06  # noqa: F401  (applies prior bundle extensions)
from . import pose_atlas_v3_03 as atlas03
from . import sam3d_relational_pose_profile_12 as v12


_v06_parse_args = base.parse_args
_v06_find_profile_dir = base._find_profile_dir
_v06_compact_record = base._compact_record

VIEW_BG = "#080a0d"
VIEW_LINE = "#ffbf3f"
VIEW_JOINT = "#ffd978"
VIEW_SUPPORT = "#36d7ff"
VIEW_MUTED = "#8b96a3"


def _parse_args_v07():
    args = _v06_parse_args()
    default_v06 = args.run_dir / "semantic-v3" / "pose-review-v0.6"
    if args.output == default_v06:
        args.output = args.run_dir / "semantic-v3" / "pose-review-v0.7"
    return args


def _find_profile_dir_v12(run_dir: Path, supplied: Path | None, sam3d_dir: Path) -> Path:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Profile directory not found: {value}")
        return value
    preferred = sam3d_dir / "relational-pose-profile-v0.12"
    if preferred.is_dir():
        return preferred
    return _v06_find_profile_dir(run_dir, supplied, sam3d_dir)


def _compact_record_v07(
    key: str,
    profile: dict[str, Any],
    original_rel: str,
    overlay_rel: str,
    raw_rel: str,
    overlay_meta: dict[str, Any],
) -> dict[str, Any]:
    record = _v06_compact_record(
        key, profile, original_rel, overlay_rel, raw_rel, overlay_meta
    )
    projected = profile.get("sam3d_projected_pose") or {}
    relations = profile.get("relations") or {}
    head = relations.get("head_supported_by_hand") or {}
    record["support_area_diagnostic"] = projected.get("support_area_diagnostic") or {}
    record["head_support_topology_guard"] = head.get("support_topology_guard") or {}
    record["head_support_topology_rejection_reason"] = head.get("rejection_reason")
    record["diagnostic_profile_schema"] = profile.get("schema_version")
    return record


base.parse_args = _parse_args_v07
base._find_profile_dir = _find_profile_dir_v12
base._compact_record = _compact_record_v07


def _keypoints(arrays: dict[str, np.ndarray]) -> np.ndarray:
    points = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if points.ndim > 2:
        points = points.reshape((-1, points.shape[-1]))
    return points[:, :3]


def _body_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for a, b in atlas03.MHR_BODY_EDGES:
        ia, ib = atlas03.MHR_BODY.get(a), atlas03.MHR_BODY.get(b)
        if ia is not None and ib is not None:
            edges.append((ia, ib))
    return edges


def _fit_transform(points: np.ndarray, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    finite = points[np.all(np.isfinite(points), axis=1)]
    if not len(finite):
        return lambda p: (float(x0), float(y0))
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.10
    mins -= span * pad
    maxs += span * pad
    span = np.maximum(maxs - mins, 1e-6)
    scale = min((x1 - x0) / span[0], (y1 - y0) / span[1])
    cx = (mins[0] + maxs[0]) / 2.0
    cy = (mins[1] + maxs[1]) / 2.0
    bx = (x0 + x1) / 2.0
    by = (y0 + y1) / 2.0

    def transform(p):
        return (
            bx + (float(p[0]) - cx) * scale,
            by + (float(p[1]) - cy) * scale,
        )
    return transform


def _draw_skeleton(
    draw: ImageDraw.ImageDraw,
    projected: np.ndarray,
    transform,
    edges: list[tuple[int, int]],
) -> None:
    for ia, ib in edges:
        if max(ia, ib) >= len(projected):
            continue
        pa, pb = projected[ia], projected[ib]
        if not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)):
            continue
        draw.line((*transform(pa), *transform(pb)), fill=VIEW_LINE, width=4)
    for p in projected:
        if not np.all(np.isfinite(p)):
            continue
        x, y = transform(p)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=VIEW_JOINT)


def _draw_sam3d_views(arrays: dict[str, np.ndarray]) -> Image.Image:
    points = _keypoints(arrays)
    canvas = Image.new("RGB", (920, 540), VIEW_BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "SAM3D side view (depth / vertical)", fill=VIEW_MUTED)
    draw.text((480, 12), "SAM3D support plane (X / depth)", fill=VIEW_MUTED)

    if not len(points):
        draw.text((20, 50), "No 3D keypoints", fill=VIEW_MUTED)
        return canvas

    body_indices = sorted(set(atlas03.MHR_BODY.values()))
    body = np.full((max(body_indices) + 1, 3), np.nan, dtype=np.float64)
    for idx in body_indices:
        if idx < len(points):
            body[idx] = points[idx]

    side = body[:, [2, 1]]
    top = body[:, [0, 2]]
    side_tf = _fit_transform(side, (20, 42, 440, 515))
    top_tf = _fit_transform(top, (480, 42, 900, 515))
    edges = _body_edges()
    _draw_skeleton(draw, side, side_tf, edges)
    _draw_skeleton(draw, top, top_tf, edges)

    support = v12._support_area_geometry(points)
    hull = support.get("support_hull_xz_shoulder_widths") or []
    shoulder_width = float(support.get("shoulder_width_3d") or 0.0)
    if hull and shoulder_width > 1e-9:
        top_norm = top / shoulder_width
        top_tf_norm = _fit_transform(top_norm, (480, 42, 900, 515))
        _draw_skeleton(draw, top_norm, top_tf_norm, edges)
        hp = [top_tf_norm(np.asarray(p, dtype=np.float64)) for p in hull]
        if len(hp) >= 3:
            draw.polygon(hp, outline=VIEW_SUPPORT)
        elif len(hp) == 2:
            draw.line((*hp[0], *hp[1]), fill=VIEW_SUPPORT, width=5)
        draw.text((490, 490), "cyan = reconstructed foot support area", fill=VIEW_SUPPORT)

    draw.line((460, 36, 460, 520), fill="#303740", width=1)
    return canvas


def _add_side_views(args) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    sam3d_dir = (args.sam3d_dir or (run_dir / "sam3d-pose-discovery-01")).expanduser().resolve()
    index_path = output / "pose_review.index.json"
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    media_side = output / "media" / "side"
    media_side.mkdir(parents=True, exist_ok=True)

    for record in index.get("records") or []:
        key = str(record.get("image_key") or "")
        if not key:
            continue
        npz = sam3d_dir / f"{key}.sam3d_arrays.npz"
        if not npz.is_file():
            matches = list(sam3d_dir.rglob(f"{key}.sam3d_arrays.npz"))
            npz = matches[0] if matches else npz
        if not npz.is_file():
            continue
        out = media_side / f"{key}.sam3d_views.webp"
        if args.overwrite or not out.exists():
            with np.load(npz) as loaded:
                arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
            view = _draw_sam3d_views(arrays)
            view.save(out, format="WEBP", quality=max(1, min(100, int(args.quality))), method=6)
        record["side_view"] = f"media/side/{out.name}"

    index["schema_version"] = "pose-review-bundle-0.7"
    index["includes_sam3d_side_and_support_view"] = True
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = base.parse_args()
    result = base.main()
    _add_side_views(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
