from __future__ import annotations

"""Patch-level vertical-axis correction for Pose Atlas v0.3.

SAM3D ``pred_keypoints_3d`` uses camera/image-style Y: increasing Y projects
farther down the image.  Pose Atlas v0.3 inherited a Cartesian plotting
convention and negated Y, which made both the camera and side/depth 3D panels
appear vertically inverted.  Keep v0.3's data/epistemic logic unchanged and
replace only the 3D display projection.
"""

import numpy as np
from PIL import Image, ImageDraw

from . import pose_atlas_v3 as base
from . import pose_atlas_v3_03 as v03


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
        idx for idx in v03.MHR_BODY.values()
        if idx < len(keypoints) and np.all(np.isfinite(keypoints[idx]))
    ]
    if not usable_indices:
        draw.text(
            (16, 64),
            "No SAM3D 3D keypoints available",
            fill="#c7cbd1",
            font=base._font(15),
        )
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
    dx, dy = max(1e-9, xmax - xmin), max(1e-9, ymax - ymin)
    scale = min((right - left) / dx, (bottom - top) / dy)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    ox = (left + right) / 2.0
    oy = (top + bottom) / 2.0

    def project(p: np.ndarray) -> tuple[float, float]:
        # SAM3D camera-space Y has the same screen direction as its 2D
        # projection: larger Y is lower in the image.  Do NOT apply the usual
        # Cartesian-plot ``oy - y`` inversion here.
        return (
            ox + (float(p[axes[0]]) - cx) * scale,
            oy + (float(p[axes[1]]) - cy) * scale,
        )

    if len(finite_mesh):
        step = max(1, len(finite_mesh) // 4500)
        for p in finite_mesh[::step]:
            draw.point(project(p), fill="#59616d")

    for a, b in v03.MHR_BODY_EDGES:
        ia, ib = v03.MHR_BODY[a], v03.MHR_BODY[b]
        if ia >= len(keypoints) or ib >= len(keypoints):
            continue
        pa, pb = keypoints[ia], keypoints[ib]
        if not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)):
            continue
        draw.line((*project(pa), *project(pb)), fill=v03.SAM3D_COLOR, width=4)

    for name, idx in v03.MHR_BODY.items():
        if idx >= len(keypoints) or not np.all(np.isfinite(keypoints[idx])):
            continue
        x2, y2 = project(keypoints[idx])
        r = 4 if "toe" not in name and "heel" not in name else 3
        draw.ellipse(
            (x2 - r, y2 - r, x2 + r, y2 + r),
            fill="#ffffff",
            outline="#111111",
        )

    if not len(finite_mesh):
        draw.text(
            (14, base.PANEL_H - 24),
            "3D skeleton fallback (OBJ mesh not retained)",
            fill="#9aa1ab",
            font=base._font(13),
        )
    return panel


# Patch only the v0.3 display function. All coordinate contracts, padding,
# residuals, JSON schemas, and output paths remain v0.3-compatible.
v03._draw_3d_pose_panel = _draw_3d_pose_panel


def main() -> int:
    return v03.main()


if __name__ == "__main__":
    raise SystemExit(main())
