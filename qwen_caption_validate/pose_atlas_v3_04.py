from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import pose_atlas_v3 as base
from . import pose_atlas_v3_03 as v03
from . import pose_atlas_v3_03_orientation as _orientation  # noqa: F401  (patches v03 3D Y display)
from . import sam3d_hand_geometry as hand
from . import sam3d_relational_pose_profile_02 as relprof
from .sam3d_subject_geometry_diagnostic_02 import build_subject_geometry


DISPLAY_GUTTER_PX = 8
HAND_MARGIN_SCALE = 1.70
HAND_MIN_WINDOW_PX = 140
DWPOSE_HAND_COLOR = "#66d9ef"
SAM3D_HAND_COLOR = "#ffd166"
OUT_OF_FRAME_FILL = "#ff9f43"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-atlas-v3-04",
        description=(
            "Build Pose Atlas v0.4 with padded body overlays, full SAM3D/DWPose "
            "finger skeletons, hand-detail zooms, and relational-pose diagnostics."
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


def _display_padding_with_hands(
    width: int,
    height: int,
    dw_points: np.ndarray,
    accepted_dwpose: set[str],
    dw_hands: dict[str, dict[str, np.ndarray]],
    sam2d: np.ndarray,
) -> dict[str, Any]:
    points: list[np.ndarray] = []
    for name in accepted_dwpose:
        idx = base.IDX.get(name)
        if idx is not None and idx < len(dw_points) and v03._finite_xy(dw_points[idx]):
            points.append(np.asarray(dw_points[idx], dtype=np.float64))

    for side in ("left", "right"):
        decoded = dw_hands.get(side) or {}
        hp = np.asarray(decoded.get("points", np.empty((0, 2))), dtype=np.float64)
        accepted = np.asarray(decoded.get("accepted_mask", np.empty((0,), dtype=bool)), dtype=bool)
        for index in range(min(len(hp), len(accepted), 21)):
            if bool(accepted[index]) and v03._finite_xy(hp[index]):
                points.append(hp[index])

    body_indices = set(v03.MHR_BODY.values())
    hand_indices = set(hand.mhr_hand_order("left")) | set(hand.mhr_hand_order("right"))
    for idx in sorted(body_indices | hand_indices):
        if idx < len(sam2d) and v03._finite_xy(sam2d[idx]):
            points.append(np.asarray(sam2d[idx], dtype=np.float64))

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
        "source": "union_of_dwpose_accepted_body_and_hands_plus_sam3d_mhr_body_and_hands",
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


def _padding_offset(padding: dict[str, Any]) -> tuple[int, int]:
    pad = padding.get("display_padding_px") or {}
    return int(pad.get("left") or 0), int(pad.get("top") or 0)


def _draw_dwpose_with_hands(
    image: Image.Image,
    dw_points: np.ndarray,
    accepted_names: set[str],
    dw_hands: dict[str, dict[str, np.ndarray]],
    padding: dict[str, Any],
) -> Image.Image:
    out = v03._draw_dwpose_padded(image, dw_points, accepted_names, padding)
    draw = ImageDraw.Draw(out)
    offset = _padding_offset(padding)

    for side in ("left", "right"):
        decoded = dw_hands[side]
        points = np.asarray(decoded["points"], dtype=np.float64)
        accepted = np.asarray(decoded["accepted_mask"], dtype=bool)
        if len(points) < 21 or len(accepted) < 21:
            continue
        for a, b in hand.dw_hand_edges():
            if not bool(accepted[a]) or not bool(accepted[b]):
                continue
            pa, pb = points[a], points[b]
            if v03._finite_xy(pa) and v03._finite_xy(pb):
                draw.line((*v03._shift(pa, offset), *v03._shift(pb, offset)), fill=DWPOSE_HAND_COLOR, width=3)
        for index in range(21):
            if not bool(accepted[index]) or not v03._finite_xy(points[index]):
                continue
            x, y = v03._shift(points[index], offset)
            r = 3
            fill = "#ffffff" if v03._in_frame(points[index], image.width, image.height) else OUT_OF_FRAME_FILL
            draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline="#111111")
    return out


def _draw_sam3d_with_hands(
    image: Image.Image,
    sam2d: np.ndarray,
    padding: dict[str, Any],
) -> Image.Image:
    out = v03._draw_sam3d_padded(image, sam2d, padding)
    draw = ImageDraw.Draw(out)
    offset = _padding_offset(padding)
    for side in ("left", "right"):
        for ia, ib in hand.mhr_hand_edges(side):
            if ia >= len(sam2d) or ib >= len(sam2d):
                continue
            pa, pb = sam2d[ia], sam2d[ib]
            if v03._finite_xy(pa) and v03._finite_xy(pb):
                draw.line((*v03._shift(pa, offset), *v03._shift(pb, offset)), fill=SAM3D_HAND_COLOR, width=3)
        for idx in hand.mhr_hand_order(side):
            if idx >= len(sam2d) or not v03._finite_xy(sam2d[idx]):
                continue
            p = sam2d[idx]
            x, y = v03._shift(p, offset)
            r = 3
            fill = "#ffffff" if v03._in_frame(p, image.width, image.height) else OUT_OF_FRAME_FILL
            draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline="#111111")
    return out


def _hand_window(
    side: str,
    dw_hand: dict[str, np.ndarray],
    sam2d: np.ndarray,
) -> tuple[int, int, int, int] | None:
    points: list[np.ndarray] = []
    hp = np.asarray(dw_hand.get("points", np.empty((0, 2))), dtype=np.float64)
    accepted = np.asarray(dw_hand.get("accepted_mask", np.empty((0,), dtype=bool)), dtype=bool)
    for index in range(min(21, len(hp), len(accepted))):
        if bool(accepted[index]) and v03._finite_xy(hp[index]):
            points.append(hp[index])
    for idx in hand.mhr_hand_order(side):
        if idx < len(sam2d) and v03._finite_xy(sam2d[idx]):
            points.append(np.asarray(sam2d[idx], dtype=np.float64))
    if not points:
        return None
    cloud = np.stack(points, axis=0)
    x0, y0 = np.min(cloud[:, :2], axis=0)
    x1, y1 = np.max(cloud[:, :2], axis=0)
    span = max(float(x1 - x0), float(y1 - y0), float(HAND_MIN_WINDOW_PX)) * HAND_MARGIN_SCALE
    cx, cy = float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)
    half = span / 2.0
    return (
        int(math.floor(cx - half)),
        int(math.floor(cy - half)),
        int(math.ceil(cx + half)),
        int(math.ceil(cy + half)),
    )


def _draw_hand_crop(
    image: Image.Image,
    side: str,
    window: tuple[int, int, int, int] | None,
    source: str,
    dw_hand: dict[str, np.ndarray],
    sam2d: np.ndarray,
    size: tuple[int, int],
) -> Image.Image:
    if window is None:
        out = Image.new("RGB", size, "#101215")
        ImageDraw.Draw(out).text((12, 12), f"{side}: unavailable", fill="#9aa1ab", font=base._font(14))
        return out

    x0, y0, x1, y1 = window
    crop = image.convert("RGB").crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)

    def local(p: np.ndarray) -> tuple[float, float]:
        return float(p[0]) - x0, float(p[1]) - y0

    if source == "dwpose":
        points = np.asarray(dw_hand.get("points", np.empty((0, 2))), dtype=np.float64)
        accepted = np.asarray(dw_hand.get("accepted_mask", np.empty((0,), dtype=bool)), dtype=bool)
        if len(points) >= 21 and len(accepted) >= 21:
            for a, b in hand.dw_hand_edges():
                if bool(accepted[a]) and bool(accepted[b]) and v03._finite_xy(points[a]) and v03._finite_xy(points[b]):
                    draw.line((*local(points[a]), *local(points[b])), fill=DWPOSE_HAND_COLOR, width=5)
            for index in range(21):
                if bool(accepted[index]) and v03._finite_xy(points[index]):
                    px, py = local(points[index])
                    draw.ellipse((px-4, py-4, px+4, py+4), fill="#ffffff", outline="#111111")
    else:
        for a, b in hand.mhr_hand_edges(side):
            if a < len(sam2d) and b < len(sam2d) and v03._finite_xy(sam2d[a]) and v03._finite_xy(sam2d[b]):
                draw.line((*local(sam2d[a]), *local(sam2d[b])), fill=SAM3D_HAND_COLOR, width=5)
        for idx in hand.mhr_hand_order(side):
            if idx < len(sam2d) and v03._finite_xy(sam2d[idx]):
                px, py = local(sam2d[idx])
                draw.ellipse((px-4, py-4, px+4, py+4), fill="#ffffff", outline="#111111")

    fitted, _ = base._fit_image(crop, size[0], size[1])
    return fitted


def _hand_detail_panel(
    title: str,
    image: Image.Image,
    dw_hands: dict[str, dict[str, np.ndarray]],
    sam2d: np.ndarray,
    source: str,
) -> Image.Image:
    panel = base._panel(title)
    draw = ImageDraw.Draw(panel)
    half = base.PANEL_W // 2
    body_top = base.TITLE_H
    label_h = 24
    crop_h = base.PANEL_H - body_top - label_h - 8
    crop_w = half - 8

    for index, side in enumerate(("left", "right")):
        x = index * half + 4
        draw.text((x + 8, body_top + 4), side, fill="#f3f4f6", font=base._font(14, bold=True))
        window = _hand_window(side, dw_hands[side], sam2d)
        detail = _draw_hand_crop(
            image,
            side,
            window,
            source,
            dw_hands[side],
            sam2d,
            (crop_w, crop_h),
        )
        panel.paste(detail, (x, body_top + label_h))
    return panel


def _hand_relational_summary(profile: dict[str, Any]) -> Image.Image:
    panel = base._panel("Hand / relational pose summary")
    draw = ImageDraw.Draw(panel)
    y = 50
    projected = profile.get("sam3d_projected_pose") or {}
    rows = [
        (
            f"Projected pose: {projected.get('pose')}  recon={projected.get('reconstruction_match_percent')}%  "
            f"crop={projected.get('crop_support_percent')}%  coverage={projected.get('crop_coverage_percent')}%"
        )
    ]
    for side in ("left", "right"):
        h = (profile.get("hand_geometry") or {}).get(side) or {}
        dw = h.get("dwpose_hand") or {}
        sam = h.get("sam3d_hand") or {}
        rows.append(
            f"{side} hand: preferred={h.get('preferred_shape_label')} ({h.get('preferred_shape_source')}); "
            f"DWPose support={dw.get('crop_support_percent')}% conf={dw.get('mean_confidence')} residual={dw.get('median_sam3d_residual_px')}px"
        )
        rows.append(
            f"  shape: DWPose={dw.get('shape_label')}  SAM3D={sam.get('shape_label')}  agreement={h.get('cross_model_shape_agreement')}"
        )

    matches = []
    for name, value in (profile.get("relations") or {}).items():
        if isinstance(value, dict) and value.get("geometry_match"):
            suffix = f" ({value.get('side')})" if value.get("side") else ""
            matches.append(name + suffix)
    rows.append("Relations: " + (", ".join(matches) if matches else "none"))

    for row in rows:
        y = base._draw_text_wrapped(draw, (14, y), row, width_chars=58)
        y += 5
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
    if dwpose:
        dw_points, accepted_names, in_frame_names = v03._dwpose_target_points(dwpose, width, height)
    else:
        dw_points, accepted_names, in_frame_names = np.empty((0, 2)), set(), set()
    dw_hands = hand.decode_dwpose_target_hands(dwpose or None, width, height)
    sam2d = v03._sam2d_points(arrays)
    padding = _display_padding_with_hands(width, height, dw_points, accepted_names, dw_hands, sam2d)

    residual = (
        v03._reprojection_residual(
            dw_points,
            sam2d,
            accepted_names,
            in_frame_names,
            width=width,
            height=height,
            dwpose=dwpose,
        )
        if dwpose and len(dw_points) and len(sam2d)
        else v03._empty_residual()
    )
    relational = relprof.build_profile(arrays, dwpose or None, width, height)

    original = base._panel("Original", image)
    dw_overlay = base._panel(
        "DWPose accepted body + hands — padded",
        _draw_dwpose_with_hands(image, dw_points, accepted_names, dw_hands, padding),
    )
    sam_overlay = base._panel(
        "SAM3D projected body + fingers — padded",
        _draw_sam3d_with_hands(image, sam2d, padding),
    )
    dw_hands_panel = _hand_detail_panel("DWPose hand detail", image, dw_hands, sam2d, "dwpose")
    sam_hands_panel = _hand_detail_panel("SAM3D hand detail", image, dw_hands, sam2d, "sam3d")
    hand_summary = _hand_relational_summary(relational)

    keypoints3d = v03._camera_keypoints_3d(arrays)
    mesh = base._load_obj_vertices(sam_obj_path)
    camera_front = v03._draw_3d_pose_panel(
        keypoints3d, mesh, (0, 1), "SAM3D reconstructed 3D — camera view"
    )
    side_depth = v03._draw_3d_pose_panel(
        keypoints3d, mesh, (2, 1), "SAM3D reconstructed 3D — side/depth"
    )
    summary = v03._summary_panel(
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

    panels = [
        original,
        dw_overlay,
        sam_overlay,
        dw_hands_panel,
        sam_hands_panel,
        hand_summary,
        camera_front,
        side_depth,
        summary,
    ]
    card = Image.new(
        "RGB",
        (base.PANEL_W * 3, base.PANEL_H * 3),
        "#0b0d10",
    )
    for index, panel in enumerate(panels):
        x = (index % 3) * base.PANEL_W
        y = (index // 3) * base.PANEL_H
        card.paste(panel, (x, y))

    record = {
        "schema_version": "pose-atlas-v3-record-0.4",
        "image_key": base._image_key(image_path),
        "image": str(image_path),
        "dwpose": str(dwpose_path) if dwpose_path else None,
        "sam3d_arrays": str(sam_npz_path),
        "sam3d_mesh": str(sam_obj_path) if sam_obj_path else None,
        "sam3d_mesh_available": bool(len(mesh)),
        "sam3d_diagnostic": diagnostic,
        "sam3d_relational_pose_profile": relational,
        "dwpose_derived": dwpose.get("derived") if dwpose else None,
        "dwpose_accepted_joint_names": sorted(accepted_names),
        "dwpose_in_frame_accepted_joint_names": sorted(in_frame_names),
        "display_padding": padding,
        "projected_fit_residual": residual,
        "human_annotation": annotation or None,
        "interpretation_policy": {
            "purpose": "visual calibration of crop-supported body and hand geometry",
            "dwpose_hand_scores_are_retained_confidences": True,
            "finger_geometry_is_displayed_for_both_dwpose_and_sam3d": True,
            "sam3d_reconstruction_is_not_direct_observation": True,
            "waving_candidate_requires_vlm_confirmation": True,
        },
    }
    return card, record


def _html_index(records: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for record in records:
        key = html.escape(str(record["image_key"]))
        webp = html.escape(str(record["card_webp"]))
        profile = record.get("sam3d_relational_pose_profile") or {}
        projected = profile.get("sam3d_projected_pose") or {}
        cards.append(
            f'<article class="card"><h2>{key}</h2>'
            f'<img src="{webp}" loading="lazy" alt="Pose atlas card for {key}">'
            f'<p><b>Projected:</b> {html.escape(str(projected.get("pose")))} &nbsp; '
            f'<b>reconstruction:</b> {html.escape(str(projected.get("reconstruction_match_percent")))}% &nbsp; '
            f'<b>crop support:</b> {html.escape(str(projected.get("crop_support_percent")))}%</p>'
            f'</article>'
        )
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Semantic Fusion V3 Pose Atlas v0.4</title>
<style>
body{font-family:system-ui,sans-serif;background:#0d0f12;color:#edf0f3;margin:24px}
header{max-width:1100px;margin:auto auto 28px}.card{max-width:1560px;margin:0 auto 34px;background:#171a1f;padding:18px;border-radius:12px}
.card img{display:block;width:100%;height:auto;border-radius:8px;background:#08090b}.card h2{margin:0 0 12px}.card p{color:#c9ced6}
</style></head><body><header><h1>Semantic Fusion V3 — Pose Atlas v0.4</h1>
<p>Hand-aware calibration. DWPose and SAM3D finger chains are shown separately, with dedicated hand zooms. DWPose hand confidence is retained and cross-model finger disagreement remains visible rather than being averaged away.</p>
<p>Projected-pose reconstruction match is separate from crop support. Raised-open-hand geometry is only a waving candidate and requires VLM confirmation.</p></header>
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
        else run_dir / "semantic-v3" / "pose-atlas-v0.4"
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
        sam_obj = v03._resolve_mesh(sam3d_dir, key)

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
        projected = (record.get("sam3d_relational_pose_profile") or {}).get("sam3d_projected_pose") or {}
        print(
            f"{key}: {out_webp} projected={projected.get('pose')} "
            f"recon={projected.get('reconstruction_match_percent')}% "
            f"crop={projected.get('crop_support_percent')}% mesh={'yes' if record.get('sam3d_mesh_available') else 'no'}"
        )

    index = {
        "schema_version": "pose-atlas-v3-run-0.4",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
