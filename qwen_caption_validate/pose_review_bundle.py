from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import pose_atlas_v3 as base
from . import pose_atlas_v3_03 as v03
from . import pose_atlas_v3_04 as v04
from . import sam3d_hand_geometry as hand


DWPOSE_COLOR = "#36d7ff"
SAM3D_COLOR = "#ffbf3f"
OUT_OF_FRAME_FILL = "#ff9f43"
FRAME_COLOR = "#7d8590"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-review-bundle",
        description=(
            "Build a self-contained local pose-review bundle from original images, "
            "DWPose, SAM3D arrays and a relational pose profile. No model inference is run."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--sam3d-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tar", action="store_true")
    return parser.parse_args()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _find_profile_dir(run_dir: Path, supplied: Path | None, sam3d_dir: Path) -> Path:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Profile directory not found: {value}")
        return value
    preferred = sam3d_dir / "relational-pose-profile-v0.6"
    if preferred.is_dir():
        return preferred
    candidates = sorted(
        p for p in sam3d_dir.iterdir()
        if p.is_dir() and p.name.startswith("relational-pose-profile-v")
    )
    if not candidates:
        raise SystemExit("Could not find relational pose profile directory; pass --profile-dir.")
    return candidates[-1]


def _image_map(images_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in base._discover_images(images_dir):
        result.setdefault(base._image_key(path), path)
    return result


def _profile_record(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    wrapper = _read_json(path)
    key = str(wrapper.get("image_key") or path.name.removesuffix(".sam3d_relational_pose.json"))
    profile = wrapper.get("profile") if isinstance(wrapper.get("profile"), dict) else wrapper
    return key, wrapper, profile if isinstance(profile, dict) else {}


def _shift(point: np.ndarray, offset: tuple[int, int]) -> tuple[float, float]:
    return float(point[0]) + offset[0], float(point[1]) + offset[1]


def _draw_combined_overlay(
    image: Image.Image,
    dwpose: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[Image.Image, dict[str, Any]]:
    width, height = image.size
    if dwpose:
        dw_points, accepted_names, in_frame_names = v03._dwpose_target_points(dwpose, width, height)
    else:
        dw_points, accepted_names, in_frame_names = np.empty((0, 2)), set(), set()
    dw_hands = hand.decode_dwpose_target_hands(dwpose or None, width, height)
    sam2d = v03._sam2d_points(arrays)
    padding = v04._display_padding_with_hands(
        width, height, dw_points, accepted_names, dw_hands, sam2d
    )
    out, offset = v03._padded_canvas(image, padding)
    draw = ImageDraw.Draw(out)

    for a, b in base.DWPOSE_EDGES:
        if a not in accepted_names or b not in accepted_names:
            continue
        ia, ib = base.IDX[a], base.IDX[b]
        if ia >= len(dw_points) or ib >= len(dw_points):
            continue
        pa, pb = dw_points[ia], dw_points[ib]
        if v03._finite_xy(pa) and v03._finite_xy(pb):
            draw.line((*_shift(pa, offset), *_shift(pb, offset)), fill=DWPOSE_COLOR, width=5)

    for side in ("left", "right"):
        decoded = dw_hands.get(side) or {}
        points = np.asarray(decoded.get("points", np.empty((0, 2))), dtype=np.float64)
        accepted = np.asarray(decoded.get("accepted_mask", np.empty((0,), dtype=bool)), dtype=bool)
        if len(points) >= 21 and len(accepted) >= 21:
            for ia, ib in hand.dw_hand_edges():
                if (
                    bool(accepted[ia])
                    and bool(accepted[ib])
                    and v03._finite_xy(points[ia])
                    and v03._finite_xy(points[ib])
                ):
                    draw.line(
                        (*_shift(points[ia], offset), *_shift(points[ib], offset)),
                        fill=DWPOSE_COLOR,
                        width=3,
                    )

    def sam_point(name: str) -> np.ndarray | None:
        idx = v03.MHR_BODY.get(name)
        if idx is None or idx >= len(sam2d):
            return None
        point = np.asarray(sam2d[idx], dtype=np.float64)
        return point if v03._finite_xy(point) else None

    for a, b in v03.MHR_BODY_EDGES:
        pa, pb = sam_point(a), sam_point(b)
        if pa is not None and pb is not None:
            draw.line((*_shift(pa, offset), *_shift(pb, offset)), fill=SAM3D_COLOR, width=4)

    for side in ("left", "right"):
        for ia, ib in hand.mhr_hand_edges(side):
            if ia < len(sam2d) and ib < len(sam2d):
                pa, pb = sam2d[ia], sam2d[ib]
                if v03._finite_xy(pa) and v03._finite_xy(pb):
                    draw.line((*_shift(pa, offset), *_shift(pb, offset)), fill=SAM3D_COLOR, width=2)

    for name in base.BODY18:
        if name not in accepted_names:
            continue
        idx = base.IDX[name]
        if idx >= len(dw_points) or not v03._finite_xy(dw_points[idx]):
            continue
        x, y = _shift(dw_points[idx], offset)
        r = 5
        fill = DWPOSE_COLOR if v03._in_frame(dw_points[idx], width, height) else OUT_OF_FRAME_FILL
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline="#071018", width=1)

    sam_indices = (
        set(v03.MHR_BODY.values())
        | set(hand.mhr_hand_order("left"))
        | set(hand.mhr_hand_order("right"))
    )
    for idx in sorted(sam_indices):
        if idx >= len(sam2d) or not v03._finite_xy(sam2d[idx]):
            continue
        x, y = _shift(sam2d[idx], offset)
        r = 3
        fill = SAM3D_COLOR if v03._in_frame(sam2d[idx], width, height) else OUT_OF_FRAME_FILL
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline="#071018", width=1)

    left, top = offset
    draw.rectangle((left, top, left + width - 1, top + height - 1), outline=FRAME_COLOR, width=2)

    metadata = {
        "dwpose_color": DWPOSE_COLOR,
        "sam3d_color": SAM3D_COLOR,
        "out_of_frame_color": OUT_OF_FRAME_FILL,
        "accepted_dwpose_body_landmarks": sorted(accepted_names),
        "in_frame_dwpose_body_landmarks": sorted(in_frame_names),
        "display_padding": padding,
    }
    return out, metadata


def _compact_record(
    key: str,
    profile: dict[str, Any],
    original_rel: str,
    overlay_rel: str,
    raw_rel: str,
    overlay_meta: dict[str, Any],
) -> dict[str, Any]:
    projected = profile.get("sam3d_projected_pose") or {}
    support = projected.get("support_state") if isinstance(projected.get("support_state"), dict) else {}
    kneel = projected.get("kneeling_candidate") if isinstance(projected.get("kneeling_candidate"), dict) else {}
    relations: list[dict[str, Any]] = []
    for name, value in (profile.get("relations") or {}).items():
        if not isinstance(value, dict) or not value.get("geometry_match"):
            continue
        relations.append({
            "name": name,
            "side": value.get("side"),
            "crop_support_percent": value.get("crop_support_percent"),
            "support_class": value.get("support_class"),
        })

    return {
        "image_key": key,
        "original": original_rel,
        "overlay": overlay_rel,
        "raw_json": raw_rel,
        "pose": projected.get("pose"),
        "best_candidate_pose": projected.get("best_candidate_pose"),
        "posture_score_percent": projected.get("posture_score_percent") or {},
        "posture_score_percent_before_support_topology": projected.get(
            "posture_score_percent_before_support_topology"
        ) or {},
        "winner_margin_percent": projected.get("winner_margin_percent"),
        "reconstruction_match_percent": projected.get("reconstruction_match_percent"),
        "crop_support_percent": projected.get("crop_support_percent"),
        "crop_coverage_percent": projected.get("crop_coverage_percent"),
        "support_class": projected.get("support_class"),
        "support_state": support,
        "modifiers": projected.get("modifiers") or {},
        "kneeling_candidate": kneel,
        "relations": relations,
        "overlay_meta": overlay_meta,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    images_dir = (args.images_dir or (run_dir / "images")).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    sam3d_dir = (args.sam3d_dir or (run_dir / "sam3d-pose-discovery-01")).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not dwpose_dir.is_dir():
        raise SystemExit(f"DWPose directory not found: {dwpose_dir}")
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")

    profile_dir = _find_profile_dir(run_dir, args.profile_dir, sam3d_dir)
    output = (args.output or (run_dir / "semantic-v3" / "pose-review-v0.1")).expanduser().resolve()
    media_original = output / "media" / "original"
    media_overlay = output / "media" / "overlay"
    raw_dir = output / "records"
    media_original.mkdir(parents=True, exist_ok=True)
    media_overlay.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    wanted = {str(value).lower() for value in args.only}
    images = _image_map(images_dir)
    profile_paths = sorted(profile_dir.glob("*.sam3d_relational_pose.json"))
    if not profile_paths:
        raise SystemExit(f"No relational pose JSON files found in {profile_dir}")

    records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for profile_path in profile_paths:
        key, wrapper, profile = _profile_record(profile_path)
        if wanted and key.lower() not in wanted:
            continue
        image_path = images.get(key)
        if image_path is None:
            missing.append({"image_key": key, "reason": "missing_original_image"})
            continue
        sam_npz = sam3d_dir / f"{key}.sam3d_arrays.npz"
        if not sam_npz.is_file():
            matches = list(sam3d_dir.rglob(f"{key}.sam3d_arrays.npz"))
            sam_npz = matches[0] if matches else sam_npz
        if not sam_npz.is_file():
            missing.append({"image_key": key, "reason": "missing_sam3d_arrays"})
            continue
        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        if not dwpose_path.is_file():
            matches = list(dwpose_dir.rglob(f"{key}.dwpose.json"))
            dwpose_path = matches[0] if matches else dwpose_path

        original_name = f"{key}{image_path.suffix.lower()}"
        original_out = media_original / original_name
        overlay_out = media_overlay / f"{key}.overlay.webp"
        raw_out = raw_dir / f"{key}.json"

        if args.overwrite or not original_out.exists():
            shutil.copy2(image_path, original_out)
        if args.overwrite or not raw_out.exists():
            raw_out.write_text(
                json.dumps(wrapper, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        overlay_meta: dict[str, Any]
        if args.overwrite or not overlay_out.exists():
            image = Image.open(image_path).convert("RGB")
            dwpose = _read_json(dwpose_path)
            with np.load(sam_npz) as loaded:
                arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
            overlay, overlay_meta = _draw_combined_overlay(image, dwpose, arrays)
            overlay.save(
                overlay_out,
                format="WEBP",
                quality=max(1, min(100, int(args.quality))),
                method=6,
            )
        else:
            overlay_meta = {
                "dwpose_color": DWPOSE_COLOR,
                "sam3d_color": SAM3D_COLOR,
                "out_of_frame_color": OUT_OF_FRAME_FILL,
            }

        records.append(_compact_record(
            key,
            profile,
            f"media/original/{original_name}",
            f"media/overlay/{overlay_out.name}",
            f"records/{raw_out.name}",
            overlay_meta,
        ))
        print(f"{key}: original={original_out.name} overlay={overlay_out.name}")

    index = {
        "schema_version": "pose-review-bundle-0.1",
        "run_dir": str(run_dir),
        "profile_dir": str(profile_dir),
        "record_count": len(records),
        "missing": missing,
        "legend": {
            "dwpose": DWPOSE_COLOR,
            "sam3d": SAM3D_COLOR,
            "out_of_frame": OUT_OF_FRAME_FILL,
        },
        "records": records,
    }
    base._write_json(output / "pose_review.index.json", index)
    annotation_path = output / "pose_review_annotations.json"
    if not annotation_path.exists():
        base._write_json(annotation_path, {
            "schema_version": "pose-review-annotations-0.1",
            "records": {},
        })

    (output / "README.txt").write_text(
        "Pose Review Bundle v0.1\n\n"
        "Run the repository's local review server against this directory:\n"
        "  python -m qwen_caption_validate.pose_review_server <bundle-dir> --open\n\n"
        "Cyan = DWPose accepted 2D geometry. Amber = SAM3D projected 3D geometry.\n"
        "Orange points are outside the source frame. The gray rectangle marks the source image.\n",
        encoding="utf-8",
    )

    print(f"Review bundle: {output}")
    print(f"Records: {len(records)}")
    if missing:
        print(f"Missing: {len(missing)} (see pose_review.index.json)")

    if args.tar:
        tar_path = output.with_suffix(".tar")
        with tarfile.open(tar_path, "w") as archive:
            archive.add(output, arcname=output.name)
        print(f"Tar: {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
