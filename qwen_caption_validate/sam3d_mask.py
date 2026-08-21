from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-mask",
        description=(
            "Rasterize cached SAM 3D Body meshes into prototype loss-weight masks. "
            "This is an entropy-focus subject-zone proposal, not a segmentation matte."
        ),
    )
    parser.add_argument("dataset", type=Path, help="Dataset directory containing the source images.")
    parser.add_argument("--sam3d-dir", type=Path, required=True, help="Directory containing cached *.sam3d.json and *.sam3d.obj files.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for masks and metadata.")
    parser.add_argument(
        "--fusion-dir",
        type=Path,
        help="Optional Fusion-v2.3 model directory. If supplied, SAM3D authority/provenance is copied into mask metadata.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Exact image basename or dataset-relative path to include; repeatable. If omitted, process all images having cached SAM3D meshes.",
    )
    parser.add_argument(
        "--dilate-frac",
        type=float,
        default=0.04,
        help="Subject-zone dilation radius as a fraction of rendered body silhouette size (default: 0.04).",
    )
    parser.add_argument(
        "--feather-frac",
        type=float,
        default=0.04,
        help="Outward soft-halo width as a fraction of rendered body silhouette size (default: 0.04).",
    )
    parser.add_argument(
        "--background-weight",
        type=float,
        default=0.35,
        help="Loss weight outside the subject halo, 0..1 (default: 0.35).",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.01,
        help="Rendered alpha threshold used to define the binary body core, 0..1 (default: 0.01).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _result_key(relative_path: Path) -> str:
    return str(relative_path.with_suffix("")).replace("/", "__").replace("\\", "__")


def _discover_images(dataset: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _apply_include(images: list[Path], dataset: Path, includes: list[str]) -> list[Path]:
    if not includes:
        return images
    wanted = {item.replace("\\", "/") for item in includes}
    selected: list[Path] = []
    for path in images:
        rel = path.relative_to(dataset).as_posix()
        if rel in wanted or path.name in wanted:
            selected.append(path)
    missing = sorted(
        item
        for item in wanted
        if not any(item == path.name or item == path.relative_to(dataset).as_posix() for path in selected)
    )
    if missing:
        raise SystemExit(f"Requested --include image(s) not found: {missing}")
    return selected


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Expected scalar camera value, got empty array")
    return float(arr[0])


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:4]
                if len(parts) == 3:
                    faces.append([int(token.split("/")[0]) - 1 for token in parts])
    if not vertices or not faces:
        raise ValueError(f"OBJ lacks vertices or faces: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _silhouette_bbox(binary: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        raise ValueError("Rendered SAM3D silhouette is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _entropy_masks(
    alpha: np.ndarray,
    *,
    alpha_threshold: float,
    dilate_frac: float,
    feather_frac: float,
    background_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    binary = (alpha >= alpha_threshold).astype(np.uint8)
    x0, y0, x1, y1 = _silhouette_bbox(binary)
    body_scale = max(x1 - x0 + 1, y1 - y0 + 1)

    dilate_px = max(0, int(round(body_scale * dilate_frac)))
    feather_px = max(0, int(round(body_scale * feather_frac)))

    if dilate_px > 0:
        size = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        zone = cv2.dilate(binary, kernel)
    else:
        zone = binary.copy()

    if feather_px > 0:
        outside = (zone == 0).astype(np.uint8)
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        proximity = np.clip(1.0 - distance / float(feather_px), 0.0, 1.0)
    else:
        proximity = zone.astype(np.float32)

    weight = background_weight + (1.0 - background_weight) * proximity
    weight[zone > 0] = 1.0
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)

    stats = {
        "body_bbox_xyxy": [x0, y0, x1, y1],
        "body_scale_px": body_scale,
        "dilate_px": dilate_px,
        "feather_px": feather_px,
        "core_frame_fraction": round(float(binary.mean()), 6),
        "subject_zone_frame_fraction": round(float(zone.mean()), 6),
        "mean_loss_weight": round(float(weight.mean()), 6),
    }
    return binary, zone, weight, stats


def _fusion_context(fusion_dir: Path | None, key: str) -> dict[str, Any] | None:
    if fusion_dir is None:
        return None
    path = fusion_dir / f"{key}.fused_v2_3.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit = ((payload.get("fusion") or {}).get("sam3d_geometry_audit") or {})
    return {
        "fusion_path": str(path),
        "target_provenance": audit.get("target_provenance"),
        "shoulder_depth_rotation": audit.get("shoulder_depth_rotation"),
        "hip_depth_rotation": audit.get("hip_depth_rotation"),
        "torso_depth_rotation": audit.get("torso_depth_rotation"),
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    fusion_dir = args.fusion_dir.expanduser().resolve() if args.fusion_dir else None

    if not dataset.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset}")
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory does not exist: {sam3d_dir}")
    if fusion_dir is not None and not fusion_dir.is_dir():
        raise SystemExit(f"Fusion directory does not exist: {fusion_dir}")
    if not 0.0 <= args.background_weight <= 1.0:
        raise SystemExit("--background-weight must be between 0 and 1")
    if not 0.0 <= args.alpha_threshold <= 1.0:
        raise SystemExit("--alpha-threshold must be between 0 and 1")
    if args.dilate_frac < 0.0 or args.feather_frac < 0.0:
        raise SystemExit("--dilate-frac and --feather-frac must be >= 0")

    try:
        from sam_3d_body.visualization.renderer import Renderer
    except ImportError as exc:
        raise SystemExit(
            "SAM 3D Body renderer is not importable. Use run_sam3d_mask_workspace.sh with the isolated SAM3D workspace."
        ) from exc

    images = _apply_include(_discover_images(dataset), dataset, args.include)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    written = 0
    skipped = 0
    missing_cache = 0

    for image_path in images:
        rel = image_path.relative_to(dataset)
        key = _result_key(rel)
        sam3d_json = sam3d_dir / f"{key}.sam3d.json"
        fallback_obj = sam3d_dir / f"{key}.sam3d.obj"
        if not sam3d_json.exists():
            if args.include:
                raise SystemExit(f"Missing SAM3D record for requested image {rel}: {sam3d_json}")
            missing_cache += 1
            continue

        record = json.loads(sam3d_json.read_text(encoding="utf-8"))
        mesh_value = record.get("mesh_obj")
        mesh_path = Path(str(mesh_value)).expanduser() if mesh_value else fallback_obj
        if not mesh_path.exists():
            mesh_path = fallback_obj
        if not mesh_path.exists():
            raise SystemExit(
                f"Missing cached SAM3D OBJ for {rel}. Re-run the SAM3D probe with mesh saving enabled: {fallback_obj}"
            )

        meta_path = output_dir / f"{key}.entropy_mask.json"
        if meta_path.exists() and not args.overwrite:
            skipped += 1
            records.append(json.loads(meta_path.read_text(encoding="utf-8")))
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        vertices, faces = _load_obj(mesh_path)
        camera = record.get("camera") or {}
        focal_length = _scalar(camera.get("focal_length"))
        cam_t = np.asarray(camera.get("pred_cam_t"), dtype=np.float32).reshape(-1)
        if cam_t.size < 3:
            raise ValueError(f"SAM3D record lacks a usable pred_cam_t: {sam3d_json}")

        renderer = Renderer(focal_length=focal_length, faces=faces)
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        rgba = renderer(vertices, cam_t[:3], blank, return_rgba=True)
        alpha = np.clip(np.asarray(rgba[:, :, 3], dtype=np.float32), 0.0, 1.0)

        core, zone, weight, stats = _entropy_masks(
            alpha,
            alpha_threshold=args.alpha_threshold,
            dilate_frac=args.dilate_frac,
            feather_frac=args.feather_frac,
            background_weight=args.background_weight,
        )

        core_path = output_dir / f"{key}.body_core.png"
        zone_path = output_dir / f"{key}.subject_zone.png"
        weight_path = output_dir / f"{key}.entropy_weight.png"
        preview_path = output_dir / f"{key}.entropy_preview.png"

        cv2.imwrite(str(core_path), np.clip(alpha * 255.0, 0, 255).astype(np.uint8))
        cv2.imwrite(str(zone_path), (zone * 255).astype(np.uint8))
        cv2.imwrite(str(weight_path), np.clip(weight * 255.0, 0, 255).astype(np.uint8))
        preview = np.clip(image.astype(np.float32) * weight[:, :, None], 0, 255).astype(np.uint8)
        cv2.imwrite(str(preview_path), preview)

        payload = {
            "schema_version": "sam3d-entropy-mask-prototype-0.1",
            "image": str(rel),
            "sam3d_record": str(sam3d_json),
            "mesh_obj": str(mesh_path),
            "renderer": "facebookresearch/sam-3d-body upstream Renderer alpha projection",
            "parameters": {
                "alpha_threshold": args.alpha_threshold,
                "dilate_fraction_of_body_scale": args.dilate_frac,
                "feather_fraction_of_body_scale": args.feather_frac,
                "background_weight": args.background_weight,
                "subject_zone_weight": 1.0,
            },
            "stats": stats,
            "outputs": {
                "body_core": str(core_path),
                "subject_zone": str(zone_path),
                "entropy_weight": str(weight_path),
                "preview": str(preview_path),
            },
            "fusion_context": _fusion_context(fusion_dir, key),
            "authority": "prototype_loss_weight_mask_not_segmentation_matte",
            "limitations": [
                "The body core is a projection of a reconstructed parametric human mesh, not a pixel segmentation result.",
                "SAM3D can reconstruct anatomy outside the crop or behind foreground occluders; viewport clipping does not solve scene occlusion.",
                "Hair, loose clothing, carried objects and other subject-associated pixels may extend beyond the body mesh; dilation intentionally provides a safety margin.",
                "Use the entropy-weight image for loss weighting experiments, not as ground-truth subject segmentation.",
            ],
        }
        _write_json(meta_path, payload)
        records.append(payload)
        written += 1
        print(
            f"{rel}: core={stats['core_frame_fraction']:.3f}; zone={stats['subject_zone_frame_fraction']:.3f}; "
            f"dilate={stats['dilate_px']}px; feather={stats['feather_px']}px"
        )

    index = {
        "schema_version": "sam3d-entropy-mask-prototype-0.1-run",
        "dataset": str(dataset),
        "sam3d_dir": str(sam3d_dir),
        "fusion_dir": str(fusion_dir) if fusion_dir else None,
        "output_dir": str(output_dir),
        "written": written,
        "skipped_existing": skipped,
        "missing_sam3d_cache": missing_cache,
        "record_count": len(records),
        "records": [
            {
                "image": record.get("image"),
                "stats": record.get("stats"),
                "outputs": record.get("outputs"),
            }
            for record in records
        ],
    }
    index_path = output_dir / "sam3d_entropy_masks.index.json"
    _write_json(index_path, index)
    print(f"Done. Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
