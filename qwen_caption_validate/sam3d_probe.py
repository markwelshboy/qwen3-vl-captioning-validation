from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_MODEL_REPO = "facebook/sam-3d-body-dinov3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-probe",
        description=(
            "Experimental SAM 3D Body probe for the geometry gap between Qwen semantics "
            "and DWPose projected 2-D evidence. Uses cached DWPose target boxes and writes "
            "compact, unsigned 3-D orientation/recline metrics plus optional OBJ meshes."
        ),
    )
    parser.add_argument("dataset", type=Path, help="Image dataset directory.")
    parser.add_argument("--dwpose-dir", type=Path, required=True, help="Existing DWPose cache directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sam3d-probe"),
        help="Output directory (default: runs/sam3d-probe).",
    )
    parser.add_argument(
        "--model-repo",
        default=os.environ.get("SAM3D_HF_REPO", DEFAULT_MODEL_REPO),
        help=f"Hugging Face SAM 3D Body repo (default: {DEFAULT_MODEL_REPO}).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Exact image basename or dataset-relative path to include; repeatable. If omitted, process all images.",
    )
    parser.add_argument(
        "--bbox-source",
        choices=["dwpose", "full"],
        default="dwpose",
        help="Use padded DWPose target bbox or the full image (default: dwpose).",
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.20,
        help="Fractional padding added to each side of the DWPose bbox (default: 0.20).",
    )
    parser.add_argument(
        "--inference-type",
        choices=["body", "full"],
        default="body",
        help="SAM 3D Body inference path. body is sufficient for torso geometry and is the default.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device passed to SAM 3D Body (default: cuda).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing probe JSON files.")
    parser.add_argument(
        "--save-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an OBJ mesh for visual inspection (default: true).",
    )
    return parser.parse_args()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


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
        if not any(
            item == path.name or item == path.relative_to(dataset).as_posix()
            for path in selected
        )
    )
    if missing:
        raise SystemExit(f"Requested --include image(s) not found: {missing}")
    return selected


def _read_dwpose(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dwpose_bbox_pixels(record: dict[str, Any], padding: float) -> tuple[np.ndarray, dict[str, Any]]:
    width = int(record.get("image_width") or 0)
    height = int(record.get("image_height") or 0)
    target = ((record.get("derived") or {}).get("target") or {})
    bbox = target.get("keypoint_bbox") or {}
    if width <= 0 or height <= 0 or not bbox:
        raise ValueError("DWPose record lacks image dimensions or target keypoint bbox")

    x0 = float(bbox["x0"]) * width
    y0 = float(bbox["y0"]) * height
    x1 = float(bbox["x1"]) * width
    y1 = float(bbox["y1"]) * height
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    xpad = bw * padding
    ypad = bh * padding

    padded = np.array(
        [
            max(0.0, x0 - xpad),
            max(0.0, y0 - ypad),
            min(float(width - 1), x1 + xpad),
            min(float(height - 1), y1 + ypad),
        ],
        dtype=np.float32,
    )
    meta = {
        "source": "dwpose_target_keypoint_bbox",
        "padding_fraction_each_side": padding,
        "raw_normalized_bbox": bbox,
        "padded_pixel_bbox_xyxy": [round(float(v), 3) for v in padded],
    }
    return padded.reshape(1, 4), meta


def _full_image_bbox(image_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]
    bbox = np.array([[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32)
    return bbox, {
        "source": "full_image",
        "padding_fraction_each_side": None,
        "padded_pixel_bbox_xyxy": [0.0, 0.0, float(width - 1), float(height - 1)],
    }


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(b, dtype=np.float64)[:3] - np.asarray(a, dtype=np.float64)[:3]


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64)[:3] + np.asarray(b, dtype=np.float64)[:3]) / 2.0


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _out_of_image_plane_angle_deg(v: np.ndarray) -> float | None:
    """Unsigned angle between a 3-D vector and the x/y image plane.

    This assumes SAM 3D Body's third coordinate is depth, but deliberately uses
    only |z|. Sign/direction is not granted semantic authority by this probe.
    """
    v = np.asarray(v, dtype=np.float64)[:3]
    if not np.all(np.isfinite(v)) or _norm(v) < 1e-9:
        return None
    in_plane = math.hypot(float(v[0]), float(v[1]))
    return round(math.degrees(math.atan2(abs(float(v[2])), in_plane)), 3)


def _signed_depth_fraction(v: np.ndarray) -> float | None:
    v = np.asarray(v, dtype=np.float64)[:3]
    length = _norm(v)
    if not np.all(np.isfinite(v)) or length < 1e-9:
        return None
    return round(float(v[2]) / length, 6)


def _mean(values: list[float | None]) -> float | None:
    good = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return round(sum(good) / len(good), 3) if good else None


def _orientation_metrics(keypoints_3d: np.ndarray, names: list[str]) -> dict[str, Any]:
    name_to_idx = {name.replace("-", "_"): i for i, name in enumerate(names)}

    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    missing = [name for name in required if name not in name_to_idx]
    if missing:
        raise ValueError(f"SAM 3D Body keypoint metadata is missing required joints: {missing}")

    def kp(name: str) -> np.ndarray:
        idx = name_to_idx[name]
        if idx >= len(keypoints_3d):
            raise ValueError(f"pred_keypoints_3d does not contain index {idx} for {name}")
        return np.asarray(keypoints_3d[idx], dtype=np.float64)[:3]

    ls = kp("left_shoulder")
    rs = kp("right_shoulder")
    lh = kp("left_hip")
    rh = kp("right_hip")
    shoulder_mid = _midpoint(ls, rs)
    hip_mid = _midpoint(lh, rh)

    shoulder_lr = _vec(ls, rs)
    hip_lr = _vec(lh, rh)
    torso = _vec(hip_mid, shoulder_mid)

    shoulder_depth = _out_of_image_plane_angle_deg(shoulder_lr)
    hip_depth = _out_of_image_plane_angle_deg(hip_lr)
    torso_depth = _out_of_image_plane_angle_deg(torso)

    selected = {
        "left_shoulder": ls.tolist(),
        "right_shoulder": rs.tolist(),
        "left_hip": lh.tolist(),
        "right_hip": rh.tolist(),
        "shoulder_midpoint": shoulder_mid.tolist(),
        "hip_midpoint": hip_mid.tolist(),
    }
    for optional in ("neck", "left_ankle", "right_ankle"):
        if optional in name_to_idx and name_to_idx[optional] < len(keypoints_3d):
            selected[optional] = np.asarray(keypoints_3d[name_to_idx[optional]], dtype=np.float64)[:3].tolist()

    return {
        "coordinate_assumption": (
            "pred_keypoints_3d is treated as x/y/depth for this experiment. Only unsigned depth-angle "
            "magnitudes are used for interpretation until SAM 3D Body's coordinate sign convention is "
            "empirically verified on the regression meshes."
        ),
        "selected_keypoints_xyz": selected,
        "shoulder_left_right_vector_xyz": shoulder_lr.tolist(),
        "hip_left_right_vector_xyz": hip_lr.tolist(),
        "hip_to_shoulder_midpoint_vector_xyz": torso.tolist(),
        "shoulder_out_of_image_plane_deg": shoulder_depth,
        "hip_out_of_image_plane_deg": hip_depth,
        "torso_depth_tilt_deg": torso_depth,
        "torso_depth_rotation_proxy_deg": _mean([shoulder_depth, hip_depth]),
        "signed_depth_fraction_diagnostics": {
            "shoulder_left_to_right": _signed_depth_fraction(shoulder_lr),
            "hip_left_to_right": _signed_depth_fraction(hip_lr),
            "hip_mid_to_shoulder_mid": _signed_depth_fraction(torso),
            "authority": "diagnostic_only_sign_not_validated",
        },
    }


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# SAM 3D Body probe mesh\n")
        for vertex in np.asarray(vertices):
            fh.write(f"v {float(vertex[0]):.8f} {float(vertex[1]):.8f} {float(vertex[2]):.8f}\n")
        for face in np.asarray(faces, dtype=np.int64):
            fh.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _print_row(image: str, metrics: dict[str, Any], seconds: float) -> None:
    print(
        f"{image}: "
        f"depth-rotation≈{metrics.get('torso_depth_rotation_proxy_deg')}°; "
        f"torso-depth-tilt≈{metrics.get('torso_depth_tilt_deg')}°; "
        f"shoulder={metrics.get('shoulder_out_of_image_plane_deg')}°; "
        f"hip={metrics.get('hip_out_of_image_plane_deg')}°; "
        f"{seconds:.2f}s"
    )


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    dwpose_dir = args.dwpose_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not dataset.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset}")
    if args.bbox_source == "dwpose" and not dwpose_dir.is_dir():
        raise SystemExit(f"DWPose directory does not exist: {dwpose_dir}")
    if args.bbox_padding < 0.0 or args.bbox_padding > 1.0:
        raise SystemExit("--bbox-padding must be between 0.0 and 1.0")

    images = _apply_include(_discover_images(dataset), dataset, args.include)
    if not images:
        raise SystemExit(f"No supported images found in {dataset}")

    try:
        import torch
        from huggingface_hub import snapshot_download
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from sam_3d_body.metadata.mhr70 import mhr_names
    except ImportError as exc:
        raise SystemExit(
            "SAM 3D Body is not importable. Run build_sam3d_workspace.sh and use run_sam3d_probe_workspace.sh."
        ) from exc

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")

    print(f"Resolving SAM 3D Body checkpoint: {args.model_repo} ...")
    snapshot_dir = Path(snapshot_download(repo_id=args.model_repo))
    checkpoint_path = snapshot_dir / "model.ckpt"
    mhr_path = snapshot_dir / "assets" / "mhr_model.pt"
    if not checkpoint_path.exists() or not mhr_path.exists():
        raise SystemExit(
            f"Checkpoint snapshot is missing expected files: {checkpoint_path} / {mhr_path}"
        )

    print(f"Loading SAM 3D Body on {args.device} ...")
    load_started = time.perf_counter()
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        mhr_path=str(mhr_path),
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    load_seconds = time.perf_counter() - load_started
    print(f"Loaded in {load_seconds:.2f}s. Processing {len(images)} image(s).")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for image_path in images:
        rel = image_path.relative_to(dataset)
        key = _result_key(rel)
        out_json = output_dir / f"{key}.sam3d.json"
        if out_json.exists() and not args.overwrite:
            record = json.loads(out_json.read_text(encoding="utf-8"))
            records.append(record)
            _print_row(str(rel), record.get("metrics") or {}, float(record.get("inference_seconds") or 0.0))
            continue

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        if args.bbox_source == "dwpose":
            if not dwpose_path.exists():
                raise SystemExit(f"Missing DWPose record for {rel}: {dwpose_path}")
            dwpose_record = _read_dwpose(dwpose_path)
            bboxes, bbox_meta = _dwpose_bbox_pixels(dwpose_record, args.bbox_padding)
        else:
            bboxes, bbox_meta = _full_image_bbox(image_path)

        started = time.perf_counter()
        outputs = estimator.process_one_image(
            str(image_path),
            bboxes=bboxes,
            inference_type=args.inference_type,
        )
        inference_seconds = time.perf_counter() - started
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one SAM 3D Body output for one supplied bbox; got {len(outputs)} for {rel}")

        person = outputs[0]
        keypoints_3d = np.asarray(person["pred_keypoints_3d"], dtype=np.float64)
        metrics = _orientation_metrics(keypoints_3d, list(mhr_names))

        mesh_path = None
        if args.save_mesh:
            mesh_path = output_dir / f"{key}.sam3d.obj"
            _write_obj(mesh_path, np.asarray(person["pred_vertices"]), np.asarray(estimator.faces))

        npz_path = output_dir / f"{key}.sam3d_arrays.npz"
        np.savez_compressed(
            npz_path,
            pred_keypoints_3d=np.asarray(person["pred_keypoints_3d"]),
            pred_keypoints_2d=np.asarray(person["pred_keypoints_2d"]),
            pred_cam_t=np.asarray(person["pred_cam_t"]),
            global_rot=np.asarray(person["global_rot"]),
            pred_joint_coords=np.asarray(person["pred_joint_coords"]),
            pred_global_rots=np.asarray(person["pred_global_rots"]),
        )

        record = {
            "schema_version": "sam3d-geometry-probe-0.1",
            "image": str(rel),
            "model_repo": args.model_repo,
            "model_snapshot": str(snapshot_dir),
            "device": args.device,
            "inference_type": args.inference_type,
            "bbox": bbox_meta,
            "inference_seconds": round(inference_seconds, 4),
            "metrics": metrics,
            "camera": {
                "focal_length": _to_builtin(person.get("focal_length")),
                "pred_cam_t": _to_builtin(person.get("pred_cam_t")),
                "authority": "diagnostic_only_default_fov_no_external_fov_estimator",
            },
            "global_rot_raw": _to_builtin(person.get("global_rot")),
            "arrays_npz": str(npz_path),
            "mesh_obj": str(mesh_path) if mesh_path else None,
            "limitations": [
                "This is a validation probe, not selection-authoritative evidence.",
                "DWPose supplies the target crop; SAM 3D Body does not independently select the target person in this mode.",
                "No external FOV estimator is used in this first experiment; the SAM 3D Body default camera model is used.",
                "Depth-angle metrics are unsigned until coordinate sign/orientation is verified visually on saved meshes.",
                "A single-image 3-D reconstruction can be plausible yet geometrically wrong; mesh review is required before integration.",
            ],
        }
        _write_json(out_json, record)
        records.append(record)
        _print_row(str(rel), metrics, inference_seconds)

    summary = {
        "schema_version": "sam3d-geometry-probe-0.1-run",
        "dataset": str(dataset),
        "dwpose_dir": str(dwpose_dir),
        "model_repo": args.model_repo,
        "model_load_seconds": round(load_seconds, 4),
        "bbox_source": args.bbox_source,
        "bbox_padding": args.bbox_padding,
        "image_count": len(records),
        "records": [
            {
                "image": record.get("image"),
                "inference_seconds": record.get("inference_seconds"),
                "metrics": record.get("metrics"),
                "mesh_obj": record.get("mesh_obj"),
                "arrays_npz": record.get("arrays_npz"),
            }
            for record in records
        ],
    }
    index_path = output_dir / "sam3d_probe.index.json"
    _write_json(index_path, summary)
    print(f"Done. Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
