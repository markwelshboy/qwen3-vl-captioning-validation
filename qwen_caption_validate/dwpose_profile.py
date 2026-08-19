from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .runner import discover_images


# DWPose's ControlNet/OpenPose representation uses the standard 18-body-joint
# ordering below after converting the COCO-WholeBody output. These labels are
# useful as a second, deterministic source of evidence; they are not treated as
# ground truth for front/back orientation or anatomical laterality.
BODY18 = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
]
IDX = {name: i for i, name in enumerate(BODY18)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dwpose-profile",
        description=(
            "Run DWPose over an image dataset and cache raw 2D pose data plus "
            "small deterministic geometry/coverage summaries."
        ),
    )
    parser.add_argument("dataset", type=Path, help="Folder containing images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/dwpose-v1"),
        help="Output folder (default: runs/dwpose-v1).",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan dataset recursively.")
    parser.add_argument("--limit", type=int, help="Process only the first N images.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing .dwpose.json files.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="DWPose execution device. auto prefers the ONNX CUDA provider when available.",
    )
    return parser.parse_args()


def _result_key(relative_path: Path) -> str:
    text = str(relative_path.with_suffix(""))
    return text.replace("/", "__").replace("\\", "__")


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


def _candidate_array(pose_data: dict[str, Any]) -> np.ndarray:
    bodies = pose_data.get("bodies") or {}
    candidate = np.asarray(bodies.get("candidate", []), dtype=np.float64)
    if candidate.size == 0:
        return np.empty((0, 18, 2), dtype=np.float64)
    if candidate.ndim == 2:
        candidate = candidate[None, ...]
    if candidate.ndim != 3 or candidate.shape[-1] < 2:
        return np.empty((0, 18, 2), dtype=np.float64)
    # The exported DWPose/OpenPose body representation is normally 18 points.
    # Keep the first 18 if an implementation returns additional body landmarks.
    return candidate[:, :18, :2]


def _is_visible(point: np.ndarray) -> bool:
    return bool(
        point.shape[0] >= 2
        and np.isfinite(point[0])
        and np.isfinite(point[1])
        and point[0] >= 0.0
        and point[1] >= 0.0
    )


def _point(person: np.ndarray, name: str) -> np.ndarray | None:
    idx = IDX[name]
    if idx >= len(person):
        return None
    p = person[idx]
    return p if _is_visible(p) else None


def _bbox(person: np.ndarray) -> dict[str, float] | None:
    pts = np.asarray([p for p in person if _is_visible(p)], dtype=np.float64)
    if not len(pts):
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    width = max(0.0, float(x1 - x0))
    height = max(0.0, float(y1 - y0))
    return {
        "x0": round(float(x0), 6),
        "y0": round(float(y0), 6),
        "x1": round(float(x1), 6),
        "y1": round(float(y1), 6),
        "width_fraction": round(width, 6),
        "height_fraction": round(height, 6),
        "area_fraction": round(width * height, 6),
        "center_x": round(float((x0 + x1) / 2.0), 6),
        "center_y": round(float((y0 + y1) / 2.0), 6),
    }


def _line_angle_deg(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return round(float(math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0])))), 2)


def _axis_angle_from_vertical_deg(top: np.ndarray | None, bottom: np.ndarray | None) -> float | None:
    if top is None or bottom is None:
        return None
    dx = float(bottom[0] - top[0])
    dy = float(bottom[1] - top[1])
    # 0 = vertical downward in image coordinates; signed values indicate cant.
    return round(float(math.degrees(math.atan2(dx, dy))), 2)


def _midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _chain(person: np.ndarray, names: list[str]) -> dict[str, Any]:
    present = [name for name in names if _point(person, name) is not None]
    return {
        "landmarks": names,
        "visible": present,
        "visible_count": len(present),
        "complete": len(present) == len(names),
    }


def _extent_hint(person: np.ndarray) -> tuple[str, str]:
    """Return a deliberately coarse shot-extent hint from lowest visible joints.

    This is not a photographic classifier. It is useful as an independent
    checkpoint against VLM framing labels and for dataset coverage counts.
    """
    ankles = sum(_point(person, n) is not None for n in ("left_ankle", "right_ankle"))
    knees = sum(_point(person, n) is not None for n in ("left_knee", "right_knee"))
    hips = sum(_point(person, n) is not None for n in ("left_hip", "right_hip"))
    shoulders = sum(_point(person, n) is not None for n in ("left_shoulder", "right_shoulder"))

    if ankles:
        return "full_length", "ankle landmark visible"
    if knees:
        return "three_quarter_or_long", "knee visible but no ankle landmark"
    if hips:
        return "waist_or_upper_body", "hip visible but no knee/ankle landmark"
    if shoulders:
        return "close_or_medium_close", "shoulders visible but no hip/knee/ankle landmark"
    return "face_or_partial_body", "insufficient torso/leg landmarks"


def _person_summary(person: np.ndarray, index: int) -> dict[str, Any]:
    bbox = _bbox(person)
    visible_names = [BODY18[i] for i, p in enumerate(person[:18]) if _is_visible(p)]

    right_shoulder = _point(person, "right_shoulder")
    left_shoulder = _point(person, "left_shoulder")
    right_hip = _point(person, "right_hip")
    left_hip = _point(person, "left_hip")
    neck = _point(person, "neck")
    hip_mid = _midpoint(right_hip, left_hip)

    extent, extent_reason = _extent_hint(person)

    return {
        "person_index": index,
        "visible_body_landmarks": visible_names,
        "visible_body_landmark_count": len(visible_names),
        "body_completeness_fraction": round(len(visible_names) / 18.0, 4),
        "keypoint_bbox": bbox,
        "pose_extent_hint": extent,
        "pose_extent_reason": extent_reason,
        "geometry": {
            "shoulder_line_angle_from_horizontal_deg": _line_angle_deg(right_shoulder, left_shoulder),
            "hip_line_angle_from_horizontal_deg": _line_angle_deg(right_hip, left_hip),
            "torso_axis_angle_from_vertical_deg": _axis_angle_from_vertical_deg(neck, hip_mid),
        },
        "connectivity": {
            "right_arm": _chain(person, ["right_shoulder", "right_elbow", "right_wrist"]),
            "left_arm": _chain(person, ["left_shoulder", "left_elbow", "left_wrist"]),
            "right_leg": _chain(person, ["right_hip", "right_knee", "right_ankle"]),
            "left_leg": _chain(person, ["left_hip", "left_knee", "left_ankle"]),
        },
    }


def _choose_target(people: list[dict[str, Any]]) -> tuple[int | None, str]:
    if not people:
        return None, "no body skeleton detected"

    # Identity datasets normally have one dominant target. Pick the skeleton with
    # the largest normalized keypoint bounding box, with a mild center preference
    # as a tie-breaker. Keep every person in JSON so this choice is auditable.
    def score(person: dict[str, Any]) -> tuple[float, float]:
        bbox = person.get("keypoint_bbox") or {}
        area = float(bbox.get("area_fraction", 0.0))
        cx = float(bbox.get("center_x", 0.5))
        cy = float(bbox.get("center_y", 0.5))
        center_distance = math.hypot(cx - 0.5, cy - 0.5)
        return area, -center_distance

    chosen = max(range(len(people)), key=lambda i: score(people[i]))
    return chosen, "largest keypoint bbox; center proximity used as tie-breaker"


def derive_pose_summary(pose_data: dict[str, Any]) -> dict[str, Any]:
    candidate = _candidate_array(pose_data)
    people = [_person_summary(candidate[i], i) for i in range(candidate.shape[0])]
    target_index, reason = _choose_target(people)
    target = people[target_index] if target_index is not None else None

    return {
        "person_count": len(people),
        "target_person_index": target_index,
        "target_selection": reason,
        "target": target,
        "people": people,
        "limitations": [
            "DWPose provides 2D keypoint evidence; it does not independently establish front-vs-back torso orientation or metric depth.",
            "Anatomical left/right labels are model predictions and can still be wrong in ambiguous or rear-facing poses.",
            "keypoint_bbox is a skeleton extent proxy, not a segmentation-derived subject coverage measurement.",
        ],
    }


def _onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed; re-run build_workspace.sh") from exc

    # Import torch before this helper is called. Modern ORT releases can also
    # preload CUDA/cuDNN libraries from PyTorch/NVIDIA site-packages.
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:
            print(f"WARNING: ONNX Runtime CUDA preload failed: {exc}", file=sys.stderr)
    return list(ort.get_available_providers())


def _resolve_device(requested: str) -> tuple[str, list[str]]:
    providers = _onnxruntime_providers()
    ort_cuda = "CUDAExecutionProvider" in providers
    torch_cuda = torch.cuda.is_available()

    if requested == "cpu":
        return "cpu", providers
    if requested == "cuda":
        if not torch_cuda:
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        if not ort_cuda:
            raise RuntimeError(
                "--device cuda requested but ONNX Runtime has no CUDAExecutionProvider. "
                f"Available providers: {providers}"
            )
        return "cuda:0", providers

    if torch_cuda and ort_cuda:
        return "cuda:0", providers
    return "cpu", providers


def _load_detector(device: str):
    try:
        from easy_dwpose import DWposeDetector
    except ImportError as exc:
        raise RuntimeError(
            "DWPose support is not installed. Re-run build_workspace.sh."
        ) from exc

    started = time.perf_counter()
    detector = DWposeDetector(device=device)
    return detector, time.perf_counter() - started


def _run_pose(detector: Any, image: Image.Image) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    # easy-dwpose returns its numerical pose dictionary when drawing is disabled.
    pose_data = detector(image, draw_pose=False)
    seconds = time.perf_counter() - started
    if not isinstance(pose_data, dict):
        raise RuntimeError(f"Unexpected easy-dwpose output type: {type(pose_data)!r}")
    return pose_data, seconds


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not dataset.is_dir():
        print(f"Dataset folder does not exist: {dataset}", file=sys.stderr)
        return 2

    images = discover_images(dataset, recursive=args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print(f"No supported images found in {dataset}", file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    device, providers = _resolve_device(args.device)

    print(f"ONNX Runtime providers: {providers}")
    if args.device == "auto" and device == "cpu" and torch.cuda.is_available():
        print("DWPose auto-selection: CUDA GPU exists, but ORT CUDA provider is unavailable; using CPU.")
    print(f"Loading DWPose on {device} ...")
    detector, load_seconds = _load_detector(device)
    print(f"Loaded DWPose in {load_seconds:.2f}s. Processing {len(images)} image(s).")

    records: list[dict[str, Any]] = []
    for image_path in tqdm(images, desc="DWPose"):
        rel = image_path.relative_to(dataset)
        key = _result_key(rel)
        result_path = output / f"{key}.dwpose.json"

        if result_path.exists() and not args.overwrite:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                records.append(result)
                continue
            except Exception:
                pass

        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            pose_data, seconds = _run_pose(detector, rgb)

        derived = derive_pose_summary(pose_data)
        result = {
            "schema_version": "dwpose-profile-1.0",
            "image": str(rel),
            "image_width": width,
            "image_height": height,
            "device": device,
            "onnxruntime_providers": providers,
            "inference_seconds": round(seconds, 4),
            "derived": derived,
            "raw_pose": _to_builtin(pose_data),
        }
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        records.append(result)

    extent_counts: Counter[str] = Counter()
    person_counts: Counter[int] = Counter()
    no_pose = 0
    bbox_heights: list[float] = []
    bbox_areas: list[float] = []
    inference_times: list[float] = []

    for record in records:
        derived = record.get("derived") or {}
        count = int(derived.get("person_count") or 0)
        person_counts[count] += 1
        if count == 0:
            no_pose += 1
        target = derived.get("target") or {}
        extent = target.get("pose_extent_hint")
        if extent:
            extent_counts[str(extent)] += 1
        bbox = target.get("keypoint_bbox") or {}
        if "height_fraction" in bbox:
            bbox_heights.append(float(bbox["height_fraction"]))
        if "area_fraction" in bbox:
            bbox_areas.append(float(bbox["area_fraction"]))
        if "inference_seconds" in record:
            inference_times.append(float(record["inference_seconds"]))

    dataset_summary = {
        "schema_version": "dwpose-dataset-profile-1.0",
        "dataset": str(dataset),
        "image_count": len(records),
        "device": device,
        "onnxruntime_providers": providers,
        "model_load_seconds": round(load_seconds, 4),
        "average_inference_seconds": round(sum(inference_times) / len(inference_times), 4) if inference_times else None,
        "no_body_pose_count": no_pose,
        "person_count_histogram": {str(k): v for k, v in sorted(person_counts.items())},
        "pose_extent_hint_counts": dict(sorted(extent_counts.items())),
        "average_target_keypoint_bbox_height_fraction": round(sum(bbox_heights) / len(bbox_heights), 4) if bbox_heights else None,
        "average_target_keypoint_bbox_area_fraction": round(sum(bbox_areas) / len(bbox_areas), 4) if bbox_areas else None,
        "files": [
            {
                "image": r.get("image"),
                "dwpose_json": f"{_result_key(Path(str(r.get('image'))))}.dwpose.json",
                "inference_seconds": r.get("inference_seconds"),
                "person_count": (r.get("derived") or {}).get("person_count"),
                "pose_extent_hint": ((r.get("derived") or {}).get("target") or {}).get("pose_extent_hint"),
                "target_keypoint_bbox": ((r.get("derived") or {}).get("target") or {}).get("keypoint_bbox"),
            }
            for r in records
        ],
    }
    summary_path = output / "dataset.dwpose.json"
    summary_path.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done. Dataset summary: {summary_path}")
    print(f"Average inference: {dataset_summary['average_inference_seconds']}s/image")
    print(f"Extent hints: {dataset_summary['pose_extent_hint_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
