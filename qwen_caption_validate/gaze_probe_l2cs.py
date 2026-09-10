from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .dwpose_compat import target_points_from_profile_record

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SCHEMA_VERSION = "gaze-probe-l2cs-0.2"
BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
IDX = {name: i for i, name in enumerate(BODY18)}
FACE_NAMES = ("nose", "right_eye", "left_eye", "right_ear", "left_ear")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _discover_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    chosen: dict[str, Path] = {}
    for p in sorted(q for q in path.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_EXTENSIONS):
        chosen.setdefault(p.stem.lower(), p)
    return list(chosen.values())


def _matches(path: Path, only: list[str]) -> bool:
    if not only:
        return True
    wanted = {item.lower() for item in only}
    return path.stem.lower() in wanted or path.name.lower() in wanted


def _find_for_key(directory: Path | None, key: str) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    direct = directory / f"{key}.dwpose.json"
    if direct.is_file():
        return direct
    direct = directory / f"{key}.json"
    if direct.is_file():
        return direct
    matches = sorted(directory.rglob(f"{key}*.json"))
    return matches[0] if matches else None


def _usable(point: np.ndarray) -> bool:
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    return bool(p.size >= 2 and np.isfinite(p[:2]).all() and p[0] >= 0 and p[1] >= 0)


def _dwpose_geometry(record: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    points = target_points_from_profile_record(record, width, height)
    if len(points) < 18:
        return {"head_center_xy": None, "retry_crop": None, "evidence": []}

    face_points: list[tuple[str, np.ndarray]] = []
    for name in FACE_NAMES:
        p = points[IDX[name]]
        if _usable(p):
            face_points.append((name, np.asarray(p[:2], dtype=np.float64)))

    evidence = [name for name, _ in face_points]
    center: np.ndarray | None = None
    if face_points:
        cloud = np.stack([p for _, p in face_points])
        center = np.median(cloud, axis=0)
        extent = max(float(np.ptp(cloud[:, 0])), float(np.ptp(cloud[:, 1])))
        side = int(round(max(96.0, extent * 2.25))) if extent > 1 else 128
    else:
        neck = points[IDX["neck"]]
        rs = points[IDX["right_shoulder"]]
        ls = points[IDX["left_shoulder"]]
        if _usable(neck) and _usable(rs) and _usable(ls):
            shoulder_mid = (np.asarray(rs[:2]) + np.asarray(ls[:2])) * 0.5
            center = np.asarray(neck[:2]) + 0.85 * (np.asarray(neck[:2]) - shoulder_mid)
            shoulder_width = float(np.linalg.norm(np.asarray(rs[:2]) - np.asarray(ls[:2])))
            side = int(round(max(128.0, shoulder_width * 1.1)))
            evidence = ["neck", "right_shoulder", "left_shoulder"]
        else:
            return {"head_center_xy": None, "retry_crop": None, "evidence": []}

    side = max(64, min(side, width, height, int(round(min(width, height) * 0.42))))
    x0 = int(round(float(center[0]) - side / 2))
    y0 = int(round(float(center[1]) - side / 2))
    x0 = max(0, min(x0, width - side))
    y0 = max(0, min(y0, height - side))
    crop = [x0, y0, x0 + side, y0 + side]
    return {
        "head_center_xy": [float(center[0]), float(center[1])],
        "retry_crop": crop,
        "evidence": evidence,
    }


def _detect(detector: Any, image: np.ndarray, threshold: float, offset: tuple[int, int] = (0, 0)) -> list[dict[str, Any]]:
    raw = detector(image)
    out: list[dict[str, Any]] = []
    if raw is None:
        return out
    ox, oy = offset
    for box, landmarks, score in raw:
        score_f = float(score)
        if score_f < threshold:
            continue
        bbox = np.asarray(box, dtype=np.float64).reshape(-1)[:4].copy()
        bbox[[0, 2]] += ox
        bbox[[1, 3]] += oy
        lm = np.asarray(landmarks, dtype=np.float64).copy()
        if lm.ndim >= 2 and lm.shape[-1] >= 2:
            lm[..., 0] += ox
            lm[..., 1] += oy
        out.append({"bbox": bbox, "landmarks": lm, "score": score_f})
    return out


def _choose_face(candidates: list[dict[str, Any]], expected_head: list[float] | None) -> tuple[int, str]:
    if not candidates:
        raise ValueError("no candidates")
    if expected_head is not None:
        expected = np.asarray(expected_head, dtype=np.float64)

        def dist(i: int) -> float:
            b = candidates[i]["bbox"]
            center = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5])
            return float(np.linalg.norm(center - expected))

        return min(range(len(candidates)), key=dist), "nearest_dwpose_head_center"
    return max(range(len(candidates)), key=lambda i: candidates[i]["score"]), "highest_retinaface_score"


def _clamp_bbox(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if b.size < 4:
        return None
    x0 = max(0, min(int(math.floor(b[0])), width - 1))
    y0 = max(0, min(int(math.floor(b[1])), height - 1))
    x1 = max(1, min(int(math.ceil(b[2])), width))
    y1 = max(1, min(int(math.ceil(b[3])), height))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _forward_deviation_deg(pitch_rad: float, yaw_rad: float) -> float:
    # Magnitude from the model's zero-angle direction. This is intentionally
    # NOT called gaze-to-lens: L2CS does not reconstruct the 3-D face center.
    dot = float(np.clip(math.cos(pitch_rad) * math.cos(yaw_rad), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _annotate(image: np.ndarray, bbox: tuple[int, int, int, int], pitch: float, yaw: float, acquisition: str) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = bbox
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    length = max(40, x1 - x0)
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    # Upstream L2CS vis.py expects the model's first returned quantity on the
    # horizontal axis. model.py shows that quantity is actually yaw; the second
    # is pitch. Keep the upstream arrow geometry but use corrected semantic names.
    dx = -length * math.sin(yaw) * math.cos(pitch)
    dy = -length * math.sin(pitch)
    end = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.arrowedLine(out, (cx, cy), end, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.16)
    lines = [
        f"L2CS gaze pitch={math.degrees(pitch):+.1f} yaw={math.degrees(yaw):+.1f}",
        f"face acquisition={acquisition}",
    ]
    y = 28
    for text in lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return out


def _process_one(
    image_path: Path,
    output_dir: Path,
    pipeline: Any,
    dwpose_dir: Path | None,
    threshold: float,
) -> dict[str, Any]:
    image = cv2.imread(image_path.as_posix())
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    dwpose_path = _find_for_key(dwpose_dir, image_path.stem)
    dwpose = _read_json(dwpose_path)
    geometry = _dwpose_geometry(dwpose, w, h) if dwpose else {"head_center_xy": None, "retry_crop": None, "evidence": []}

    candidates = _detect(pipeline.detector, image, threshold)
    acquisition = "full_image"
    full_count = len(candidates)
    retry_count = 0

    if not candidates and geometry.get("retry_crop"):
        x0, y0, x1, y1 = geometry["retry_crop"]
        crop = image[y0:y1, x0:x1]
        if crop.size:
            candidates = _detect(pipeline.detector, crop, threshold, (x0, y0))
            retry_count = len(candidates)
            if candidates:
                acquisition = "dwpose_head_crop"

    if not candidates:
        record = {
            "schema_version": SCHEMA_VERSION,
            "backend": "l2cs",
            "image_key": image_path.stem,
            "image": image_path.as_posix(),
            "status": "no_face",
            "face_acquisition": {
                "method": "full_image",
                "full_image_face_count": full_count,
                "crop_face_count": retry_count,
                "dwpose_path": str(dwpose_path) if dwpose_path else None,
                "expected_head_center_xy": geometry.get("head_center_xy"),
                "retry_crop": geometry.get("retry_crop"),
            },
        }
        if geometry.get("retry_crop"):
            x0, y0, x1, y1 = geometry["retry_crop"]
            crop_path = output_dir / f"{image_path.stem}.l2cs_retry_crop.png"
            cv2.imwrite(crop_path.as_posix(), image[y0:y1, x0:x1])
            record["face_acquisition"]["retry_crop_asset"] = crop_path.as_posix()
        _write_json(output_dir / f"{image_path.stem}.l2cs.json", record)
        return record

    selected_index, strategy = _choose_face(candidates, geometry.get("head_center_xy"))
    selected = candidates[selected_index]
    bbox = _clamp_bbox(selected["bbox"], w, h)
    if bbox is None:
        raise RuntimeError(f"Invalid selected L2CS bbox for {image_path.name}")
    x0, y0, x1, y1 = bbox
    face = image[y0:y1, x0:x1]
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))

    # L2CS-Net model.py returns (yaw_logits, pitch_logits), but the upstream
    # Pipeline.predict_gaze() assigns those outputs to (pitch, yaw). Therefore
    # its returned tuple is semantically (yaw, pitch). Correct that here.
    upstream_first, upstream_second = pipeline.predict_gaze(np.stack([rgb]))
    yaw = float(np.asarray(upstream_first).reshape(-1)[0])
    pitch = float(np.asarray(upstream_second).reshape(-1)[0])

    overlay_path = output_dir / f"{image_path.stem}.l2cs.png"
    cv2.imwrite(overlay_path.as_posix(), _annotate(image, bbox, pitch, yaw, acquisition))

    record = {
        "schema_version": SCHEMA_VERSION,
        "backend": "l2cs",
        "model": "L2CSNet_gaze360",
        "architecture": "ResNet50",
        "image_key": image_path.stem,
        "image": image_path.as_posix(),
        "status": "ok",
        "face_count": len(candidates),
        "selected_face_index": selected_index,
        "selected_face_bbox_xyxy": [float(v) for v in selected["bbox"]],
        "selected_face_score": float(selected["score"]),
        "face_acquisition": {
            "method": acquisition,
            "full_image_face_count": full_count,
            "crop_face_count": retry_count,
            "selection_strategy": strategy,
            "dwpose_path": str(dwpose_path) if dwpose_path else None,
            "expected_head_center_xy": geometry.get("head_center_xy"),
            "retry_crop": geometry.get("retry_crop"),
        },
        "gaze": {
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "forward_deviation_deg": _forward_deviation_deg(pitch, yaw),
            "upstream_pipeline_return_order": ["yaw", "pitch"],
        },
        "overlay": overlay_path.as_posix(),
        "notes": [
            "Corrected known upstream L2CS Pipeline pitch/yaw label swap: model.py returns yaw logits then pitch logits.",
            "Overlay keeps upstream L2CS arrow geometry with corrected semantic axis names.",
            "forward_deviation_deg is angular magnitude from L2CS zero direction; it is not a gaze-to-camera-lens measurement.",
            "DWPose is used only for target-face selection and optional detector retry, not for gaze inference.",
        ],
    }
    _write_json(output_dir / f"{image_path.stem}.l2cs.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic L2CS-Net Gaze360 probe for static images.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--only", nargs="+", default=[])
    return parser.parse_args()


def main() -> int:
    from l2cs import Pipeline

    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if not weights.is_file():
        raise SystemExit(f"L2CS weights not found: {weights}")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    pipeline = Pipeline(weights=weights, arch="ResNet50", device=device, include_detector=True, confidence_threshold=args.confidence_threshold)

    images = [p for p in _discover_images(source) if _matches(p, args.only)]
    if not images:
        raise SystemExit("No matching images found")

    records = []
    for image_path in images:
        record = _process_one(image_path, output_dir, pipeline, dwpose_dir, args.confidence_threshold)
        records.append(record)
        if record["status"] == "ok":
            gaze = record["gaze"]
            print(
                f"{record['image_key']}: "
                f"gaze(p={gaze['pitch_deg']:+.1f}, y={gaze['yaw_deg']:+.1f}, "
                f"forward={gaze['forward_deviation_deg']:.1f}) "
                f"face={record['face_acquisition']['method']} score={record['selected_face_score']:.3f}"
            )
        else:
            acq = record["face_acquisition"]
            retry = "attempted" if acq.get("retry_crop") else "unavailable"
            print(f"{record['image_key']}: no_face retry={retry}")

    _write_json(output_dir / "gaze_probe_l2cs.index.json", {
        "schema_version": SCHEMA_VERSION + "-run",
        "backend": "l2cs",
        "model": "L2CSNet_gaze360",
        "record_count": len(records),
        "records": records,
    })
    print(f"L2CS gaze probe bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
