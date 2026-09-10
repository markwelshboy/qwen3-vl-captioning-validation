from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dwpose_compat import target_points_from_profile_record

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SCHEMA_VERSION = "gaze-probe-pyfeat-v28-0.1"
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
    for suffix in (".dwpose.json", ".json"):
        direct = directory / f"{key}{suffix}"
        if direct.is_file():
            return direct
    matches = sorted(directory.rglob(f"{key}*.json"))
    return matches[0] if matches else None


def _usable(point: np.ndarray) -> bool:
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    return bool(p.size >= 2 and np.isfinite(p[:2]).all() and p[0] >= 0 and p[1] >= 0)


def _dwpose_head_center(record: dict[str, Any], width: int, height: int) -> list[float] | None:
    points = target_points_from_profile_record(record, width, height)
    if len(points) < 18:
        return None

    face = []
    for name in FACE_NAMES:
        point = points[IDX[name]]
        if _usable(point):
            face.append(np.asarray(point[:2], dtype=np.float64))
    if face:
        center = np.median(np.stack(face), axis=0)
        return [float(center[0]), float(center[1])]

    neck = points[IDX["neck"]]
    rs = points[IDX["right_shoulder"]]
    ls = points[IDX["left_shoulder"]]
    if _usable(neck) and _usable(rs) and _usable(ls):
        shoulder_mid = (np.asarray(rs[:2]) + np.asarray(ls[:2])) * 0.5
        center = np.asarray(neck[:2]) + 0.85 * (np.asarray(neck[:2]) - shoulder_mid)
        return [float(center[0]), float(center[1])]
    return None


def _as_float(row: Any, name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bbox_from_row(row: Any) -> np.ndarray | None:
    x = _as_float(row, "FaceRectX")
    y = _as_float(row, "FaceRectY")
    w = _as_float(row, "FaceRectWidth")
    h = _as_float(row, "FaceRectHeight")
    if None in (x, y, w, h) or w <= 0 or h <= 0:
        return None
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def _select_row(fex: Any, expected_head: list[float] | None, threshold: float) -> tuple[int, str] | None:
    candidates: list[int] = []
    for i in range(len(fex)):
        row = fex.iloc[i]
        bbox = _bbox_from_row(row)
        score = _as_float(row, "FaceScore")
        if bbox is not None and score is not None and score >= threshold:
            candidates.append(i)
    if not candidates:
        return None

    if expected_head is not None:
        expected = np.asarray(expected_head, dtype=np.float64)

        def dist(i: int) -> float:
            box = _bbox_from_row(fex.iloc[i])
            assert box is not None
            center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
            return float(np.linalg.norm(center - expected))

        return min(candidates, key=dist), "nearest_dwpose_head_center"

    return max(candidates, key=lambda i: float(fex.iloc[i]["FaceScore"])), "highest_retinaface_score"


def _extract_landmarks68(row: Any) -> np.ndarray:
    xs = [_as_float(row, f"x_{i}") for i in range(68)]
    ys = [_as_float(row, f"y_{i}") for i in range(68)]
    out = np.full((68, 2), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x is not None and y is not None:
            out[i] = [x, y]
    return out


def _extract_mesh478(row: Any) -> np.ndarray:
    out = np.full((478, 3), np.nan, dtype=np.float32)
    for i in range(478):
        x = _as_float(row, f"mesh_x_{i}")
        y = _as_float(row, f"mesh_y_{i}")
        z = _as_float(row, f"mesh_z_{i}")
        if x is not None and y is not None and z is not None:
            out[i] = [x, y, z]
    return out


def _zero_direction_deviation_deg(pitch_rad: float, yaw_rad: float) -> float:
    dot = float(np.clip(math.cos(pitch_rad) * math.cos(yaw_rad), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _draw_overlay(
    image: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray,
    head_pitch: float,
    head_yaw: float,
    head_roll: float,
    gaze_pitch: float,
    gaze_yaw: float,
    gaze_mag: float,
) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)

    for point in landmarks:
        if np.isfinite(point).all():
            cv2.circle(out, (int(round(float(point[0]))), int(round(float(point[1])))), 2, (255, 180, 0), -1)

    # Canonical py-feat Fex convention: +pitch is up; +yaw is subject-right,
    # which projects toward image-left for a frontal face.
    cx, cy = int(round((x0 + x1) * 0.5)), int(round((y0 + y1) * 0.5))
    length = max(45, int(round(max(x1 - x0, y1 - y0) * 0.75)))
    dx = -length * math.sin(gaze_yaw) * math.cos(gaze_pitch)
    dy = -length * math.sin(gaze_pitch)
    end = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.arrowedLine(out, (cx, cy), end, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.17)

    lines = [
        f"head P={math.degrees(head_pitch):+.1f} Y={math.degrees(head_yaw):+.1f} R={math.degrees(head_roll):+.1f}",
        f"gaze P={math.degrees(gaze_pitch):+.1f} Y={math.degrees(gaze_yaw):+.1f} mag={gaze_mag:.1f}",
    ]
    y = 28
    for text in lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return out


def _process_one(
    detector: Any,
    image_path: Path,
    output_dir: Path,
    dwpose_dir: Path | None,
    threshold: float,
) -> dict[str, Any]:
    image = cv2.imread(image_path.as_posix())
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    dwpose_path = _find_for_key(dwpose_dir, image_path.stem)
    dwpose = _read_json(dwpose_path)
    expected_head = _dwpose_head_center(dwpose, width, height) if dwpose else None

    fex = detector.detect(
        inputs=image_path.as_posix(),
        data_type="image",
        face_detection_threshold=threshold,
        batch_size=1,
        num_workers=0,
    )
    selected = _select_row(fex, expected_head, threshold)
    if selected is None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "backend": "pyfeat_detectorv2",
            "model": "face_multitask_v28",
            "image_key": image_path.stem,
            "image": image_path.as_posix(),
            "status": "no_face",
            "face_acquisition": {
                "method": "retinaface_full_image",
                "rows_returned": int(len(fex)),
                "dwpose_path": str(dwpose_path) if dwpose_path else None,
                "expected_head_center_xy": expected_head,
            },
        }
        _write_json(output_dir / f"{image_path.stem}.pyfeat.json", record)
        return record

    row_index, strategy = selected
    row = fex.iloc[row_index]
    bbox = _bbox_from_row(row)
    assert bbox is not None
    score = _as_float(row, "FaceScore")

    hp = _as_float(row, "Pitch")
    hr = _as_float(row, "Roll")
    hy = _as_float(row, "Yaw")
    gp = _as_float(row, "gaze_pitch")
    gy = _as_float(row, "gaze_yaw")
    ga = _as_float(row, "gaze_angle")
    required = (hp, hr, hy, gp, gy)
    if any(v is None for v in required):
        raise RuntimeError(f"py-feat returned selected face without complete head/gaze outputs: {image_path.name}")

    assert hp is not None and hr is not None and hy is not None and gp is not None and gy is not None
    gaze_mag = _zero_direction_deviation_deg(gp, gy)
    landmarks = _extract_landmarks68(row)
    mesh = _extract_mesh478(row)

    arrays_path = output_dir / f"{image_path.stem}.pyfeat_arrays.npz"
    np.savez_compressed(arrays_path, landmarks68=landmarks, mesh478=mesh)

    overlay_path = output_dir / f"{image_path.stem}.pyfeat.png"
    overlay = _draw_overlay(image, bbox, landmarks, hp, hy, hr, gp, gy, gaze_mag)
    cv2.imwrite(overlay_path.as_posix(), overlay)

    record = {
        "schema_version": SCHEMA_VERSION,
        "backend": "pyfeat_detectorv2",
        "model": "face_multitask_v28",
        "image_key": image_path.stem,
        "image": image_path.as_posix(),
        "status": "ok",
        "face": {
            "score": score,
            "bbox_xyxy": [float(v) for v in bbox],
            "selected_row_index": int(row_index),
            "candidate_count": int(sum(_bbox_from_row(fex.iloc[i]) is not None for i in range(len(fex)))),
        },
        "face_acquisition": {
            "method": "retinaface_full_image",
            "selection_strategy": strategy,
            "dwpose_path": str(dwpose_path) if dwpose_path else None,
            "expected_head_center_xy": expected_head,
        },
        "head_pose": {
            "pitch_rad": hp,
            "yaw_rad": hy,
            "roll_rad": hr,
            "pitch_deg": math.degrees(hp),
            "yaw_deg": math.degrees(hy),
            "roll_deg": math.degrees(hr),
            "translation": {
                "x": _as_float(row, "X"),
                "y": _as_float(row, "Y"),
                "z": _as_float(row, "Z"),
            },
            "convention": "+pitch=up, +yaw=subject-right, +roll=subject-right",
        },
        "gaze": {
            "pitch_rad": gp,
            "yaw_rad": gy,
            "pitch_deg": math.degrees(gp),
            "yaw_deg": math.degrees(gy),
            "gaze_angle_rad_from_fex": ga,
            "gaze_angle_deg_from_fex": math.degrees(ga) if ga is not None else None,
            "zero_direction_deviation_deg": gaze_mag,
            "convention": "canonical py-feat Fex values; +pitch=up; +yaw=subject-right/image-left for frontal face",
        },
        "arrays": arrays_path.as_posix(),
        "overlay": overlay_path.as_posix(),
        "notes": [
            "Head and gaze values are read from Detectorv2's canonical Fex columns, not raw multitask tensor order.",
            "zero_direction_deviation_deg is only angular magnitude from zero gaze; it is NOT a gaze-to-lens metric.",
            "DWPose is used only to select the intended face when multiple RetinaFace detections are returned.",
            "No semantic gaze labels or trust thresholds are applied in this diagnostic probe.",
        ],
    }
    _write_json(output_dir / f"{image_path.stem}.pyfeat.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static-image probe for py-feat Detectorv2 / face_multitask_v28.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--only", nargs="+", default=[])
    return parser.parse_args()


def main() -> int:
    from feat import Detectorv2
    import feat

    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = Detectorv2(
        device=args.device,
        face_detection_threshold=args.confidence_threshold,
        identity_model=None,
        compile=False,
    )

    images = [p for p in _discover_images(source) if _matches(p, args.only)]
    if not images:
        raise SystemExit("No matching images found")

    records = []
    for image_path in images:
        record = _process_one(detector, image_path, output_dir, dwpose_dir, args.confidence_threshold)
        records.append(record)
        if record["status"] == "ok":
            hp = record["head_pose"]
            gaze = record["gaze"]
            print(
                f"{record['image_key']}: "
                f"head(p={hp['pitch_deg']:+.1f}, y={hp['yaw_deg']:+.1f}, r={hp['roll_deg']:+.1f}) "
                f"gaze(p={gaze['pitch_deg']:+.1f}, y={gaze['yaw_deg']:+.1f}, mag={gaze['zero_direction_deviation_deg']:.1f}) "
                f"score={record['face']['score']:.3f}"
            )
        else:
            print(f"{record['image_key']}: no_face")

    version = getattr(feat, "__version__", None)
    _write_json(output_dir / "gaze_probe_pyfeat_v2.index.json", {
        "schema_version": SCHEMA_VERSION + "-run",
        "backend": "pyfeat_detectorv2",
        "pyfeat_version": version,
        "model": "face_multitask_v28",
        "record_count": len(records),
        "records": records,
    })
    print(f"py-feat multitask gaze/head bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
