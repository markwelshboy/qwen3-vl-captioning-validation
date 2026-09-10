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
SCHEMA_VERSION = "face-authority-uniface-0.1"

BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
IDX = {name: i for i, name in enumerate(BODY18)}
FACE_NAMES = ("nose", "right_eye", "left_eye", "right_ear", "left_ear")

PARSING_CLASSES = {
    0: "background",
    1: "skin",
    2: "left_eyebrow",
    3: "right_eyebrow",
    4: "left_eye",
    5: "right_eye",
    6: "eyeglasses",
    7: "left_ear",
    8: "right_ear",
    9: "earring",
    10: "nose",
    11: "mouth",
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}

PARSE_COLORS = np.array([
    [0, 0, 0],
    [180, 180, 180],
    [120, 80, 40],
    [120, 80, 40],
    [0, 255, 255],
    [0, 255, 255],
    [255, 180, 0],
    [180, 120, 80],
    [180, 120, 80],
    [255, 0, 255],
    [0, 180, 255],
    [0, 0, 255],
    [0, 80, 255],
    [0, 80, 255],
    [120, 120, 255],
    [255, 255, 0],
    [80, 80, 80],
    [80, 40, 20],
    [255, 255, 255],
], dtype=np.uint8)


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


def _face_center(face: Any) -> np.ndarray:
    box = np.asarray(face.bbox, dtype=np.float64).reshape(-1)
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)


def _select_face(faces: list[Any], expected_head: list[float] | None) -> tuple[Any, str] | None:
    if not faces:
        return None
    if expected_head is not None:
        expected = np.asarray(expected_head, dtype=np.float64)
        face = min(faces, key=lambda f: float(np.linalg.norm(_face_center(f) - expected)))
        return face, "nearest_dwpose_head_center"
    face = max(faces, key=lambda f: float(f.confidence))
    return face, "highest_retinaface_score"


def _clip_bbox(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in np.asarray(bbox).reshape(-1)[:4]]
    ix0 = max(0, min(width - 1, int(math.floor(x0))))
    iy0 = max(0, min(height - 1, int(math.floor(y0))))
    ix1 = max(ix0 + 1, min(width, int(math.ceil(x1))))
    iy1 = max(iy0 + 1, min(height, int(math.ceil(y1))))
    return ix0, iy0, ix1, iy1


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _iris_stats(points: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(arr).all(axis=1) if arr.ndim == 2 else np.zeros((0,), dtype=bool)
    result: dict[str, Any] = {
        "point_count": int(arr.shape[0]) if arr.ndim == 2 else 0,
        "finite_count": int(finite.sum()),
        "all_finite": bool(arr.ndim == 2 and len(arr) == 5 and finite.all()),
        "center_xy": None,
        "radius_px": None,
    }
    if arr.ndim == 2 and arr.shape[0] >= 5 and arr.shape[1] >= 2 and finite[:5].all():
        center = arr[0, :2]
        radius = float(np.mean(np.linalg.norm(arr[1:5, :2] - center[None, :], axis=1)))
        result["center_xy"] = [float(center[0]), float(center[1])]
        result["radius_px"] = radius
    return result


def _fraction_map(mask: np.ndarray) -> dict[str, float]:
    total = int(mask.size)
    if total <= 0:
        return {}
    return {
        name: float(np.count_nonzero(mask == class_id) / total)
        for class_id, name in PARSING_CLASSES.items()
        if np.any(mask == class_id)
    }


def _circle_region(mask: np.ndarray, center_xy: tuple[float, float], radius: float) -> np.ndarray:
    h, w = mask.shape[:2]
    cx, cy = center_xy
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def _region_class_fractions(mask: np.ndarray, region: np.ndarray) -> dict[str, float]:
    count = int(np.count_nonzero(region))
    if count <= 0:
        return {}
    values = mask[region]
    eye = np.count_nonzero((values == 4) | (values == 5))
    glasses = np.count_nonzero(values == 6)
    hair = np.count_nonzero(values == 17)
    background = np.count_nonzero(values == 0)
    skin = np.count_nonzero(values == 1)
    return {
        "eye": float(eye / count),
        "eyeglasses": float(glasses / count),
        "hair": float(hair / count),
        "background": float(background / count),
        "skin": float(skin / count),
        "other": float(max(0, count - eye - glasses - hair - background - skin) / count),
    }


def _eye_parse_stats(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    x0, y0, x1, _ = bbox
    centers = []
    for item in (left, right):
        if item.get("center_xy") is not None:
            centers.append(np.asarray(item["center_xy"], dtype=np.float64))
    inter_eye = float(np.linalg.norm(centers[0] - centers[1])) if len(centers) == 2 else None

    fallback_radius = 0.10 * max(1, x1 - x0)
    radius = max(5.0, 0.28 * inter_eye) if inter_eye is not None else max(5.0, fallback_radius)
    radius = min(radius, max(6.0, 0.22 * max(1, x1 - x0)))

    local: list[dict[str, Any]] = []
    for label, item in (("iris_left", left), ("iris_right", right)):
        center = item.get("center_xy")
        if center is None:
            local.append({"label": label, "available": False})
            continue
        cx = float(center[0]) - x0
        cy = float(center[1]) - y0
        region = _circle_region(mask, (cx, cy), radius)
        local.append({
            "label": label,
            "available": True,
            "center_crop_xy": [cx, cy],
            "radius_px": float(radius),
            "fractions": _region_class_fractions(mask, region),
        })

    return {
        "inter_iris_distance_px": inter_eye,
        "roi_radius_px": float(radius),
        "iris_regions": local,
    }


def _parse_visual(mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(mask.astype(np.int32), 0, len(PARSE_COLORS) - 1)
    return PARSE_COLORS[clipped]


def _draw_full_overlay(
    image: np.ndarray,
    face: Any,
    head: Any,
    states: dict[str, float],
    quality: float | None,
    mesh_points: np.ndarray,
) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(face.bbox).reshape(-1)[:4]]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)

    for point in np.asarray(face.landmarks):
        if np.asarray(point).size >= 2 and np.isfinite(point[:2]).all():
            cv2.circle(out, (int(round(float(point[0]))), int(round(float(point[1])))), 4, (0, 255, 255), -1)

    if mesh_points.ndim == 2 and mesh_points.shape[0] >= 478:
        for idx in range(468, 478):
            p = mesh_points[idx, :2]
            if np.isfinite(p).all():
                cv2.circle(out, (int(round(float(p[0]))), int(round(float(p[1])))), 2, (255, 0, 255), -1)

    lines = [
        f"face={float(face.confidence):.3f} quality={quality:.3f}" if quality is not None else f"face={float(face.confidence):.3f} quality=n/a",
        f"head raw: P={head.pitch:+.1f} Y={head.yaw:+.1f} R={head.roll:+.1f}",
        f"eyes open L={states.get('left_eye_open', float('nan')):.2f} R={states.get('right_eye_open', float('nan')):.2f}",
        f"glasses={states.get('eyeglasses', float('nan')):.2f} sunglasses={states.get('sunglasses', float('nan')):.2f} mask={states.get('mask', float('nan')):.2f}",
    ]
    y = 28
    for text in lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 27
    return out


def _draw_face_diag(
    crop: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    left: dict[str, Any],
    right: dict[str, Any],
    eye_stats: dict[str, Any],
) -> np.ndarray:
    x0, y0, _, _ = bbox
    face_view = crop.copy()
    parse_view = _parse_visual(mask)

    radius = float(eye_stats.get("roi_radius_px") or 0.0)
    for item in (left, right):
        center = item.get("center_xy")
        if center is None:
            continue
        cx = int(round(float(center[0]) - x0))
        cy = int(round(float(center[1]) - y0))
        if radius > 0:
            cv2.circle(face_view, (cx, cy), int(round(radius)), (0, 255, 255), 2)
            cv2.circle(parse_view, (cx, cy), int(round(radius)), (255, 255, 255), 2)
        cv2.circle(face_view, (cx, cy), 3, (255, 0, 255), -1)
        cv2.circle(parse_view, (cx, cy), 3, (255, 0, 255), -1)

    return np.concatenate([face_view, parse_view], axis=1)


def _process_one(
    image_path: Path,
    output_dir: Path,
    dwpose_dir: Path | None,
    models: dict[str, Any],
) -> dict[str, Any]:
    image = cv2.imread(image_path.as_posix())
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    dwpose_path = _find_for_key(dwpose_dir, image_path.stem)
    dwpose = _read_json(dwpose_path)
    expected_head = _dwpose_head_center(dwpose, width, height) if dwpose else None

    faces = models["detector"].detect(image)
    selected = _select_face(faces, expected_head)
    if selected is None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "backend": "uniface",
            "image_key": image_path.stem,
            "image": image_path.as_posix(),
            "status": "no_face",
            "face_acquisition": {
                "candidate_count": 0,
                "dwpose_path": str(dwpose_path) if dwpose_path else None,
                "expected_head_center_xy": expected_head,
            },
        }
        _write_json(output_dir / f"{image_path.stem}.uniface.json", record)
        return record

    face, strategy = selected
    bbox = _clip_bbox(face.bbox, width, height)
    x0, y0, x1, y1 = bbox
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise RuntimeError(f"Selected UniFace bbox produced empty crop: {image_path.name}")

    state_result = models["facestate"].predict(image, face)
    states = {k: float(v) for k, v in state_result.as_dict().items()}

    quality_result = models["quality"].predict(image, np.asarray(face.landmarks))
    quality_score = _safe_float(quality_result.score)

    mesh_results = models["mesh"].predict(image, faces=[face])
    mesh_result = mesh_results[0] if mesh_results else None
    mesh_points = (
        np.asarray(mesh_result.landmarks, dtype=np.float32)
        if mesh_result is not None
        else np.empty((0, 3), dtype=np.float32)
    )

    from uniface.landmark import IRIS_LEFT, IRIS_RIGHT
    left_points = mesh_points[IRIS_LEFT] if len(mesh_points) >= 478 else np.empty((0, 3), dtype=np.float32)
    right_points = mesh_points[IRIS_RIGHT] if len(mesh_points) >= 478 else np.empty((0, 3), dtype=np.float32)
    left_iris = _iris_stats(left_points)
    right_iris = _iris_stats(right_points)

    parse_mask = models["parser"].parse(crop)
    eye_parse = _eye_parse_stats(parse_mask, bbox, left_iris, right_iris)

    head = models["headpose"].estimate(crop)

    arrays_path = output_dir / f"{image_path.stem}.uniface_arrays.npz"
    np.savez_compressed(
        arrays_path,
        mesh478=mesh_points,
        parse_mask=parse_mask.astype(np.uint8),
        detector_landmarks5=np.asarray(face.landmarks, dtype=np.float32),
    )

    overlay_path = output_dir / f"{image_path.stem}.uniface.png"
    overlay = _draw_full_overlay(image, face, head, states, quality_score, mesh_points)
    cv2.imwrite(overlay_path.as_posix(), overlay)

    face_diag_path = output_dir / f"{image_path.stem}.uniface_face_diag.png"
    face_diag = _draw_face_diag(crop, parse_mask, bbox, left_iris, right_iris, eye_parse)
    cv2.imwrite(face_diag_path.as_posix(), face_diag)

    finite_mesh = int(np.isfinite(mesh_points).all(axis=1).sum()) if mesh_points.ndim == 2 else 0
    parse_fractions = _fraction_map(parse_mask)

    record = {
        "schema_version": SCHEMA_VERSION,
        "backend": "uniface",
        "models": {
            "detector": "RetinaFace MNET_V2",
            "face_state": "FaceAttribNet",
            "face_mesh": "FaceMesh V2_478",
            "face_quality": "eDifFIQA-T",
            "face_parsing": "BiSeNet ResNet18",
            "head_pose": "HeadPose ResNet50",
        },
        "image_key": image_path.stem,
        "image": image_path.as_posix(),
        "status": "ok",
        "face": {
            "score": float(face.confidence),
            "bbox_xyxy": [float(v) for v in np.asarray(face.bbox).reshape(-1)[:4]],
            "selection_strategy": strategy,
            "candidate_count": int(len(faces)),
            "landmarks5": np.asarray(face.landmarks, dtype=float).tolist(),
        },
        "face_acquisition": {
            "dwpose_path": str(dwpose_path) if dwpose_path else None,
            "expected_head_center_xy": expected_head,
        },
        "face_state": states,
        "face_quality": {
            "score": quality_score,
            "note": "eDifFIQA is a relative face-quality score, not a calibrated probability.",
        },
        "face_mesh": {
            "presence_score": _safe_float(mesh_result.score) if mesh_result is not None else None,
            "finite_landmarks": finite_mesh,
            "landmark_count": int(mesh_points.shape[0]) if mesh_points.ndim == 2 else 0,
            "iris_left": left_iris,
            "iris_right": right_iris,
            "inter_iris_distance_px": eye_parse.get("inter_iris_distance_px"),
            "note": "FaceMesh presence score saturates near 1 and is diagnostic only.",
        },
        "face_parsing": {
            "crop_bbox_xyxy": [x0, y0, x1, y1],
            "class_fractions": parse_fractions,
            "eye_region_diagnostics": eye_parse,
            "note": "Eye-region fractions are measured in circles around predicted iris centers; diagnostic only.",
        },
        "head_pose": {
            "pitch_deg_raw": float(head.pitch),
            "yaw_deg_raw": float(head.yaw),
            "roll_deg_raw": float(head.roll),
            "convention": "UniFace raw: +pitch=down, +yaw=looking right, +roll=clockwise",
            "pitch_deg_pyfeat_compare": float(-head.pitch),
            "note": "Only pitch is sign-normalized for py-feat comparison; yaw/roll remain raw until visually calibrated.",
        },
        "arrays": arrays_path.as_posix(),
        "overlay": overlay_path.as_posix(),
        "face_diag": face_diag_path.as_posix(),
        "notes": [
            "This probe records raw quality/visibility evidence and deliberately applies no final gaze-authority threshold.",
            "Sunglasses, eye openness, parsing and iris geometry must be validated empirically before becoming hard gates.",
        ],
    }
    _write_json(output_dir / f"{image_path.stem}.uniface.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniFace diagnostic probe for gaze-authority signals and head pose.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--provider", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--only", nargs="+", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else None

    providers = ["CPUExecutionProvider"] if args.provider == "cpu" else None

    from uniface.attribute import FaceAttribNet
    from uniface.constants import FaceMeshWeights, HeadPoseWeights
    from uniface.detection import RetinaFace
    from uniface.headpose import HeadPose
    from uniface.landmark import FaceMesh
    from uniface.parsing import BiSeNet
    from uniface.quality import EDifFIQA

    models = {
        "detector": RetinaFace(confidence_threshold=args.confidence_threshold, providers=providers),
        "facestate": FaceAttribNet(providers=providers),
        "mesh": FaceMesh(model_name=FaceMeshWeights.V2_478, providers=providers),
        "quality": EDifFIQA(providers=providers),
        "parser": BiSeNet(providers=providers),
        "headpose": HeadPose(model_name=HeadPoseWeights.RESNET50, providers=providers),
    }

    images = [p for p in _discover_images(source) if _matches(p, args.only)]
    if not images:
        raise SystemExit(f"No matching images found under {source}")

    records: list[dict[str, Any]] = []
    for image_path in images:
        try:
            record = _process_one(image_path, output_dir, dwpose_dir, models)
        except Exception as exc:
            record = {
                "schema_version": SCHEMA_VERSION,
                "backend": "uniface",
                "image_key": image_path.stem,
                "image": image_path.as_posix(),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_json(output_dir / f"{image_path.stem}.uniface.json", record)
        records.append(record)

        if record["status"] == "ok":
            state = record["face_state"]
            q = record["face_quality"]["score"]
            hp = record["head_pose"]
            mesh = record["face_mesh"]
            prefix = (
                f"{image_path.stem}: face={record['face']['score']:.3f} q={q:.3f} "
                if q is not None
                else f"{image_path.stem}: face={record['face']['score']:.3f} q=n/a "
            )
            print(
                prefix
                + f"eyes(L={state['left_eye_open']:.2f},R={state['right_eye_open']:.2f}) "
                + f"gl={state['eyeglasses']:.2f} sun={state['sunglasses']:.2f} mask={state['mask']:.2f} "
                + f"iris={mesh['inter_iris_distance_px'] if mesh['inter_iris_distance_px'] is not None else 'n/a'} "
                + f"head(raw P={hp['pitch_deg_raw']:+.1f},Y={hp['yaw_deg_raw']:+.1f},R={hp['roll_deg_raw']:+.1f})"
            )
        else:
            print(f"{image_path.stem}: {record['status']} {record.get('error', '')}".rstrip())

    index = {
        "schema_version": SCHEMA_VERSION,
        "source": source.as_posix(),
        "output_dir": output_dir.as_posix(),
        "record_count": len(records),
        "ok_count": sum(1 for r in records if r.get("status") == "ok"),
        "records": records,
    }
    _write_json(output_dir / "face_authority_uniface.index.json", index)
    print(f"UniFace authority probe bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
