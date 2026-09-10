from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import OmegaConf

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SCHEMA_VERSION = "gaze-probe-ptgaze-0.3"

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
BODY18_IDX = {name: index for index, name in enumerate(BODY18)}


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
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


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
    candidates = sorted(directory.rglob(f"{key}*.json"))
    return candidates[0] if candidates else None


def _bbox_area(bbox: np.ndarray) -> float:
    box = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if box.size < 4:
        return 0.0
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    if va.size != 3 or vb.size != 3 or not np.isfinite(va).all() or not np.isfinite(vb).all():
        return None
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-9 or nb <= 1e-9:
        return None
    dot = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _angle_from_optical_axis_deg(gaze_vector: np.ndarray) -> float | None:
    # ptgaze's zero pitch/yaw gaze vector is [0, 0, -1]. This measures
    # deviation from the optical axis, NOT whether the subject is looking
    # at the camera lens when the face is off-axis in the image.
    return _angle_between_deg(gaze_vector, np.array([0.0, 0.0, -1.0]))


def _angle_to_camera_origin_deg(gaze_vector: np.ndarray, face_center: np.ndarray | None) -> float | None:
    if face_center is None:
        return None
    center = np.asarray(face_center, dtype=np.float64).reshape(-1)
    if center.size != 3 or not np.isfinite(center).all():
        return None
    # ptgaze coordinates place the camera at the origin. The ray from the
    # reconstructed face center back to the lens is therefore -face_center.
    return _angle_between_deg(gaze_vector, -center)


def _normalized_to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)[..., :2].copy()
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr).all(axis=-1)]
    if not finite.size:
        return arr
    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    if minimum >= -0.05 and maximum <= 1.25:
        arr[..., 0] *= width
        arr[..., 1] *= height
    elif minimum >= -1.25 and maximum <= 1.25:
        arr[..., 0] = (arr[..., 0] + 1.0) * 0.5 * width
        arr[..., 1] = (arr[..., 1] + 1.0) * 0.5 * height
    return arr


def _dwpose_target_points(record: dict[str, Any], width: int, height: int) -> np.ndarray:
    raw = record.get("raw_pose") or {}
    bodies = raw.get("bodies") or {}
    candidate = np.asarray(bodies.get("candidate", []), dtype=np.float64)
    if candidate.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if candidate.ndim == 2:
        candidate = candidate[None, ...]
    if candidate.ndim != 3 or candidate.shape[-1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    target_index = int(((record.get("derived") or {}).get("target_person_index") or 0))
    if target_index < 0 or target_index >= candidate.shape[0]:
        target_index = 0
    return _normalized_to_pixels(candidate[target_index, :18, :2], width, height)


def _usable_point(point: np.ndarray) -> bool:
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    return bool(
        value.size >= 2
        and np.isfinite(value[:2]).all()
        and float(value[0]) >= 0.0
        and float(value[1]) >= 0.0
    )


def _dwpose_head_crop(record: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    """Build a conservative square face-search crop from cached DWPose landmarks.

    This is only a detector assist. If MediaPipe finds a face in the crop, its
    landmarks are translated back to full-image coordinates before ptgaze runs,
    so head/gaze geometry still uses the original image camera model.
    """
    points = _dwpose_target_points(record, width, height)
    if len(points) < 18:
        return None

    def point(name: str) -> np.ndarray | None:
        index = BODY18_IDX[name]
        if index >= len(points) or not _usable_point(points[index]):
            return None
        return np.asarray(points[index, :2], dtype=np.float64)

    face_names = ("nose", "right_eye", "left_eye", "right_ear", "left_ear")
    face_pairs = [(name, point(name)) for name in face_names]
    face_pairs = [(name, value) for name, value in face_pairs if value is not None]

    neck = point("neck")
    right_shoulder = point("right_shoulder")
    left_shoulder = point("left_shoulder")

    evidence: list[str] = [name for name, _ in face_pairs]
    if face_pairs:
        face_cloud = np.stack([value for _, value in face_pairs])
        center = np.median(face_cloud, axis=0)
    elif neck is not None and right_shoulder is not None and left_shoulder is not None:
        shoulder_mid = (right_shoulder + left_shoulder) * 0.5
        # Continue from the shoulder midpoint through the neck to estimate the
        # face center. This remains valid for rotated/reclining image-plane poses.
        center = neck + 0.85 * (neck - shoulder_mid)
        evidence.extend(["neck", "right_shoulder", "left_shoulder"])
    else:
        return None

    scale_candidates = [72.0, min(width, height) * 0.10]
    if len(face_pairs) >= 2:
        face_cloud = np.stack([value for _, value in face_pairs])
        face_extent = max(float(np.ptp(face_cloud[:, 0])), float(np.ptp(face_cloud[:, 1])))
        if face_extent > 0:
            scale_candidates.append(face_extent * 4.0)
    if right_shoulder is not None and left_shoulder is not None:
        shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))
        if shoulder_width > 0:
            scale_candidates.append(shoulder_width * 1.15)
            evidence.extend(name for name in ("right_shoulder", "left_shoulder") if name not in evidence)
    nose = point("nose")
    if nose is not None and neck is not None:
        nose_neck = float(np.linalg.norm(nose - neck))
        if nose_neck > 0:
            scale_candidates.append(nose_neck * 2.8)
            if "neck" not in evidence:
                evidence.append("neck")

    side = int(round(max(scale_candidates)))
    side = max(48, min(side, width, height, int(round(min(width, height) * 0.55))))
    if side < 16:
        return None

    x0 = int(round(float(center[0]) - side / 2.0))
    y0 = int(round(float(center[1]) - side / 2.0))
    x0 = max(0, min(x0, width - side))
    y0 = max(0, min(y0, height - side))
    x1 = x0 + side
    y1 = y0 + side

    return {
        "bbox_xyxy": [x0, y0, x1, y1],
        "center_xy": [float(center[0]), float(center[1])],
        "side_px": side,
        "evidence_landmarks": evidence,
    }


def _translate_face_to_full_image(face, x0: int, y0: int) -> None:
    offset = np.array([float(x0), float(y0)], dtype=np.float64)
    face.bbox = np.asarray(face.bbox, dtype=np.float64) + offset
    face.landmarks = np.asarray(face.landmarks, dtype=np.float64) + offset


def _make_config(image_path: Path, device: str):
    from ptgaze.main import load_mode_config
    from ptgaze.utils import (
        check_path_all,
        download_ethxgaze_model,
        download_mediapipe_face_landmarker,
        expanduser_all,
        set_dummy_camera_params,
    )

    args = argparse.Namespace(
        config=None,
        mode="eth-xgaze",
        face_detector="mediapipe",
        device=device,
        image=image_path.as_posix(),
        video=None,
        camera=None,
        output_dir=None,
        ext=None,
        no_screen=True,
        debug=False,
    )
    config = load_mode_config(args)
    config.gaze_estimator.checkpoint = download_ethxgaze_model().as_posix()
    download_mediapipe_face_landmarker()
    expanduser_all(config)
    set_dummy_camera_params(config)
    check_path_all(config)
    OmegaConf.set_readonly(config, True)
    return config


def _annotate(
    image: np.ndarray,
    estimator,
    face,
    *,
    head_angles: np.ndarray,
    gaze_angles: np.ndarray,
    camera_origin_angle: float | None,
    retry_crop: dict[str, Any] | None,
):
    from ptgaze.common import Visualizer
    from ptgaze.utils import get_3d_face_model

    face_model = get_3d_face_model(estimator._config)  # ptgaze has no public accessor for this helper.
    viz = Visualizer(estimator.camera, face_model.NOSE_INDEX)
    viz.set_image(image.copy())
    viz.draw_bbox(face.bbox)
    viz.draw_model_axes(face, 0.05, lw=2)
    if face.center is not None and face.gaze_vector is not None:
        viz.draw_3d_line(face.center, face.center + 0.05 * face.gaze_vector)

    annotated = viz.image if viz.image is not None else image.copy()
    if retry_crop is not None:
        x0, y0, x1, y1 = [int(v) for v in retry_crop["bbox_xyxy"]]
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 0), 2)

    hp, hy, hr = [float(x) for x in head_angles]
    gp, gy = [float(x) for x in gaze_angles]
    lines = [
        f"head pitch={hp:+.1f} yaw={hy:+.1f} roll={hr:+.1f}",
        f"gaze pitch={gp:+.1f} yaw={gy:+.1f}",
        f"gaze-to-lens={camera_origin_angle:.1f} deg" if camera_origin_angle is not None else "gaze-to-lens=n/a",
    ]
    if retry_crop is not None:
        lines.append("face acquisition=DWPose-assisted crop")
    y = 28
    for text in lines:
        cv2.putText(annotated, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return annotated


def _process_one(image_path: Path, output_dir: Path, device: str, dwpose_dir: Path | None) -> dict[str, Any]:
    from ptgaze.gaze_estimator import GazeEstimator

    image = cv2.imread(image_path.as_posix())
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    config = _make_config(image_path, device)
    estimator = GazeEstimator(config)
    dwpose_path = _find_for_key(dwpose_dir, image_path.stem)
    retry_crop: dict[str, Any] | None = None
    acquisition: dict[str, Any] = {
        "method": "full_image",
        "full_image_face_count": 0,
        "dwpose_path": str(dwpose_path) if dwpose_path else None,
    }

    try:
        undistorted = cv2.undistort(image, estimator.camera.camera_matrix, estimator.camera.dist_coefficients)
        faces = estimator.detect_faces(undistorted)
        acquisition["full_image_face_count"] = len(faces)

        if not faces and dwpose_path is not None:
            dwpose_record = _read_json(dwpose_path)
            retry_crop = _dwpose_head_crop(dwpose_record, image.shape[1], image.shape[0])
            acquisition["retry_crop"] = retry_crop
            if retry_crop is not None:
                x0, y0, x1, y1 = [int(v) for v in retry_crop["bbox_xyxy"]]
                crop = undistorted[y0:y1, x0:x1]
                crop_faces = estimator.detect_faces(crop) if crop.size else []
                acquisition["crop_face_count"] = len(crop_faces)
                if crop_faces:
                    for crop_face in crop_faces:
                        _translate_face_to_full_image(crop_face, x0, y0)
                    faces = crop_faces
                    acquisition["method"] = "dwpose_head_crop"

        if not faces:
            record = {
                "schema_version": SCHEMA_VERSION,
                "image_key": image_path.stem,
                "image": image_path.as_posix(),
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "face_count": 0,
                "face_acquisition": acquisition,
                "status": "no_face",
            }
            _write_json(output_dir / f"{image_path.stem}.gaze.json", record)
            return record

        selected_index = max(range(len(faces)), key=lambda i: _bbox_area(faces[i].bbox))
        face = faces[selected_index]
        estimator.estimate_gaze(undistorted, face)

        if face.head_pose_rot is None or face.gaze_vector is None:
            raise RuntimeError(f"ptgaze did not populate head/gaze outputs for {image_path.name}")

        euler = face.head_pose_rot.as_euler("XYZ", degrees=True)
        head_angles = np.asarray(face.change_coordinate_system(euler), dtype=np.float64)
        gaze_angles = np.rad2deg(face.vector_to_angle(face.gaze_vector)).astype(np.float64)
        gaze_vector = np.asarray(face.gaze_vector, dtype=np.float64)
        face_center = np.asarray(face.center, dtype=np.float64) if face.center is not None else None
        optical_axis_angle = _angle_from_optical_axis_deg(gaze_vector)
        camera_origin_angle = _angle_to_camera_origin_deg(gaze_vector, face_center)

        overlay_path = output_dir / f"{image_path.stem}.gaze.png"
        annotated = _annotate(
            image,
            estimator,
            face,
            head_angles=head_angles,
            gaze_angles=gaze_angles,
            camera_origin_angle=camera_origin_angle,
            retry_crop=retry_crop if acquisition["method"] == "dwpose_head_crop" else None,
        )
        cv2.imwrite(overlay_path.as_posix(), annotated)

        bbox = np.asarray(face.bbox, dtype=np.float64).reshape(-1)
        record = {
            "schema_version": SCHEMA_VERSION,
            "backend": "ptgaze",
            "mode": "eth-xgaze",
            "image_key": image_path.stem,
            "image": image_path.as_posix(),
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "face_count": len(faces),
            "face_acquisition": acquisition,
            "selected_face_index": int(selected_index),
            "selected_face_bbox_xyxy": [float(x) for x in bbox[:4]],
            "face_center_camera": [float(x) for x in face_center] if face_center is not None else None,
            "head_pose": {
                "pitch_deg": float(head_angles[0]),
                "yaw_deg": float(head_angles[1]),
                "roll_deg": float(head_angles[2]),
                "rotation_matrix": np.asarray(face.head_pose_rot.as_matrix(), dtype=np.float64).tolist(),
            },
            "gaze": {
                "pitch_deg": float(gaze_angles[0]),
                "yaw_deg": float(gaze_angles[1]),
                "vector_camera": [float(x) for x in gaze_vector],
                "angle_from_optical_axis_deg": optical_axis_angle,
                "angle_to_camera_origin_deg": camera_origin_angle,
                "angle_from_camera_deg": camera_origin_angle,
            },
            "overlay": overlay_path.as_posix(),
            "status": "ok",
            "notes": [
                "Angles are raw ptgaze outputs; no caption-language thresholds are applied.",
                "angle_from_optical_axis_deg is not equivalent to looking at the camera when the face is off-axis.",
                "angle_to_camera_origin_deg compares gaze with the ray from the reconstructed face center to the camera origin/lens.",
                "Camera intrinsics are ptgaze dummy parameters derived from the ORIGINAL image dimensions because source-camera calibration is unavailable.",
                "For DWPose-assisted retries, face detection runs on a crop but detected landmarks are translated back to original-image coordinates before head/gaze estimation.",
            ],
        }
        _write_json(output_dir / f"{image_path.stem}.gaze.json", record)
        return record
    finally:
        estimator.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe head pose and camera-relative gaze with ptgaze ETH-XGaze.")
    parser.add_argument("input", type=Path, help="Image file or directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dwpose-dir", type=Path, help="Optional cached DWPose directory for face-crop retry when full-image detection fails.")
    parser.add_argument("--only", nargs="+", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else None
    if dwpose_dir is not None and not dwpose_dir.is_dir():
        raise SystemExit(f"DWPose directory not found: {dwpose_dir}")

    images = [p for p in _discover_images(source) if _matches(p, args.only)]
    if not images:
        raise SystemExit("No matching images found")

    records = []
    for image_path in images:
        record = _process_one(image_path, output_dir, args.device, dwpose_dir)
        records.append(record)
        if record.get("status") == "ok":
            head = record["head_pose"]
            gaze = record["gaze"]
            lens = gaze.get("angle_to_camera_origin_deg")
            lens_text = f"{lens:.1f}" if lens is not None else "n/a"
            optical = gaze.get("angle_from_optical_axis_deg")
            optical_text = f"{optical:.1f}" if optical is not None else "n/a"
            acquisition = record.get("face_acquisition") or {}
            method = acquisition.get("method") or "unknown"
            print(
                f"{record['image_key']}: "
                f"head(p={head['pitch_deg']:+.1f}, y={head['yaw_deg']:+.1f}, r={head['roll_deg']:+.1f}) "
                f"gaze(p={gaze['pitch_deg']:+.1f}, y={gaze['yaw_deg']:+.1f}, "
                f"lens={lens_text}, optical={optical_text}) face={method}"
            )
        else:
            acquisition = record.get("face_acquisition") or {}
            retry = acquisition.get("retry_crop")
            retry_text = " retry=attempted" if retry is not None else ""
            print(f"{record['image_key']}: {record.get('status')}{retry_text}")

    _write_json(output_dir / "gaze_probe.index.json", {
        "schema_version": SCHEMA_VERSION + "-run",
        "backend": "ptgaze",
        "mode": "eth-xgaze",
        "record_count": len(records),
        "dwpose_dir": str(dwpose_dir) if dwpose_dir else None,
        "records": records,
    })
    print(f"Gaze probe bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
