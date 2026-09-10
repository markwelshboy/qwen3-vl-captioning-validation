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
SCHEMA_VERSION = "gaze-probe-ptgaze-0.2"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _discover_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _matches(path: Path, only: list[str]) -> bool:
    if not only:
        return True
    wanted = {item.lower() for item in only}
    return path.stem.lower() in wanted or path.name.lower() in wanted


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
    hp, hy, hr = [float(x) for x in head_angles]
    gp, gy = [float(x) for x in gaze_angles]
    lines = [
        f"head pitch={hp:+.1f} yaw={hy:+.1f} roll={hr:+.1f}",
        f"gaze pitch={gp:+.1f} yaw={gy:+.1f}",
        f"gaze-to-lens={camera_origin_angle:.1f} deg" if camera_origin_angle is not None else "gaze-to-lens=n/a",
    ]
    y = 28
    for text in lines:
        cv2.putText(annotated, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return annotated


def _process_one(image_path: Path, output_dir: Path, device: str) -> dict[str, Any]:
    from ptgaze.gaze_estimator import GazeEstimator

    image = cv2.imread(image_path.as_posix())
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    config = _make_config(image_path, device)
    estimator = GazeEstimator(config)
    try:
        undistorted = cv2.undistort(image, estimator.camera.camera_matrix, estimator.camera.dist_coefficients)
        faces = estimator.detect_faces(undistorted)
        if not faces:
            record = {
                "schema_version": SCHEMA_VERSION,
                "image_key": image_path.stem,
                "image": image_path.as_posix(),
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "face_count": 0,
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
                # Backward-compatible field name; from v0.2 onward this means
                # gaze error relative to the lens/camera origin, not optical axis.
                "angle_from_camera_deg": camera_origin_angle,
            },
            "overlay": overlay_path.as_posix(),
            "status": "ok",
            "notes": [
                "Angles are raw ptgaze outputs; no caption-language thresholds are applied.",
                "angle_from_optical_axis_deg is not equivalent to looking at the camera when the face is off-axis.",
                "angle_to_camera_origin_deg compares gaze with the ray from the reconstructed face center to the camera origin/lens.",
                "Camera intrinsics are ptgaze dummy parameters derived from image dimensions because source-camera calibration is unavailable.",
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
    parser.add_argument("--only", nargs="+", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in _discover_images(source) if _matches(p, args.only)]
    if not images:
        raise SystemExit("No matching images found")

    records = []
    for image_path in images:
        record = _process_one(image_path, output_dir, args.device)
        records.append(record)
        if record.get("status") == "ok":
            head = record["head_pose"]
            gaze = record["gaze"]
            lens = gaze.get("angle_to_camera_origin_deg")
            lens_text = f"{lens:.1f}" if lens is not None else "n/a"
            optical = gaze.get("angle_from_optical_axis_deg")
            optical_text = f"{optical:.1f}" if optical is not None else "n/a"
            print(
                f"{record['image_key']}: "
                f"head(p={head['pitch_deg']:+.1f}, y={head['yaw_deg']:+.1f}, r={head['roll_deg']:+.1f}) "
                f"gaze(p={gaze['pitch_deg']:+.1f}, y={gaze['yaw_deg']:+.1f}, "
                f"lens={lens_text}, optical={optical_text})"
            )
        else:
            print(f"{record['image_key']}: {record.get('status')}")

    _write_json(output_dir / "gaze_probe.index.json", {
        "schema_version": SCHEMA_VERSION + "-run",
        "backend": "ptgaze",
        "mode": "eth-xgaze",
        "record_count": len(records),
        "records": records,
    })
    print(f"Gaze probe bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
