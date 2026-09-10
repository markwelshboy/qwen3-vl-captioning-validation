from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import cv2
import numpy as np

from . import gaze_probe_pyfeat_v2 as base
from .face_target_binding import select_face_for_target, target_body_geometry

SCHEMA_VERSION = "gaze-probe-pyfeat-v28-0.2"


@dataclass
class _RowFace:
    row_index: int
    bbox: np.ndarray
    confidence: float


def _target_summary(target: dict[str, Any] | None) -> dict[str, Any] | None:
    if not target:
        return None
    return {k: v for k, v in target.items() if k != "points"}


def _select_row_for_target(fex: Any, target: dict[str, Any] | None, threshold: float):
    faces: list[_RowFace] = []
    for i in range(len(fex)):
        row = fex.iloc[i]
        bbox = base._bbox_from_row(row)
        score = base._as_float(row, "FaceScore")
        if bbox is not None and score is not None and score >= threshold:
            faces.append(_RowFace(i, bbox, score))
    selected = select_face_for_target(faces, target)
    if selected is None:
        return None
    face, strategy, diagnostics = selected
    return face.row_index, strategy, diagnostics


def _process_one(detector: Any, image_path: Path, output_dir: Path, dwpose_dir: Path | None, threshold: float) -> dict[str, Any]:
    image = cv2.imread(image_path.as_posix())
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    dwpose_path = base._find_for_key(dwpose_dir, image_path.stem)
    dwpose = base._read_json(dwpose_path)
    target = target_body_geometry(dwpose, width, height) if dwpose else None

    fex = detector.detect(
        inputs=image_path.as_posix(),
        data_type="image",
        face_detection_threshold=threshold,
        batch_size=1,
        num_workers=0,
    )
    selected = _select_row_for_target(fex, target, threshold)
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
                "target_body": _target_summary(target),
            },
        }
        base._write_json(output_dir / f"{image_path.stem}.pyfeat.json", record)
        return record

    row_index, strategy, diagnostics = selected
    row = fex.iloc[row_index]
    bbox = base._bbox_from_row(row)
    assert bbox is not None
    score = base._as_float(row, "FaceScore")

    hp = base._as_float(row, "Pitch")
    hr = base._as_float(row, "Roll")
    hy = base._as_float(row, "Yaw")
    gp = base._as_float(row, "gaze_pitch")
    gy = base._as_float(row, "gaze_yaw")
    ga = base._as_float(row, "gaze_angle")
    if any(v is None for v in (hp, hr, hy, gp, gy)):
        raise RuntimeError(f"py-feat returned selected face without complete head/gaze outputs: {image_path.name}")
    assert hp is not None and hr is not None and hy is not None and gp is not None and gy is not None

    gaze_mag = base._zero_direction_deviation_deg(gp, gy)
    landmarks = base._extract_landmarks68(row)
    mesh = base._extract_mesh478(row)

    arrays_path = output_dir / f"{image_path.stem}.pyfeat_arrays.npz"
    np.savez_compressed(arrays_path, landmarks68=landmarks, mesh478=mesh)

    overlay_path = output_dir / f"{image_path.stem}.pyfeat.png"
    cv2.imwrite(overlay_path.as_posix(), base._draw_overlay(image, bbox, landmarks, hp, hy, hr, gp, gy, gaze_mag))

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
            "candidate_count": int(len(diagnostics)),
        },
        "face_acquisition": {
            "method": "retinaface_full_image",
            "selection_strategy": strategy,
            "dwpose_path": str(dwpose_path) if dwpose_path else None,
            "target_body": _target_summary(target),
            "candidate_diagnostics": diagnostics,
        },
        "head_pose": {
            "pitch_rad": hp,
            "yaw_rad": hy,
            "roll_rad": hr,
            "pitch_deg": math.degrees(hp),
            "yaw_deg": math.degrees(hy),
            "roll_deg": math.degrees(hr),
            "translation": {
                "x": base._as_float(row, "X"),
                "y": base._as_float(row, "Y"),
                "z": base._as_float(row, "Z"),
            },
            "convention": "+pitch=up, +yaw=subject-right/frame-left, +roll=subject-right",
        },
        "gaze": {
            "pitch_rad": gp,
            "yaw_rad": gy,
            "pitch_deg": math.degrees(gp),
            "yaw_deg": math.degrees(gy),
            "gaze_angle_rad_from_fex": ga,
            "gaze_angle_deg_from_fex": math.degrees(ga) if ga is not None else None,
            "zero_direction_deviation_deg": gaze_mag,
            "convention": "canonical py-feat Fex values; diagnostic gaze only",
        },
        "arrays": arrays_path.as_posix(),
        "overlay": overlay_path.as_posix(),
        "notes": [
            "Face selection uses DWPose target-body membership, scale and head proximity.",
            "Head/gaze values remain canonical py-feat Fex outputs.",
            "py-feat gaze is diagnostic and is not the primary camera-relative gaze authority.",
        ],
    }
    base._write_json(output_dir / f"{image_path.stem}.pyfeat.json", record)
    return record


def main() -> int:
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base._process_one = _process_one
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
