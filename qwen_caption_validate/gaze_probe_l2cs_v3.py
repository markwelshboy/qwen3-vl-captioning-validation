from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import cv2
import numpy as np

from . import gaze_probe_l2cs as base
from .face_target_binding import select_face_for_target, target_body_geometry

SCHEMA_VERSION = "gaze-probe-l2cs-0.3"


@dataclass
class _CandidateFace:
    candidate_index: int
    bbox: np.ndarray
    confidence: float


def _target_summary(target: dict[str, Any] | None) -> dict[str, Any] | None:
    if not target:
        return None
    return {k: v for k, v in target.items() if k != "points"}


def _select_candidate(candidates: list[dict[str, Any]], target: dict[str, Any] | None):
    faces = [
        _CandidateFace(i, np.asarray(item["bbox"], dtype=np.float64), float(item["score"]))
        for i, item in enumerate(candidates)
    ]
    selected = select_face_for_target(faces, target)
    if selected is None:
        return None
    face, strategy, diagnostics = selected
    return face.candidate_index, strategy, diagnostics


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

    dwpose_path = base._find_for_key(dwpose_dir, image_path.stem)
    dwpose = base._read_json(dwpose_path)
    geometry = base._dwpose_geometry(dwpose, w, h) if dwpose else {"head_center_xy": None, "retry_crop": None, "evidence": []}
    target = target_body_geometry(dwpose, w, h) if dwpose else None

    candidates = base._detect(pipeline.detector, image, threshold)
    acquisition = "full_image"
    full_count = len(candidates)
    retry_count = 0

    if not candidates and geometry.get("retry_crop"):
        x0, y0, x1, y1 = geometry["retry_crop"]
        crop = image[y0:y1, x0:x1]
        if crop.size:
            candidates = base._detect(pipeline.detector, crop, threshold, (x0, y0))
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
                "target_body": _target_summary(target),
                "retry_crop": geometry.get("retry_crop"),
            },
        }
        if geometry.get("retry_crop"):
            x0, y0, x1, y1 = geometry["retry_crop"]
            crop_path = output_dir / f"{image_path.stem}.l2cs_retry_crop.png"
            cv2.imwrite(crop_path.as_posix(), image[y0:y1, x0:x1])
            record["face_acquisition"]["retry_crop_asset"] = crop_path.as_posix()
        base._write_json(output_dir / f"{image_path.stem}.l2cs.json", record)
        return record

    chosen = _select_candidate(candidates, target)
    if chosen is None:
        raise RuntimeError(f"Could not select L2CS face for {image_path.name}")
    selected_index, strategy, diagnostics = chosen
    selected = candidates[selected_index]
    bbox = base._clamp_bbox(selected["bbox"], w, h)
    if bbox is None:
        raise RuntimeError(f"Invalid selected L2CS bbox for {image_path.name}")
    x0, y0, x1, y1 = bbox
    face = image[y0:y1, x0:x1]
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))

    upstream_first, upstream_second = pipeline.predict_gaze(np.stack([rgb]))
    yaw = float(np.asarray(upstream_first).reshape(-1)[0])
    pitch = float(np.asarray(upstream_second).reshape(-1)[0])

    overlay_path = output_dir / f"{image_path.stem}.l2cs.png"
    cv2.imwrite(overlay_path.as_posix(), base._annotate(image, bbox, pitch, yaw, acquisition))

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
            "target_body": _target_summary(target),
            "candidate_diagnostics": diagnostics,
            "retry_crop": geometry.get("retry_crop"),
        },
        "gaze": {
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "forward_deviation_deg": base._forward_deviation_deg(pitch, yaw),
            "upstream_pipeline_return_order": ["yaw", "pitch"],
            "convention": "+pitch=up/-pitch=down; +yaw=frame-left/-yaw=frame-right",
        },
        "overlay": overlay_path.as_posix(),
        "notes": [
            "Corrected upstream L2CS pitch/yaw return-order bug.",
            "Face selection uses DWPose target-body membership, scale and head proximity.",
            "L2CS is the primary camera-relative gaze-direction specialist.",
        ],
    }
    base._write_json(output_dir / f"{image_path.stem}.l2cs.json", record)
    return record


def main() -> int:
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base._process_one = _process_one
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
