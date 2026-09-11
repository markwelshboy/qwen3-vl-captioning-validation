from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import face_authority_uniface as v1
from .face_target_binding import select_face_for_target, target_body_geometry

SCHEMA_VERSION = "face-authority-uniface-0.2"


def _target_summary(target: dict[str, Any] | None) -> dict[str, Any] | None:
    if not target:
        return None
    return {
        "valid_joint_count": target.get("valid_joint_count"),
        "body_bbox_xyxy": target.get("body_bbox_xyxy"),
        "body_diagonal_px": target.get("body_diagonal_px"),
        "body_scale_px": target.get("body_scale_px"),
        "head_center_xy": target.get("head_center_xy"),
        "neck_xy": target.get("neck_xy"),
        "shoulder_mid_xy": target.get("shoulder_mid_xy"),
        "shoulder_span_px": target.get("shoulder_span_px"),
    }


def _apply_affine(point_xy: list[float] | tuple[float, float] | np.ndarray, matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(point_xy, dtype=np.float64).reshape(-1)[:2]
    return p @ matrix[:, :2].T + matrix[:, 2]


def _aligned_eye_parse_stats(
    mask: np.ndarray,
    full_to_crop: np.ndarray,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    centers: list[np.ndarray] = []
    transformed: dict[str, np.ndarray] = {}
    for label, item in (("iris_left", left), ("iris_right", right)):
        center = item.get("center_xy")
        if center is None:
            continue
        local = _apply_affine(center, full_to_crop)
        transformed[label] = local
        centers.append(local)

    inter_eye = float(np.linalg.norm(centers[0] - centers[1])) if len(centers) == 2 else None
    crop_width = int(mask.shape[1])
    fallback_radius = 0.10 * max(1, crop_width)
    radius = max(5.0, 0.28 * inter_eye) if inter_eye is not None else max(5.0, fallback_radius)
    radius = min(radius, max(6.0, 0.22 * max(1, crop_width)))

    local_stats: list[dict[str, Any]] = []
    for label in ("iris_left", "iris_right"):
        center = transformed.get(label)
        if center is None:
            local_stats.append({"label": label, "available": False})
            continue
        region = v1._circle_region(mask, (float(center[0]), float(center[1])), radius)
        local_stats.append({
            "label": label,
            "available": True,
            "center_aligned_crop_xy": [float(center[0]), float(center[1])],
            "radius_px": float(radius),
            "fractions": v1._region_class_fractions(mask, region),
        })

    return {
        "inter_iris_distance_aligned_px": inter_eye,
        "roi_radius_px": float(radius),
        "iris_regions": local_stats,
    }


def _draw_aligned_face_diag(
    aligned_crop: np.ndarray,
    mask: np.ndarray,
    eye_stats: dict[str, Any],
) -> np.ndarray:
    face_view = aligned_crop.copy()
    parse_view = v1._parse_visual(mask)
    radius = float(eye_stats.get("roi_radius_px") or 0.0)

    for item in eye_stats.get("iris_regions", []):
        center = item.get("center_aligned_crop_xy")
        if center is None:
            continue
        cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
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

    dwpose_path = v1._find_for_key(dwpose_dir, image_path.stem)
    dwpose = v1._read_json(dwpose_path)
    target = target_body_geometry(dwpose, width, height) if dwpose else None

    faces = models["detector"].detect(image)
    selected = select_face_for_target(faces, target)
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
                "target_body": _target_summary(target),
            },
        }
        v1._write_json(output_dir / f"{image_path.stem}.uniface.json", record)
        return record

    face, strategy, candidate_diagnostics = selected
    bbox = v1._clip_bbox(face.bbox, width, height)
    x0, y0, x1, y1 = bbox
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise RuntimeError(f"Selected UniFace bbox produced empty crop: {image_path.name}")

    state_result = models["facestate"].predict(image, face)
    states = {k: float(v) for k, v in state_result.as_dict().items()}

    quality_result = models["quality"].predict(image, np.asarray(face.landmarks))
    quality_score = v1._safe_float(quality_result.score)

    mesh_results = models["mesh"].predict(image, faces=[face])
    mesh_result = mesh_results[0] if mesh_results else None
    mesh_points = (
        np.asarray(mesh_result.landmarks, dtype=np.float32)
        if mesh_result is not None
        else np.empty((0, 3), dtype=np.float32)
    )

    from uniface.landmark import IRIS_LEFT, IRIS_RIGHT, roi_from_box, warp_roi

    left_points = mesh_points[IRIS_LEFT] if len(mesh_points) >= 478 else np.empty((0, 3), dtype=np.float32)
    right_points = mesh_points[IRIS_RIGHT] if len(mesh_points) >= 478 else np.empty((0, 3), dtype=np.float32)
    left_iris = v1._iris_stats(left_points)
    right_iris = v1._iris_stats(right_points)

    full_inter_iris = None
    if left_iris.get("center_xy") is not None and right_iris.get("center_xy") is not None:
        full_inter_iris = float(
            np.linalg.norm(
                np.asarray(left_iris["center_xy"], dtype=np.float64)
                - np.asarray(right_iris["center_xy"], dtype=np.float64)
            )
        )

    # Parse the same roll-normalized 1.5x face ROI family used by FaceMesh rather
    # than an axis-aligned detector crop.  This removes roll as a confounder for
    # eye/hair/background fractions.
    parse_roi = roi_from_box(np.asarray(face.bbox), np.asarray(face.landmarks), margin=0.25)
    aligned_crop, crop_to_full = warp_roi(image, parse_roi, int(models["mesh"].input_size))
    full_to_crop = cv2.invertAffineTransform(crop_to_full)
    parse_mask = models["parser"].parse(aligned_crop)
    eye_parse = _aligned_eye_parse_stats(parse_mask, full_to_crop, left_iris, right_iris)

    head = models["headpose"].estimate(crop)

    arrays_path = output_dir / f"{image_path.stem}.uniface_arrays.npz"
    np.savez_compressed(
        arrays_path,
        mesh478=mesh_points,
        parse_mask=parse_mask.astype(np.uint8),
        detector_landmarks5=np.asarray(face.landmarks, dtype=np.float32),
        parse_crop_to_full=np.asarray(crop_to_full, dtype=np.float64),
        parse_full_to_crop=np.asarray(full_to_crop, dtype=np.float64),
    )

    overlay_path = output_dir / f"{image_path.stem}.uniface.png"
    overlay = v1._draw_full_overlay(image, face, head, states, quality_score, mesh_points)
    cv2.imwrite(overlay_path.as_posix(), overlay)

    face_diag_path = output_dir / f"{image_path.stem}.uniface_face_diag.png"
    face_diag = _draw_aligned_face_diag(aligned_crop, parse_mask, eye_parse)
    cv2.imwrite(face_diag_path.as_posix(), face_diag)

    finite_mesh = int(np.isfinite(mesh_points).all(axis=1).sum()) if mesh_points.ndim == 2 else 0
    parse_fractions = v1._fraction_map(parse_mask)

    selected_index = None
    selected_box = np.asarray(face.bbox, dtype=np.float64).reshape(-1)[:4]
    for candidate in candidate_diagnostics:
        box = np.asarray(candidate.get("bbox_xyxy", []), dtype=np.float64)
        if box.size >= 4 and np.allclose(box[:4], selected_box, atol=1e-4):
            selected_index = int(candidate["index"])
            break

    record = {
        "schema_version": SCHEMA_VERSION,
        "backend": "uniface",
        "models": {
            "detector": "RetinaFace MNET_V2",
            "face_state": "FaceAttribNet",
            "face_mesh": "FaceMesh V2_478",
            "face_quality": "eDifFIQA-T",
            "face_parsing": "BiSeNet ResNet18, roll-normalized ROI",
            "head_pose": "HeadPose ResNet50",
        },
        "image_key": image_path.stem,
        "image": image_path.as_posix(),
        "status": "ok",
        "face": {
            "score": float(face.confidence),
            "bbox_xyxy": [float(v) for v in selected_box],
            "selection_strategy": strategy,
            "selected_candidate_index": selected_index,
            "candidate_count": int(len(faces)),
            "landmarks5": np.asarray(face.landmarks, dtype=float).tolist(),
        },
        "face_acquisition": {
            "dwpose_path": str(dwpose_path) if dwpose_path else None,
            "target_body": _target_summary(target),
            "candidate_diagnostics": candidate_diagnostics,
        },
        "face_state": states,
        "face_quality": {
            "score": quality_score,
            "note": "eDifFIQA is a relative face-quality score, not a calibrated gaze probability.",
        },
        "face_mesh": {
            "presence_score": v1._safe_float(mesh_result.score) if mesh_result is not None else None,
            "finite_landmarks": finite_mesh,
            "landmark_count": int(mesh_points.shape[0]) if mesh_points.ndim == 2 else 0,
            "iris_left": left_iris,
            "iris_right": right_iris,
            "inter_iris_distance_px": full_inter_iris,
            "note": "FaceMesh presence score saturates near 1 and is diagnostic only.",
        },
        "face_parsing": {
            "roi": {
                "center_x": float(parse_roi[0]),
                "center_y": float(parse_roi[1]),
                "side_px": float(parse_roi[2]),
                "roll_angle_deg": float(parse_roi[3]),
                "output_size_px": int(aligned_crop.shape[1]),
            },
            "class_fractions": parse_fractions,
            "eye_region_diagnostics": eye_parse,
            "note": "BiSeNet runs on a roll-normalized FaceMesh-style ROI; iris centers are transformed into that crop before measuring class fractions.",
        },
        "head_pose": {
            "pitch_deg_raw": float(head.pitch),
            "yaw_deg_raw": float(head.yaw),
            "roll_deg_raw": float(head.roll),
            "upstream_documented_convention": "UniFace types.py documents +pitch=down, +yaw=looking right, +roll=clockwise.",
            "pyfeat_empirical_compare": {
                "pitch_deg": float(head.pitch),
                "yaw_deg": float(head.yaw),
                "roll_deg": float(-head.roll),
            },
            "note": "On this calibration set raw UniFace pitch/yaw align in sign with py-feat; UniFace roll aligns after sign reversal. Keep raw values authoritative until conventions are formally calibrated.",
        },
        "arrays": arrays_path.as_posix(),
        "overlay": overlay_path.as_posix(),
        "face_diag": face_diag_path.as_posix(),
        "notes": [
            "v0.2 binds detected faces to the selected DWPose body using body membership, head/neck proximity, and target-person scale.",
            "v0.2 roll-normalizes BiSeNet parsing before computing iris-region visibility fractions.",
            "This remains a diagnostic probe: no final gaze-authority threshold is applied.",
        ],
    }
    v1._write_json(output_dir / f"{image_path.stem}.uniface.json", record)
    return record


def _providers_for_mode(mode: str) -> list[str] | None:
    if mode == "cpu":
        return ["CPUExecutionProvider"]
    if mode == "cuda":
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise SystemExit(
                "ERROR: --provider cuda requested but CUDAExecutionProvider is unavailable. "
                f"Registered providers: {available}"
            )
        # Be explicit: do not allow TensorRT to become the implicit first choice
        # merely because it is registered by the onnxruntime-gpu wheel.
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if mode == "auto":
        return None
    raise ValueError(f"Unknown provider mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniFace v0.2 diagnostic probe for target-bound gaze-authority signals and head pose.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--provider", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--only", nargs="+", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else None
    providers = _providers_for_mode(args.provider)
    print(f"UniFace provider mode: {args.provider}; session providers: {providers if providers is not None else 'ORT default order'}")

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

    images = [p for p in v1._discover_images(source) if v1._matches(p, args.only)]
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
            v1._write_json(output_dir / f"{image_path.stem}.uniface.json", record)
        records.append(record)

        if record["status"] == "ok":
            state = record["face_state"]
            q = record["face_quality"]["score"]
            hp = record["head_pose"]
            mesh = record["face_mesh"]
            parse = record["face_parsing"]["eye_region_diagnostics"]["iris_regions"]
            parts = []
            for item in parse:
                if item.get("available"):
                    f = item.get("fractions", {})
                    parts.append(
                        f"{item['label']}[eye={f.get('eye', 0):.2f},hair={f.get('hair', 0):.2f},bg={f.get('background', 0):.2f},gl={f.get('eyeglasses', 0):.2f}]"
                    )
            prefix = (
                f"{image_path.stem}: face={record['face']['score']:.3f} q={q:.3f} "
                if q is not None
                else f"{image_path.stem}: face={record['face']['score']:.3f} q=n/a "
            )
            print(
                prefix
                + f"eyes(L={state['left_eye_open']:.2f},R={state['right_eye_open']:.2f}) "
                + f"sun={state['sunglasses']:.2f} iris={mesh['inter_iris_distance_px'] if mesh['inter_iris_distance_px'] is not None else 'n/a'} "
                + f"head(P={hp['pitch_deg_raw']:+.1f},Y={hp['yaw_deg_raw']:+.1f},R={hp['roll_deg_raw']:+.1f}) "
                + " ".join(parts)
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
    v1._write_json(output_dir / "face_authority_uniface_v2.index.json", index)
    print(f"UniFace authority v0.2 bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
