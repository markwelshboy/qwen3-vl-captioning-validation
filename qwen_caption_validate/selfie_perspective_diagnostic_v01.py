from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import camera_composition_shadow_v01 as camera_shadow
from . import mesh_arm_occupancy_shadow_v01 as mesh_shadow

SCHEMA_VERSION = "selfie-perspective-diagnostic-0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_FRAMING_SUBDIR = Path("semantic-v3") / "framing-semantics-v1.0"
DEFAULT_SELFIE_SUBDIR = Path("semantic-v3") / "selfie-evidence-shadow-v0.5"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "selfie-perspective-diagnostic-v0.1"

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _finite_vec(value: Any, n: int = 3) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < n or not np.isfinite(arr[:n]).all():
        return None
    return arr[:n]


def _camera_distance_geometry(
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    points = np.asarray(arrays.get("pred_keypoints_3d"), dtype=np.float64)
    cam_t = _finite_vec(arrays.get("pred_cam_t"), 3)
    if (
        points.ndim != 2
        or points.shape[0] <= RIGHT_SHOULDER
        or points.shape[1] < 3
        or cam_t is None
    ):
        return {
            "status": "unavailable",
            "reason": "sam3d_shoulder_or_camera_translation_missing",
        }

    left = np.asarray(points[LEFT_SHOULDER, :3], dtype=np.float64)
    right = np.asarray(points[RIGHT_SHOULDER, :3], dtype=np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return {
            "status": "unavailable",
            "reason": "nonfinite_sam3d_shoulder_geometry",
        }

    shoulder_width = float(np.linalg.norm(left - right))
    midpoint_root = (left + right) / 2.0
    midpoint_cam = midpoint_root + cam_t

    camera_distance = float(np.linalg.norm(midpoint_cam))
    optical_depth = float(midpoint_cam[2])
    if shoulder_width <= 1e-9:
        return {
            "status": "unavailable",
            "reason": "degenerate_reconstructed_shoulder_width",
        }

    return {
        "status": "available",
        "shoulder_width_model_units": round(shoulder_width, 6),
        "camera_to_shoulder_midpoint_model_units": round(camera_distance, 6),
        "camera_to_shoulder_midpoint_shoulder_widths": round(
            camera_distance / shoulder_width, 6
        ),
        "shoulder_midpoint_optical_depth_model_units": round(optical_depth, 6),
        "shoulder_midpoint_optical_depth_shoulder_widths": round(
            optical_depth / shoulder_width, 6
        ),
        "shoulder_midpoint_camera_xyz": [
            round(float(v), 6) for v in midpoint_cam
        ],
        "authority": "sam3d_reconstructed_scale_free_camera_geometry_diagnostic",
        "note": (
            "The ratios are scale-free within the reconstructed SAM3D coordinate "
            "system. They are diagnostic perspective cues, not metric physical distance."
        ),
    }


def _projection_geometry(
    arrays: dict[str, np.ndarray],
    width: int,
    height: int,
) -> dict[str, Any]:
    k3 = np.asarray(arrays.get("pred_keypoints_3d"), dtype=np.float64)
    k2 = np.asarray(arrays.get("pred_keypoints_2d"), dtype=np.float64)
    cam_t = np.asarray(arrays.get("pred_cam_t"), dtype=np.float64)

    focal, meta = mesh_shadow._recover_focal_length(
        k3,
        k2,
        cam_t,
        width,
        height,
    )
    if focal is None:
        return {
            "status": "unavailable",
            "focal_length_recovery": meta,
        }

    f = float(focal)
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * f))) if f > 0 else None
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * f))) if f > 0 else None

    residual = mesh_shadow._projection_residual(
        k3,
        k2,
        cam_t,
        f,
        width,
        height,
    )

    return {
        "status": "available",
        "focal_length_px": round(f, 6),
        "focal_over_image_width": round(f / width, 6) if width > 0 else None,
        "focal_over_image_height": round(f / height, 6) if height > 0 else None,
        "horizontal_fov_deg": round(hfov, 4) if hfov is not None else None,
        "vertical_fov_deg": round(vfov, 4) if vfov is not None else None,
        "projection_residual": residual,
        "focal_length_recovery": meta,
        "authority": "recovered_from_cached_sam3d_projection_diagnostic",
    }


def _framing_geometry(framing: dict[str, Any]) -> dict[str, Any]:
    span = (
        framing.get("anatomical_span")
        if isinstance(framing.get("anatomical_span"), dict)
        else {}
    )
    face = (
        framing.get("face_scale_geometry")
        if isinstance(framing.get("face_scale_geometry"), dict)
        else {}
    )
    person = (
        framing.get("person_scale_geometry")
        if isinstance(framing.get("person_scale_geometry"), dict)
        else {}
    )
    scale = (
        framing.get("standard_shot_scale")
        if isinstance(framing.get("standard_shot_scale"), dict)
        else {}
    )
    return {
        "anatomical_span": {
            "upper": span.get("upper_anchor"),
            "lower": span.get("lower_anchor"),
        },
        "shot_scale": scale.get("label"),
        "face_height_fraction": face.get("height_fraction"),
        "face_width_fraction": face.get("width_fraction"),
        "person_visible_height_fraction": person.get("visible_height_fraction"),
        "person_visible_width_fraction": person.get("visible_width_fraction"),
        "face_person_height_ratio": framing.get("face_person_height_ratio"),
        "person_crop_edges": person.get("crop_edges"),
        "authority": "production_framing_semantics_observation",
    }


def evaluate(
    policy: dict[str, Any],
    framing: dict[str, Any],
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any],
    selfie_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    size = policy.get("image_size")
    if (
        not isinstance(size, list)
        or len(size) < 2
        or not all(isinstance(v, (int, float)) for v in size[:2])
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "image_key": policy.get("image_key"),
            "reason": "image_size_missing",
        }

    width, height = int(size[0]), int(size[1])

    subject_geometry = camera_shadow.build_subject_geometry(arrays, dwpose)
    viewpoint = camera_shadow._camera_viewpoint(subject_geometry)
    shoulder_depth = camera_shadow._shoulder_depth(arrays, dwpose)

    prior = {}
    if isinstance(selfie_record, dict):
        decision = (
            selfie_record.get("decision")
            if isinstance(selfie_record.get("decision"), dict)
            else {}
        )
        evidence = (
            selfie_record.get("evidence")
            if isinstance(selfie_record.get("evidence"), dict)
            else {}
        )
        prior = {
            "status": decision.get("status"),
            "publishable_selfie": decision.get("publishable_selfie"),
            "neutral_semantic_grade": (
                (evidence.get("neutral_semantic") or {}).get("grade")
                if isinstance(evidence.get("neutral_semantic"), dict)
                else None
            ),
            "camera_viewpoint_grade": (
                (evidence.get("camera_viewpoint") or {}).get("grade")
                if isinstance(evidence.get("camera_viewpoint"), dict)
                else None
            ),
            "foreground_arm_grade": (
                (evidence.get("foreground_arm") or {}).get("grade")
                if isinstance(evidence.get("foreground_arm"), dict)
                else None
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": policy.get("image_key"),
        "image_size": [width, height],
        "camera_distance_geometry": _camera_distance_geometry(arrays),
        "projection_geometry": _projection_geometry(arrays, width, height),
        "framing_geometry": _framing_geometry(framing),
        "camera_viewpoint_geometry": {
            "classification": viewpoint.get("classification"),
            "vertical_vs_eye": viewpoint.get("vertical_vs_eye"),
            "vertical_band": viewpoint.get("vertical_band"),
            "optical_axis_pitch_deg": viewpoint.get("optical_axis_pitch_deg"),
            "camera_pose_pattern": viewpoint.get("camera_pose_pattern"),
        },
        "shoulder_depth_geometry": {
            "status": shoulder_depth.get("status"),
            "nearer_anatomical_side_internal": shoulder_depth.get(
                "anatomical_side_nearer"
            ),
            "distance_delta_shoulder_widths": shoulder_depth.get(
                "distance_delta_shoulder_widths"
            ),
            "distance_and_z_order_agree": shoulder_depth.get(
                "distance_and_z_order_agree"
            ),
        },
        "prior_selfie_shadow_v05": prior or None,
        "interpretation": {
            "classification": "diagnostic_only",
            "note": (
                "No close-range selfie threshold is applied in v0.1. The purpose is "
                "to compare scale-free camera distance, recovered projection, crop/face "
                "geometry and shoulder perspective across known positives, misses and controls."
            ),
        },
        "invariants": {
            "no_new_model_call": True,
            "no_selfie_label_is_created_by_this_diagnostic": True,
            "sam3d_distance_is_scale_free_not_metric": True,
            "framing_observation_and_reconstruction_are_kept_separate": True,
            "shoulder_anatomical_side_is_internal_diagnostic_only": True,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raw close-range photographic perspective diagnostics for selfie validation."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--framing-dir", type=Path)
    p.add_argument("--selfie-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = (
        args.policy_dir.expanduser().resolve()
        if args.policy_dir
        else run_dir / DEFAULT_POLICY_SUBDIR
    )
    framing_dir = (
        args.framing_dir.expanduser().resolve()
        if args.framing_dir
        else run_dir / DEFAULT_FRAMING_SUBDIR
    )
    selfie_dir = (
        args.selfie_dir.expanduser().resolve()
        if args.selfie_dir
        else run_dir / DEFAULT_SELFIE_SUBDIR
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / DEFAULT_OUTPUT_SUBDIR
    )

    if not run_dir.is_dir() or not policy_dir.is_dir() or not framing_dir.is_dir():
        print(
            f"Required input missing: run={run_dir} policy={policy_dir} framing={framing_dir}",
            file=sys.stderr,
        )
        return 2

    requested = set(args.only)
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        paths = [
            p
            for p in paths
            if p.name.removesuffix(".perception_policy.json") in requested
        ]
    if not paths:
        print("No matching policy records.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for policy_path in paths:
        key = policy_path.name.removesuffix(".perception_policy.json")
        out_path = output_dir / f"{key}.selfie_perspective.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
            records.append(record)
            continue

        policy = _read_json(policy_path)
        sources = (
            policy.get("sources")
            if isinstance(policy.get("sources"), dict)
            else {}
        )
        arrays_path = Path(str(sources.get("sam3d_arrays") or "")).expanduser()
        dwpose_path = Path(str(sources.get("dwpose") or "")).expanduser()
        framing_path = framing_dir / f"{key}.framing.json"
        selfie_path = selfie_dir / f"{key}.selfie_evidence.json"

        if (
            not arrays_path.is_file()
            or not dwpose_path.is_file()
            or not framing_path.is_file()
        ):
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "unavailable",
                "image_key": key,
                "reason": "missing_sam3d_dwpose_or_framing_input",
            }
        else:
            record = evaluate(
                policy,
                _read_json(framing_path),
                _load_npz(arrays_path),
                _read_json(dwpose_path),
                _read_json(selfie_path) if selfie_path.is_file() else None,
            )
            record["sources"] = {
                "policy": str(policy_path),
                "framing": str(framing_path),
                "sam3d_arrays": str(arrays_path),
                "dwpose": str(dwpose_path),
                "prior_selfie_shadow_v05": (
                    str(selfie_path) if selfie_path.is_file() else None
                ),
            }

        _write_json(out_path, record)
        records.append(record)

        dist = (
            (record.get("camera_distance_geometry") or {}).get(
                "camera_to_shoulder_midpoint_shoulder_widths"
            )
        )
        focal = (
            (record.get("projection_geometry") or {}).get(
                "focal_over_image_width"
            )
        )
        ratio = (
            (record.get("framing_geometry") or {}).get(
                "face_person_height_ratio"
            )
        )
        prior = (
            (record.get("prior_selfie_shadow_v05") or {}).get("status")
            if isinstance(record.get("prior_selfie_shadow_v05"), dict)
            else None
        )
        print(
            f"{key}: distance_sw={dist if dist is not None else '-'} "
            f"focal/w={focal if focal is not None else '-'} "
            f"face/person={ratio if ratio is not None else '-'} "
            f"prior={prior or '-'}"
        )

    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "records": records,
    }
    _write_json(output_dir / "selfie_perspective_diagnostic.index.json", index)
    print(f"Index: {output_dir / 'selfie_perspective_diagnostic.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
