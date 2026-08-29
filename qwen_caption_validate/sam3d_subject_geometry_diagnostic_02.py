from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np

from . import sam3d_subject_geometry_diagnostic as v01


def _orientation_band(yaw_deg: float | None) -> str | None:
    """Map continuous body yaw to human-readable orientation bands.

    v0.1 jumped directly from frontal to three-quarter at 20 degrees.  The
    calibration set contains several visually modest ~20-30 degree turns, so
    v0.2 keeps those as ``slightly_angled`` and reserves three-quarter for the
    clearer 35-65 degree range.
    """
    if yaw_deg is None:
        return None
    a = abs(float(yaw_deg))
    if a <= 20.0:
        return "frontal"
    if a <= 35.0:
        return "slightly_angled"
    if a <= 65.0:
        return "three_quarter"
    if a <= 115.0:
        return "side_on"
    if a <= 160.0:
        return "rear_three_quarter"
    return "rear"


def build_subject_geometry(
    arrays: dict[str, np.ndarray],
    dwpose_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refine the v0.1 diagnostic without changing its calibrated transforms."""
    out = copy.deepcopy(v01.build_subject_geometry(arrays, dwpose_record))
    out["schema_version"] = "sam3d-subject-geometry-diagnostic-0.2"

    body = out.get("body_camera_relation") or {}
    yaw = body.get("yaw_deg")
    body_band = _orientation_band(yaw)
    body["orientation_band"] = body_band
    if yaw is None or abs(float(yaw)) <= 20.0:
        body["faces_frame"] = None
    out["body_camera_relation"] = body

    camera = out.get("camera_relative_subject") or {}
    # The geometry itself does not establish capture mode.  A below-eye/upward
    # or above-eye/downward relation can occur in an external portrait as well
    # as a selfie, so use a neutral name and leave capture interpretation to a
    # later semantic governor.
    camera["camera_pose_pattern"] = camera.pop("selfie_like_geometry", None)
    out["camera_relative_subject"] = camera

    visibility = out.get("dwpose_visibility_gate") or {}
    face = out.get("face_camera_relation") or {}
    face_band = face.get("orientation_band")
    turn = face.get("head_turn_toward_camera_deg")

    compound = None
    if (
        visibility.get("body_yaw_observation_gate")
        and visibility.get("face_yaw_observation_gate")
        and body_band in {"slightly_angled", "three_quarter", "side_on"}
        and face_band == "toward_camera"
        and turn is not None
        and float(turn) >= 20.0
    ):
        compound = {
            "body_orientation": body_band,
            "body_faces_frame": body.get("faces_frame"),
            "head_relation": "turned_toward_camera",
            "face_orientation": face_band,
            "head_turn_toward_camera_deg": turn,
        }
    out["compound_pose_hint"] = compound

    policy = out.setdefault("policy", {})
    policy.update(
        {
            "report_only": True,
            "body_orientation_bands": {
                "frontal": "abs(yaw)<=20",
                "slightly_angled": "20<abs(yaw)<=35",
                "three_quarter": "35<abs(yaw)<=65",
                "side_on": "65<abs(yaw)<=115",
                "rear_three_quarter": "115<abs(yaw)<=160",
                "rear": "abs(yaw)>160",
            },
            "camera_pose_pattern_is_capture_mode_neutral": True,
            "subject_relative_camera_use": (
                "camera position/aim is subject-relative geometry only; capture mode and world high/low "
                "require separate qualification"
            ),
        }
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-subject-geometry-diagnostic-02",
        description=(
            "Inspect cached SAM3D root rotation/camera translation as subject-relative camera/body/head geometry "
            "using calibrated v0.2 orientation bands."
        ),
    )
    parser.add_argument("sam3d_dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")
    dwpose_dir = v01._resolve_dwpose(sam3d_dir, args.dwpose_dir)

    paths = v01._discover_npz(sam3d_dir, args.include)
    if not paths:
        raise SystemExit("No matching *.sam3d_arrays.npz records found")

    rows: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        dwpose_path = dwpose_dir / f"{key}.dwpose.json" if dwpose_dir else None
        dwpose = v01._read_json(dwpose_path) if dwpose_path and dwpose_path.exists() else None
        diagnostic = build_subject_geometry(arrays, dwpose)
        rows.append(
            {
                "image_key": key,
                "source": str(path),
                "dwpose_source": str(dwpose_path) if dwpose_path and dwpose_path.exists() else None,
                "diagnostic": diagnostic,
            }
        )

        body = diagnostic["body_camera_relation"]
        camera = diagnostic["camera_relative_subject"]
        face = diagnostic["face_camera_relation"]
        compound = diagnostic.get("compound_pose_hint")
        print(
            f"{key}: bodyYaw={body['yaw_deg']}° {body['orientation_band']} {body['faces_frame'] or '-'}; "
            f"cam-eye={camera['vertical_vs_eye']:+.3f}m pitch={camera['optical_axis_pitch_deg']:+.1f}° side={camera['side'] or '-'}; "
            f"faceYaw={face['yaw_deg']}° {face['orientation_band']}; compound={'yes' if compound else 'no'}"
        )

    payload = {
        "schema_version": "sam3d-subject-geometry-diagnostic-run-0.2",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir) if dwpose_dir else None,
        "record_count": len(rows),
        "records": rows,
    }
    out = (args.output or (sam3d_dir / "sam3d_subject_geometry_diagnostic.json")).expanduser().resolve()
    v01._write_json(out, payload)
    print(f"Diagnostic: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
