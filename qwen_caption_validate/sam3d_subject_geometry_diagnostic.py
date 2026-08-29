from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# First 70 MHR keypoints used by the existing SAM3D probe.
MHR70 = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "neck": 69,
}

# mhr_head.py converts the MHR output to its camera coordinate convention by
# multiplying Y and Z by -1 after applying global_rot.
CAMERA_SYSTEM_FLIP = np.diag([1.0, -1.0, -1.0])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _round_vec(value: np.ndarray | None, digits: int = 6) -> list[float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return [round(float(v), digits) for v in arr[:3]]


def _round_matrix(value: np.ndarray, digits: int = 6) -> list[list[float]]:
    return [[round(float(v), digits) for v in row] for row in np.asarray(value, dtype=np.float64)]


def euler_zyx_to_rotmat(euler_zyx: np.ndarray) -> np.ndarray:
    """Reconstruct roma's ZYX body rotation convention used by MHRHead.

    global_rot is emitted by roma.rotmat_to_euler("ZYX", global_rot_rotmat),
    so the corresponding matrix is Rz(z) @ Ry(y) @ Rx(x).
    """
    z, y, x = [float(v) for v in np.asarray(euler_zyx, dtype=np.float64).reshape(3)]
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return rz @ ry @ rx


def _angle_deg(y: float, x: float) -> float:
    return math.degrees(math.atan2(float(y), float(x)))


def _wrap_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _orientation_band(yaw_deg: float | None) -> str | None:
    if yaw_deg is None:
        return None
    a = abs(float(yaw_deg))
    if a <= 20.0:
        return "frontal"
    if a <= 65.0:
        return "three_quarter"
    if a <= 115.0:
        return "side_on"
    if a <= 160.0:
        return "rear_three_quarter"
    return "rear"


def _face_orientation_band(yaw_deg: float | None) -> str | None:
    if yaw_deg is None:
        return None
    a = abs(float(yaw_deg))
    if a <= 20.0:
        return "toward_camera"
    if a <= 60.0:
        return "three_quarter"
    if a <= 120.0:
        return "profile"
    return "away_from_camera"


def _mid(points: np.ndarray, a: str, b: str) -> np.ndarray:
    return (points[MHR70[a]] + points[MHR70[b]]) / 2.0


def _dwpose_gate(record: dict[str, Any] | None) -> dict[str, Any]:
    target = (((record or {}).get("derived") or {}).get("target") or {})
    visible = {
        str(item)
        for item in (target.get("visible_body_landmarks") or [])
        if isinstance(item, str)
    }

    shoulders = {"left_shoulder", "right_shoulder"}.issubset(visible)
    hips = {"left_hip", "right_hip"}.issubset(visible)
    face = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}.issubset(visible)
    return {
        "dwpose_available": bool(record),
        "visible_body_landmarks": sorted(visible),
        "pose_extent_hint": target.get("pose_extent_hint"),
        "bilateral_shoulders_observed": shoulders,
        "bilateral_hips_observed": hips,
        "face_five_observed": face,
        "body_yaw_observation_gate": shoulders,
        "face_yaw_observation_gate": face,
        "authority": "dwpose_image_observation_gate",
    }


def build_subject_geometry(
    arrays: dict[str, np.ndarray],
    dwpose_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global_rot = np.asarray(arrays["global_rot"], dtype=np.float64).reshape(3)
    cam_t = np.asarray(arrays["pred_cam_t"], dtype=np.float64).reshape(3)
    keypoints_cam_root = np.asarray(arrays["pred_keypoints_3d"], dtype=np.float64)
    if keypoints_cam_root.ndim != 2 or keypoints_cam_root.shape[0] < 70 or keypoints_cam_root.shape[1] < 3:
        raise ValueError("pred_keypoints_3d must contain at least the first 70 MHR keypoints")
    keypoints_cam_root = keypoints_cam_root[:, :3]

    # MHR canonical/body -> MHR rotated frame -> SAM3D camera-system frame.
    body_to_mhr = euler_zyx_to_rotmat(global_rot)
    body_to_camera = CAMERA_SYSTEM_FLIP @ body_to_mhr

    # SAM3D projects root-relative reconstructed points after adding pred_cam_t:
    # P_camera = R_body_to_camera * P_body + pred_cam_t.
    # Therefore camera center in body coordinates is -R^T t.
    camera_center_body = -(body_to_camera.T @ cam_t)
    camera_forward_body = body_to_camera.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    body_forward_camera = body_to_camera @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Remove only root/global rotation, leaving articulated body/head pose intact.
    keypoints_body = (body_to_camera.T @ keypoints_cam_root.T).T

    shoulder_body = _mid(keypoints_body, "left_shoulder", "right_shoulder")
    hip_body = _mid(keypoints_body, "left_hip", "right_hip")
    knee_body = _mid(keypoints_body, "left_knee", "right_knee")
    ankle_body = _mid(keypoints_body, "left_ankle", "right_ankle")
    eye_body = _mid(keypoints_body, "left_eye", "right_eye")
    ear_body = _mid(keypoints_body, "left_ear", "right_ear")
    nose_body = keypoints_body[MHR70["nose"]]
    neck_body = keypoints_body[MHR70["neck"]]

    shoulder_width = float(
        np.linalg.norm(
            keypoints_body[MHR70["left_shoulder"]]
            - keypoints_body[MHR70["right_shoulder"]]
        )
    )

    # Body/root yaw is measured from canonical +Z body-forward against camera -Z
    # (a frontal subject faces back toward the camera). Sign also gives frame-facing direction.
    body_yaw = _angle_deg(body_forward_camera[0], -body_forward_camera[2])
    body_pitch = _angle_deg(
        body_forward_camera[1],
        math.hypot(body_forward_camera[0], body_forward_camera[2]),
    )
    frame_facing = None
    if abs(body_yaw) > 20.0:
        frame_facing = "left" if body_forward_camera[0] < 0.0 else "right"

    # Camera optical-axis aim relative to body frame. Positive pitch looks upward
    # through the subject frame; negative pitch looks downward.
    optical_pitch = math.degrees(
        math.asin(max(-1.0, min(1.0, float(camera_forward_body[1]))))
    )
    optical_yaw = _angle_deg(camera_forward_body[0], -camera_forward_body[2])

    # Reconstructed face-forward proxy: ear midpoint -> nose. It is not a head
    # joint rotation, but on observed faces gives a useful camera-relative yaw.
    face_forward_camera = keypoints_cam_root[MHR70["nose"]] - _mid(
        keypoints_cam_root, "left_ear", "right_ear"
    )
    face_norm = float(np.linalg.norm(face_forward_camera))
    face_yaw = None
    face_pitch = None
    if face_norm > 1e-9 and np.all(np.isfinite(face_forward_camera)):
        face_forward_camera = face_forward_camera / face_norm
        face_yaw = _angle_deg(face_forward_camera[0], -face_forward_camera[2])
        face_pitch = _angle_deg(
            face_forward_camera[1],
            math.hypot(face_forward_camera[0], face_forward_camera[2]),
        )

    head_turn_relative_body = (
        _wrap_deg(float(face_yaw) - body_yaw) if face_yaw is not None else None
    )
    turn_toward_camera = None
    if face_yaw is not None:
        turn_toward_camera = max(0.0, abs(body_yaw) - abs(float(face_yaw)))

    canonical_left_x = float(keypoints_body[MHR70["left_shoulder"], 0])
    canonical_right_x = float(keypoints_body[MHR70["right_shoulder"], 0])
    lateral_axis_valid = canonical_left_x > canonical_right_x
    camera_side = None
    if lateral_axis_valid and abs(float(camera_center_body[0])) > 1e-6:
        camera_side = "subject_left" if camera_center_body[0] > 0.0 else "subject_right"

    camera_eye_delta = float(camera_center_body[1] - eye_body[1])
    camera_shoulder_delta = float(camera_center_body[1] - shoulder_body[1])
    camera_hip_delta = float(camera_center_body[1] - hip_body[1])

    vertical_band = "near_eye_level"
    if camera_eye_delta >= 0.15:
        vertical_band = "above_eye_level"
    elif camera_eye_delta <= -0.15:
        vertical_band = "below_eye_level"

    selfie_like_geometry = None
    if camera_eye_delta >= 0.10 and optical_pitch <= -5.0:
        selfie_like_geometry = "camera_above_subject_aimed_down"
    elif camera_eye_delta <= -0.15 and optical_pitch >= 5.0:
        selfie_like_geometry = "camera_below_subject_aimed_up"

    visibility = _dwpose_gate(dwpose_record)
    body_band = _orientation_band(body_yaw)
    face_band = _face_orientation_band(face_yaw)

    compound_hint = None
    if (
        visibility["body_yaw_observation_gate"]
        and visibility["face_yaw_observation_gate"]
        and body_band in {"three_quarter", "side_on"}
        and face_band == "toward_camera"
        and turn_toward_camera is not None
        and turn_toward_camera >= 20.0
    ):
        compound_hint = {
            "body_orientation": body_band,
            "body_faces_frame": frame_facing,
            "head_relation": "turned_toward_camera",
            "face_orientation": face_band,
            "head_turn_toward_camera_deg": _round(turn_toward_camera),
        }

    return {
        "schema_version": "sam3d-subject-geometry-diagnostic-0.1",
        "source_conventions": {
            "global_rot": "MHR body/root global rotation expressed as roma ZYX Euler angles",
            "projection": "P_camera = reconstructed_root_relative_point + pred_cam_t; global body rotation is already baked into reconstructed points",
            "camera_axis": "+Z is camera optical/depth axis for perspective projection",
            "camera_system_flip_xyz": [1, -1, -1],
            "canonical_body_axis_hypothesis": "+Y=body up; +Z=body forward; +X=subject anatomical left, empirically checked from reconstructed shoulders",
        },
        "raw": {
            "global_rot_rad_zyx": [_round(v, 6) for v in global_rot],
            "global_rot_deg_zyx": [_round(math.degrees(v), 3) for v in global_rot],
            "pred_cam_t": _round_vec(cam_t),
        },
        "transforms": {
            "body_to_mhr_rotation": _round_matrix(body_to_mhr),
            "body_to_camera_rotation": _round_matrix(body_to_camera),
            "camera_center_body_xyz": _round_vec(camera_center_body),
            "camera_optical_axis_body_xyz": _round_vec(camera_forward_body),
            "body_forward_camera_xyz": _round_vec(body_forward_camera),
        },
        "body_camera_relation": {
            "yaw_deg": _round(body_yaw),
            "pitch_deg": _round(body_pitch),
            "orientation_band": body_band,
            "faces_frame": frame_facing,
            "authority": "candidate_until_observation_gate_and_cross_source_validation",
        },
        "camera_relative_subject": {
            "center_body_xyz": _round_vec(camera_center_body),
            "side": camera_side,
            "lateral_shoulder_widths": _round(
                float(camera_center_body[0]) / shoulder_width if shoulder_width > 1e-9 else None
            ),
            "vertical_vs_eye": _round(camera_eye_delta),
            "vertical_vs_shoulders": _round(camera_shoulder_delta),
            "vertical_vs_hips": _round(camera_hip_delta),
            "vertical_band": vertical_band,
            "optical_axis_pitch_deg": _round(optical_pitch),
            "optical_axis_yaw_deg": _round(optical_yaw),
            "selfie_like_geometry": selfie_like_geometry,
            "authority": "subject_relative_geometry_not_world_camera_elevation",
        },
        "face_camera_relation": {
            "face_forward_camera_xyz": _round_vec(face_forward_camera),
            "yaw_deg": _round(face_yaw),
            "pitch_deg": _round(face_pitch),
            "orientation_band": face_band,
            "head_turn_relative_body_yaw_deg": _round(head_turn_relative_body),
            "head_turn_toward_camera_deg": _round(turn_toward_camera),
            "authority": "reconstructed_face_proxy_requires_observed_face_gate",
        },
        "body_frame_landmarks": {
            "nose": _round_vec(nose_body),
            "eye_mid": _round_vec(eye_body),
            "ear_mid": _round_vec(ear_body),
            "neck": _round_vec(neck_body),
            "shoulder_mid": _round_vec(shoulder_body),
            "hip_mid": _round_vec(hip_body),
            "knee_mid": _round_vec(knee_body),
            "ankle_mid": _round_vec(ankle_body),
            "shoulder_width": _round(shoulder_width, 6),
            "canonical_lateral_axis_check": {
                "left_shoulder_x": _round(canonical_left_x, 6),
                "right_shoulder_x": _round(canonical_right_x, 6),
                "plus_x_is_subject_left": lateral_axis_valid,
            },
        },
        "dwpose_visibility_gate": visibility,
        "compound_pose_hint": compound_hint,
        "policy": {
            "report_only": True,
            "body_yaw_may_validate_analyze_when": "bilateral shoulders are independently observed and SAM3D agrees with other torso-orientation evidence",
            "face_yaw_may_validate_analyze_when": "nose, both eyes, and both ears are independently observed",
            "subject_relative_camera_use": "useful for selfies and approximately upright/studio-like poses; do not convert to world high/low for bent or reclining bodies without posture/world-frame qualification",
        },
    }


def _discover_npz(sam3d_dir: Path, includes: list[str]) -> list[Path]:
    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if not includes:
        return paths
    wanted = [item.lower() for item in includes]
    return [p for p in paths if any(token in p.stem.lower() for token in wanted)]


def _resolve_dwpose(sam3d_dir: Path, supplied: Path | None) -> Path | None:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        return value if value.is_dir() else None
    sibling = sam3d_dir.parent / "dwpose"
    return sibling if sibling.is_dir() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-subject-geometry-diagnostic",
        description="Inspect cached SAM3D root rotation/camera translation as subject-relative camera/body/head geometry.",
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
    dwpose_dir = _resolve_dwpose(sam3d_dir, args.dwpose_dir)

    paths = _discover_npz(sam3d_dir, args.include)
    if not paths:
        raise SystemExit("No matching *.sam3d_arrays.npz records found")

    rows: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        dwpose_path = dwpose_dir / f"{key}.dwpose.json" if dwpose_dir else None
        dwpose = _read_json(dwpose_path) if dwpose_path and dwpose_path.exists() else None
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
        "schema_version": "sam3d-subject-geometry-diagnostic-run-0.1",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir) if dwpose_dir else None,
        "record_count": len(rows),
        "records": rows,
    }
    out = (args.output or (sam3d_dir / "sam3d_subject_geometry_diagnostic.json")).expanduser().resolve()
    _write_json(out, payload)
    print(f"Diagnostic: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
