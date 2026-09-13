from __future__ import annotations

import math
from typing import Any

import numpy as np

from .sam3d_subject_geometry_diagnostic_02 import build_subject_geometry as build_legacy_geometry

MHR = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 9,
    "right_hip": 10,
}


def _round(value: float | None, digits: int = 1) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _unit_xz(value: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    out = np.array([arr[0], arr[2]], dtype=np.float64)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-9:
        return None
    return out / norm


def _signed_yaw_between_deg(forward_xyz: np.ndarray, toward_camera_xyz: np.ndarray) -> float | None:
    """Signed yaw from a body-forward vector to the actual subject->camera ray.

    Both vectors are projected into camera X/Z. Zero means the body segment
    faces the physical camera center, regardless of where the subject sits in
    the image. This intentionally differs from yaw against the camera optical
    axis when the subject is off-axis.
    """
    a = _unit_xz(forward_xyz)
    b = _unit_xz(toward_camera_xyz)
    if a is None or b is None:
        return None
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    cross = float(a[0] * b[1] - a[1] * b[0])
    return math.degrees(math.atan2(cross, dot))


def _wrap_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _orientation_band(yaw_deg: float | None) -> str | None:
    """Map caption-facing yaw to semantic bands without changing legacy diagnostics.

    The legacy SAM3D diagnostic deliberately keeps its calibrated v0.2 bands.
    Caption-facing orientation uses an additional ``oblique`` band so a clear
    ~25-35 degree turn is not weakened to the phrase ``slightly angled`` while
    still remaining distinct from a genuine three-quarter ~35-65 degree turn.
    """
    if yaw_deg is None:
        return None
    a = abs(float(yaw_deg))
    if a <= 15.0:
        return "frontal"
    if a <= 25.0:
        return "slightly_angled"
    if a <= 35.0:
        return "oblique"
    if a <= 65.0:
        return "three_quarter"
    if a <= 115.0:
        return "side_on"
    if a <= 160.0:
        return "rear_three_quarter"
    return "rear"


def _approx_angle(value: float | None) -> int | None:
    if value is None:
        return None
    # Caption-facing angles should be approximate, not forensic measurements.
    return int(5 * round(abs(float(value)) / 5.0))


def _orientation_payload(yaw_deg: float | None, *, reference: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "yaw_deg": _round(yaw_deg),
        "yaw_magnitude_deg": _round(abs(float(yaw_deg))) if yaw_deg is not None else None,
        "approx_yaw_deg": _approx_angle(yaw_deg),
        "orientation_band": _orientation_band(yaw_deg),
        "reference": reference,
    }
    if extra:
        payload.update(extra)
    return payload


def _caption_summary(root: dict[str, Any], upper: dict[str, Any]) -> dict[str, Any]:
    root_band = root.get("orientation_band")
    upper_band = upper.get("orientation_band")
    root_yaw = root.get("yaw_deg")
    upper_yaw = upper.get("yaw_deg")
    delta = None
    if isinstance(root_yaw, (int, float)) and isinstance(upper_yaw, (int, float)):
        delta = _wrap_deg(float(upper_yaw) - float(root_yaw))
    twist_mag = abs(delta) if delta is not None else None

    if upper_band is None:
        mode = "root_only"
        preferred = root_band
    elif root_band is None:
        mode = "upper_torso_only"
        preferred = upper_band
    elif root_band == upper_band:
        mode = "combined"
        preferred = upper_band
    elif twist_mag is not None and twist_mag >= 12.0:
        mode = "articulated"
        preferred = upper_band
    else:
        # A small numerical difference can straddle a bucket boundary. Do not
        # manufacture torso twist prose merely because adjacent semantic bands
        # differ; prefer the caption-relevant upper torso.
        mode = "upper_torso_dominant"
        preferred = upper_band

    return {
        "mode": mode,
        "preferred_orientation_band": preferred,
        "preferred_approx_yaw_deg": upper.get("approx_yaw_deg") if upper_band is not None else root.get("approx_yaw_deg"),
        "relative_twist_yaw_deg": _round(delta),
        "relative_twist_magnitude_deg": _round(twist_mag),
        "meaningful_twist": bool(twist_mag is not None and twist_mag >= 12.0),
        "note": (
            "Collapse root and upper-torso orientation when they substantially agree; preserve both only when "
            "articulation creates a meaningful camera-relative torso twist."
        ),
    }


def build_caption_orientation(
    arrays: dict[str, np.ndarray],
    dwpose_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build caption-facing root and upper-torso camera orientation.

    The older diagnostic measured global/root yaw against the camera optical
    axis. That is useful diagnostically but is not exactly "angle to camera"
    for an off-axis subject. Here the reference is the ray from each body
    segment to the physical camera center. Upper-torso orientation is derived
    from the reconstructed shoulder/hip plane and kept separate from body/root
    orientation so seated/twisted poses are not collapsed prematurely.
    """
    legacy = build_legacy_geometry(arrays, dwpose_record)
    keypoints = np.asarray(arrays["pred_keypoints_3d"], dtype=np.float64)
    cam_t = np.asarray(arrays["pred_cam_t"], dtype=np.float64).reshape(3)
    if keypoints.ndim != 2 or keypoints.shape[0] < 11 or keypoints.shape[1] < 3:
        raise ValueError("pred_keypoints_3d must contain shoulders and hips")
    points = keypoints[:, :3]

    transforms = legacy.get("transforms") if isinstance(legacy.get("transforms"), dict) else {}
    root_forward = np.asarray(transforms.get("body_forward_camera_xyz"), dtype=np.float64).reshape(3)
    root_to_camera = -cam_t
    root_yaw_camera_center = _signed_yaw_between_deg(root_forward, root_to_camera)

    ls = points[MHR["left_shoulder"]]
    rs = points[MHR["right_shoulder"]]
    lh = points[MHR["left_hip"]]
    rh = points[MHR["right_hip"]]
    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    shoulder_axis = ls - rs
    torso_up = shoulder_mid - hip_mid

    upper_forward = np.cross(shoulder_axis, torso_up)
    upper_norm = float(np.linalg.norm(upper_forward))
    if upper_norm <= 1e-9 or not np.all(np.isfinite(upper_forward)):
        upper_forward = root_forward.copy()
        upper_source = "root_forward_fallback_degenerate_torso_plane"
    else:
        upper_forward = upper_forward / upper_norm
        if float(np.dot(upper_forward, root_forward)) < 0.0:
            upper_forward = -upper_forward
        upper_source = "reconstructed_shoulder_hip_plane"

    upper_center_root_relative = (shoulder_mid + hip_mid) / 2.0
    upper_center_camera = upper_center_root_relative + cam_t
    upper_to_camera = -upper_center_camera
    upper_yaw_camera_center = _signed_yaw_between_deg(upper_forward, upper_to_camera)

    legacy_body = legacy.get("body_camera_relation") if isinstance(legacy.get("body_camera_relation"), dict) else {}
    root = _orientation_payload(
        root_yaw_camera_center,
        reference="body_root_forward_to_physical_camera_center",
        extra={
            "optical_axis_yaw_deg": legacy_body.get("yaw_deg"),
            "optical_axis_orientation_band": legacy_body.get("orientation_band"),
            "forward_camera_xyz": [round(float(v), 6) for v in root_forward],
            "to_camera_xyz": [round(float(v), 6) for v in root_to_camera],
        },
    )
    upper = _orientation_payload(
        upper_yaw_camera_center,
        reference="upper_torso_plane_forward_to_physical_camera_center",
        extra={
            "source_geometry": upper_source,
            "forward_camera_xyz": [round(float(v), 6) for v in upper_forward],
            "to_camera_xyz": [round(float(v), 6) for v in upper_to_camera],
        },
    )
    summary = _caption_summary(root, upper)

    return {
        "schema_version": "sam3d-caption-orientation-0.1",
        "body_root_orientation": root,
        "upper_torso_orientation": upper,
        "caption_orientation": summary,
        "legacy_optical_axis": {
            "yaw_deg": legacy_body.get("yaw_deg"),
            "orientation_band": legacy_body.get("orientation_band"),
            "note": "Legacy root/global yaw against camera optical axis; retained for diagnostics, not preferred caption wording.",
        },
        "policy": {
            "caption_reference_is_physical_camera_center": True,
            "upper_torso_is_default_caption_orientation": True,
            "root_and_upper_torso_remain_separate_when_meaningfully_twisted": True,
            "legacy_diagnostic_orientation_thresholds_unchanged": True,
            "caption_orientation_has_intermediate_oblique_band": True,
            "caption_orientation_bands": {
                "frontal": "abs(yaw)<=15",
                "slightly_angled": "15<abs(yaw)<=25",
                "oblique": "25<abs(yaw)<=35",
                "three_quarter": "35<abs(yaw)<=65",
                "side_on": "65<abs(yaw)<=115",
                "rear_three_quarter": "115<abs(yaw)<=160",
                "rear": "abs(yaw)>160",
            },
        },
    }
