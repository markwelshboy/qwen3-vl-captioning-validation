from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LOW_TORSO_CAMERA_LONGITUDINAL_FRACTION_MAX = -0.25
LOW_HIP_TO_SHOULDER_SIGNED_DEPTH_FRACTION_MIN = 0.15


def _vec3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out) else None


def _midpoint(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if a is None or b is None:
        return None
    return [(a[i] + b[i]) / 2.0 for i in range(3)]


def _add(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if a is None or b is None:
        return None
    return [a[i] + b[i] for i in range(3)]


def _sub(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if a is None or b is None:
        return None
    return [a[i] - b[i] for i in range(3)]


def _scale(v: list[float] | None, factor: float) -> list[float] | None:
    if v is None:
        return None
    return [float(x) * float(factor) for x in v]


def _dot(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _norm(v: list[float] | None) -> float | None:
    if v is None:
        return None
    n = math.sqrt(sum(x * x for x in v))
    return n if n > 1e-12 and math.isfinite(n) else None


def _unit(v: list[float] | None) -> list[float] | None:
    n = _norm(v)
    if v is None or n is None:
        return None
    return [x / n for x in v]


def _signed_depth_fraction(a: list[float] | None, b: list[float] | None) -> float | None:
    """Signed z component of vector a->b divided by its 3-D length."""
    v = _sub(b, a)
    n = _norm(v)
    if v is None or n is None:
        return None
    return round(v[2] / n, 6)


def _depth_delta_nearer_lower(lower: list[float] | None, upper: list[float] | None) -> float | None:
    """Positive when lower body's z is smaller/nearer than upper body's z.

    SAM3D's projection uses pred_keypoints_3d + pred_cam_t, so translation cancels
    for this relative depth comparison. This is deliberately diagnostic only.
    """
    if lower is None or upper is None:
        return None
    return round(upper[2] - lower[2], 6)


def _ray_elevation_deg(camera_space_point: list[float] | None) -> float | None:
    """Vertical ray angle in SAM3D camera coordinates, assuming +Y is image-up.

    This is not a world/gravity camera elevation estimate. It is useful only as a
    camera-relative diagnostic and must be calibrated empirically before use.
    """
    if camera_space_point is None:
        return None
    y = float(camera_space_point[1])
    z = float(camera_space_point[2])
    if not math.isfinite(y) or not math.isfinite(z) or abs(z) < 1e-12:
        return None
    return round(math.degrees(math.atan2(y, z)), 3)


def _round_vec(value: list[float] | None) -> list[float] | None:
    return [round(v, 6) for v in value] if value is not None else None


def _camera_longitudinal_metric(
    *,
    axis_start: list[float] | None,
    axis_end: list[float] | None,
    anchor: list[float] | None,
    cam_t: list[float] | None,
) -> dict[str, Any]:
    """Project the camera direction onto a reconstructed body-longitudinal axis.

    SAM3D keypoints are root-relative and official projection first adds pred_cam_t.
    The camera is therefore the origin in camera coordinates. We form the vector
    from an anchor body point to the camera, project it onto a body axis, and
    normalize by that axis length.

    Negative values mean the camera lies toward the axis-start/footward direction;
    positive values mean it lies toward the axis-end/headward direction. This is
    body-relative, not gravity/world-relative, and strong forward torso pitch can
    contaminate the positive/high direction.
    """
    axis = _sub(axis_end, axis_start)
    axis_len = _norm(axis)
    axis_unit = _unit(axis)
    anchor_cam = _add(anchor, cam_t)
    anchor_to_camera = _scale(anchor_cam, -1.0)
    camera_distance = _norm(anchor_to_camera)
    camera_unit = _unit(anchor_to_camera)
    longitudinal = _dot(anchor_to_camera, axis_unit)
    cosine = _dot(camera_unit, axis_unit)

    if longitudinal is None or axis_len is None:
        fraction = None
    else:
        fraction = round(longitudinal / axis_len, 6)

    if cosine is None:
        angle = None
    else:
        angle = round(math.degrees(math.asin(max(-1.0, min(1.0, cosine)))), 3)

    return {
        "axis_start_xyz": _round_vec(axis_start),
        "axis_end_xyz": _round_vec(axis_end),
        "anchor_xyz": _round_vec(anchor),
        "axis_length": round(axis_len, 6) if axis_len is not None else None,
        "anchor_to_camera_distance": round(camera_distance, 6) if camera_distance is not None else None,
        "camera_longitudinal_distance": round(longitudinal, 6) if longitudinal is not None else None,
        "camera_longitudinal_fraction": fraction,
        "camera_longitudinal_angle_deg": angle,
        "sign_convention": "negative=axis-start/footward of anchor; positive=axis-end/headward of anchor",
        "authority": "sam3d_body_relative_diagnostic_not_world_gravity",
    }


def _dwpose_visibility(dwpose_record: dict[str, Any] | None) -> dict[str, Any]:
    target = (((dwpose_record or {}).get("derived") or {}).get("target") or {})
    visible = {
        str(item)
        for item in (target.get("visible_body_landmarks") or [])
        if isinstance(item, str)
    }

    def bilateral(a: str, b: str) -> bool:
        return a in visible and b in visible

    shoulders = bilateral("left_shoulder", "right_shoulder")
    hips = bilateral("left_hip", "right_hip")
    ankles = bilateral("left_ankle", "right_ankle")
    knees = bilateral("left_knee", "right_knee")

    return {
        "dwpose_available": bool(dwpose_record),
        "visible_body_landmarks": sorted(visible),
        "visible_body_landmark_count": len(visible),
        "pose_extent_hint": target.get("pose_extent_hint"),
        "shoulders_observed_bilateral": shoulders,
        "hips_observed_bilateral": hips,
        "knees_observed_bilateral": knees,
        "ankles_observed_bilateral": ankles,
        "torso_axis_visibility_qualified": shoulders and hips,
        "whole_body_axis_visibility_qualified": shoulders and hips and ankles,
        "authority": "dwpose_image_observation_gate",
    }


def _low_angle_support(
    *,
    torso_metric: dict[str, Any],
    ordering: dict[str, Any],
    visibility: dict[str, Any],
) -> dict[str, Any]:
    longitudinal = torso_metric.get("camera_longitudinal_fraction")
    hip_to_shoulder = ordering.get("hip_to_shoulder_signed_depth_fraction")
    ankle_to_shoulder = ordering.get("ankle_to_shoulder_signed_depth_fraction")

    longitudinal_support = (
        isinstance(longitudinal, (int, float))
        and longitudinal <= LOW_TORSO_CAMERA_LONGITUDINAL_FRACTION_MAX
    )
    depth_support = (
        isinstance(hip_to_shoulder, (int, float))
        and hip_to_shoulder >= LOW_HIP_TO_SHOULDER_SIGNED_DEPTH_FRACTION_MIN
    )
    geometry_candidate = bool(longitudinal_support and depth_support)

    reasons: list[str] = []
    limitations: list[str] = []
    if longitudinal_support:
        reasons.append(
            f"torso camera longitudinal fraction {longitudinal:.3f} is footward of the calibrated {LOW_TORSO_CAMERA_LONGITUDINAL_FRACTION_MAX:.2f} threshold"
        )
    if depth_support:
        reasons.append(
            f"hip-to-shoulder signed depth fraction {hip_to_shoulder:.3f} shows lower torso reconstructed nearer than shoulders"
        )

    torso_visible = bool(visibility.get("torso_axis_visibility_qualified"))
    ankles_visible = bool(visibility.get("ankles_observed_bilateral"))
    ankle_depth_support = (
        isinstance(ankle_to_shoulder, (int, float))
        and ankle_to_shoulder >= LOW_HIP_TO_SHOULDER_SIGNED_DEPTH_FRACTION_MIN
    )

    if not geometry_candidate:
        action = "withheld"
        confidence = "withheld"
        qualified = None
        authority = "insufficient_low_angle_geometry"
        limitations.append("calibrated strong-low SAM3D body-axis signature is absent")
    elif torso_visible:
        action = "qualified"
        qualified = "low"
        authority = "dwpose_visible_torso_plus_sam3d_camera_geometry"
        if ankles_visible and ankle_depth_support:
            confidence = "strong"
            reasons.append("bilateral DWPose ankles are observed and ankle-to-shoulder depth independently agrees")
        else:
            confidence = "moderate"
            limitations.append("distal lower-body visibility/depth does not independently strengthen the torso-axis result")
    else:
        action = "supporting"
        confidence = "weak"
        qualified = None
        authority = "sam3d_camera_geometry_visibility_insufficient"
        limitations.append(
            "SAM3D low-angle signature cannot independently create a caption fact because DWPose does not observe both shoulders and both hips"
        )

    return {
        "candidate_elevation": "low" if geometry_candidate else None,
        "action": action,
        "qualified_elevation": qualified,
        "confidence_band": confidence,
        "authority": authority,
        "reasons": reasons,
        "limitations": limitations,
        "thresholds": {
            "torso_camera_longitudinal_fraction_max": LOW_TORSO_CAMERA_LONGITUDINAL_FRACTION_MAX,
            "hip_to_shoulder_signed_depth_fraction_min": LOW_HIP_TO_SHOULDER_SIGNED_DEPTH_FRACTION_MIN,
        },
        "policy": (
            "strong negative body-axis camera position may qualify LOW only with independently observed DWPose shoulders+hips; "
            "positive/headward geometry never independently qualifies HIGH because forward torso pitch contaminates that direction"
        ),
    }


def build_camera_diagnostic(
    record: dict[str, Any],
    dwpose_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    selected = metrics.get("selected_keypoints_xyz") or {}
    camera = record.get("camera") or {}
    cam_t = _vec3(camera.get("pred_cam_t"))

    left_shoulder = _vec3(selected.get("left_shoulder"))
    right_shoulder = _vec3(selected.get("right_shoulder"))
    left_hip = _vec3(selected.get("left_hip"))
    right_hip = _vec3(selected.get("right_hip"))
    left_ankle = _vec3(selected.get("left_ankle"))
    right_ankle = _vec3(selected.get("right_ankle"))
    neck = _vec3(selected.get("neck"))

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    ankle_mid = _midpoint(left_ankle, right_ankle)

    shoulder_cam = _add(shoulder_mid, cam_t)
    hip_cam = _add(hip_mid, cam_t)
    ankle_cam = _add(ankle_mid, cam_t)
    neck_cam = _add(neck, cam_t)

    torso_len = _norm(_sub(shoulder_mid, hip_mid))
    shoulder_minus_hip_depth = _depth_delta_nearer_lower(hip_mid, shoulder_mid)
    shoulder_minus_ankle_depth = _depth_delta_nearer_lower(ankle_mid, shoulder_mid)

    def normalized(value: float | None) -> float | None:
        if value is None or torso_len is None:
            return None
        return round(value / torso_len, 6)

    focal_raw = camera.get("focal_length")
    try:
        focal = float(focal_raw) if focal_raw is not None else None
    except (TypeError, ValueError):
        focal = None

    ordering = {
        "hip_to_shoulder_signed_depth_fraction": _signed_depth_fraction(hip_mid, shoulder_mid),
        "ankle_to_shoulder_signed_depth_fraction": _signed_depth_fraction(ankle_mid, shoulder_mid),
        "shoulder_minus_hip_depth": shoulder_minus_hip_depth,
        "shoulder_minus_ankle_depth": shoulder_minus_ankle_depth,
        "shoulder_minus_hip_depth_over_torso_length": normalized(shoulder_minus_hip_depth),
        "shoulder_minus_ankle_depth_over_torso_length": normalized(shoulder_minus_ankle_depth),
        "positive_depth_delta_means": "lower reference point is reconstructed closer to camera than upper reference point",
    }
    rays = {
        "shoulder_mid": _ray_elevation_deg(shoulder_cam),
        "hip_mid": _ray_elevation_deg(hip_cam),
        "ankle_mid": _ray_elevation_deg(ankle_cam),
        "neck": _ray_elevation_deg(neck_cam),
        "authority": "diagnostic_only_not_world_gravity_camera_elevation",
    }
    torso_camera = _camera_longitudinal_metric(
        axis_start=hip_mid,
        axis_end=shoulder_mid,
        anchor=hip_mid,
        cam_t=cam_t,
    )
    whole_body_camera = _camera_longitudinal_metric(
        axis_start=ankle_mid,
        axis_end=shoulder_mid,
        anchor=hip_mid,
        cam_t=cam_t,
    )
    visibility = _dwpose_visibility(dwpose_record)
    low_support = _low_angle_support(
        torso_metric=torso_camera,
        ordering=ordering,
        visibility=visibility,
    )

    return {
        "schema_version": "sam3d-camera-diagnostic-0.2",
        "source_sam3d_schema": record.get("schema_version"),
        "camera_translation_xyz": _round_vec(cam_t),
        "focal_length": focal,
        "root_relative_points": {
            "shoulder_mid": _round_vec(shoulder_mid),
            "hip_mid": _round_vec(hip_mid),
            "ankle_mid": _round_vec(ankle_mid),
            "neck": _round_vec(neck),
        },
        "camera_space_points": {
            "shoulder_mid": _round_vec(shoulder_cam),
            "hip_mid": _round_vec(hip_cam),
            "ankle_mid": _round_vec(ankle_cam),
            "neck": _round_vec(neck_cam),
        },
        "vertical_depth_ordering": ordering,
        "camera_ray_elevation_deg": rays,
        "body_axis_camera_position": {
            "torso_hip_to_shoulder": torso_camera,
            "whole_body_ankle_to_shoulder": whole_body_camera,
            "interpretation": (
                "camera position projected onto reconstructed body-longitudinal axes; negative is footward, positive is headward. "
                "This is body-relative and not a gravity/world camera pose."
            ),
        },
        "dwpose_visibility_gate": visibility,
        "low_angle_support": low_support,
        "existing_signed_depth_diagnostics": metrics.get("signed_depth_fraction_diagnostics"),
        "interpretation_policy": {
            "categorical_low_high_disabled": True,
            "low_angle_qualification_enabled": True,
            "categorical_high_disabled": True,
            "reason": (
                "legacy direct low/high classification from SAM3D reconstruction remains disabled. A calibrated strong-negative body-axis signature "
                "may qualify LOW only through the DWPose visibility gate; positive/headward geometry remains diagnostic because forward torso pitch can mimic HIGH."
            ),
            "projection_fact": "official SAM3D projection adds pred_cam_t to pred_keypoints_3d before perspective projection",
            "visibility_warning": (
                "ankle/hip geometry may be reconstructed outside the visible crop; reconstructed points remain report-only unless independently observed by DWPose"
            ),
        },
    }


def _discover_records(sam3d_dir: Path, includes: list[str]) -> list[Path]:
    paths = sorted(
        p for p in sam3d_dir.rglob("*.sam3d.json")
        if p.is_file() and p.name != "sam3d_probe.index.json"
    )
    if not includes:
        return paths
    wanted = [item.lower() for item in includes]
    return [p for p in paths if any(token in p.stem.lower() for token in wanted)]


def _find_dwpose_record(dwpose_dir: Path | None, key: str) -> tuple[dict[str, Any] | None, Path | None]:
    if dwpose_dir is None or not dwpose_dir.is_dir():
        return None, None
    direct = dwpose_dir / f"{key}.dwpose.json"
    candidates = [direct] if direct.exists() else list(dwpose_dir.rglob(f"{key}.dwpose.json"))
    if not candidates:
        return None, None
    path = candidates[0]
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except Exception:
        return None, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-camera-diagnostic",
        description=(
            "Read cached SAM3D records and report body-relative camera geometry plus a DWPose-visibility-gated low-angle diagnostic without rerunning inference."
        ),
    )
    parser.add_argument("sam3d_dir", type=Path)
    parser.add_argument(
        "--dwpose-dir",
        type=Path,
        help="DWPose cache directory. Defaults to a sibling 'dwpose' directory when present.",
    )
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")

    if args.dwpose_dir is not None:
        dwpose_dir: Path | None = args.dwpose_dir.expanduser().resolve()
        if not dwpose_dir.is_dir():
            raise SystemExit(f"DWPose directory not found: {dwpose_dir}")
    else:
        sibling = sam3d_dir.parent / "dwpose"
        dwpose_dir = sibling.resolve() if sibling.is_dir() else None

    paths = _discover_records(sam3d_dir, args.include)
    if not paths:
        raise SystemExit("No matching *.sam3d.json records found")

    rows: list[dict[str, Any]] = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = path.name.removesuffix(".sam3d.json")
        dwpose_record, dwpose_path = _find_dwpose_record(dwpose_dir, key)
        diagnostic = build_camera_diagnostic(record, dwpose_record=dwpose_record)
        rows.append({
            "image_key": key,
            "source": str(path),
            "dwpose_source": str(dwpose_path) if dwpose_path is not None else None,
            "diagnostic": diagnostic,
        })
        ordering = diagnostic["vertical_depth_ordering"]
        rays = diagnostic["camera_ray_elevation_deg"]
        torso_axis = diagnostic["body_axis_camera_position"]["torso_hip_to_shoulder"]
        low = diagnostic["low_angle_support"]
        print(
            f"{key}: "
            f"hip->shoulder dzfrac={ordering['hip_to_shoulder_signed_depth_fraction']}; "
            f"ankle->shoulder dzfrac={ordering['ankle_to_shoulder_signed_depth_fraction']}; "
            f"torso-cam-long={torso_axis['camera_longitudinal_fraction']}; "
            f"ray shoulder/hip/ankle={rays['shoulder_mid']}/{rays['hip_mid']}/{rays['ankle_mid']}°; "
            f"low={low['action']}/{low['confidence_band']}"
        )

    payload = {
        "schema_version": "sam3d-camera-diagnostic-run-0.2",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir) if dwpose_dir is not None else None,
        "record_count": len(rows),
        "records": rows,
    }
    out = (args.output or (sam3d_dir / "sam3d_camera_diagnostic.json")).expanduser().resolve()
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Diagnostic: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
