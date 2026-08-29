from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


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


def _norm(v: list[float] | None) -> float | None:
    if v is None:
        return None
    n = math.sqrt(sum(x * x for x in v))
    return n if n > 1e-12 and math.isfinite(n) else None


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


def build_camera_diagnostic(record: dict[str, Any]) -> dict[str, Any]:
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

    return {
        "schema_version": "sam3d-camera-diagnostic-0.1",
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
        "vertical_depth_ordering": {
            "hip_to_shoulder_signed_depth_fraction": _signed_depth_fraction(hip_mid, shoulder_mid),
            "ankle_to_shoulder_signed_depth_fraction": _signed_depth_fraction(ankle_mid, shoulder_mid),
            "shoulder_minus_hip_depth": shoulder_minus_hip_depth,
            "shoulder_minus_ankle_depth": shoulder_minus_ankle_depth,
            "shoulder_minus_hip_depth_over_torso_length": normalized(shoulder_minus_hip_depth),
            "shoulder_minus_ankle_depth_over_torso_length": normalized(shoulder_minus_ankle_depth),
            "positive_depth_delta_means": "lower reference point is reconstructed closer to camera than upper reference point",
        },
        "camera_ray_elevation_deg": {
            "shoulder_mid": _ray_elevation_deg(shoulder_cam),
            "hip_mid": _ray_elevation_deg(hip_cam),
            "ankle_mid": _ray_elevation_deg(ankle_cam),
            "neck": _ray_elevation_deg(neck_cam),
            "authority": "diagnostic_only_not_world_gravity_camera_elevation",
        },
        "existing_signed_depth_diagnostics": metrics.get("signed_depth_fraction_diagnostics"),
        "interpretation_policy": {
            "categorical_low_high_disabled": True,
            "reason": (
                "single-image SAM3D camera/body reconstruction is model-prior dependent; first calibrate signed vertical depth ordering "
                "against known low/eye/high examples before granting camera-elevation authority"
            ),
            "projection_fact": "official SAM3D projection adds pred_cam_t to pred_keypoints_3d before perspective projection",
            "visibility_warning": (
                "ankle/hip geometry may be reconstructed outside the visible crop; only use a vertical-depth segment after independent image visibility support"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-camera-diagnostic",
        description="Read cached SAM3D records and report camera-relative vertical depth geometry without rerunning inference.",
    )
    parser.add_argument("sam3d_dir", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")

    paths = _discover_records(sam3d_dir, args.include)
    if not paths:
        raise SystemExit("No matching *.sam3d.json records found")

    rows: list[dict[str, Any]] = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        diagnostic = build_camera_diagnostic(record)
        key = path.name.removesuffix(".sam3d.json")
        rows.append({"image_key": key, "source": str(path), "diagnostic": diagnostic})
        ordering = diagnostic["vertical_depth_ordering"]
        rays = diagnostic["camera_ray_elevation_deg"]
        print(
            f"{key}: "
            f"hip->shoulder dzfrac={ordering['hip_to_shoulder_signed_depth_fraction']}; "
            f"ankle->shoulder dzfrac={ordering['ankle_to_shoulder_signed_depth_fraction']}; "
            f"shoulder-ankle dz={ordering['shoulder_minus_ankle_depth']}; "
            f"ray shoulder/hip/ankle={rays['shoulder_mid']}/{rays['hip_mid']}/{rays['ankle_mid']}°"
        )

    payload = {
        "schema_version": "sam3d-camera-diagnostic-run-0.1",
        "sam3d_dir": str(sam3d_dir),
        "record_count": len(rows),
        "records": rows,
    }
    out = (args.output or (sam3d_dir / "sam3d_camera_diagnostic.json")).expanduser().resolve()
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Diagnostic: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
