from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .caption_perception_policy import _dwpose_points


SCHEMA_VERSION = "pose-support-shape-diagnostic-0.1"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "support-shape-diagnostic-v01"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _mid(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    return ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0)


def _joint_angle(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
    c: tuple[float, float] | None,
) -> float | None:
    if a is None or b is None or c is None:
        return None
    ux, uy = float(a[0] - b[0]), float(a[1] - b[1])
    vx, vy = float(c[0] - b[0]), float(c[1] - b[1])
    denom = math.hypot(ux, uy) * math.hypot(vx, vy)
    if denom <= 1e-8:
        return None
    cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / denom))
    return math.degrees(math.acos(cosine))


def _angle_from_down_vertical(
    origin: tuple[float, float] | None,
    distal: tuple[float, float] | None,
) -> float | None:
    if origin is None or distal is None:
        return None
    dx = float(distal[0] - origin[0])
    dy = float(distal[1] - origin[1])
    length = math.hypot(dx, dy)
    if length <= 1e-8:
        return None
    cosine = max(-1.0, min(1.0, dy / length))
    return math.degrees(math.acos(cosine))


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(float(value)) else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1e-8:
        return None
    return float(numerator) / float(denominator)


def _torso_geometry(points: dict[str, tuple[float, float] | None]) -> dict[str, Any]:
    shoulder_mid = _mid(points.get("left_shoulder"), points.get("right_shoulder"))
    hip_mid = _mid(points.get("left_hip"), points.get("right_hip"))
    torso_length = _distance(shoulder_mid, hip_mid)
    return {
        "shoulder_midpoint": list(shoulder_mid) if shoulder_mid is not None else None,
        "hip_midpoint": list(hip_mid) if hip_mid is not None else None,
        "shoulder_midpoint_to_hip_midpoint_px": _round(torso_length, 2),
        "axis_angle_from_down_vertical_deg": _round(_angle_from_down_vertical(shoulder_mid, hip_mid), 1),
    }


def _leg_metrics(
    points: dict[str, tuple[float, float] | None],
    side: str,
    torso_length: float | None,
) -> dict[str, Any]:
    hip = points.get(f"{side}_hip")
    knee = points.get(f"{side}_knee")
    ankle = points.get(f"{side}_ankle")
    observed = hip is not None and knee is not None and ankle is not None

    thigh = _distance(hip, knee)
    shin = _distance(knee, ankle)
    path = thigh + shin if thigh is not None and shin is not None else None
    hip_ankle_euclidean = _distance(hip, ankle)
    vertical_drop = float(ankle[1] - hip[1]) if hip is not None and ankle is not None else None
    horizontal_offset = abs(float(ankle[0] - hip[0])) if hip is not None and ankle is not None else None

    return {
        "observed_full_chain": observed,
        "knee_angle_deg": _round(_joint_angle(hip, knee, ankle), 1),
        "hip_to_ankle_angle_from_down_vertical_deg": _round(_angle_from_down_vertical(hip, ankle), 1),
        "hip_to_ankle_vertical_drop_px": _round(vertical_drop, 2),
        "hip_to_ankle_horizontal_offset_px": _round(horizontal_offset, 2),
        "hip_to_ankle_euclidean_px": _round(hip_ankle_euclidean, 2),
        "thigh_length_px": _round(thigh, 2),
        "shin_length_px": _round(shin, 2),
        "thigh_plus_shin_path_length_px": _round(path, 2),
        "hip_to_ankle_vertical_drop_over_torso": _round(_safe_ratio(vertical_drop, torso_length), 3),
        "hip_to_ankle_euclidean_over_torso": _round(_safe_ratio(hip_ankle_euclidean, torso_length), 3),
        "chain_straightness_euclidean_over_path": _round(_safe_ratio(hip_ankle_euclidean, path), 3),
        "vertical_efficiency_drop_over_path": _round(_safe_ratio(vertical_drop, path), 3),
    }


def build_diagnostic(
    *,
    image_key: str,
    policy: dict[str, Any],
    dwpose_record: dict[str, Any],
) -> dict[str, Any]:
    size = policy.get("image_size")
    if not isinstance(size, list) or len(size) < 2:
        raise ValueError(f"{image_key}: perception policy missing image_size")
    width, height = int(size[0]), int(size[1])
    points = _dwpose_points(dwpose_record, width, height)

    torso = _torso_geometry(points)
    torso_length_raw = _distance(
        _mid(points.get("left_shoulder"), points.get("right_shoulder")),
        _mid(points.get("left_hip"), points.get("right_hip")),
    )
    legs = {
        side: _leg_metrics(points, side, torso_length_raw)
        for side in ("left", "right")
    }

    ankle_y = {
        side: float(points[f"{side}_ankle"][1])
        for side in ("left", "right")
        if points.get(f"{side}_ankle") is not None
    }
    lower_ankle_side = max(ankle_y, key=ankle_y.get) if ankle_y else None

    hip_mid = _mid(points.get("left_hip"), points.get("right_hip"))
    lower_ankle = points.get(f"{lower_ankle_side}_ankle") if lower_ankle_side else None
    pelvis_to_lower_ankle_vertical = (
        float(lower_ankle[1] - hip_mid[1])
        if hip_mid is not None and lower_ankle is not None
        else None
    )
    pelvis_to_lower_ankle_euclidean = _distance(hip_mid, lower_ankle)

    vertical_ratios = [
        value
        for side in ("left", "right")
        for value in [legs[side].get("hip_to_ankle_vertical_drop_over_torso")]
        if isinstance(value, (int, float))
    ]
    straightness = [
        value
        for side in ("left", "right")
        for value in [legs[side].get("chain_straightness_euclidean_over_path")]
        if isinstance(value, (int, float))
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "image_key": image_key,
        "route_mode": ((policy.get("policy") or {}).get("mode") if isinstance(policy.get("policy"), dict) else None),
        "pose_relevance": policy.get("pose_relevance"),
        "image_size": [width, height],
        "sources": {
            "perception_policy": policy.get("_source_path"),
            "dwpose": ((policy.get("sources") or {}).get("dwpose") if isinstance(policy.get("sources"), dict) else None),
        },
        "torso": torso,
        "legs": legs,
        "global": {
            "lower_ankle_side_in_frame": lower_ankle_side,
            "pelvis_mid_to_lower_ankle_vertical_drop_px": _round(pelvis_to_lower_ankle_vertical, 2),
            "pelvis_mid_to_lower_ankle_euclidean_px": _round(pelvis_to_lower_ankle_euclidean, 2),
            "pelvis_mid_to_lower_ankle_vertical_drop_over_torso": _round(
                _safe_ratio(pelvis_to_lower_ankle_vertical, torso_length_raw), 3
            ),
            "pelvis_mid_to_lower_ankle_euclidean_over_torso": _round(
                _safe_ratio(pelvis_to_lower_ankle_euclidean, torso_length_raw), 3
            ),
            "max_leg_vertical_drop_over_torso": _round(max(vertical_ratios), 3) if vertical_ratios else None,
            "min_leg_vertical_drop_over_torso": _round(min(vertical_ratios), 3) if vertical_ratios else None,
            "max_leg_chain_straightness": _round(max(straightness), 3) if straightness else None,
            "min_leg_chain_straightness": _round(min(straightness), 3) if straightness else None,
        },
        "interpretation": {
            "classification": None,
            "thresholds_applied": False,
            "note": (
                "Diagnostic only. Measures whole-body support/compression geometry without deciding standing, crouched, seated, "
                "or changing any caption fact. Compare distributions before adding specialist authority."
            ),
        },
    }


def _discover_policy_files(policy_dir: Path, only: list[str]) -> list[Path]:
    files = sorted(policy_dir.glob("*.perception_policy.json"))
    if not only:
        return files
    wanted = set(only)
    return [p for p in files if p.name.removesuffix(".perception_policy.json") in wanted]


def _tsv_row(record: dict[str, Any]) -> list[str]:
    torso = record.get("torso") if isinstance(record.get("torso"), dict) else {}
    legs = record.get("legs") if isinstance(record.get("legs"), dict) else {}
    global_metrics = record.get("global") if isinstance(record.get("global"), dict) else {}
    row: list[Any] = [
        record.get("image_key"),
        record.get("route_mode"),
        torso.get("shoulder_midpoint_to_hip_midpoint_px"),
    ]
    for side in ("left", "right"):
        leg = legs.get(side) if isinstance(legs.get(side), dict) else {}
        row.extend([
            leg.get("knee_angle_deg"),
            leg.get("hip_to_ankle_angle_from_down_vertical_deg"),
            leg.get("hip_to_ankle_vertical_drop_over_torso"),
            leg.get("chain_straightness_euclidean_over_path"),
            leg.get("vertical_efficiency_drop_over_path"),
        ])
    row.extend([
        global_metrics.get("lower_ankle_side_in_frame"),
        global_metrics.get("pelvis_mid_to_lower_ankle_vertical_drop_over_torso"),
        global_metrics.get("max_leg_vertical_drop_over_torso"),
        global_metrics.get("min_leg_chain_straightness"),
    ])
    return ["-" if value is None else str(value) for value in row]


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic comparison of DWPose support/compression geometry.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = (
        args.policy_dir.expanduser().resolve()
        if args.policy_dir
        else run_dir / "semantic-v3" / "caption-perception-policy-v0.1"
    )
    output = args.output.expanduser().resolve() if args.output else run_dir / DEFAULT_OUTPUT_SUBDIR
    if not policy_dir.is_dir():
        raise SystemExit(f"Perception-policy directory not found: {policy_dir}")
    output.mkdir(parents=True, exist_ok=True)

    policy_files = _discover_policy_files(policy_dir, args.only)
    if args.only:
        found = {p.name.removesuffix(".perception_policy.json") for p in policy_files}
        missing = sorted(set(args.only) - found)
        if missing:
            raise SystemExit("Missing perception-policy records: " + ", ".join(missing))

    records: list[dict[str, Any]] = []
    for policy_path in policy_files:
        image_key = policy_path.name.removesuffix(".perception_policy.json")
        out_path = output / f"{image_key}.support_shape_diagnostic.json"
        if out_path.is_file() and not args.overwrite:
            records.append(_read_json(out_path))
            continue

        policy = _read_json(policy_path)
        policy["_source_path"] = str(policy_path)
        sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
        dwpose_text = sources.get("dwpose")
        if not dwpose_text:
            raise SystemExit(f"{image_key}: perception policy has no DWPose source")
        dwpose_path = Path(str(dwpose_text)).expanduser()
        if not dwpose_path.is_file():
            raise SystemExit(f"{image_key}: DWPose source not found: {dwpose_path}")

        record = build_diagnostic(
            image_key=image_key,
            policy=policy,
            dwpose_record=_read_json(dwpose_path),
        )
        _write_json(out_path, record)
        records.append(record)

    header = [
        "image_key", "route", "torso_px",
        "left_knee_deg", "left_axis_deg", "left_vdrop_torso", "left_straightness", "left_vertical_eff",
        "right_knee_deg", "right_axis_deg", "right_vdrop_torso", "right_straightness", "right_vertical_eff",
        "lower_ankle_side", "pelvis_to_lower_ankle_vdrop_torso", "max_leg_vdrop_torso", "min_leg_straightness",
    ]
    tsv_lines = ["\t".join(header)] + ["\t".join(_tsv_row(record)) for record in records]
    tsv_path = output / "support_shape_diagnostic.tsv"
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "record_count": len(records),
        "records": [
            {
                "image_key": r.get("image_key"),
                "route_mode": r.get("route_mode"),
                "global": r.get("global"),
            }
            for r in records
        ],
        "tsv": str(tsv_path),
        "invariants": {
            "diagnostic_only": True,
            "no_caption_facts_modified": True,
            "no_pose_classification_performed": True,
            "no_thresholds_applied": True,
        },
    }
    _write_json(output / "support_shape_diagnostic.index.json", index)

    for line in tsv_lines:
        print(line)
    print(f"TSV: {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
