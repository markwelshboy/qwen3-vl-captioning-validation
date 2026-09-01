from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_06 as v06


MHR = v06.MHR
BALANCE_CONFLICT_REVIEW_THRESHOLD = 0.55
RECLINE_REVIEW_THRESHOLD = 0.60
KNEEL_CONTEXT_CONFLICT_THRESHOLD = 0.50


def _round(value: float | None, digits: int = 3) -> float | None:
    return v06._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v06._ramp(value, low, high)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _point(keypoints: np.ndarray, name: str) -> np.ndarray | None:
    idx = MHR.get(name)
    if idx is None or idx >= len(keypoints):
        return None
    point = np.asarray(keypoints[idx, :3], dtype=np.float64)
    return point if point.size >= 3 and np.all(np.isfinite(point)) else None


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0


def _foot_point(keypoints: np.ndarray, side: str) -> np.ndarray | None:
    points: list[np.ndarray] = []
    for suffix in ("ankle", "big_toe", "heel"):
        point = _point(keypoints, f"{side}_{suffix}")
        if point is not None:
            points.append(point)
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def _horizontal_distance(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    if scale <= 1e-9:
        return 0.0
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(delta[[0, 2]]) / scale)


def _distance_to_support_segment(
    point: np.ndarray,
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    scale: float,
) -> float:
    if scale <= 1e-9:
        return 0.0
    p = np.asarray(point, dtype=np.float64)[[0, 2]]
    a = np.asarray(left_foot, dtype=np.float64)[[0, 2]]
    b = np.asarray(right_foot, dtype=np.float64)[[0, 2]]
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a) / scale)
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest) / scale)


def _joint_support(profile: dict[str, Any], name: str) -> float:
    support = ((profile.get("evidence_support") or {}).get("joint_crop_support") or {})
    return float(support.get(name) or 0.0)


def _paired_support(profile: dict[str, Any], suffix: str) -> float:
    return float(np.mean([
        _joint_support(profile, f"left_{suffix}"),
        _joint_support(profile, f"right_{suffix}"),
    ]))


def _balance_support(profile: dict[str, Any]) -> float:
    return min(
        _paired_support(profile, "shoulder"),
        _paired_support(profile, "hip"),
        _paired_support(profile, "ankle"),
    )


def _recline_support(profile: dict[str, Any]) -> float:
    return min(
        _paired_support(profile, "shoulder"),
        _paired_support(profile, "hip"),
    )


def _body_support_geometry(keypoints: np.ndarray) -> dict[str, Any]:
    required = {
        name: _point(keypoints, name)
        for name in (
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        )
    }
    if any(value is None for value in required.values()):
        return {"available": False}

    ls = required["left_shoulder"]
    rs = required["right_shoulder"]
    lh = required["left_hip"]
    rh = required["right_hip"]
    lk = required["left_knee"]
    rk = required["right_knee"]
    la = required["left_ankle"]
    ra = required["right_ankle"]
    assert all(value is not None for value in (ls, rs, lh, rh, lk, rk, la, ra))

    shoulder_width = float(np.linalg.norm(ls - rs))
    if shoulder_width <= 1e-9:
        return {"available": False}

    shoulder_mid = _midpoint(ls, rs)
    hip_mid = _midpoint(lh, rh)
    knee_mid = _midpoint(lk, rk)
    ankle_mid = _midpoint(la, ra)
    left_foot = _foot_point(keypoints, "left") or la
    right_foot = _foot_point(keypoints, "right") or ra
    foot_mid = _midpoint(left_foot, right_foot)
    torso_proxy = 0.42 * shoulder_mid + 0.58 * hip_mid

    pelvis_to_support = _distance_to_support_segment(hip_mid, left_foot, right_foot, shoulder_width)
    shoulder_to_support = _distance_to_support_segment(shoulder_mid, left_foot, right_foot, shoulder_width)
    torso_to_support = _distance_to_support_segment(torso_proxy, left_foot, right_foot, shoulder_width)
    pelvis_to_foot_centroid = _horizontal_distance(hip_mid, foot_mid, shoulder_width)
    shoulder_to_foot_centroid = _horizontal_distance(shoulder_mid, foot_mid, shoulder_width)
    torso_to_foot_centroid = _horizontal_distance(torso_proxy, foot_mid, shoulder_width)
    support_base_span = _horizontal_distance(left_foot, right_foot, shoulder_width)
    shoulder_hip_horizontal = _horizontal_distance(shoulder_mid, hip_mid, shoulder_width)
    shoulder_hip_vertical = abs(float(hip_mid[1] - shoulder_mid[1])) / shoulder_width
    torso_angle = float(np.degrees(np.arctan2(
        shoulder_hip_horizontal,
        max(1e-9, shoulder_hip_vertical),
    )))

    centers = np.stack([shoulder_mid, hip_mid, knee_mid, ankle_mid], axis=0)
    vertical_span = float(np.ptp(centers[:, 1]) / shoulder_width)
    horizontal_span = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            horizontal_span = max(
                horizontal_span,
                _horizontal_distance(centers[i], centers[j], shoulder_width),
            )
    body_flatness_ratio = float(horizontal_span / max(0.10, vertical_span))

    shoulder_closure = pelvis_to_support - shoulder_to_support
    shoulder_retreat = shoulder_to_support - pelvis_to_support

    pelvis_balanced = 1.0 - _ramp(pelvis_to_support, 0.25, 1.05)
    torso_balanced = 1.0 - _ramp(torso_to_support, 0.25, 1.10)
    shoulder_moves_toward_support = _ramp(shoulder_closure, 0.05, 0.50)
    low_stance_balance = _clamp(
        0.45 * pelvis_balanced
        + 0.35 * torso_balanced
        + 0.20 * shoulder_moves_toward_support
    )

    pelvis_displaced = _ramp(pelvis_to_support, 0.40, 1.30)
    torso_upright = 1.0 - _ramp(torso_angle, 20.0, 62.0)
    shoulder_not_recovering = 1.0 - _ramp(shoulder_closure, 0.03, 0.45)
    seated_displacement = _clamp(
        0.45 * pelvis_displaced
        + 0.30 * torso_upright
        + 0.25 * shoulder_not_recovering
    )

    torso_recline = _ramp(torso_angle, 28.0, 76.0)
    whole_body_horizontal = _ramp(body_flatness_ratio, 0.55, 1.80)
    support_separation = _ramp(max(pelvis_to_support, shoulder_to_support), 0.55, 1.45)
    recline_geometry = _clamp(
        0.45 * torso_recline
        + 0.25 * whole_body_horizontal
        + 0.20 * support_separation
        + 0.10 * (1.0 - low_stance_balance)
    )
    lying_plane = _clamp(0.65 * torso_recline + 0.35 * whole_body_horizontal)

    return {
        "available": True,
        "shoulder_width_3d": _round(shoulder_width),
        "support_base_span_shoulder_widths": _round(support_base_span),
        "pelvis_to_support_segment_shoulder_widths": _round(pelvis_to_support),
        "shoulder_to_support_segment_shoulder_widths": _round(shoulder_to_support),
        "torso_proxy_to_support_segment_shoulder_widths": _round(torso_to_support),
        "pelvis_to_foot_centroid_shoulder_widths": _round(pelvis_to_foot_centroid),
        "shoulder_to_foot_centroid_shoulder_widths": _round(shoulder_to_foot_centroid),
        "torso_proxy_to_foot_centroid_shoulder_widths": _round(torso_to_foot_centroid),
        "shoulder_closure_toward_support_shoulder_widths": _round(shoulder_closure),
        "shoulder_retreat_from_support_shoulder_widths": _round(shoulder_retreat),
        "torso_horizontal_span_shoulder_widths": _round(shoulder_hip_horizontal),
        "torso_vertical_span_shoulder_widths": _round(shoulder_hip_vertical),
        "torso_axis_from_vertical_deg": _round(torso_angle),
        "whole_body_vertical_span_shoulder_widths": _round(vertical_span),
        "whole_body_horizontal_span_shoulder_widths": _round(horizontal_span),
        "body_flatness_ratio": _round(body_flatness_ratio),
        "low_stance_balance_score": _round(low_stance_balance, 4),
        "seated_displacement_score": _round(seated_displacement, 4),
        "torso_recline_score": _round(torso_recline, 4),
        "whole_body_horizontal_score": _round(whole_body_horizontal, 4),
        "recline_geometry_score": _round(recline_geometry, 4),
        "lying_plane_score": _round(lying_plane, 4),
    }


def _report_diagnostics(profile: dict[str, Any], keypoints: np.ndarray) -> None:
    geometry = _body_support_geometry(keypoints)
    projected = profile.get("sam3d_projected_pose") or {}
    posture_scores = {
        str(name): float(value or 0.0)
        for name, value in (projected.get("posture_scores") or {}).items()
    }

    if not geometry.get("available"):
        projected["support_balance_diagnostic"] = {"report_only": True, "available": False}
        projected["recline_diagnostic"] = {"report_only": True, "available": False}
        projected["kneeling_context_diagnostic"] = {"report_only": True, "available": False}
        profile["sam3d_projected_pose"] = projected
        return

    balance_support = _balance_support(profile)
    low_stance_score = max(
        float(posture_scores.get("crouching") or 0.0),
        float(posture_scores.get("squatting") or 0.0),
    )
    low_stance_balance = float(geometry.get("low_stance_balance_score") or 0.0)
    seated_displacement = float(geometry.get("seated_displacement_score") or 0.0)
    low_stance_conflict = _clamp(
        _ramp(low_stance_score, 0.45, 0.80) * (1.0 - low_stance_balance)
    )
    projected["support_balance_diagnostic"] = {
        "report_only": True,
        "geometry": geometry,
        "low_stance_balance_score": _round(low_stance_balance, 4),
        "low_stance_balance_score_percent": int(round(100.0 * low_stance_balance)),
        "seated_displacement_score": _round(seated_displacement, 4),
        "seated_displacement_score_percent": int(round(100.0 * seated_displacement)),
        "low_stance_balance_conflict": _round(low_stance_conflict, 4),
        "low_stance_balance_conflict_percent": int(round(100.0 * low_stance_conflict)),
        "review_match": bool(low_stance_conflict >= BALANCE_CONFLICT_REVIEW_THRESHOLD),
        "crop_support": _round(balance_support, 4),
        "crop_support_percent": int(round(100.0 * balance_support)),
        "support_class": v06.v05._projected_support_class(balance_support),
        "interpretation": (
            "Low crouch/squat balance means the reconstructed pelvis/torso is not plausibly "
            "supported over the reconstructed feet. This is a diagnostic, not a contact claim."
        ),
    }

    recline_support = _recline_support(profile)
    recline_score = float(geometry.get("recline_geometry_score") or 0.0)
    current_recline = float(posture_scores.get("reclined") or 0.0)
    recline_gap = max(0.0, recline_score - current_recline)
    projected["recline_diagnostic"] = {
        "report_only": True,
        "score": _round(recline_score, 4),
        "score_percent": int(round(100.0 * recline_score)),
        "current_posture_reclined_score": _round(current_recline, 4),
        "current_posture_reclined_score_percent": int(round(100.0 * current_recline)),
        "diagnostic_gap": _round(recline_gap, 4),
        "diagnostic_gap_percent": int(round(100.0 * recline_gap)),
        "torso_axis_from_vertical_deg": geometry.get("torso_axis_from_vertical_deg"),
        "body_flatness_ratio": geometry.get("body_flatness_ratio"),
        "lying_plane_score": geometry.get("lying_plane_score"),
        "review_match": bool(recline_score >= RECLINE_REVIEW_THRESHOLD and recline_gap >= 0.15),
        "crop_support": _round(recline_support, 4),
        "crop_support_percent": int(round(100.0 * recline_support)),
        "support_class": v06.v05._projected_support_class(recline_support),
        "interpretation": (
            "Recline diagnostic emphasizes shoulder/hip horizontalization and whole-body "
            "flatness. It does not change the v0.6 posture score."
        ),
    }

    kneel = projected.get("kneeling_candidate") or {}
    kneel_score = float(kneel.get("score") or 0.0)
    conflicting_context = max(seated_displacement, recline_score)
    kneel_context_conflict = _clamp(kneel_score * conflicting_context)
    projected["kneeling_context_diagnostic"] = {
        "report_only": True,
        "candidate_score": _round(kneel_score, 4),
        "candidate_score_percent": int(round(100.0 * kneel_score)),
        "seated_displacement_score_percent": int(round(100.0 * seated_displacement)),
        "recline_geometry_score_percent": int(round(100.0 * recline_score)),
        "context_conflict": _round(kneel_context_conflict, 4),
        "context_conflict_percent": int(round(100.0 * kneel_context_conflict)),
        "review_match": bool(kneel_context_conflict >= KNEEL_CONTEXT_CONFLICT_THRESHOLD),
        "interpretation": (
            "A high kneeling candidate accompanied by strong seated/reclined support geometry "
            "is flagged for review; kneeling remains report-only."
        ),
    }

    profile["sam3d_projected_pose"] = projected


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v06.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.7"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]
    _report_diagnostics(profile, keypoints)

    policy = profile.get("policy") or {}
    policy.update({
        "v07_posture_scores_are_frozen_from_v06": True,
        "balance_support_diagnostic_is_report_only": True,
        "recline_diagnostic_is_report_only": True,
        "kneeling_context_diagnostic_is_report_only": True,
        "support_base_is_reconstructed_geometry_not_observed_contact": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-07",
        description=(
            "Build v0.7 report-only balance/support and recline diagnostics on top of "
            "the frozen v0.6 posture classifier. No new model inference is performed."
        ),
    )
    parser.add_argument("sam3d_dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")

    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else sam3d_dir.parent / "dwpose"
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else sam3d_dir.parent / "images"
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.7")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    helpers = v06.v05.v04.v03
    rows: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        dwpose = helpers._read_json(dwpose_path)
        width = int(dwpose.get("image_width") or 0)
        height = int(dwpose.get("image_height") or 0)
        if width <= 0 or height <= 0:
            image_matches = [p for p in images_dir.rglob(f"{key}.*") if p.is_file()] if images_dir.is_dir() else []
            if not image_matches:
                raise SystemExit(f"Cannot determine image size for {key}")
            with Image.open(image_matches[0]) as im:
                width, height = im.size

        profile = build_profile(arrays, dwpose or None, width, height)
        record = {
            "image_key": key,
            "sam3d_arrays": str(path),
            "dwpose": str(dwpose_path) if dwpose_path.is_file() else None,
            "image_width": width,
            "image_height": height,
            "profile": profile,
        }
        out_path = output / f"{key}.sam3d_relational_pose.json"
        helpers._write_json(out_path, record)
        rows.append(record)

        projected = profile["sam3d_projected_pose"]
        scores = projected.get("posture_score_percent") or {}
        state = projected.get("support_state") or {}
        balance = projected.get("support_balance_diagnostic") or {}
        recline = projected.get("recline_diagnostic") or {}
        kneel_context = projected.get("kneeling_context_diagnostic") or {}
        state_label = "-"
        if state.get("geometry_match"):
            state_label = (
                f"single_leg:{state.get('candidate_support_side')} "
                f"free:{state.get('candidate_free_leg')}@{state.get('crop_support_percent', 0)}%"
            )
        balance_label = (
            f"low:{balance.get('low_stance_balance_score_percent', 0)} "
            f"seat:{balance.get('seated_displacement_score_percent', 0)} "
            f"conflict:{balance.get('low_stance_balance_conflict_percent', 0)}"
            + ("*" if balance.get("review_match") else "")
        )
        recline_label = (
            f"{recline.get('score_percent', 0)}% gap:{recline.get('diagnostic_gap_percent', 0)}"
            + ("*" if recline.get("review_match") else "")
        )
        kneel_label = (
            f"{(projected.get('kneeling_candidate') or {}).get('score_percent', 0)}% "
            f"ctx:{kneel_context.get('context_conflict_percent', 0)}"
            + ("*" if kneel_context.get("review_match") else "")
        )
        print(
            f"{key}: projected={projected['pose']} best={projected['best_candidate_pose']} "
            f"scores=stand:{scores.get('standing', 0)} crouch:{scores.get('crouching', 0)} "
            f"squat:{scores.get('squatting', 0)} sit:{scores.get('sitting', 0)} recl:{scores.get('reclined', 0)} "
            f"crop={projected['crop_support_percent']}%[{projected.get('support_class')}] "
            f"support={state_label} balance={balance_label} recline_diag={recline_label} kneel={kneel_label}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.7",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    index_path = output / "sam3d_relational_pose.index.json"
    helpers._write_json(index_path, index)
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
