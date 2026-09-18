from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import mesh_arm_occupancy_shadow_v01 as base

SCHEMA_VERSION = "mesh-arm-occupancy-shadow-0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "mesh-arm-occupancy-shadow-v0.2"

_BASE_EVALUATE = base.evaluate

HAND_SCORE_THRESHOLD = 0.30
HAND_STRONG_POINT_COUNT = 8
HAND_MODERATE_POINT_COUNT = 4

DISTAL_LOWER_Y_MIN = 0.55
DISTAL_SIDE_X_MAX = 0.43
DISTAL_SIDE_X_MIN = 0.57
DISTAL_EDGE_DISTANCE_MAX = 0.10
ARM_SPAN_DIAG_MIN = 0.12
ARM_OUTWARD_DELTA_MIN = 0.04
ARM_STRAIGHTNESS_MIN = 0.72

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _to_point(value: Any, width: int, height: int) -> tuple[float, float] | None:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return None
    x, y = float(arr[0]), float(arr[1])
    if x < 0 or y < 0:
        return None
    if -0.05 <= x <= 1.25 and -0.05 <= y <= 1.25:
        x *= width
        y *= height
    if not (0 <= x < width and 0 <= y < height):
        return None
    return x, y


def _hand_support(
    dwpose: dict[str, Any],
    side: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    raw = dwpose.get("raw_pose") if isinstance(dwpose.get("raw_pose"), dict) else {}
    hands = np.asarray(raw.get("hands") or [], dtype=np.float64)
    scores = np.asarray(raw.get("hands_scores") or [], dtype=np.float64)
    body_scores = np.asarray(raw.get("body_scores") or [], dtype=np.float64)

    if hands.ndim != 3 or hands.shape[1] < 21:
        return {
            "grade": "unavailable",
            "confident_point_count": 0,
            "points": [],
            "reason": "raw_easy_dwpose_hand_array_unavailable",
        }

    person_count = int(body_scores.shape[0]) if body_scores.ndim >= 2 else max(1, hands.shape[0] // 2)
    target_idx = int(((dwpose.get("derived") or {}).get("target_person_index") or 0))
    target_idx = max(0, min(target_idx, max(0, person_count - 1)))

    # easy-dwpose _format_pose stacks all left hands first, then all right hands.
    hand_idx = target_idx if side == "left" else person_count + target_idx
    if hand_idx >= len(hands):
        return {
            "grade": "unavailable",
            "confident_point_count": 0,
            "points": [],
            "reason": "target_hand_index_out_of_range",
        }

    hand = hands[hand_idx, :21]
    score_row = scores[hand_idx, :21] if scores.ndim == 2 and hand_idx < len(scores) else np.ones(21)
    points: list[tuple[float, float] | None] = []
    confident: list[tuple[float, float]] = []
    confident_indices: list[int] = []
    for idx in range(21):
        score = float(score_row[idx]) if idx < len(score_row) and np.isfinite(score_row[idx]) else 0.0
        p = _to_point(hand[idx], width, height) if score > HAND_SCORE_THRESHOLD else None
        points.append(p)
        if p is not None:
            confident.append(p)
            confident_indices.append(idx)

    count = len(confident)
    if count >= HAND_STRONG_POINT_COUNT:
        grade = "strong"
    elif count >= HAND_MODERATE_POINT_COUNT:
        grade = "moderate"
    elif count:
        grade = "weak"
    else:
        grade = "none"

    bbox = None
    if confident:
        xs = [p[0] for p in confident]
        ys = [p[1] for p in confident]
        bbox = {
            "x0": round(min(xs) / width, 4),
            "y0": round(min(ys) / height, 4),
            "x1": round(max(xs) / width, 4),
            "y1": round(max(ys) / height, 4),
            "center_x": round(((min(xs) + max(xs)) / 2) / width, 4),
            "center_y": round(((min(ys) + max(ys)) / 2) / height, 4),
        }

    return {
        "grade": grade,
        "confident_point_count": count,
        "confident_point_indices": confident_indices,
        "points": [list(p) if p is not None else None for p in points],
        "bbox": bbox,
        "reason": "easy_dwpose_target_bound_hand_keypoints",
        "authority": "direct_dwpose_hand_landmark_observation",
        "hand_array_order": "left_block_then_right_block_per_easy_dwpose_format_pose",
    }


def _frame_region(point: tuple[float, float] | None, width: int, height: int) -> str | None:
    if point is None:
        return None
    x = point[0] / width
    y = point[1] / height
    horizontal = "left" if x < DISTAL_SIDE_X_MAX else ("right" if x > DISTAL_SIDE_X_MIN else "center")
    if y >= DISTAL_LOWER_Y_MIN:
        if horizontal == "left":
            return "lower_frame_left"
        if horizontal == "right":
            return "lower_frame_right"
        return "lower_center"
    if horizontal == "left":
        return "frame_left"
    if horizontal == "right":
        return "frame_right"
    return "center"


def _edge_distance(point: tuple[float, float] | None, width: int, height: int) -> float | None:
    if point is None:
        return None
    x = point[0] / width
    y = point[1] / height
    return max(0.0, min(x, 1.0 - x, y, 1.0 - y))


def _center_radius(point: tuple[float, float] | None, width: int, height: int) -> float | None:
    if point is None:
        return None
    x = point[0] / width - 0.5
    y = point[1] / height - 0.5
    return math.hypot(x, y)


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))


def _sam_wrist_projection(
    side: str,
    sam3d_projected_keypoints: np.ndarray | None,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if sam3d_projected_keypoints is None:
        return None
    idx = base.MHR[f"{side}_wrist"]
    if idx >= len(sam3d_projected_keypoints):
        return None
    p = np.asarray(sam3d_projected_keypoints[idx], dtype=np.float64)
    if p.size < 2 or not np.isfinite(p[:2]).all():
        return None
    x, y = float(p[0]), float(p[1])
    return {
        "x_fraction": round(x / width, 4),
        "y_fraction": round(y / height, 4),
        "inside_frame": bool(0 <= x < width and 0 <= y < height),
        "point": [x, y],
    }


def _distal_arm_path_proxy(
    side: str,
    *,
    points: dict[str, tuple[float, float] | None],
    hand_support: dict[str, Any],
    sam3d_projected_keypoints: np.ndarray | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    shoulder = points.get(f"{side}_shoulder")
    elbow = points.get(f"{side}_elbow")
    wrist = points.get(f"{side}_wrist")

    if shoulder is None or elbow is None:
        return {
            "grade": "none",
            "reason": "requires_observed_shoulder_and_elbow",
            "anatomical_side_internal": side,
        }

    wrist_observed = wrist is not None
    distal = wrist if wrist_observed else elbow
    distal_kind = "wrist" if wrist_observed else "elbow"

    diag = math.hypot(width, height)
    shoulder_distal = _distance(shoulder, distal)
    span_diag = shoulder_distal / diag if shoulder_distal is not None and diag > 0 else None
    distal_region = _frame_region(distal, width, height)
    distal_edge = _edge_distance(distal, width, height)
    shoulder_radius = _center_radius(shoulder, width, height)
    distal_radius = _center_radius(distal, width, height)
    outward_delta = (
        distal_radius - shoulder_radius
        if distal_radius is not None and shoulder_radius is not None
        else None
    )

    straightness = None
    path_diag = None
    if wrist_observed:
        se = _distance(shoulder, elbow)
        ew = _distance(elbow, wrist)
        sw = _distance(shoulder, wrist)
        path = (se or 0.0) + (ew or 0.0)
        if path > 1e-8 and sw is not None:
            straightness = sw / path
            path_diag = path / diag if diag > 0 else None
    else:
        path_diag = span_diag

    sam_wrist = _sam_wrist_projection(side, sam3d_projected_keypoints, width, height)
    sam_continuation = None
    if not wrist_observed and sam_wrist and sam_wrist.get("point") is not None:
        sx, sy = sam_wrist["point"]
        v1 = np.asarray([elbow[0] - shoulder[0], elbow[1] - shoulder[1]], dtype=np.float64)
        v2 = np.asarray([sx - elbow[0], sy - elbow[1]], dtype=np.float64)
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom > 1e-8:
            sam_continuation = float(np.dot(v1, v2) / denom)

    lower_side = distal_region in {"lower_frame_left", "lower_frame_right"}
    near_edge = bool(distal_edge is not None and distal_edge <= DISTAL_EDGE_DISTANCE_MAX)
    long_span = bool(span_diag is not None and span_diag >= ARM_SPAN_DIAG_MIN)
    outward = bool(outward_delta is not None and outward_delta >= ARM_OUTWARD_DELTA_MIN)
    straight = bool(straightness is None or straightness >= ARM_STRAIGHTNESS_MIN)
    sam_support = bool(
        not wrist_observed
        and sam_wrist
        and (
            sam_wrist.get("inside_frame") is False
            or (sam_continuation is not None and sam_continuation >= 0.35)
        )
    )

    score = sum([
        1 if lower_side else 0,
        1 if near_edge else 0,
        1 if long_span else 0,
        1 if outward else 0,
        1 if straight else 0,
        1 if sam_support else 0,
    ])

    hand_grade = str(hand_support.get("grade") or "none")
    hand_conflict = (not wrist_observed) and hand_grade in {"strong", "moderate"}
    if hand_conflict:
        score -= 2

    if lower_side and near_edge and long_span and score >= 4:
        grade = "strong"
    elif lower_side and long_span and score >= 3:
        grade = "moderate"
    elif score >= 2:
        grade = "weak"
    else:
        grade = "none"

    return {
        "grade": grade,
        "score": score,
        "anatomical_side_internal": side,
        "distal_observed_joint": distal_kind,
        "distal_frame_region": distal_region,
        "distal_edge_distance_fraction": round(distal_edge, 4) if distal_edge is not None else None,
        "shoulder_to_distal_length_image_diagonal": round(span_diag, 4) if span_diag is not None else None,
        "path_length_image_diagonal": round(path_diag, 4) if path_diag is not None else None,
        "outward_center_radius_delta": round(outward_delta, 4) if outward_delta is not None else None,
        "complete_chain_straightness": round(straightness, 4) if straightness is not None else None,
        "wrist_observed": wrist_observed,
        "same_side_hand_support": hand_support,
        "hand_presence_conflicts_with_missing_wrist_crop_exit": hand_conflict,
        "sam3d_projected_wrist": sam_wrist,
        "sam3d_wrist_continuation_cosine": round(sam_continuation, 4)
        if sam_continuation is not None
        else None,
        "components": {
            "distal_in_lower_frame_side": lower_side,
            "distal_near_frame_edge": near_edge,
            "arm_span_long_enough": long_span,
            "distal_moves_outward_from_image_center": outward,
            "observed_chain_sufficiently_straight": straight,
            "sam3d_supports_missing_wrist_continuation": sam_support,
        },
        "reason": "observed_arm_path_toward_lower_frame_boundary",
        "note": (
            "This is a composition diagnostic, not proof that the arm holds the camera. "
            "Observed DWPose geometry owns the path; SAM3D wrist continuation is diagnostic only."
        ),
    }


def _combined_arm_grade(mesh_grade: str, path_grade: str) -> str:
    rank = {"none": 0, "insufficient": 0, "weak": 1, "moderate": 2, "strong": 3}
    m = rank.get(str(mesh_grade), 0)
    p = rank.get(str(path_grade), 0)
    if m >= 2 and p >= 2 and max(m, p) >= 3:
        return "strong"
    if m >= 2 and p >= 2:
        return "moderate"
    if (m >= 3 and p >= 1) or (p >= 3 and m >= 1):
        return "moderate"
    if max(m, p) >= 2:
        return "weak"
    if max(m, p) == 1:
        return "weak"
    return "insufficient"


def _draw_hand_debug(
    skeleton_overlay_path: Path | None,
    *,
    left_hand: dict[str, Any],
    right_hand: dict[str, Any],
) -> None:
    if skeleton_overlay_path is None or not skeleton_overlay_path.is_file():
        return
    image = Image.open(skeleton_overlay_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for side, record, color in (
        ("left", left_hand, (50, 255, 80)),
        ("right", right_hand, (255, 210, 40)),
    ):
        points_raw = record.get("points") if isinstance(record.get("points"), list) else []
        points: list[tuple[float, float] | None] = []
        for value in points_raw[:21]:
            if isinstance(value, list) and len(value) >= 2:
                points.append((float(value[0]), float(value[1])))
            else:
                points.append(None)
        while len(points) < 21:
            points.append(None)

        for a, b in HAND_EDGES:
            pa, pb = points[a], points[b]
            if pa is not None and pb is not None:
                draw.line([pa, pb], fill=color, width=2)
        radius = max(2, round(min(image.size) / 260))
        for p in points:
            if p is None:
                continue
            x, y = p
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
                outline=(20, 20, 20),
            )
        bbox = record.get("bbox") if isinstance(record.get("bbox"), dict) else None
        if bbox:
            x = float(bbox["x0"]) * image.width
            y = float(bbox["y0"]) * image.height
            draw.text((x, max(0, y - 14)), f"{side[0].upper()}H:{record.get('grade')}", fill=color)

    image.save(skeleton_overlay_path, quality=92)


def evaluate(
    policy: dict[str, Any],
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any],
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    image_path: Path | None = None,
    debug_overlay_path: Path | None = None,
    debug_skeleton_overlay_path: Path | None = None,
) -> dict[str, Any]:
    out = _BASE_EVALUATE(
        policy,
        arrays,
        dwpose,
        vertices,
        faces,
        image_path=image_path,
        debug_overlay_path=debug_overlay_path,
        debug_skeleton_overlay_path=debug_skeleton_overlay_path,
    )
    out["schema_version"] = SCHEMA_VERSION
    if out.get("status") != "ok":
        return out

    width, height = (int(v) for v in out["image_size"])
    focal = ((out.get("focal_length_recovery") or {}).get("median_px"))
    keypoints3d = np.asarray(arrays.get("pred_keypoints_3d"), dtype=np.float64)
    cam_t = np.asarray(arrays.get("pred_cam_t"), dtype=np.float64)
    sam_projected = None
    if isinstance(focal, (int, float)):
        sam_projected, _ = base._project(keypoints3d, cam_t, float(focal), width, height)

    points = base._dwpose_points_pixels(dwpose, width, height)
    left_hand = _hand_support(dwpose, "left", width, height)
    right_hand = _hand_support(dwpose, "right", width, height)

    _draw_hand_debug(
        debug_skeleton_overlay_path,
        left_hand=left_hand,
        right_hand=right_hand,
    )

    for side, hand in (("left", left_hand), ("right", right_hand)):
        arm = out["arms"][side]
        path = _distal_arm_path_proxy(
            side,
            points=points,
            hand_support=hand,
            sam3d_projected_keypoints=sam_projected,
            width=width,
            height=height,
        )
        combined = _combined_arm_grade(
            str(arm.get("evidence_grade") or "insufficient"),
            str(path.get("grade") or "none"),
        )
        arm["dwpose_hand_support"] = hand
        arm["distal_arm_path_proxy"] = path
        arm["combined_foreground_arm_grade"] = combined
        arm["combined_grade_basis"] = {
            "mesh_occupancy_grade": arm.get("evidence_grade"),
            "observed_arm_path_grade": path.get("grade"),
        }

    out["mesh"]["debug_overlay_legend"].update({
        "left_hand_observation": "green",
        "right_hand_observation": "yellow",
    })
    out["invariants"].update({
        "dwpose_hand_landmarks_are_direct_observation": True,
        "missing_wrist_is_supportive_not_required_for_foreground_arm": True,
        "complete_arm_chain_can_support_foreground_arm_when_distal_joint_reaches_frame_edge": True,
        "hand_presence_can_veto_missing_wrist_crop_exit_interpretation": True,
        "combined_foreground_arm_grade_requires_mesh_or_observed_path_support": True,
    })
    return out


def main() -> int:
    original_argv = list(sys.argv)
    old_evaluate = base.evaluate
    old_schema = base.SCHEMA_VERSION
    old_output = base.DEFAULT_OUTPUT_SUBDIR
    try:
        base.evaluate = evaluate
        base.SCHEMA_VERSION = SCHEMA_VERSION
        base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
        return base.main()
    finally:
        base.evaluate = old_evaluate
        base.SCHEMA_VERSION = old_schema
        base.DEFAULT_OUTPUT_SUBDIR = old_output
        sys.argv[:] = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
