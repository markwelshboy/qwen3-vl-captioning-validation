from __future__ import annotations

import math
from typing import Any

import numpy as np

from .dwpose_compat import _candidate_array


BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
IDX = {name: i for i, name in enumerate(BODY18)}


def _visible(point: np.ndarray) -> bool:
    return bool(
        point.shape[0] >= 2
        and np.isfinite(point[0])
        and np.isfinite(point[1])
        and point[0] >= 0.0
        and point[1] >= 0.0
    )


def _xy(point: np.ndarray | None) -> list[float] | None:
    if point is None or not _visible(point):
        return None
    return [round(float(point[0]), 5), round(float(point[1]), 5)]


def _clipped_bbox(person: np.ndarray) -> dict[str, float] | None:
    pts = np.asarray([p[:2] for p in person if _visible(p)], dtype=np.float64)
    if not len(pts):
        return None
    clipped = np.clip(pts, 0.0, 1.0)
    x0, y0 = clipped.min(axis=0)
    x1, y1 = clipped.max(axis=0)
    width = max(0.0, float(x1 - x0))
    height = max(0.0, float(y1 - y0))
    return {
        "x0": round(float(x0), 5),
        "y0": round(float(y0), 5),
        "x1": round(float(x1), 5),
        "y1": round(float(y1), 5),
        "width_fraction": round(width, 5),
        "height_fraction": round(height, 5),
        "area_fraction": round(width * height, 5),
    }


def _hand_candidates(raw_pose: dict[str, Any], target: np.ndarray | None) -> list[dict[str, Any]]:
    hands_raw = raw_pose.get("hands")
    scores_raw = raw_pose.get("hands_scores")
    if hands_raw is None or scores_raw is None:
        return []

    hands = np.asarray(hands_raw, dtype=np.float64)
    scores = np.asarray(scores_raw, dtype=np.float64)
    if hands.ndim != 3 or hands.shape[-1] < 2 or scores.ndim != 2:
        return []

    wrists: dict[str, np.ndarray] = {}
    if target is not None and len(target) >= 18:
        for side in ("left", "right"):
            p = target[IDX[f"{side}_wrist"]]
            if _visible(p):
                wrists[side] = p[:2]

    out: list[dict[str, Any]] = []
    count = min(len(hands), len(scores))
    for i in range(count):
        n = min(len(hands[i]), len(scores[i]))
        if n == 0:
            continue
        conf = scores[i, :n]
        pts = hands[i, :n, :2]
        mask = np.isfinite(conf) & (conf >= 0.30) & np.isfinite(pts).all(axis=1)
        if not mask.any():
            continue

        visible_pts = pts[mask]
        visible_conf = conf[mask]
        centroid = visible_pts.mean(axis=0)
        nearest_side = None
        nearest_distance = None
        for side, wrist in wrists.items():
            distance = float(np.linalg.norm(centroid - wrist))
            if nearest_distance is None or distance < nearest_distance:
                nearest_side = side
                nearest_distance = distance

        out.append(
            {
                "candidate_index": i,
                "visible_keypoints": int(mask.sum()),
                "mean_confidence": round(float(visible_conf.mean()), 4),
                "max_confidence": round(float(visible_conf.max()), 4),
                "centroid_xy": [round(float(centroid[0]), 5), round(float(centroid[1]), 5)],
                "nearest_visible_target_wrist": nearest_side,
                "distance_to_nearest_visible_target_wrist": round(nearest_distance, 5)
                if nearest_distance is not None
                else None,
                "supported_by_nearby_visible_target_wrist": bool(
                    nearest_distance is not None and nearest_distance <= 0.10
                ),
            }
        )
    return out


def build_pose_evidence(dwpose_record: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, caption-safe evidence block from a cached DWPose record.

    This intentionally keeps DWPose in the role of secondary geometric evidence.
    It does not promote front/back, depth, or anatomical laterality predictions to
    ground truth.
    """
    raw_pose = dwpose_record.get("raw_pose") or {}
    derived = dwpose_record.get("derived") or {}
    candidate = _candidate_array(raw_pose)

    target_index = derived.get("target_person_index")
    target: np.ndarray | None = None
    if isinstance(target_index, int) and 0 <= target_index < len(candidate):
        target = candidate[target_index]

    target_derived = derived.get("target") or {}
    geometry = target_derived.get("geometry") or {}
    connectivity = target_derived.get("connectivity") or {}

    wrists = {"left": None, "right": None}
    if target is not None:
        wrists = {
            "left": _xy(target[IDX["left_wrist"]]),
            "right": _xy(target[IDX["right_wrist"]]),
        }

    people = derived.get("people") or []
    target_area = float((target_derived.get("keypoint_bbox") or {}).get("area_fraction") or 0.0)
    significant_people = 0
    small_secondary_people = 0
    for person in people:
        area = float((person.get("keypoint_bbox") or {}).get("area_fraction") or 0.0)
        if person.get("person_index") == target_index:
            continue
        significant_threshold = max(0.02, target_area * 0.20)
        if area >= significant_threshold:
            significant_people += 1
        elif area > 0:
            small_secondary_people += 1

    return {
        "schema_version": "dwpose-caption-evidence-1.0",
        "person_evidence": {
            "detected_person_count": int(derived.get("person_count") or 0),
            "significant_secondary_people": significant_people,
            "small_secondary_people": small_secondary_people,
            "target_person_index": target_index,
        },
        "target_2d_geometry": {
            "pose_extent_hint": target_derived.get("pose_extent_hint"),
            "raw_keypoint_bbox": target_derived.get("keypoint_bbox"),
            "clipped_in_frame_keypoint_bbox": _clipped_bbox(target) if target is not None else None,
            "shoulder_line_angle_from_horizontal_deg": geometry.get("shoulder_line_angle_from_horizontal_deg"),
            "hip_line_angle_from_horizontal_deg": geometry.get("hip_line_angle_from_horizontal_deg"),
            "torso_axis_angle_from_vertical_deg": geometry.get("torso_axis_angle_from_vertical_deg"),
            "predicted_wrists_xy": wrists,
            "connectivity": connectivity,
        },
        "hand_candidates": _hand_candidates(raw_pose, target),
        "interpretation_rules": [
            "Use DWPose as strong secondary evidence for projected 2D geometry, visible-joint extent, and limb-chain consistency.",
            "DWPose does not independently establish front-vs-back torso orientation or metric depth.",
            "DWPose anatomical left/right labels are predictions and can be wrong in ambiguous or rear-facing poses.",
            "A hand candidate without a nearby visible target wrist is evidence against confidently assigning that hand to a visible target arm, not proof that it belongs to another person.",
            "Do not mention numeric angles, DWPose, keypoints, or detector confidence in the final caption; translate only useful supported geometry into natural language.",
        ],
    }
