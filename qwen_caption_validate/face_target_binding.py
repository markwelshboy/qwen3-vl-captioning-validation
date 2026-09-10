from __future__ import annotations

import math
from typing import Any

import numpy as np

from .dwpose_compat import target_points_from_profile_record

BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
IDX = {name: i for i, name in enumerate(BODY18)}
FACE_NAMES = ("nose", "right_eye", "left_eye", "right_ear", "left_ear")


def _usable(point: np.ndarray) -> bool:
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    return bool(p.size >= 2 and np.isfinite(p[:2]).all() and p[0] >= 0 and p[1] >= 0)


def _point(points: np.ndarray, name: str) -> np.ndarray | None:
    if len(points) <= IDX[name]:
        return None
    p = np.asarray(points[IDX[name], :2], dtype=np.float64)
    return p if _usable(p) else None


def target_body_geometry(record: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    """Build a lightweight target-person geometry descriptor from cached DWPose.

    The descriptor intentionally uses only the already-selected BODY18 person.  It is
    used to bind a face detector result to that person; it does not infer semantics.
    """
    points = target_points_from_profile_record(record, width, height)
    if len(points) < 18:
        return None

    valid = np.array([_usable(p) for p in points], dtype=bool)
    if int(valid.sum()) < 2:
        return None

    observed = np.asarray(points[valid, :2], dtype=np.float64)
    lo = observed.min(axis=0)
    hi = observed.max(axis=0)
    body_bbox = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
    body_diag = float(np.linalg.norm(hi - lo))

    face_points = [p for name in FACE_NAMES if (p := _point(points, name)) is not None]
    head_center = np.median(np.stack(face_points), axis=0) if face_points else None

    rs = _point(points, "right_shoulder")
    ls = _point(points, "left_shoulder")
    neck = _point(points, "neck")
    shoulder_mid = None
    shoulder_span = None
    if rs is not None and ls is not None:
        shoulder_mid = (rs + ls) * 0.5
        shoulder_span = float(np.linalg.norm(rs - ls))

    # If facial DWPose points are absent, keep a weak neck/shoulder fallback.  BODY18's
    # neck is often close to the shoulder midpoint, so use body scale rather than
    # pretending we can reconstruct a precise face center from it.
    if head_center is None and neck is not None:
        head_center = neck.copy()

    scale_candidates = [v for v in (shoulder_span, body_diag * 0.22) if v is not None and math.isfinite(v) and v > 1.0]
    body_scale = float(max(scale_candidates)) if scale_candidates else max(1.0, body_diag * 0.22)

    return {
        "points": np.asarray(points[:, :2], dtype=np.float64),
        "valid_joint_count": int(valid.sum()),
        "body_bbox_xyxy": body_bbox,
        "body_diagonal_px": body_diag,
        "body_scale_px": body_scale,
        "head_center_xy": [float(v) for v in head_center] if head_center is not None else None,
        "neck_xy": [float(v) for v in neck] if neck is not None else None,
        "shoulder_mid_xy": [float(v) for v in shoulder_mid] if shoulder_mid is not None else None,
        "shoulder_span_px": shoulder_span,
    }


def _face_box(face: Any) -> np.ndarray:
    box = np.asarray(face.bbox, dtype=np.float64).reshape(-1)
    if box.size < 4:
        raise ValueError("face bbox must contain x1,y1,x2,y2")
    return box[:4]


def _face_center(face: Any) -> np.ndarray:
    x0, y0, x1, y1 = _face_box(face)
    return np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)


def _expanded_bbox_contains(center: np.ndarray, bbox: list[float], margin_px: float) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return bool(
        x0 - margin_px <= center[0] <= x1 + margin_px
        and y0 - margin_px <= center[1] <= y1 + margin_px
    )


def select_face_for_target(
    faces: list[Any],
    target: dict[str, Any] | None,
) -> tuple[Any, str, list[dict[str, Any]]] | None:
    """Select the detector face that most plausibly belongs to the target DWPose body.

    Head-center distance remains important, but is no longer the only signal.  We also
    require spatial membership in the target body's expanded extent, measure distance
    to the neck, and penalize faces that are implausibly tiny relative to the target
    shoulder/body scale.  The latter is specifically useful when a small background
    face overlaps the foreground subject, as in imageblind-01_00049.

    Candidate diagnostics are returned so selection errors remain inspectable.
    """
    if not faces:
        return None

    if not target:
        best = max(faces, key=lambda f: float(getattr(f, "confidence", 0.0)))
        diagnostics = []
        for i, face in enumerate(faces):
            box = _face_box(face)
            diagnostics.append({
                "index": i,
                "bbox_xyxy": [float(v) for v in box],
                "detector_score": float(getattr(face, "confidence", 0.0)),
                "cost": None,
            })
        return best, "highest_retinaface_score_no_dwpose_body", diagnostics

    scale = max(1.0, float(target.get("body_scale_px") or 1.0))
    shoulder_span = target.get("shoulder_span_px")
    if shoulder_span is not None and (not math.isfinite(float(shoulder_span)) or float(shoulder_span) <= 1.0):
        shoulder_span = None

    head = target.get("head_center_xy")
    head_xy = np.asarray(head, dtype=np.float64) if head is not None else None
    neck = target.get("neck_xy")
    neck_xy = np.asarray(neck, dtype=np.float64) if neck is not None else None
    body_bbox = target.get("body_bbox_xyxy")
    margin = max(12.0, 0.55 * scale)

    diagnostics: list[dict[str, Any]] = []
    costs: list[tuple[float, float, int]] = []
    for i, face in enumerate(faces):
        box = _face_box(face)
        center = _face_center(face)
        fw = max(1.0, float(box[2] - box[0]))
        fh = max(1.0, float(box[3] - box[1]))
        area = fw * fh

        head_norm = float(np.linalg.norm(center - head_xy) / scale) if head_xy is not None else 0.0
        neck_norm = float(np.linalg.norm(center - neck_xy) / scale) if neck_xy is not None else 0.0
        inside_body = _expanded_bbox_contains(center, body_bbox, margin) if body_bbox is not None else True

        # Shoulder width is a useful target-person scale reference.  A face much less
        # than ~1/4 of the selected body's shoulder span is more likely to be a
        # background face.  This is a soft penalty, not a universal hard rejection.
        size_ratio = float(fw / float(shoulder_span)) if shoulder_span is not None else float(fw / scale)
        tiny_penalty = max(0.0, (0.27 - size_ratio) / 0.27)

        # Head proximity dominates; neck/body membership and target-scale plausibility
        # resolve ambiguous multi-face cases.  A tiny area epsilon makes a larger face
        # win only when geometric costs are otherwise essentially equal.
        cost = head_norm + 0.18 * neck_norm + 1.25 * tiny_penalty + (0.0 if inside_body else 2.0)
        area_tiebreak = -math.log(max(area, 1.0)) * 1e-4
        ranked_cost = cost + area_tiebreak
        costs.append((ranked_cost, -area, i))

        diagnostics.append({
            "index": i,
            "bbox_xyxy": [float(v) for v in box],
            "center_xy": [float(v) for v in center],
            "detector_score": float(getattr(face, "confidence", 0.0)),
            "inside_expanded_target_body": inside_body,
            "head_distance_norm": head_norm if head_xy is not None else None,
            "neck_distance_norm": neck_norm if neck_xy is not None else None,
            "face_width_to_shoulder_or_scale": size_ratio,
            "tiny_face_penalty": tiny_penalty,
            "cost": cost,
        })

    _, _, best_index = min(costs)
    return faces[best_index], "dwpose_body_membership_head_scale", diagnostics
