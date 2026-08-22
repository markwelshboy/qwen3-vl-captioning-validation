from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
BODY_IDX = {name: i for i, name in enumerate(BODY18)}
MHR70 = {
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10, "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14, "right_wrist": 41, "left_wrist": 62,
}
SIDE_RE = re.compile(r"(?<![A-Za-z0-9])(left|right)(?![A-Za-z0-9])", re.I)
ARM_RE = re.compile(r"\b(?:arm|forearm|elbow|hand|wrist|finger|fingers)\b", re.I)
LEG_RE = re.compile(r"\b(?:leg|thigh|calf|knee|ankle|foot|feet)\b", re.I)
DISTAL_RE = re.compile(r"\b(?:hand|wrist|finger|fingers|fingertip|fingertips)\b", re.I)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mirror_sensitive(analysis: dict[str, Any]) -> bool:
    framing = analysis.get("framing") or {}
    text = " ".join((str(framing.get("photographic_archetype") or ""), str(analysis.get("image_summary") or ""))).lower()
    return "mirror" in text or "reflection selfie" in text


def _target_points(dw: dict[str, Any]) -> dict[str, np.ndarray]:
    raw = np.asarray((dw.get("raw_pose") or {}).get("bodies") or [], dtype=float)
    scores = np.asarray((dw.get("raw_pose") or {}).get("body_scores") or [], dtype=float)
    target = (dw.get("derived") or {}).get("target_person_index")
    if raw.ndim != 2 or scores.ndim != 2 or not isinstance(target, int) or not (0 <= target < len(scores)):
        return {}
    mapping = scores[target]
    out: dict[str, np.ndarray] = {}
    for name, body_index in BODY_IDX.items():
        if body_index >= len(mapping):
            continue
        raw_index = int(mapping[body_index])
        if raw_index < 0 or raw_index >= len(raw):
            continue
        point = np.asarray(raw[raw_index], dtype=float).reshape(-1)[:2]
        if len(point) == 2 and np.isfinite(point).all() and (point >= 0).all():
            out[name] = point
    return out


def _load_sam2d(sam_path: Path, sam: dict[str, Any]) -> np.ndarray | None:
    candidates = [Path(str(sam.get("arrays_npz") or "")), sam_path.with_name(sam_path.name.replace(".sam3d.json", ".sam3d_arrays.npz"))]
    for path in candidates:
        try:
            if path.is_file():
                with np.load(path) as arrays:
                    value = np.asarray(arrays["pred_keypoints_2d"], dtype=float)
                if value.ndim == 2 and value.shape[0] >= 70 and value.shape[1] >= 2:
                    return value[:, :2]
        except Exception:
            continue
    return None


def _sam_vote(name: str, dw: dict[str, Any], points: dict[str, np.ndarray], sam2d: np.ndarray | None) -> dict[str, Any]:
    if sam2d is None or name not in points or name not in MHR70:
        return {"status": "unavailable"}
    side, family = name.split("_", 1)
    other = ("right" if side == "left" else "left") + "_" + family
    width, height = float(dw.get("image_width") or 0), float(dw.get("image_height") or 0)
    if width <= 0 or height <= 0 or other not in MHR70:
        return {"status": "unavailable"}
    scale = np.array([width, height], dtype=float)
    same = sam2d[MHR70[name]] / scale
    opposite = sam2d[MHR70[other]] / scale
    if not np.isfinite(same).all() or not np.isfinite(opposite).all():
        return {"status": "unavailable"}
    same_d = float(np.linalg.norm(points[name] - same))
    opposite_d = float(np.linalg.norm(points[name] - opposite))
    status = "unresolved"
    if same_d <= 0.20 and same_d <= opposite_d * 0.70:
        status = "agrees"
    elif opposite_d <= 0.12 and opposite_d <= same_d * 0.70:
        status = "conflicts"
    return {"status": status, "same_side_distance": round(same_d, 5), "opposite_side_distance": round(opposite_d, 5)}


def _connectivity(dw: dict[str, Any]) -> dict[str, Any]:
    return (((dw.get("derived") or {}).get("target") or {}).get("connectivity") or {})


def _hand_entities(dw: dict[str, Any], points: dict[str, np.ndarray], sam2d: np.ndarray | None) -> list[dict[str, Any]]:
    raw = dw.get("raw_pose") or {}
    hands, scores = raw.get("hands") or [], raw.get("hands_scores") or []
    conn = _connectivity(dw)
    entities: list[dict[str, Any]] = []
    for index in range(min(len(hands), len(scores))):
        if not scores[index] or float(scores[index][0]) < 0.30:
            continue
        root = np.asarray(hands[index][0], dtype=float).reshape(-1)[:2]
        distances = {side: float(np.linalg.norm(root - points[f"{side}_wrist"])) for side in ("left", "right") if f"{side}_wrist" in points}
        supported = sorted(((side, distance) for side, distance in distances.items() if distance <= 0.10), key=lambda item: item[1])
        if not supported:
            continue
        side: str | None = None
        reason = ""
        if len(supported) == 1:
            side, _ = supported[0]
            reason = "single_observed_target_wrist"
        else:
            (first, d1), (second, d2) = supported[:2]
            first_complete = bool((conn.get(f"{first}_arm") or {}).get("complete"))
            second_complete = bool((conn.get(f"{second}_arm") or {}).get("complete"))
            ratio, margin = d1 / (d2 + 1e-9), d2 - d1
            if first_complete != second_complete and (margin < 0.02 or ratio > 0.65):
                side = first if first_complete else second
                reason = "near_tied_wrists_prefer_complete_chain"
            elif margin >= 0.02 or ratio <= 0.65:
                side = first
                reason = "clear_hand_root_to_wrist_association"
            else:
                reason = "ambiguous_between_observed_wrists"
        vote = _sam_vote(f"{side}_wrist", dw, points, sam2d) if side else {"status": "unavailable"}
        if vote.get("status") == "conflicts":
            side = None
            reason += ";sam3d_disagrees_with_dwpose_wrist_label"
        complete = bool((conn.get(f"{side}_arm") or {}).get("complete")) if side else False
        authority = "unknown"
        if side and vote.get("status") == "agrees":
            authority = "dwpose_sam_correlated"
        elif side and complete:
            authority = "dwpose_complete_chain"
        elif side:
            authority = "dwpose_observed_wrist"
        entities.append({
            "candidate_index": index,
            "root_xy": [float(root[0]), float(root[1])],
            "root_confidence": float(scores[index][0]),
            "distances_to_observed_target_wrists": {key: round(value, 5) for key, value in distances.items()},
            "qualified_side": side,
            "resolution_reason": reason,
            "chain_complete": complete,
            "sam3d_vote": vote,
            "authority": authority,
        })
    clustered: list[dict[str, Any]] = []
    for entity in sorted(entities, key=lambda item: item["root_confidence"], reverse=True):
        root = np.asarray(entity["root_xy"])
        duplicate = next((item for item in clustered if np.linalg.norm(root - np.asarray(item["root_xy"])) <= 0.06), None)
        if duplicate is None:
            clustered.append(entity)
        elif duplicate.get("qualified_side") is None and entity.get("qualified_side") is not None:
            duplicate.update(entity)
    return clustered


