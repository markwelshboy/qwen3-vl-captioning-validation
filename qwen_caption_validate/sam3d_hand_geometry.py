from __future__ import annotations

from typing import Any

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# DWPose/easy-dwpose hand ordering follows the standard 21-point whole-body
# convention: wrist, then four points along each finger from base -> tip.
DW_HAND_CHAINS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

# Meta MHR70 names/indexing are tip-first for each finger.  Reorder each chain
# below into wrist/base -> ... -> tip so it matches the DWPose 21-point layout.
MHR_HANDS = {
    "right": {
        "wrist": 41,
        "thumb": [24, 23, 22, 21],
        "index": [28, 27, 26, 25],
        "middle": [32, 31, 30, 29],
        "ring": [36, 35, 34, 33],
        "pinky": [40, 39, 38, 37],
    },
    "left": {
        "wrist": 62,
        "thumb": [45, 44, 43, 42],
        "index": [49, 48, 47, 46],
        "middle": [53, 52, 51, 50],
        "ring": [57, 56, 55, 54],
        "pinky": [61, 60, 59, 58],
    },
}


def mhr_hand_order(side: str) -> list[int]:
    spec = MHR_HANDS[side]
    out = [int(spec["wrist"])]
    for finger in FINGERS:
        out.extend(int(v) for v in spec[finger])
    return out


def mhr_hand_edges(side: str) -> list[tuple[int, int]]:
    spec = MHR_HANDS[side]
    wrist = int(spec["wrist"])
    edges: list[tuple[int, int]] = []
    for finger in FINGERS:
        chain = [wrist] + [int(v) for v in spec[finger]]
        edges.extend(zip(chain[:-1], chain[1:]))
    return edges


def dw_hand_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for finger in FINGERS:
        chain = [0] + DW_HAND_CHAINS[finger]
        edges.extend(zip(chain[:-1], chain[1:]))
    return edges


def decode_dwpose_target_hands(
    record: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Decode target-person DWPose hands into original-image pixel coordinates.

    easy-dwpose serializes hands as ``[all left hands, all right hands]`` where
    each hand contains 21 normalized image coordinates. Unlike ``body_scores``,
    ``hands_scores`` retains the original confidence values.
    """
    empty = {
        side: {
            "points": np.empty((0, 2), dtype=np.float64),
            "scores": np.empty((0,), dtype=np.float64),
            "accepted_mask": np.empty((0,), dtype=bool),
        }
        for side in ("left", "right")
    }
    if not record:
        return empty

    raw = record.get("raw_pose") or {}
    try:
        hands = np.asarray(raw.get("hands", []), dtype=np.float64)
        scores = np.asarray(raw.get("hands_scores", []), dtype=np.float64)
        body_scores = np.asarray(raw.get("body_scores", []), dtype=np.float64)
    except (TypeError, ValueError):
        return empty

    if hands.ndim != 3 or hands.shape[1] < 21 or hands.shape[2] < 2:
        return empty
    if scores.ndim != 2 or scores.shape[1] < 21:
        return empty

    if body_scores.ndim == 2 and body_scores.shape[0] > 0:
        people = int(body_scores.shape[0])
    else:
        people = max(1, int(hands.shape[0] // 2))
    if hands.shape[0] < 2 * people or scores.shape[0] < 2 * people:
        return empty

    value = (record.get("derived") or {}).get("target_person_index")
    try:
        target = int(value) if value is not None else 0
    except (TypeError, ValueError):
        target = 0
    if target < 0 or target >= people:
        target = 0

    result: dict[str, dict[str, np.ndarray]] = {}
    for side, raw_index in (("left", target), ("right", people + target)):
        points = np.asarray(hands[raw_index, :21, :2], dtype=np.float64).copy()
        points[:, 0] *= float(width)
        points[:, 1] *= float(height)
        confidence = np.asarray(scores[raw_index, :21], dtype=np.float64).copy()
        accepted = np.isfinite(confidence) & (confidence > 0.3)
        result[side] = {
            "points": points,
            "scores": confidence,
            "accepted_mask": accepted,
        }
    return result


def _chain_extension_ratio(points: np.ndarray, indices: list[int]) -> float | None:
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    if any(idx >= len(points) for idx in indices):
        return None
    chain = np.asarray(points[indices], dtype=np.float64)
    if not np.all(np.isfinite(chain)):
        return None
    segments = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    path = float(np.sum(segments))
    if path <= 1e-9:
        return None
    chord = float(np.linalg.norm(chain[-1] - chain[0]))
    return chord / path


def dwpose_finger_extension_ratios(points: np.ndarray) -> dict[str, float | None]:
    return {
        finger: _chain_extension_ratio(points, [0] + DW_HAND_CHAINS[finger])
        for finger in FINGERS
    }


def sam3d_finger_extension_ratios(
    keypoints: np.ndarray,
    side: str,
) -> dict[str, float | None]:
    if keypoints.ndim != 2 or keypoints.shape[1] < 3:
        return {finger: None for finger in FINGERS}
    spec = MHR_HANDS[side]
    wrist = int(spec["wrist"])
    return {
        finger: _chain_extension_ratio(
            keypoints,
            [wrist] + [int(v) for v in spec[finger]],
        )
        for finger in FINGERS
    }


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def summarize_finger_shape(
    ratios: dict[str, float | None],
) -> dict[str, Any]:
    states: dict[str, str | None] = {}
    for finger, value in ratios.items():
        if value is None:
            states[finger] = None
        elif value >= 0.88:
            states[finger] = "extended"
        elif value <= 0.72:
            states[finger] = "curled"
        else:
            states[finger] = "partially_flexed"

    non_thumb = [ratios[f] for f in ("index", "middle", "ring", "pinky") if ratios.get(f) is not None]
    if non_thumb:
        open_score = float(np.mean([_ramp(float(v), 0.78, 0.95) for v in non_thumb]))
        closed_score = float(np.mean([1.0 - _ramp(float(v), 0.62, 0.82) for v in non_thumb]))
        mean_ratio = float(np.mean(non_thumb))
    else:
        open_score = closed_score = mean_ratio = None

    return {
        "finger_extension_ratio": {
            name: round(float(value), 4) if value is not None else None
            for name, value in ratios.items()
        },
        "finger_state": states,
        "mean_non_thumb_extension_ratio": (
            round(mean_ratio, 4) if mean_ratio is not None else None
        ),
        "open_hand_score": round(open_score, 4) if open_score is not None else None,
        "closed_fist_score": round(closed_score, 4) if closed_score is not None else None,
    }
