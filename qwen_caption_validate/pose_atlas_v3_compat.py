from __future__ import annotations

"""Compatibility entry point for the V3 pose atlas.

Historical DWPose caches in this project do not all serialize ``raw_pose.bodies``
with the same wrapper shape.  Newer caches use ``{"candidate": ...}``, while
some older calibration runs contain the candidate body array directly as a
JSON list.  The atlas is intentionally able to consume both because it is a
calibration/review tool over already-frozen evidence; it must not require
regenerating DWPose merely to draw an overlay.

This module patches only the atlas' DWPose point reader, then delegates to the
normal ``pose_atlas_v3.main`` implementation.  Once all historical cache shapes
have been characterized, this small adapter can be folded into the main atlas
module without changing its external behavior.
"""

from typing import Any

import numpy as np

from . import pose_atlas_v3 as atlas


def _empty_candidates() -> np.ndarray:
    return np.empty((0, 18, 2), dtype=np.float64)


def _normalize_candidate_array(value: Any) -> np.ndarray:
    """Return candidate bodies as ``(people, joints, coords)`` when possible.

    Accepted historical shapes include:

    * ``{"candidate": [[[x, y], ...]]}``
    * ``[[[x, y], ...]]`` (bodies is the candidate array directly)
    * ``[[x, y], ...]`` (single person)
    * a list of per-person dictionaries containing ``candidate``
    * wrapper dictionaries/lists containing one of the above

    Unknown/ragged metadata such as a DWPose ``subset`` array is ignored rather
    than being coerced into a skeleton.
    """

    if value is None:
        return _empty_candidates()

    if isinstance(value, dict):
        if "candidate" in value:
            return _normalize_candidate_array(value.get("candidate"))
        for key in ("bodies", "body", "people", "persons", "poses"):
            if key in value:
                candidate = _normalize_candidate_array(value.get(key))
                if candidate.size:
                    return candidate
        return _empty_candidates()

    if isinstance(value, (list, tuple)) and value and all(isinstance(item, dict) for item in value):
        people: list[np.ndarray] = []
        for item in value:
            candidate = _normalize_candidate_array(item)
            if candidate.ndim == 3 and candidate.size:
                people.extend(candidate)
        if people:
            try:
                return np.stack(people, axis=0).astype(np.float64, copy=False)
            except ValueError:
                return _empty_candidates()
        return _empty_candidates()

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        arr = np.empty((0,), dtype=np.float64)

    # Strip harmless singleton wrappers, but do not flatten the person/joint axes.
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2 and arr.shape[0] >= 18 and arr.shape[1] >= 2:
        return arr[None, ...]
    if arr.ndim == 3 and arr.shape[1] >= 18 and arr.shape[2] >= 2:
        return arr

    # Some old wrappers are heterogeneous lists such as [candidate, subset].
    # Prefer the first child that actually looks like body keypoints.
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _normalize_candidate_array(item)
            if candidate.size:
                return candidate

    return _empty_candidates()


def _dwpose_target_points_compat(record: dict[str, Any], width: int, height: int) -> np.ndarray:
    raw = record.get("raw_pose") or {}
    if isinstance(raw, dict):
        bodies: Any = raw.get("bodies")
        if bodies is None:
            # Be tolerant of a cache that stored the candidate directly in raw_pose.
            bodies = raw
    else:
        bodies = raw

    candidate = _normalize_candidate_array(bodies)
    if candidate.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    target_index_raw = (record.get("derived") or {}).get("target_person_index")
    try:
        target_index = int(target_index_raw) if target_index_raw is not None else 0
    except (TypeError, ValueError):
        target_index = 0
    if target_index < 0 or target_index >= candidate.shape[0]:
        target_index = 0

    points = candidate[target_index, :18, :2]
    return atlas._normalized_to_pixels(points, width, height)


def main() -> int:
    # Deliberately patch only this cache-decoding seam.  All atlas rendering,
    # SAM3D diagnostics, card generation, and output schemas remain unchanged.
    atlas._dwpose_target_points = _dwpose_target_points_compat
    return atlas.main()


if __name__ == "__main__":
    raise SystemExit(main())
