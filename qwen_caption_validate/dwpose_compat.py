from __future__ import annotations

from typing import Any

import numpy as np

from . import dwpose_profile as impl


_EMPTY = np.empty((0, 18, 2), dtype=np.float64)


def _candidate_array(pose_data: dict[str, Any]) -> np.ndarray:
    """Normalize supported DWPose body formats to [person, 18, xy].

    easy-dwpose 1.0.x returns:
      bodies: flattened ndarray shaped [people * 18, 2]
      body_scores: ndarray shaped [people, 18], where -1 means not visible

    Historical Fizgig caches may store a single BODY18 person directly as:
      bodies: [[x, y], ...]  # 18 rows

    Other DWPose/OpenPose wrappers may instead return:
      bodies: {candidate: ...}

    The profiler consumes one normalized representation and explicitly marks
    low-confidence/unavailable easy-dwpose joints as [-1, -1].
    """
    bodies_raw = pose_data.get("bodies")
    if bodies_raw is None:
        return _EMPTY.copy()

    # Common OpenPose-style wrapper: {"candidate": ... , "subset": ...}
    if isinstance(bodies_raw, dict):
        candidate_raw = bodies_raw.get("candidate")
        if candidate_raw is None:
            return _EMPTY.copy()
        candidate = np.asarray(candidate_raw, dtype=np.float64)
        if candidate.size == 0:
            return _EMPTY.copy()
        if candidate.ndim == 2:
            if candidate.shape[1] >= 2 and candidate.shape[0] % 18 == 0:
                candidate = candidate.reshape(-1, 18, candidate.shape[1])
            else:
                candidate = candidate[None, ...]
        if candidate.ndim != 3 or candidate.shape[1] < 18 or candidate.shape[-1] < 2:
            return _EMPTY.copy()
        return candidate[:, :18, :2]

    # easy-dwpose / historical direct-array format.
    bodies = np.asarray(bodies_raw, dtype=np.float64)
    if bodies.size == 0:
        return _EMPTY.copy()

    # Be tolerant if a future release stops flattening the body array.
    if bodies.ndim == 3 and bodies.shape[1] >= 18 and bodies.shape[-1] >= 2:
        return bodies[:, :18, :2]
    if bodies.ndim != 2 or bodies.shape[1] < 2:
        return _EMPTY.copy()

    scores_raw = pose_data.get("body_scores")
    if scores_raw is None:
        # Historical single-person BODY18 and last-resort compatibility path.
        # Without scores we cannot distinguish low-confidence joints, but can
        # still recover person grouping. Existing negative sentinels are kept.
        if bodies.shape[0] % 18 != 0:
            return _EMPTY.copy()
        return bodies[:, :2].reshape(-1, 18, 2)

    scores = np.asarray(scores_raw)
    if scores.size == 0:
        return _EMPTY.copy()
    if scores.ndim == 1:
        scores = scores[None, ...]
    if scores.ndim != 2 or scores.shape[1] < 18:
        return _EMPTY.copy()

    people = scores.shape[0]
    needed_rows = people * 18
    if bodies.shape[0] < needed_rows:
        return _EMPTY.copy()

    candidate = bodies[:needed_rows, :2].reshape(people, 18, 2).copy()
    visible = scores[:people, :18] >= 0
    candidate[~visible] = -1.0
    return candidate


def _points_to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized DWPose points without letting -1 sentinels mimic NDC.

    DWPose coordinates are [0,1]-ish when normalized. Missing joints in historical
    caches are commonly negative. Range detection therefore considers only finite,
    non-negative joints and leaves missing sentinels untouched.
    """
    arr = np.asarray(points, dtype=np.float64)[..., :2].copy()
    if arr.size == 0:
        return arr

    valid = (
        np.isfinite(arr).all(axis=-1)
        & (arr[..., 0] >= 0.0)
        & (arr[..., 1] >= 0.0)
    )
    if not np.any(valid):
        return arr

    observed = arr[valid]
    if float(np.nanmax(observed)) <= 1.25 and float(np.nanmin(observed)) >= -0.05:
        arr[valid, 0] *= float(width)
        arr[valid, 1] *= float(height)
    return arr


def target_points_from_profile_record(record: dict[str, Any], width: int, height: int) -> np.ndarray:
    """Return the selected BODY18 person from a cached profiler record in pixels."""
    raw = record.get("raw_pose") if isinstance(record.get("raw_pose"), dict) else {}
    candidate = _candidate_array(raw)
    if candidate.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    derived = record.get("derived") if isinstance(record.get("derived"), dict) else {}
    try:
        target_index = int(derived.get("target_person_index") or 0)
    except (TypeError, ValueError):
        target_index = 0
    if target_index < 0 or target_index >= candidate.shape[0]:
        target_index = 0

    return _points_to_pixels(candidate[target_index, :18, :2], width, height)


def main() -> int:
    # dwpose_profile's analysis helpers resolve _candidate_array from that
    # module's globals at runtime, so replace only the adapter while retaining
    # the profiler/reporting implementation in one place.
    impl._candidate_array = _candidate_array
    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
