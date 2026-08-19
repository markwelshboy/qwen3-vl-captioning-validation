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

    # easy-dwpose 1.0.x format. Its _format_pose() flattens the first 18
    # whole-body candidates into [people*18, xy] and stores visibility/index
    # information separately in body_scores.
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
        # Last-resort compatibility path. Without scores we cannot distinguish
        # low-confidence joints, but can still recover person grouping.
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


def main() -> int:
    # dwpose_profile's analysis helpers resolve _candidate_array from that
    # module's globals at runtime, so replace only the adapter while retaining
    # the profiler/reporting implementation in one place.
    impl._candidate_array = _candidate_array
    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
