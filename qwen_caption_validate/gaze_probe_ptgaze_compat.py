from __future__ import annotations

from . import gaze_probe_ptgaze as impl
from .dwpose_compat import target_points_from_profile_record


def main() -> int:
    # The gaze probe predates the historical DWPose adapter and resolves this
    # helper from module globals at runtime. Reuse the shared compatibility
    # reader so both direct BODY18 caches and newer candidate-mapping caches work.
    impl._dwpose_target_points = target_points_from_profile_record
    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
