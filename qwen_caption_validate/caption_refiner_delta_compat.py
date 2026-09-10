from __future__ import annotations

from . import caption_refiner_delta as impl
from . import pose_atlas_v3 as atlas
from .dwpose_compat import target_points_from_profile_record


def main() -> int:
    # caption_refiner_delta delegates DWPose target extraction to pose_atlas_v3.
    # Replace that helper at runtime so historical direct BODY18 caches and newer
    # candidate-mapping caches both retain DWPose as the preferred laterality source.
    atlas._dwpose_target_points = target_points_from_profile_record
    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
