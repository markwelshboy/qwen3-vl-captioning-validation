from __future__ import annotations

from pathlib import Path
from typing import Any

from . import pose_review_bundle as base
from . import pose_review_bundle_04 as _v04  # noqa: F401  (applies prior bundle extensions)


_v04_parse_args = base.parse_args
_v04_find_profile_dir = base._find_profile_dir
_v04_compact_record = base._compact_record


def _parse_args_v05():
    args = _v04_parse_args()
    default_v04 = args.run_dir / "semantic-v3" / "pose-review-v0.4"
    if args.output == default_v04:
        args.output = args.run_dir / "semantic-v3" / "pose-review-v0.5"
    return args


def _find_profile_dir_v10(run_dir: Path, supplied: Path | None, sam3d_dir: Path) -> Path:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Profile directory not found: {value}")
        return value
    preferred = sam3d_dir / "relational-pose-profile-v0.10"
    if preferred.is_dir():
        return preferred
    return _v04_find_profile_dir(run_dir, supplied, sam3d_dir)


def _compact_record_v05(
    key: str,
    profile: dict[str, Any],
    original_rel: str,
    overlay_rel: str,
    raw_rel: str,
    overlay_meta: dict[str, Any],
) -> dict[str, Any]:
    record = _v04_compact_record(
        key,
        profile,
        original_rel,
        overlay_rel,
        raw_rel,
        overlay_meta,
    )
    projected = profile.get("sam3d_projected_pose") or {}
    record["upper_body_recline_diagnostic"] = projected.get("upper_body_recline_diagnostic") or {}
    record["v09_pose_before_sitting_recline_refine"] = projected.get(
        "v09_pose_before_sitting_recline_refine"
    )
    record["v09_best_candidate_before_sitting_recline_refine"] = projected.get(
        "v09_best_candidate_before_sitting_recline_refine"
    )
    record["v09_posture_score_percent_before_sitting_recline_refine"] = projected.get(
        "v09_posture_score_percent_before_sitting_recline_refine"
    ) or {}
    record["diagnostic_profile_schema"] = profile.get("schema_version")
    return record


base.parse_args = _parse_args_v05
base._find_profile_dir = _find_profile_dir_v10
base._compact_record = _compact_record_v05


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
