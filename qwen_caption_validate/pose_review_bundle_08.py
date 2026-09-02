from __future__ import annotations

from pathlib import Path
from typing import Any

from . import pose_review_bundle as base
from . import pose_review_bundle_07 as _v07  # noqa: F401  (applies side/support view extension)


_v07_parse_args = base.parse_args
_v07_find_profile_dir = base._find_profile_dir
_v07_compact_record = base._compact_record


def _parse_args_v08():
    args = _v07_parse_args()
    default_v07 = args.run_dir / "semantic-v3" / "pose-review-v0.7"
    if args.output == default_v07:
        args.output = args.run_dir / "semantic-v3" / "pose-review-v0.8"
    return args


def _find_profile_dir_v13(run_dir: Path, supplied: Path | None, sam3d_dir: Path) -> Path:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Profile directory not found: {value}")
        return value
    preferred = sam3d_dir / "relational-pose-profile-v0.13"
    if preferred.is_dir():
        return preferred
    return _v07_find_profile_dir(run_dir, supplied, sam3d_dir)


def _compact_record_v08(
    key: str,
    profile: dict[str, Any],
    original_rel: str,
    overlay_rel: str,
    raw_rel: str,
    overlay_meta: dict[str, Any],
) -> dict[str, Any]:
    record = _v07_compact_record(
        key, profile, original_rel, overlay_rel, raw_rel, overlay_meta
    )
    projected = profile.get("sam3d_projected_pose") or {}
    record["seated_low_stance_diagnostic"] = projected.get("seated_low_stance_diagnostic") or {}
    record["diagnostic_profile_schema"] = profile.get("schema_version")
    return record


base.parse_args = _parse_args_v08
base._find_profile_dir = _find_profile_dir_v13
base._compact_record = _compact_record_v08


def main() -> int:
    return _v07.main()


if __name__ == "__main__":
    raise SystemExit(main())
