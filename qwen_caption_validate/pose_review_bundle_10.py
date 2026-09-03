from __future__ import annotations

from pathlib import Path
from typing import Any

from . import pose_review_bundle as base
from . import pose_review_bundle_09 as _v09  # noqa: F401  (applies all v0.14 extensions)


_v09_parse_args = base.parse_args
_v09_find_profile_dir = base._find_profile_dir
_v09_compact_record = base._compact_record


def _parse_args_v10():
    args = _v09_parse_args()
    default_v09 = args.run_dir / "semantic-v3" / "pose-review-v0.9"
    if args.output == default_v09:
        args.output = args.run_dir / "semantic-v3" / "pose-review-v0.10"
    return args


def _find_profile_dir_v15(run_dir: Path, supplied: Path | None, sam3d_dir: Path) -> Path:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Profile directory not found: {value}")
        return value
    preferred = sam3d_dir / "relational-pose-profile-v0.15"
    if preferred.is_dir():
        return preferred
    return _v09_find_profile_dir(run_dir, supplied, sam3d_dir)


def _compact_record_v10(
    key: str,
    profile: dict[str, Any],
    original_rel: str,
    overlay_rel: str,
    raw_rel: str,
    overlay_meta: dict[str, Any],
) -> dict[str, Any]:
    record = _v09_compact_record(
        key, profile, original_rel, overlay_rel, raw_rel, overlay_meta
    )
    projected = profile.get("sam3d_projected_pose") or {}
    record["assertion_authority"] = projected.get("assertion_authority") or {}
    record["body_orientation_diagnostic"] = projected.get("body_orientation_diagnostic") or {}
    record["v14_public_pose_before_assertion_authority"] = projected.get(
        "v14_public_pose_before_assertion_authority"
    )
    record["diagnostic_profile_schema"] = profile.get("schema_version")
    return record


base.parse_args = _parse_args_v10
base._find_profile_dir = _find_profile_dir_v15
base._compact_record = _compact_record_v10


def main() -> int:
    return _v09.main()


if __name__ == "__main__":
    raise SystemExit(main())
