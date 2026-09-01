from __future__ import annotations

from pathlib import Path
from typing import Any

from . import pose_review_bundle as base
from . import pose_review_bundle_02 as _v02  # noqa: F401  (applies v0.2 bundle extensions)


_v02_parse_args = base.parse_args
_v02_find_profile_dir = base._find_profile_dir
_v02_compact_record = base._compact_record


def _parse_args_v03():
    args = _v02_parse_args()
    default_v02 = args.run_dir / "semantic-v3" / "pose-review-v0.2"
    if args.output == default_v02:
        args.output = args.run_dir / "semantic-v3" / "pose-review-v0.3"
    return args


def _find_profile_dir_v08(run_dir: Path, supplied: Path | None, sam3d_dir: Path) -> Path:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Profile directory not found: {value}")
        return value
    preferred = sam3d_dir / "relational-pose-profile-v0.8"
    if preferred.is_dir():
        return preferred
    return _v02_find_profile_dir(run_dir, supplied, sam3d_dir)


def _compact_record_v03(
    key: str,
    profile: dict[str, Any],
    original_rel: str,
    overlay_rel: str,
    raw_rel: str,
    overlay_meta: dict[str, Any],
) -> dict[str, Any]:
    record = _v02_compact_record(
        key,
        profile,
        original_rel,
        overlay_rel,
        raw_rel,
        overlay_meta,
    )
    projected = profile.get("sam3d_projected_pose") or {}
    record["leg_state_diagnostic"] = projected.get("leg_state_diagnostic") or {}
    record["independent_support_diagnostic"] = projected.get("independent_support_diagnostic") or {}
    record["diagnostic_profile_schema"] = profile.get("schema_version")
    return record


base.parse_args = _parse_args_v03
base._find_profile_dir = _find_profile_dir_v08
base._compact_record = _compact_record_v03


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
