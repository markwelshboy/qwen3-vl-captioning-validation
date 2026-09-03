from __future__ import annotations

from pathlib import Path
from typing import Any

from . import pose_review_bundle as base
from . import pose_review_bundle_08 as _v08  # noqa: F401  (applies v0.13 + side-view extensions)


_v08_parse_args = base.parse_args
_v08_find_profile_dir = base._find_profile_dir
_v08_compact_record = base._compact_record


def _parse_args_v09():
    args = _v08_parse_args()
    default_v08 = args.run_dir / "semantic-v3" / "pose-review-v0.8"
    if args.output == default_v08:
        args.output = args.run_dir / "semantic-v3" / "pose-review-v0.9"
    return args


def _find_profile_dir_v14(run_dir: Path, supplied: Path | None, sam3d_dir: Path) -> Path:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Profile directory not found: {value}")
        return value
    preferred = sam3d_dir / "relational-pose-profile-v0.14"
    if preferred.is_dir():
        return preferred
    return _v08_find_profile_dir(run_dir, supplied, sam3d_dir)


def _compact_record_v09(
    key: str,
    profile: dict[str, Any],
    original_rel: str,
    overlay_rel: str,
    raw_rel: str,
    overlay_meta: dict[str, Any],
) -> dict[str, Any]:
    record = _v08_compact_record(
        key, profile, original_rel, overlay_rel, raw_rel, overlay_meta
    )
    projected = profile.get("sam3d_projected_pose") or {}
    record["whole_body_recline_override"] = projected.get("whole_body_recline_override") or {}
    record["relative_squat_compensation"] = projected.get("relative_squat_compensation") or {}
    relation = (profile.get("relations") or {}).get("head_supported_by_hand") or {}
    record["head_support_proximal_guard"] = relation.get("v14_proximal_chain_guard") or {}
    record["diagnostic_profile_schema"] = profile.get("schema_version")
    return record


base.parse_args = _parse_args_v09
base._find_profile_dir = _find_profile_dir_v14
base._compact_record = _compact_record_v09


def main() -> int:
    return _v08.main()


if __name__ == "__main__":
    raise SystemExit(main())
