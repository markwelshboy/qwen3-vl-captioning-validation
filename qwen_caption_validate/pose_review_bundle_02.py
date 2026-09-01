from __future__ import annotations

from pathlib import Path
from typing import Any

from . import pose_review_bundle as base


_original_parse_args = base.parse_args
_original_compact_record = base._compact_record


def _parse_args_v02():
    args = _original_parse_args()
    if args.output is None:
        args.output = args.run_dir / "semantic-v3" / "pose-review-v0.2"
    return args


def _find_profile_dir_v07(run_dir: Path, supplied: Path | None, sam3d_dir: Path) -> Path:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"Profile directory not found: {value}")
        return value
    preferred = sam3d_dir / "relational-pose-profile-v0.7"
    if preferred.is_dir():
        return preferred
    return base._find_profile_dir(run_dir, supplied, sam3d_dir)


def _compact_record_v02(
    key: str,
    profile: dict[str, Any],
    original_rel: str,
    overlay_rel: str,
    raw_rel: str,
    overlay_meta: dict[str, Any],
) -> dict[str, Any]:
    record = _original_compact_record(
        key,
        profile,
        original_rel,
        overlay_rel,
        raw_rel,
        overlay_meta,
    )
    projected = profile.get("sam3d_projected_pose") or {}
    record["support_balance_diagnostic"] = projected.get("support_balance_diagnostic") or {}
    record["recline_diagnostic"] = projected.get("recline_diagnostic") or {}
    record["kneeling_context_diagnostic"] = projected.get("kneeling_context_diagnostic") or {}
    record["diagnostic_profile_schema"] = profile.get("schema_version")
    return record


base.parse_args = _parse_args_v02
base._find_profile_dir = _find_profile_dir_v07
base._compact_record = _compact_record_v02


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
