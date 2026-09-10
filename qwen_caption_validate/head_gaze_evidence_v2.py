from __future__ import annotations

from typing import Any

from . import head_gaze_evidence as base

SCHEMA_VERSION = "head-gaze-evidence-0.2"
_BASE_BUILD_RECORD = base.build_record


def _head_authority(record: dict[str, Any], uniface: dict[str, Any]) -> str:
    head = record.get("head") if isinstance(record.get("head"), dict) else {}
    if not head.get("available"):
        return "unavailable"

    source = head.get("primary_source")
    comparison = head.get("pyfeat_comparison") if isinstance(head.get("pyfeat_comparison"), dict) else None
    quality = None
    if uniface.get("status") == "ok":
        try:
            quality = float((uniface.get("face_quality") or {}).get("score"))
        except (TypeError, ValueError):
            quality = None

    if source == "pyfeat_v28_fallback":
        authority = "reduced"
    elif comparison and comparison.get("status") == "agree":
        authority = "corroborated"
    elif comparison and comparison.get("status") == "profile_magnitude_divergence":
        authority = "corroborated_direction"
    elif comparison and comparison.get("status") == "diverge":
        authority = "conflict"
    else:
        authority = "high"

    if quality is not None and quality < 0.30 and authority not in {"unavailable", "conflict"}:
        authority = "reduced"
    return authority


def build_record(key: str, uniface: dict[str, Any], pyfeat: dict[str, Any], l2cs: dict[str, Any], ptgaze: dict[str, Any]) -> dict[str, Any]:
    record = _BASE_BUILD_RECORD(key, uniface, pyfeat, l2cs, ptgaze)
    record["schema_version"] = SCHEMA_VERSION
    record["head"]["authority"] = _head_authority(record, uniface)
    record["notes"].append(
        "Head authority is independent of gaze authority; extreme-profile magnitude disagreement can retain direction-level corroboration."
    )
    return record


def main() -> int:
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.build_record = build_record
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
