from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import pose_library_census as base


SUPPORTED_RELATION_THRESHOLD = 0.50


def _supported_relations(profile_dir: Path) -> dict[str, dict[str, Any]]:
    rows = base._profile_records(profile_dir)
    out: dict[str, dict[str, Any]] = {}
    for name in base.NAMED_RELATIONS:
        geometry_matches: list[dict[str, Any]] = []
        supported: list[dict[str, Any]] = []
        for record in rows:
            key = str(record.get("image_key") or "")
            relation = (((record.get("profile") or {}).get("relations") or {}).get(name) or {})
            if not relation.get("geometry_match"):
                continue
            support = float(relation.get("crop_support") or 0.0)
            item = {
                "image_key": key,
                "support": round(support, 4),
                "support_percent": int(round(100.0 * support)),
                "side": relation.get("side"),
                "support_class": relation.get("support_class"),
            }
            geometry_matches.append(item)
            if support >= SUPPORTED_RELATION_THRESHOLD:
                supported.append(item)
        geometry_matches.sort(key=lambda item: (item["support"], item["image_key"]), reverse=True)
        supported.sort(key=lambda item: (item["support"], item["image_key"]), reverse=True)
        out[name] = {
            "geometry_match_count": len(geometry_matches),
            "crop_supported_count": len(supported),
            "crop_supported_threshold": SUPPORTED_RELATION_THRESHOLD,
            "geometry_match_examples": geometry_matches[:8],
            "crop_supported_examples": supported[:8],
        }
    return out


def _append_supported_section(md_path: Path, supported: dict[str, dict[str, Any]]) -> None:
    text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    lines = [
        "",
        "## Crop-supported named relations",
        "",
        "A geometry match is retained even when crop support is weak. The supported count below requires at least 50% crop support.",
        "",
        "| Relation | Geometry matches | Crop-supported |",
        "|---|---:|---:|",
    ]
    for name, value in supported.items():
        lines.append(
            f"| `{name}` | {value['geometry_match_count']} | {value['crop_supported_count']} |"
        )
    lines.append("")
    md_path.write_text(text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def main() -> int:
    # Let the established census generator produce its normal report/review set,
    # then augment it with the explicit geometry-match vs crop-supported split.
    args = base.parse_args()
    rc = base.main()
    if rc:
        return int(rc)

    profile_dir = args.profile_dir.expanduser().resolve()
    output = (args.output or (profile_dir / "pose-library-census")).expanduser().resolve()
    json_path = output / "pose_library_census.json"
    md_path = output / "pose_library_census.md"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    supported = _supported_relations(profile_dir)
    payload["schema_version"] = "sam3d-pose-library-census-0.2"
    payload.setdefault("parameters", {})["named_relation_crop_supported_threshold"] = SUPPORTED_RELATION_THRESHOLD
    payload["named_relation_support_census"] = supported
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _append_supported_section(md_path, supported)

    print(
        "Crop-supported named relations: "
        + ", ".join(
            f"{name}={value['crop_supported_count']}/{value['geometry_match_count']}"
            for name, value in supported.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
