from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BANDS = (
    ("low", 0.0, 15.0),
    ("moderate", 15.0, 30.0),
    ("high", 30.0, 50.0),
    ("very_high", 50.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-dataset-profile",
        description=(
            "Report-only dataset profile for qualified SAM3D shoulder-girdle depth rotation. "
            "This command never changes Dataset Evidence / V8.1 selection scores."
        ),
    )
    parser.add_argument(
        "fusion_dir",
        type=Path,
        help="Fusion-v2.3 model directory containing *.fused_v2_3.json files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="sam3d_shoulder_depth_profile",
        help="Output basename written next to the fusion directory (default: sam3d_shoulder_depth_profile).",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _band(angle: float | None) -> str | None:
    if angle is None:
        return None
    value = float(angle)
    for name, low, high in BANDS:
        if low <= value < high:
            return name
    return None


def _authority_status(audit: dict[str, Any]) -> str:
    shoulder = audit.get("shoulder_depth_rotation") or {}
    authority = str(shoulder.get("authority") or "unknown")
    provenance = audit.get("target_provenance") or {}
    if authority == "qualified_component_geometry":
        if provenance.get("context_risk") == "requires_review":
            return "pending_target_provenance"
        return "qualified"
    if authority == "report_only_partial_image_support":
        return "partial_image_support"
    if authority == "reconstructed_prior_only":
        return "prior_reconstructed"
    if authority == "report_only_requires_analyze_v2_1_visibility":
        return "missing_visibility_support"
    return "unresolved"


def _record(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    fusion = payload.get("fusion") or {}
    audit = fusion.get("sam3d_geometry_audit") or {}
    shoulder = audit.get("shoulder_depth_rotation") or {}
    angle = shoulder.get("magnitude_deg")
    angle_float = float(angle) if angle is not None else None
    return {
        "image": payload.get("image") or path.name.removesuffix(".fused_v2_3.json"),
        "angle_deg": round(angle_float, 3) if angle_float is not None else None,
        "band": _band(angle_float),
        "authority_status": _authority_status(audit),
        "component_authority": shoulder.get("authority"),
        "support_state": ((shoulder.get("support") or {}).get("state")),
        "target_provenance": audit.get("target_provenance"),
    }


def build_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    authority_counts = Counter(str(r["authority_status"]) for r in records)
    qualified = [r for r in records if r["authority_status"] == "qualified" and r["band"]]
    pending = [r for r in records if r["authority_status"] == "pending_target_provenance" and r["band"]]
    band_counts = Counter(str(r["band"]) for r in qualified)
    pending_band_counts = Counter(str(r["band"]) for r in pending)

    return {
        "schema_version": "sam3d-shoulder-depth-profile-0.1",
        "axis": {
            "name": "shoulder_girdle_depth_rotation",
            "quantity": "unsigned left-to-right shoulder axis rotation out of the image plane",
            "units": "degrees",
            "authority": "report_only_dataset_coverage",
            "selection_usable": False,
            "caption_usable": False,
            "bands": [
                {"name": "low", "range": "[0,15)"},
                {"name": "moderate", "range": "[15,30)"},
                {"name": "high", "range": "[30,50)"},
                {"name": "very_high", "range": "[50,+inf)"},
            ],
            "band_note": "Bands are presentation-only during validation; they do not contribute to V8.1 selection scoring.",
        },
        "image_count": len(records),
        "authority_counts": dict(sorted(authority_counts.items())),
        "qualified_image_count": len(qualified),
        "qualified_band_counts": {name: int(band_counts.get(name, 0)) for name, _, _ in BANDS},
        "pending_provenance_band_counts": {name: int(pending_band_counts.get(name, 0)) for name, _, _ in BANDS},
        "records": sorted(
            records,
            key=lambda r: (
                r["angle_deg"] is None,
                -(r["angle_deg"] or 0.0),
                str(r["image"]),
            ),
        ),
    }


def _markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# SAM3D shoulder-girdle depth profile",
        "",
        "This is a **report-only** dataset coverage view. It does not change V8.1 selection scores.",
        "",
        f"Images: **{profile['image_count']}**  ",
        f"Qualified shoulder geometry: **{profile['qualified_image_count']}**",
        "",
        "## Qualified coverage bands",
        "",
        "| Band | Range | Count |",
        "|---|---:|---:|",
    ]
    ranges = {item["name"]: item["range"] for item in profile["axis"]["bands"]}
    for name, _, _ in BANDS:
        lines.append(f"| {name} | {ranges[name]}° | {profile['qualified_band_counts'][name]} |")

    lines.extend([
        "",
        "## Authority",
        "",
        "| Status | Count |",
        "|---|---:|",
    ])
    for status, count in profile["authority_counts"].items():
        lines.append(f"| {status} | {count} |")

    lines.extend([
        "",
        "## Images",
        "",
        "| Image | Shoulder depth | Band | Authority |",
        "|---|---:|---|---|",
    ])
    for record in profile["records"]:
        angle = "—" if record["angle_deg"] is None else f"{record['angle_deg']:.2f}°"
        lines.append(
            f"| `{record['image']}` | {angle} | {record['band'] or '—'} | {record['authority_status']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "`shoulder_girdle_depth_rotation` is intentionally not called torso yaw. It is an unsigned, camera-relative 3-D component measurement from SAM 3D Body. Only records with image-supported shoulder landmarks and no unresolved target-provenance risk enter the qualified coverage histogram.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    fusion_dir = args.fusion_dir.expanduser().resolve()
    if not fusion_dir.is_dir():
        raise SystemExit(f"Fusion directory does not exist: {fusion_dir}")

    paths = sorted(fusion_dir.glob("*.fused_v2_3.json"))
    if not paths:
        raise SystemExit(f"No *.fused_v2_3.json files found in {fusion_dir}")

    profile = build_profile([_record(path) for path in paths])
    output_base = fusion_dir.parent / args.output_prefix
    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")
    json_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(profile), encoding="utf-8")

    print(f"SAM3D shoulder-depth profile: {json_path}")
    print(f"Report: {md_path}")
    print(f"Images: {profile['image_count']}; qualified: {profile['qualified_image_count']}")
    print(f"Qualified bands: {profile['qualified_band_counts']}")
    print(f"Authority: {profile['authority_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
