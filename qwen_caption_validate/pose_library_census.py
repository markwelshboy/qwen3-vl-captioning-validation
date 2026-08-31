from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NAMED_RELATIONS = (
    "hands_on_hips",
    "head_supported_by_hand",
    "head_supported_by_fist",
)

OBSERVED_HAND_SUPPORT_MIN = 0.35
DISCOVERY_SUPPORT_MIN = 0.35


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _profile_records(profile_dir: Path) -> list[dict[str, Any]]:
    index_path = profile_dir / "sam3d_relational_pose.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        records = index.get("records") or []
        if isinstance(records, list) and records:
            return [r for r in records if isinstance(r, dict)]

    records: list[dict[str, Any]] = []
    for path in sorted(profile_dir.glob("*.sam3d_relational_pose.json")):
        value = _read_json(path)
        if value:
            records.append(value)
    return records


def _flexion_rank(band: str | None) -> int:
    order = {
        "tightly_flexed": 0,
        "flexed": 1,
        "moderately_flexed": 2,
        "near_straight": 3,
    }
    return order.get(str(band), 9)


def _wrist_height_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    if value >= 0.50:
        return "well_above_shoulder"
    if value >= 0.15:
        return "above_shoulder"
    if value >= -0.25:
        return "near_shoulder_height"
    return "below_shoulder"


def _hands_distance_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    if value <= 0.45:
        return "very_close"
    if value <= 0.85:
        return "close"
    if value <= 1.50:
        return "moderate"
    return "far"


def _side_signature(side_value: dict[str, Any]) -> str:
    flags = []
    if side_value.get("hand_near_hip"):
        flags.append("near_hip")
    if side_value.get("hand_near_face"):
        flags.append("near_face")
    if side_value.get("hand_near_knee"):
        flags.append("near_knee")
    if not flags:
        flags.append("no_near_anchor")

    return ",".join(
        [
            str(side_value.get("elbow_flexion_band") or "elbow_unknown"),
            str(side_value.get("hand_shape") or "hand_unknown"),
            _wrist_height_band(side_value.get("wrist_above_shoulder_shoulder_widths")),
            *flags,
        ]
    )


def _signature(profile: dict[str, Any]) -> str:
    projected = profile.get("sam3d_projected_pose") or {}
    primitives = profile.get("discovery_primitives") or {}
    sides = primitives.get("per_side") or {}
    bilateral = primitives.get("bilateral") or {}
    return " | ".join(
        [
            f"pose={projected.get('pose') or 'unknown'}",
            f"L={_side_signature(sides.get('left') or {})}",
            f"R={_side_signature(sides.get('right') or {})}",
            (
                "hands="
                + _hands_distance_band(
                    bilateral.get("hand_centroid_distance_shoulder_widths")
                )
            ),
        ]
    )


def _pattern_evidence(
    key: str,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Emit geometry-only review patterns, not semantic pose labels."""
    primitives = profile.get("discovery_primitives") or {}
    sides = primitives.get("per_side") or {}
    bilateral = primitives.get("bilateral") or {}
    relations = profile.get("relations") or {}

    left = sides.get("left") or {}
    right = sides.get("right") or {}
    result: list[dict[str, Any]] = []

    def arm_support(side_value: dict[str, Any]) -> float:
        return float(side_value.get("arm_anchor_crop_support") or 0.0)

    def hand_support(side_value: dict[str, Any]) -> float:
        return float(side_value.get("observed_hand_crop_support") or 0.0)

    def observed_shape(side_value: dict[str, Any], label: str) -> bool:
        return (
            side_value.get("hand_shape_source") == "dwpose_observed"
            and side_value.get("hand_shape") == label
            and hand_support(side_value) >= OBSERVED_HAND_SUPPORT_MIN
        )

    for anchor_name, field in (
        ("hand_near_hip", "hand_near_hip"),
        ("hand_near_face", "hand_near_face"),
        ("hand_near_knee", "hand_near_knee"),
    ):
        matching = [
            side
            for side, value in (("left", left), ("right", right))
            if value.get(field)
        ]
        if matching:
            support = max(
                arm_support(left if side == "left" else right)
                for side in matching
            )
            result.append(
                {
                    "pattern": anchor_name,
                    "image_key": key,
                    "support": support,
                    "sides": matching,
                }
            )

    above = []
    well_above = []
    for side, value in (("left", left), ("right", right)):
        height = value.get("wrist_above_shoulder_shoulder_widths")
        if height is not None and float(height) >= 0.15:
            above.append(side)
        if height is not None and float(height) >= 0.50:
            well_above.append(side)
    if above:
        result.append(
            {
                "pattern": "wrist_above_shoulder",
                "image_key": key,
                "support": max(
                    arm_support(left if side == "left" else right)
                    for side in above
                ),
                "sides": above,
            }
        )
    if well_above:
        result.append(
            {
                "pattern": "wrist_well_above_shoulder",
                "image_key": key,
                "support": max(
                    arm_support(left if side == "left" else right)
                    for side in well_above
                ),
                "sides": well_above,
            }
        )

    open_sides = [
        side
        for side, value in (("left", left), ("right", right))
        if observed_shape(value, "open_hand")
    ]
    fist_sides = [
        side
        for side, value in (("left", left), ("right", right))
        if observed_shape(value, "closed_fist")
    ]
    if open_sides:
        result.append(
            {
                "pattern": "observed_open_hand",
                "image_key": key,
                "support": max(
                    hand_support(left if side == "left" else right)
                    for side in open_sides
                ),
                "sides": open_sides,
            }
        )
    if fist_sides:
        result.append(
            {
                "pattern": "observed_closed_fist",
                "image_key": key,
                "support": max(
                    hand_support(left if side == "left" else right)
                    for side in fist_sides
                ),
                "sides": fist_sides,
            }
        )
    if open_sides and fist_sides:
        result.append(
            {
                "pattern": "one_open_hand_one_closed_fist",
                "image_key": key,
                "support": min(
                    max(
                        hand_support(left if side == "left" else right)
                        for side in open_sides
                    ),
                    max(
                        hand_support(left if side == "left" else right)
                        for side in fist_sides
                    ),
                ),
                "sides": sorted(set(open_sides + fist_sides)),
            }
        )

    left_band = left.get("elbow_flexion_band")
    right_band = right.get("elbow_flexion_band")
    if left_band and right_band:
        if _flexion_rank(left_band) <= 2 and _flexion_rank(right_band) <= 2:
            result.append(
                {
                    "pattern": "both_elbows_flexed",
                    "image_key": key,
                    "support": min(arm_support(left), arm_support(right)),
                    "sides": ["left", "right"],
                }
            )
        if abs(_flexion_rank(left_band) - _flexion_rank(right_band)) >= 2:
            result.append(
                {
                    "pattern": "asymmetric_elbow_flexion",
                    "image_key": key,
                    "support": min(arm_support(left), arm_support(right)),
                    "sides": ["left", "right"],
                }
            )

    hand_distance = bilateral.get("hand_centroid_distance_shoulder_widths")
    if hand_distance is not None and float(hand_distance) <= 0.85:
        result.append(
            {
                "pattern": "hands_close",
                "image_key": key,
                "support": min(
                    max(arm_support(left), hand_support(left)),
                    max(arm_support(right), hand_support(right)),
                ),
                "sides": ["left", "right"],
                "hand_centroid_distance_shoulder_widths": float(hand_distance),
            }
        )

    named_match = any(
        bool((relations.get(name) or {}).get("geometry_match"))
        for name in NAMED_RELATIONS
    )
    for item in result:
        item["already_has_named_relation"] = named_match
    return result


def _representative_rows(
    rows: list[dict[str, Any]],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda item: (
            float(item.get("support") or 0.0),
            str(item.get("image_key") or ""),
        ),
        reverse=True,
    )
    return rows[:max_examples]


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# SAM3D/DWPose pose-library census",
        "",
        "This is a geometry-discovery report. Review patterns are mechanical groupings, not new semantic pose labels.",
        "",
        f"Records: **{payload['record_count']}**",
        "",
        "## Projected pose census",
        "",
        "| Pose | Count |",
        "|---|---:|",
    ]
    for name, count in payload["projected_pose_counts"].items():
        lines.append(f"| `{name}` | {count} |")

    lines += [
        "",
        "## Current named relations",
        "",
        "| Relation | Count |",
        "|---|---:|",
    ]
    for name, value in payload["named_relations"].items():
        lines.append(f"| `{name}` | {value['count']} |")

    lines += [
        "",
        "## Observed hand shapes",
        "",
        "| Shape | Count |",
        "|---|---:|",
    ]
    for name, count in payload["observed_hand_shape_counts"].items():
        lines.append(f"| `{name}` | {count} |")

    lines += [
        "",
        "## Frequent geometry review patterns",
        "",
        "| Pattern | Count | Mean support | Examples |",
        "|---|---:|---:|---|",
    ]
    for item in payload["review_patterns"]:
        examples = ", ".join(row["image_key"] for row in item["examples"])
        lines.append(
            f"| `{item['pattern']}` | {item['count']} | "
            f"{item['mean_support']:.2f} | {examples} |"
        )

    lines += [
        "",
        "## Repeated signatures",
        "",
        "| Count | Signature | Examples |",
        "|---:|---|---|",
    ]
    for item in payload["repeated_signatures"]:
        examples = ", ".join(item["examples"])
        lines.append(
            f"| {item['count']} | `{item['signature']}` | {examples} |"
        )

    lines += [
        "",
        "## Review candidates",
        "",
        "These are selected to cover named relations, frequent unnamed geometry, strong model disagreements, and reconstruction-heavy global poses.",
        "",
    ]
    for item in payload["review_candidates"]:
        reasons = "; ".join(item["reasons"])
        lines.append(f"- `{item['image_key']}` — {reasons}")

    lines += [
        "",
        "## Policy",
        "",
        "- No action semantics such as `waving` are generated here.",
        "- `open_hand`, `closed_fist`, wrist height, proximity, and flexion are geometry primitives for later Fusion.",
        "- Pattern names in this report are review buckets, not accepted caption vocabulary.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-library-census",
        description=(
            "Summarize relational-pose profiles into a geometry census and "
            "select representative images for pose-library review."
        ),
    )
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--min-support", type=float, default=DISCOVERY_SUPPORT_MIN)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-review", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_dir = args.profile_dir.expanduser().resolve()
    if not profile_dir.is_dir():
        raise SystemExit(f"Profile directory not found: {profile_dir}")

    output = (
        args.output or (profile_dir / "pose-library-census")
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = _profile_records(profile_dir)
    if not records:
        raise SystemExit(f"No relational pose profiles found in {profile_dir}")

    projected_pose_counts: Counter[str] = Counter()
    observed_hand_shape_counts: Counter[str] = Counter()
    elbow_band_counts: Counter[str] = Counter()
    named_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in NAMED_RELATIONS
    }
    pattern_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signature_rows: dict[str, list[str]] = defaultdict(list)

    review_reasons: dict[str, set[str]] = defaultdict(set)
    review_scores: dict[str, float] = defaultdict(float)

    for record in records:
        key = str(record.get("image_key") or "")
        profile = record.get("profile") or {}
        projected = profile.get("sam3d_projected_pose") or {}
        projected_pose_counts[str(projected.get("pose") or "unknown")] += 1

        for side in ("left", "right"):
            h = (profile.get("hand_geometry") or {}).get(side) or {}
            if h.get("preferred_shape_source") == "dwpose_observed":
                dw = h.get("dwpose_hand") or {}
                support = float(dw.get("crop_support") or 0.0)
                if support >= args.min_support:
                    observed_hand_shape_counts[
                        str(h.get("preferred_shape_label") or "unavailable")
                    ] += 1

            arm = (profile.get("arm_geometry") or {}).get(side) or {}
            band = arm.get("elbow_flexion_band")
            if band:
                elbow_band_counts[str(band)] += 1

        for name in NAMED_RELATIONS:
            relation = (profile.get("relations") or {}).get(name) or {}
            if relation.get("geometry_match"):
                support = float(relation.get("crop_support") or 0.0)
                row = {
                    "image_key": key,
                    "support": support,
                    "support_percent": relation.get("crop_support_percent"),
                    "side": relation.get("side"),
                }
                named_rows[name].append(row)
                review_reasons[key].add(f"named relation {name}")
                review_scores[key] = max(review_scores[key], support)

        for row in _pattern_evidence(key, profile):
            if float(row.get("support") or 0.0) < args.min_support:
                continue
            pattern_rows[str(row["pattern"])].append(row)

        sig = _signature(profile)
        signature_rows[sig].append(key)

        for side in ("left", "right"):
            h = (profile.get("hand_geometry") or {}).get(side) or {}
            dw = h.get("dwpose_hand") or {}
            support = float(dw.get("crop_support") or 0.0)
            agreement = h.get("cross_model_shape_agreement")
            if (
                h.get("preferred_shape_source") == "dwpose_observed"
                and support >= 0.50
                and agreement is not None
                and float(agreement) <= 0.50
            ):
                review_reasons[key].add(
                    f"{side} hand DWPose/SAM3D shape disagreement"
                )
                review_scores[key] = max(review_scores[key], support)

        recon = float(projected.get("reconstruction_match") or 0.0)
        crop = float(projected.get("crop_support") or 0.0)
        if recon >= 0.80 and crop <= 0.15:
            review_reasons[key].add(
                "strong projected pose with weak crop support"
            )
            review_scores[key] = max(review_scores[key], recon * 0.5)

    review_patterns: list[dict[str, Any]] = []
    for pattern, rows in pattern_rows.items():
        if len(rows) < args.min_count:
            continue
        supports = [float(row.get("support") or 0.0) for row in rows]
        review_patterns.append(
            {
                "pattern": pattern,
                "count": len(rows),
                "mean_support": round(sum(supports) / len(supports), 4),
                "examples": _representative_rows(
                    rows, max_examples=args.max_examples
                ),
            }
        )
    review_patterns.sort(
        key=lambda item: (item["count"], item["mean_support"], item["pattern"]),
        reverse=True,
    )

    for item in review_patterns:
        for row in item["examples"][:2]:
            key = row["image_key"]
            review_reasons[key].add(f"review pattern {item['pattern']}")
            review_scores[key] = max(
                review_scores[key], float(row.get("support") or 0.0)
            )

    repeated_signatures = [
        {
            "signature": signature,
            "count": len(keys),
            "examples": keys[: args.max_examples],
        }
        for signature, keys in signature_rows.items()
        if len(keys) >= args.min_count
    ]
    repeated_signatures.sort(
        key=lambda item: (item["count"], item["signature"]),
        reverse=True,
    )

    review_candidates = [
        {
            "image_key": key,
            "score": round(review_scores[key], 4),
            "reasons": sorted(reasons),
        }
        for key, reasons in review_reasons.items()
    ]
    review_candidates.sort(
        key=lambda item: (item["score"], item["image_key"]),
        reverse=True,
    )
    review_candidates = review_candidates[: args.max_review]

    named_relations = {
        name: {
            "count": len(rows),
            "examples": _representative_rows(
                rows, max_examples=args.max_examples
            ),
        }
        for name, rows in named_rows.items()
    }

    payload = {
        "schema_version": "sam3d-pose-library-census-0.1",
        "profile_dir": str(profile_dir),
        "record_count": len(records),
        "parameters": {
            "min_count": args.min_count,
            "min_support": args.min_support,
            "max_examples": args.max_examples,
            "max_review": args.max_review,
        },
        "projected_pose_counts": dict(
            sorted(projected_pose_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "named_relations": named_relations,
        "observed_hand_shape_counts": dict(
            sorted(
                observed_hand_shape_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "elbow_flexion_band_counts": dict(
            sorted(elbow_band_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "review_patterns": review_patterns,
        "repeated_signatures": repeated_signatures,
        "review_candidates": review_candidates,
        "policy": {
            "report_only": True,
            "patterns_are_geometry_review_buckets_not_semantic_pose_labels": True,
            "action_semantics_are_reserved_for_fusion_caption": True,
        },
    }

    json_path = output / "pose_library_census.json"
    md_path = output / "pose_library_census.md"
    candidates_path = output / "review_candidates.json"
    keys_path = output / "review_keys.txt"
    atlas_args_path = output / "atlas_only_args.txt"

    _write_json(json_path, payload)
    md_path.write_text(_markdown_report(payload) + "\n", encoding="utf-8")
    _write_json(
        candidates_path,
        {
            "schema_version": "sam3d-pose-library-review-candidates-0.1",
            "candidates": review_candidates,
        },
    )
    keys_path.write_text(
        "".join(f"{item['image_key']}\n" for item in review_candidates),
        encoding="utf-8",
    )
    atlas_args_path.write_text(
        "".join(f"--only {item['image_key']}\n" for item in review_candidates),
        encoding="utf-8",
    )

    print(f"Records: {len(records)}")
    print(
        "Projected poses: "
        + ", ".join(
            f"{name}={count}" for name, count in payload["projected_pose_counts"].items()
        )
    )
    print(
        "Named relations: "
        + ", ".join(
            f"{name}={value['count']}"
            for name, value in named_relations.items()
        )
    )
    print(
        "Review patterns: "
        + ", ".join(
            f"{item['pattern']}={item['count']}"
            for item in review_patterns[:12]
        )
    )
    print(f"Review candidates: {len(review_candidates)}")
    print(f"Report: {md_path}")
    print(f"JSON: {json_path}")
    print(f"Review keys: {keys_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
