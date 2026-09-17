from __future__ import annotations

"""Shadow-only crouched-stance lexical/depth qualifier.

This stage does NOT decide whether a person is crouching.  Broad-pose authority
must already have established ``crouching`` in the canonical fact sheet.  The
stage asks a much narrower question: when an authoritative crouch exists, does
high-authority bilateral SAM3D lower-body geometry support the conservative
modifier ``hips slightly lowered``?

The distinction is intentional:

* canonical broad pose ``crouching`` -> lexical base ``holds a crouched stance``
* moderate bilateral lowering -> optional ``with her hips slightly lowered``
* deeper/more compressed or unresolved geometry -> no depth modifier

Thresholds are provisional and population-review-only in v0.1.  No fact sheet
is mutated and no composer is changed by this module.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v09 as sam3d_shadow


SCHEMA_VERSION = "crouched-stance-depth-shadow-0.1"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.13"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "crouched-stance-depth-shadow-v0.1"

POSE_BEARING_ROUTES = {"pose_allowed", "pose_guided"}
CANONICAL_POSE_REQUIRED = "crouching"
SAM3D_PUBLIC_POSE_REQUIRED = "crouching"

# Conservative first-pass bounds for a *moderately lowered* crouched stance.
# These are deliberately NOT a crouch classifier.  They are only evaluated
# after the broad-pose specialist has already made crouching authoritative.
MIN_SAM3D_KNEE_AUTHORITY = 0.80
MAX_EACH_KNEE_ANGLE_DEG = 140.0
MIN_MEAN_KNEE_ANGLE_DEG = 95.0
MIN_EACH_THAI_FROM_DOWN_DEG = 20.0
MAX_EACH_THAI_FROM_DOWN_DEG = 55.0
MIN_MEAN_THAI_FROM_DOWN_DEG = 25.0
MAX_MEAN_THAI_FROM_DOWN_DEG = 50.0
MIN_EACH_LEG_EXTENSION_RATIO = 0.70
MIN_MEAN_LEG_EXTENSION_RATIO = 0.78
MAX_MEAN_LEG_EXTENSION_RATIO = 0.92

BASE_LEXICALIZATION = "holds a crouched stance"
QUALIFIED_LEXICALIZATION = "holds a crouched stance with her hips slightly lowered"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: Any, digits: int = 3) -> float | None:
    number = _finite(value)
    return round(number, digits) if number is not None else None


def _route(sheet: dict[str, Any]) -> str:
    policy = sheet.get("policy") if isinstance(sheet.get("policy"), dict) else {}
    return str(policy.get("mode") or "")


def _canonical_pose(sheet: dict[str, Any]) -> str | None:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}

    candidate = body.get("pose_candidate")
    value: Any = None
    if isinstance(candidate, dict):
        value = candidate.get("composer_text") or candidate.get("normalized_text") or candidate.get("text")
    elif isinstance(candidate, str):
        value = candidate

    if value is None:
        value = body.get("broad_pose") or body.get("pose")
    if value is None:
        return None
    return " ".join(str(value).split()).strip().lower() or None


def _side_metrics(projected: dict[str, Any], side: str) -> dict[str, float | None]:
    geometry = projected.get("geometry") if isinstance(projected.get("geometry"), dict) else {}
    lower = geometry.get("asymmetric_lower_body") if isinstance(geometry.get("asymmetric_lower_body"), dict) else {}
    per_side = lower.get("per_side") if isinstance(lower.get("per_side"), dict) else {}
    source = per_side.get(side) if isinstance(per_side.get(side), dict) else {}

    thigh = _finite(source.get("thigh_axis_from_image_down_deg"))
    return {
        "knee_flexion_deg": _round(source.get("knee_flexion_deg")),
        "hip_flexion_deg": _round(source.get("hip_flexion_deg")),
        "thigh_axis_from_image_down_deg": _round(abs(thigh) if thigh is not None else None),
        "leg_extension_ratio": _round(source.get("leg_extension_ratio")),
    }


def _knee_authority(projected: dict[str, Any]) -> float | None:
    region = projected.get("region_support") if isinstance(projected.get("region_support"), dict) else {}
    return _round(region.get("knees"))


def _aggregate(left: dict[str, float | None], right: dict[str, float | None]) -> dict[str, float | None]:
    def mean(key: str) -> float | None:
        a = _finite(left.get(key))
        b = _finite(right.get(key))
        if a is None or b is None:
            return None
        return round((a + b) / 2.0, 3)

    return {
        "mean_knee_flexion_deg": mean("knee_flexion_deg"),
        "mean_hip_flexion_deg": mean("hip_flexion_deg"),
        "mean_thigh_axis_from_image_down_deg": mean("thigh_axis_from_image_down_deg"),
        "mean_leg_extension_ratio": mean("leg_extension_ratio"),
    }


def _qualifier_gates(
    *,
    knee_authority: float | None,
    left: dict[str, float | None],
    right: dict[str, float | None],
    aggregate: dict[str, float | None],
) -> dict[str, bool]:
    lk = _finite(left.get("knee_flexion_deg"))
    rk = _finite(right.get("knee_flexion_deg"))
    lt = _finite(left.get("thigh_axis_from_image_down_deg"))
    rt = _finite(right.get("thigh_axis_from_image_down_deg"))
    le = _finite(left.get("leg_extension_ratio"))
    re = _finite(right.get("leg_extension_ratio"))
    mk = _finite(aggregate.get("mean_knee_flexion_deg"))
    mt = _finite(aggregate.get("mean_thigh_axis_from_image_down_deg"))
    me = _finite(aggregate.get("mean_leg_extension_ratio"))

    return {
        "sam3d_knee_authority_high": bool(
            knee_authority is not None and knee_authority >= MIN_SAM3D_KNEE_AUTHORITY
        ),
        "both_knees_bent": bool(
            lk is not None and rk is not None
            and lk <= MAX_EACH_KNEE_ANGLE_DEG
            and rk <= MAX_EACH_KNEE_ANGLE_DEG
        ),
        "not_extreme_bilateral_knee_flexion": bool(
            mk is not None and mk >= MIN_MEAN_KNEE_ANGLE_DEG
        ),
        "both_thighs_moderately_inclined": bool(
            lt is not None and rt is not None
            and MIN_EACH_THAI_FROM_DOWN_DEG <= lt <= MAX_EACH_THAI_FROM_DOWN_DEG
            and MIN_EACH_THAI_FROM_DOWN_DEG <= rt <= MAX_EACH_THAI_FROM_DOWN_DEG
        ),
        "mean_thigh_inclination_moderate": bool(
            mt is not None
            and MIN_MEAN_THAI_FROM_DOWN_DEG <= mt <= MAX_MEAN_THAI_FROM_DOWN_DEG
        ),
        "neither_leg_deeply_compacted": bool(
            le is not None and re is not None
            and le >= MIN_EACH_LEG_EXTENSION_RATIO
            and re >= MIN_EACH_LEG_EXTENSION_RATIO
        ),
        "mean_leg_compression_moderate": bool(
            me is not None
            and MIN_MEAN_LEG_EXTENSION_RATIO <= me <= MAX_MEAN_LEG_EXTENSION_RATIO
        ),
    }


def _depth_classification(
    *,
    left: dict[str, float | None],
    right: dict[str, float | None],
    aggregate: dict[str, float | None],
    gates: dict[str, bool],
) -> str:
    required = [
        left.get("knee_flexion_deg"),
        right.get("knee_flexion_deg"),
        left.get("thigh_axis_from_image_down_deg"),
        right.get("thigh_axis_from_image_down_deg"),
        left.get("leg_extension_ratio"),
        right.get("leg_extension_ratio"),
    ]
    if any(_finite(value) is None for value in required):
        return "unresolved"
    if all(gates.values()):
        return "moderate_lowering"

    mk = float(aggregate["mean_knee_flexion_deg"])
    mt = float(aggregate["mean_thigh_axis_from_image_down_deg"])
    me = float(aggregate["mean_leg_extension_ratio"])
    min_extension = min(float(left["leg_extension_ratio"]), float(right["leg_extension_ratio"]))

    if (
        mk < MIN_MEAN_KNEE_ANGLE_DEG
        or mt > MAX_MEAN_THAI_FROM_DOWN_DEG
        or me < MIN_MEAN_LEG_EXTENSION_RATIO
        or min_extension < MIN_EACH_LEG_EXTENSION_RATIO
    ):
        return "deeper_or_more_compressed"

    if (
        mt < MIN_MEAN_THAI_FROM_DOWN_DEG
        or me > MAX_MEAN_LEG_EXTENSION_RATIO
        or max(float(left["knee_flexion_deg"]), float(right["knee_flexion_deg"])) > MAX_EACH_KNEE_ANGLE_DEG
    ):
        return "shallow_or_mixed_crouch_geometry"

    return "crouch_depth_not_resolved"


def evaluate_shadow(
    *,
    image_key: str,
    sheet: dict[str, Any],
    projected: dict[str, Any] | None,
    specialist_error: str | None = None,
) -> dict[str, Any]:
    route = _route(sheet)
    canonical_pose = _canonical_pose(sheet)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "image_key": image_key,
        "shadow_only": True,
        "composer_authoritative": False,
        "status": "not_applicable",
        "route_mode": route,
        "canonical_broad_pose": canonical_pose,
        "sam3d_public_pose": None,
        "depth_classification": None,
        "would_qualify_hips_slightly_lowered": False,
        "reason": None,
        "lexicalization": {
            "base_predicate": BASE_LEXICALIZATION,
            "base_authority": "canonical_broad_pose_only",
            "qualified_predicate": QUALIFIED_LEXICALIZATION,
            "qualified_modifier": "hips_slightly_lowered",
            "qualified_modifier_authoritative": False,
            "promotion_candidate": False,
        },
        "thresholds": {
            "min_sam3d_knee_authority": MIN_SAM3D_KNEE_AUTHORITY,
            "max_each_knee_angle_deg": MAX_EACH_KNEE_ANGLE_DEG,
            "min_mean_knee_angle_deg": MIN_MEAN_KNEE_ANGLE_DEG,
            "min_each_thigh_from_down_deg": MIN_EACH_THAI_FROM_DOWN_DEG,
            "max_each_thigh_from_down_deg": MAX_EACH_THAI_FROM_DOWN_DEG,
            "min_mean_thigh_from_down_deg": MIN_MEAN_THAI_FROM_DOWN_DEG,
            "max_mean_thigh_from_down_deg": MAX_MEAN_THAI_FROM_DOWN_DEG,
            "min_each_leg_extension_ratio": MIN_EACH_LEG_EXTENSION_RATIO,
            "min_mean_leg_extension_ratio": MIN_MEAN_LEG_EXTENSION_RATIO,
            "max_mean_leg_extension_ratio": MAX_MEAN_LEG_EXTENSION_RATIO,
        },
    }

    if route not in POSE_BEARING_ROUTES:
        record.update(status="route_abstain", reason="route_outside_pose_bearing_scope")
        return record

    if canonical_pose != CANONICAL_POSE_REQUIRED:
        record.update(status="not_applicable", reason="canonical_broad_pose_not_crouching")
        return record

    # At this point the base lexicalization is safe because it is merely a
    # wording projection of an already-authoritative broad pose.
    record["status"] = "candidate_crouched_stance_only"
    record["reason"] = "canonical_crouching_supports_stance_explicit_base_lexicalization"

    if projected is None:
        record["depth_classification"] = "unresolved"
        record["reason"] = specialist_error or "sam3d_v16_specialist_unavailable"
        return record

    public_pose = str(projected.get("pose") or "uncertain").strip().lower()
    record["sam3d_public_pose"] = public_pose
    if public_pose != SAM3D_PUBLIC_POSE_REQUIRED:
        record["depth_classification"] = "unresolved"
        record["reason"] = "sam3d_public_pose_does_not_confirm_crouching"
        return record

    left = _side_metrics(projected, "left")
    right = _side_metrics(projected, "right")
    authority = _knee_authority(projected)
    aggregate = _aggregate(left, right)
    gates = _qualifier_gates(
        knee_authority=authority,
        left=left,
        right=right,
        aggregate=aggregate,
    )

    record["sam3d_knee_authority"] = authority
    record["per_side"] = {"left": left, "right": right}
    record["aggregate"] = aggregate
    record["qualifier_gates"] = gates
    record["depth_classification"] = _depth_classification(
        left=left,
        right=right,
        aggregate=aggregate,
        gates=gates,
    )

    qualifies = all(gates.values())
    record["would_qualify_hips_slightly_lowered"] = qualifies
    record["lexicalization"]["promotion_candidate"] = qualifies

    if qualifies:
        record.update(
            status="candidate_hips_slightly_lowered",
            reason=(
                "authoritative_crouching_plus_high_authority_bilateral_moderate_lower_body_geometry"
            ),
        )
    else:
        record["reason"] = "crouched_stance_supported_but_slight_depth_modifier_not_population_gated"

    return record


def _input_files(input_dir: Path, only: set[str]) -> list[Path]:
    paths = sorted(input_dir.glob("*.fact_sheet.json"))
    return [p for p in paths if not only or p.name.removesuffix(".fact_sheet.json") in only]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shadow-only crouched-stance lexical/depth qualifier."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else run_dir / DEFAULT_INPUT_SUBDIR
    output = args.output.expanduser().resolve() if args.output else run_dir / DEFAULT_OUTPUT_SUBDIR
    if not input_dir.is_dir():
        raise SystemExit(f"Fact-sheet directory not found: {input_dir}")
    output.mkdir(parents=True, exist_ok=True)

    only = set(args.only)
    paths = _input_files(input_dir, only)
    if only:
        found = {p.name.removesuffix(".fact_sheet.json") for p in paths}
        missing = sorted(only - found)
        if missing:
            raise SystemExit("Missing fact-sheet records: " + ", ".join(missing))
    if not paths:
        raise SystemExit(f"No fact sheets found in {input_dir}")

    records: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".fact_sheet.json")
        out_path = output / f"{key}.crouched_stance_depth_shadow.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        else:
            sheet = _read_json(path)
            projected, specialist_error = sam3d_shadow._load_specialist_projection(sheet)
            record = evaluate_shadow(
                image_key=key,
                sheet=sheet,
                projected=projected,
                specialist_error=specialist_error,
            )
            _write_json(out_path, record)
        records.append(record)

        if record.get("status") in {
            "candidate_crouched_stance_only",
            "candidate_hips_slightly_lowered",
        }:
            agg = record.get("aggregate") if isinstance(record.get("aggregate"), dict) else {}
            print(
                f"{key}: {record.get('status')} | "
                f"depth={record.get('depth_classification')} | "
                f"mean_thigh={agg.get('mean_thigh_axis_from_image_down_deg')} | "
                f"mean_knee={agg.get('mean_knee_flexion_deg')} | "
                f"mean_extension={agg.get('mean_leg_extension_ratio')}"
            )

    status_counts = Counter(str(record.get("status") or "unknown") for record in records)
    depth_counts = Counter(
        str(record.get("depth_classification") or "not_applicable") for record in records
    )
    qualifier_keys = [
        str(record.get("image_key"))
        for record in records
        if record.get("status") == "candidate_hips_slightly_lowered"
    ]
    crouch_keys = [
        str(record.get("image_key"))
        for record in records
        if record.get("status") in {
            "candidate_crouched_stance_only",
            "candidate_hips_slightly_lowered",
        }
    ]

    index = {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "depth_classification_counts": dict(sorted(depth_counts.items())),
        "authoritative_crouching_keys": crouch_keys,
        "hips_slightly_lowered_candidate_keys": qualifier_keys,
        "hips_slightly_lowered_candidate_count": len(qualifier_keys),
        "shadow_only": True,
        "fact_sheet_mutated": False,
        "composer_changed": False,
        "promotion_policy": (
            "Review the full population before promoting the optional depth modifier. "
            "The base 'holds a crouched stance' wording is a lexical projection of canonical crouching; "
            "the 'hips slightly lowered' modifier is not authoritative in v0.1."
        ),
    }
    index_path = output / "crouched_stance_depth_shadow.index.json"
    _write_json(index_path, index)

    print()
    print(f"Index: {index_path}")
    print(f"Status counts: {dict(sorted(status_counts.items()))}")
    print(f"Depth counts: {dict(sorted(depth_counts.items()))}")
    print(f"hips-slightly-lowered candidates ({len(qualifier_keys)}): {', '.join(qualifier_keys) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
