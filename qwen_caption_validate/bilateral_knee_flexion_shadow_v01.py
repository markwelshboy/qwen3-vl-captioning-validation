from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v09 as sam3d_shadow
from .caption_perception_policy import _dwpose_points

SCHEMA_VERSION = "bilateral-knee-flexion-shadow-0.1"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.11"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "bilateral-knee-flexion-shadow-v01"

# Reuse the router's existing semantic boundary for deep knee flexion rather
# than fitting a new threshold to the 00066 control.
MAX_DEEP_KNEE_ANGLE_DEG = 135.0
MIN_KNEE_REGION_AUTHORITY = 0.80
ADMISSIBLE_ROUTES = {"pose_allowed", "pose_guided"}

KNEE_TEXT_RE = re.compile(r"\bknees?\b", re.I)
BILATERAL_BENT_KNEES_RE = re.compile(
    r"\b(?:both\s+knees?|knees)\b.{0,24}\b(?:bent|flexed)\b",
    re.I,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _route(sheet: dict[str, Any]) -> str:
    policy = sheet.get("policy") if isinstance(sheet.get("policy"), dict) else {}
    return str(policy.get("mode") or "")


def _configuration_texts(sheet: dict[str, Any]) -> list[str]:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    out: list[str] = []
    for item in configuration:
        if not isinstance(item, dict):
            continue
        text = item.get("composer_text") or item.get("normalized_text") or item.get("text")
        if isinstance(text, str) and text.strip():
            out.append(" ".join(text.split()))
    return out


def _joint_angle(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
    c: tuple[float, float] | None,
) -> float | None:
    if a is None or b is None or c is None:
        return None
    ux, uy = float(a[0] - b[0]), float(a[1] - b[1])
    vx, vy = float(c[0] - b[0]), float(c[1] - b[1])
    denom = math.hypot(ux, uy) * math.hypot(vx, vy)
    if denom <= 1e-8:
        return None
    cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / denom))
    return math.degrees(math.acos(cosine))


def _round(value: Any, digits: int = 3) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _load_dwpose_observation(
    sheet: dict[str, Any],
) -> tuple[dict[str, tuple[float, float] | None] | None, str | None]:
    sources = sheet.get("sources") if isinstance(sheet.get("sources"), dict) else {}
    policy_text = sources.get("perception_policy")
    if not policy_text:
        return None, "perception_policy_source_missing"
    policy_path = Path(str(policy_text)).expanduser()
    if not policy_path.is_file():
        return None, "perception_policy_source_not_found"
    try:
        policy = _read_json(policy_path)
        size = policy.get("image_size")
        policy_sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
        dwpose_text = policy_sources.get("dwpose")
        if not isinstance(size, list) or len(size) < 2:
            return None, "image_size_missing"
        if not dwpose_text:
            return None, "dwpose_source_missing"
        dwpose_path = Path(str(dwpose_text)).expanduser()
        if not dwpose_path.is_file():
            return None, "dwpose_source_not_found"
        return _dwpose_points(_read_json(dwpose_path), int(size[0]), int(size[1])), None
    except Exception as exc:  # pragma: no cover - surfaced as diagnostic status
        return None, f"dwpose_observation_failed:{type(exc).__name__}"


def _dwpose_chain_summary(
    points: dict[str, tuple[float, float] | None],
    side: str,
) -> dict[str, Any]:
    hip = points.get(f"{side}_hip")
    knee = points.get(f"{side}_knee")
    ankle = points.get(f"{side}_ankle")
    return {
        "hip_observed": hip is not None,
        "knee_observed": knee is not None,
        "ankle_observed": ankle is not None,
        "full_chain_observed": hip is not None and knee is not None and ankle is not None,
        "knee_angle_2d_deg": _round(_joint_angle(hip, knee, ankle), 1),
    }


def evaluate_shadow(
    *,
    image_key: str,
    sheet: dict[str, Any],
    projected: dict[str, Any] | None,
    dwpose_points: dict[str, tuple[float, float] | None] | None,
    specialist_error: str | None = None,
    dwpose_error: str | None = None,
) -> dict[str, Any]:
    route = _route(sheet)
    configuration = _configuration_texts(sheet)
    knee_texts = [text for text in configuration if KNEE_TEXT_RE.search(text)]
    bilateral_already_present = any(BILATERAL_BENT_KNEES_RE.search(text) for text in knee_texts)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "image_key": image_key,
        "status": "not_candidate",
        "shadow_only": True,
        "composer_authoritative": False,
        "would_change": False,
        "route_mode": route,
        "thresholds": {
            "max_deep_knee_angle_deg": MAX_DEEP_KNEE_ANGLE_DEG,
            "min_knee_region_authority": MIN_KNEE_REGION_AUTHORITY,
            "require_both_full_dwpose_chains": True,
        },
        "current_knee_configuration_texts": knee_texts,
        "proposed_relation": None,
        "reason": None,
    }

    if route not in ADMISSIBLE_ROUTES:
        record.update(status="route_abstain", reason="route_outside_pose_bearing_shadow_scope")
        return record

    if projected is None:
        record.update(
            status="insufficient_evidence",
            reason=specialist_error or "sam3d_v16_specialist_unavailable",
        )
        return record
    if dwpose_points is None:
        record.update(
            status="insufficient_evidence",
            reason=dwpose_error or "dwpose_observation_unavailable",
        )
        return record

    left_dw = _dwpose_chain_summary(dwpose_points, "left")
    right_dw = _dwpose_chain_summary(dwpose_points, "right")
    both_full_chains = bool(left_dw["full_chain_observed"] and right_dw["full_chain_observed"])

    geometry = projected.get("geometry") if isinstance(projected.get("geometry"), dict) else {}
    region = projected.get("region_support") if isinstance(projected.get("region_support"), dict) else {}
    assertion = projected.get("assertion_authority") if isinstance(projected.get("assertion_authority"), dict) else {}

    left_sam = _round(geometry.get("left_knee_angle_deg"), 3)
    right_sam = _round(geometry.get("right_knee_angle_deg"), 3)
    knee_authority = _round(region.get("knees"), 4)

    both_angles_available = left_sam is not None and right_sam is not None
    both_deep = bool(
        both_angles_available
        and left_sam <= MAX_DEEP_KNEE_ANGLE_DEG
        and right_sam <= MAX_DEEP_KNEE_ANGLE_DEG
    )
    authority_ok = bool(
        knee_authority is not None and knee_authority >= MIN_KNEE_REGION_AUTHORITY
    )

    record["dwpose"] = {
        "left": left_dw,
        "right": right_dw,
        "both_full_chains_observed": both_full_chains,
    }
    record["sam3d_v16"] = {
        "public_pose": projected.get("pose"),
        "best_candidate_pose": projected.get("best_candidate_pose"),
        "left_knee_angle_deg": left_sam,
        "right_knee_angle_deg": right_sam,
        "knee_region_authority": knee_authority,
        "knee_region_authority_percent": (
            int(round(100.0 * knee_authority)) if knee_authority is not None else None
        ),
        "assertion_path": assertion.get("selected_path"),
    }
    record["gates"] = {
        "both_full_dwpose_chains_observed": both_full_chains,
        "both_sam3d_knee_angles_available": both_angles_available,
        "both_sam3d_knees_deep": both_deep,
        "knee_region_authority_sufficient": authority_ok,
    }

    qualifies = bool(both_full_chains and both_angles_available and both_deep and authority_ok)
    if qualifies:
        record["proposed_relation"] = "both knees bent"
        record["would_change"] = not bilateral_already_present
        record["status"] = "candidate_would_change" if record["would_change"] else "candidate_already_present"
        record["reason"] = (
            "both_observed_dwpose_leg_chains_plus_high_authority_sam3d_v16_bilateral_deep_knee_flexion"
        )
        return record

    failed: list[str] = []
    if not both_full_chains:
        failed.append("both_full_dwpose_chains_not_observed")
    if not both_angles_available:
        failed.append("bilateral_sam3d_knee_angles_unavailable")
    elif not both_deep:
        failed.append("one_or_both_sam3d_knees_above_deep_flexion_boundary")
    if not authority_ok:
        failed.append("knee_region_authority_below_threshold")
    record["reason"] = ";".join(failed) or "shadow_gate_not_satisfied"
    return record


def _input_files(input_dir: Path, only: set[str]) -> list[Path]:
    paths = sorted(input_dir.glob("*.fact_sheet.json"))
    if only:
        paths = [p for p in paths if p.name.removesuffix(".fact_sheet.json") in only]
    return paths


def _tsv_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return " | ".join(str(x) for x in value) if value else "-"
    return str(value)


def _tsv_row(record: dict[str, Any]) -> list[str]:
    dw = record.get("dwpose") if isinstance(record.get("dwpose"), dict) else {}
    left_dw = dw.get("left") if isinstance(dw.get("left"), dict) else {}
    right_dw = dw.get("right") if isinstance(dw.get("right"), dict) else {}
    sam = record.get("sam3d_v16") if isinstance(record.get("sam3d_v16"), dict) else {}
    return [
        _tsv_value(record.get("image_key")),
        _tsv_value(record.get("route_mode")),
        _tsv_value(record.get("status")),
        _tsv_value(sam.get("public_pose")),
        _tsv_value(left_dw.get("knee_angle_2d_deg")),
        _tsv_value(right_dw.get("knee_angle_2d_deg")),
        _tsv_value(sam.get("left_knee_angle_deg")),
        _tsv_value(sam.get("right_knee_angle_deg")),
        _tsv_value(sam.get("knee_region_authority_percent")),
        _tsv_value(dw.get("both_full_chains_observed")),
        _tsv_value(record.get("current_knee_configuration_texts")),
        _tsv_value(record.get("proposed_relation")),
        _tsv_value(record.get("would_change")),
        _tsv_value(record.get("reason")),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shadow-only bilateral knee-flexion specialist using DWPose observability and SAM3D-v16 3-D angles."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir
        else run_dir / DEFAULT_INPUT_SUBDIR
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / DEFAULT_OUTPUT_SUBDIR
    )
    if not input_dir.is_dir():
        raise SystemExit(f"Fact-sheet directory not found: {input_dir}")
    output.mkdir(parents=True, exist_ok=True)

    input_paths = _input_files(input_dir, set(args.only))
    if args.only:
        found = {p.name.removesuffix(".fact_sheet.json") for p in input_paths}
        missing = sorted(set(args.only) - found)
        if missing:
            raise SystemExit("Missing fact-sheet records: " + ", ".join(missing))
    if not input_paths:
        raise SystemExit(f"No fact sheets found in {input_dir}")

    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        image_key = input_path.name.removesuffix(".fact_sheet.json")
        out_path = output / f"{image_key}.bilateral_knee_flexion_shadow.json"
        if out_path.is_file() and not args.overwrite:
            records.append(_read_json(out_path))
            continue

        sheet = _read_json(input_path)
        projected, specialist_error = sam3d_shadow._load_specialist_projection(sheet)
        points, dwpose_error = _load_dwpose_observation(sheet)
        record = evaluate_shadow(
            image_key=image_key,
            sheet=sheet,
            projected=projected,
            dwpose_points=points,
            specialist_error=specialist_error,
            dwpose_error=dwpose_error,
        )
        record["sources"] = {
            "fact_sheet": str(input_path),
            "perception_policy": (
                (sheet.get("sources") or {}).get("perception_policy")
                if isinstance(sheet.get("sources"), dict)
                else None
            ),
        }
        _write_json(out_path, record)
        records.append(record)
        if record.get("status", "").startswith("candidate"):
            sam = record.get("sam3d_v16") or {}
            print(
                f"{image_key}: {record['status']} | "
                f"sam3d knees={sam.get('left_knee_angle_deg')}/{sam.get('right_knee_angle_deg')} "
                f"authority={sam.get('knee_region_authority_percent')}% | "
                f"current={record.get('current_knee_configuration_texts')}"
            )

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    counts = Counter(str(r.get("status") or "unknown") for r in records)
    candidates = [r for r in records if str(r.get("status") or "").startswith("candidate")]

    header = [
        "image_key", "route", "status", "public_pose",
        "dw_left_knee_deg", "dw_right_knee_deg",
        "sam3d_left_knee_deg", "sam3d_right_knee_deg",
        "knee_authority_percent", "both_dwpose_chains",
        "current_knee_texts", "proposed_relation", "would_change", "reason",
    ]
    tsv = ["\t".join(header)] + ["\t".join(_tsv_row(r)) for r in records]
    (output / "bilateral_knee_flexion_shadow.tsv").write_text(
        "\n".join(tsv) + "\n", encoding="utf-8"
    )

    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "candidate_count": len(candidates),
        "thresholds": {
            "max_deep_knee_angle_deg": MAX_DEEP_KNEE_ANGLE_DEG,
            "min_knee_region_authority": MIN_KNEE_REGION_AUTHORITY,
            "require_both_full_dwpose_chains": True,
        },
        "invariants": {
            "shadow_only": True,
            "no_fact_sheet_mutation": True,
            "dwpose_owns_direct_observability": True,
            "sam3d_v16_supplies_3d_knee_flexion": True,
            "both_individual_knees_must_cross_existing_deep_flexion_boundary": True,
            "missing_chain_is_not_treated_as_counterevidence_but_blocks_publication": True,
        },
        "candidates": [
            {
                "image_key": r.get("image_key"),
                "status": r.get("status"),
                "route_mode": r.get("route_mode"),
                "current_knee_configuration_texts": r.get("current_knee_configuration_texts"),
                "proposed_relation": r.get("proposed_relation"),
                "would_change": r.get("would_change"),
                "dwpose": r.get("dwpose"),
                "sam3d_v16": r.get("sam3d_v16"),
            }
            for r in candidates
        ],
    }
    _write_json(output / "bilateral_knee_flexion_shadow.index.json", index)
    print(f"\nStatus counts: {dict(sorted(counts.items()))}")
    print(f"Candidates: {len(candidates)}")
    print(f"Index: {output / 'bilateral_knee_flexion_shadow.index.json'}")
    print(f"TSV: {output / 'bilateral_knee_flexion_shadow.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
