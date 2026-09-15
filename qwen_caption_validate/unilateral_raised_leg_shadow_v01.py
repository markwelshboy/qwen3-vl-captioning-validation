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

SCHEMA_VERSION = "unilateral-raised-leg-shadow-0.1"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.12"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "unilateral-raised-leg-shadow-v01"

ADMISSIBLE_ROUTES = {"pose_allowed", "pose_guided"}
PUBLIC_POSE_REQUIRED = "standing"

# Deliberately conservative first-pass gates. These express a distinctive
# unilateral raised-thigh topology rather than trying to infer support/contact.
MAX_THAI_HORIZONTAL_OFFSET_DEG = 20.0
MAX_RAISED_KNEE_FLEXION_DEG = 100.0
MAX_RAISED_LEG_EXTENSION_RATIO = 0.70
MIN_OPPOSITE_KNEE_EXTENSION_DEG = 145.0
MIN_OPPOSITE_LEG_EXTENSION_RATIO = 0.85
MAX_OPPOSITE_THAI_FROM_DOWN_DEG = 30.0
MIN_ANKLE_ELEVATION_SHOULDER_WIDTHS = 0.75

KNEE_RAISED_RE = re.compile(r"\bknee\b.{0,28}\braised\b|\braised\b.{0,28}\bknee\b", re.I)
HIGH_RE = re.compile(r"\b(?:high|highly|near(?:ly)?\s+horizontal|horizontal)\b", re.I)


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


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))


def _round(value: Any, digits: int = 3) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _load_dwpose_points(sheet: dict[str, Any]) -> tuple[dict[str, tuple[float, float] | None] | None, str | None]:
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
        ps = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
        dwpose_text = ps.get("dwpose")
        if not isinstance(size, list) or len(size) < 2:
            return None, "image_size_missing"
        if not dwpose_text:
            return None, "dwpose_source_missing"
        dwpose_path = Path(str(dwpose_text)).expanduser()
        if not dwpose_path.is_file():
            return None, "dwpose_source_not_found"
        return _dwpose_points(_read_json(dwpose_path), int(size[0]), int(size[1])), None
    except Exception as exc:  # pragma: no cover
        return None, f"dwpose_observation_failed:{type(exc).__name__}"


def _side_metrics(
    projected: dict[str, Any],
    points: dict[str, tuple[float, float] | None],
    side: str,
) -> dict[str, Any]:
    geometry = projected.get("geometry") if isinstance(projected.get("geometry"), dict) else {}
    asym = geometry.get("asymmetric_lower_body") if isinstance(geometry.get("asymmetric_lower_body"), dict) else {}
    per_side = asym.get("per_side") if isinstance(asym.get("per_side"), dict) else {}
    sam = per_side.get(side) if isinstance(per_side.get(side), dict) else {}

    other = "right" if side == "left" else "left"
    candidate_ankle = points.get(f"{side}_ankle")
    opposite_ankle = points.get(f"{other}_ankle")
    ls, rs = points.get("left_shoulder"), points.get("right_shoulder")
    shoulder_width = _distance(ls, rs)
    ankle_elevation_px = None
    ankle_elevation_sw = None
    if candidate_ankle is not None and opposite_ankle is not None:
        # Image y increases downward. Positive means candidate ankle is higher.
        ankle_elevation_px = float(opposite_ankle[1] - candidate_ankle[1])
        if shoulder_width is not None and shoulder_width > 1e-8:
            ankle_elevation_sw = ankle_elevation_px / shoulder_width

    thigh_from_down = _round(sam.get("thigh_axis_from_image_down_deg"), 3)
    thigh_horizontal_offset = (
        abs(abs(float(thigh_from_down)) - 90.0) if thigh_from_down is not None else None
    )

    return {
        "side": side,
        "dwpose": {
            "hip_observed": points.get(f"{side}_hip") is not None,
            "knee_observed": points.get(f"{side}_knee") is not None,
            "ankle_observed": candidate_ankle is not None,
            "opposite_ankle_observed": opposite_ankle is not None,
            "shoulders_observed": ls is not None and rs is not None,
            "ankle_elevation_px": _round(ankle_elevation_px, 1),
            "ankle_elevation_shoulder_widths": _round(ankle_elevation_sw, 3),
        },
        "sam3d": {
            "knee_flexion_deg": _round(sam.get("knee_flexion_deg"), 3),
            "hip_flexion_deg": _round(sam.get("hip_flexion_deg"), 3),
            "thigh_axis_from_image_down_deg": thigh_from_down,
            "thigh_horizontal_offset_deg": _round(thigh_horizontal_offset, 3),
            "leg_extension_ratio": _round(sam.get("leg_extension_ratio"), 3),
        },
    }


def _qualifies_candidate(candidate: dict[str, Any], opposite: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    cdw = candidate["dwpose"]
    cs = candidate["sam3d"]
    os = opposite["sam3d"]

    observed = bool(
        cdw.get("hip_observed")
        and cdw.get("ankle_observed")
        and cdw.get("opposite_ankle_observed")
        and cdw.get("shoulders_observed")
    )
    thigh_near_horizontal = bool(
        cs.get("thigh_horizontal_offset_deg") is not None
        and cs["thigh_horizontal_offset_deg"] <= MAX_THAI_HORIZONTAL_OFFSET_DEG
    )
    candidate_knee_bent = bool(
        cs.get("knee_flexion_deg") is not None
        and cs["knee_flexion_deg"] <= MAX_RAISED_KNEE_FLEXION_DEG
    )
    candidate_compact = bool(
        cs.get("leg_extension_ratio") is not None
        and cs["leg_extension_ratio"] <= MAX_RAISED_LEG_EXTENSION_RATIO
    )
    opposite_knee_extended = bool(
        os.get("knee_flexion_deg") is not None
        and os["knee_flexion_deg"] >= MIN_OPPOSITE_KNEE_EXTENSION_DEG
    )
    opposite_leg_extended = bool(
        os.get("leg_extension_ratio") is not None
        and os["leg_extension_ratio"] >= MIN_OPPOSITE_LEG_EXTENSION_RATIO
    )
    opposite_thigh_vertical = bool(
        os.get("thigh_axis_from_image_down_deg") is not None
        and abs(float(os["thigh_axis_from_image_down_deg"])) <= MAX_OPPOSITE_THAI_FROM_DOWN_DEG
    )
    distal_elevation = bool(
        cdw.get("ankle_elevation_shoulder_widths") is not None
        and cdw["ankle_elevation_shoulder_widths"] >= MIN_ANKLE_ELEVATION_SHOULDER_WIDTHS
    )

    gates = {
        "candidate_observed_by_dwpose": observed,
        "candidate_thigh_near_horizontal": thigh_near_horizontal,
        "candidate_knee_strongly_flexed": candidate_knee_bent,
        "candidate_leg_compact": candidate_compact,
        "opposite_knee_extended": opposite_knee_extended,
        "opposite_leg_extended": opposite_leg_extended,
        "opposite_thigh_near_vertical": opposite_thigh_vertical,
        "candidate_distal_leg_substantially_elevated": distal_elevation,
    }
    return all(gates.values()), gates


def _already_has_specific_relation(texts: list[str], side: str) -> bool:
    side_re = re.compile(rf"\b{re.escape(side)}\b", re.I)
    return any(
        KNEE_RAISED_RE.search(text)
        and side_re.search(text)
        and HIGH_RE.search(text)
        for text in texts
    )


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
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "image_key": image_key,
        "status": "not_candidate",
        "shadow_only": True,
        "composer_authoritative": False,
        "would_change": False,
        "route_mode": route,
        "thresholds": {
            "max_thigh_horizontal_offset_deg": MAX_THAI_HORIZONTAL_OFFSET_DEG,
            "max_raised_knee_flexion_deg": MAX_RAISED_KNEE_FLEXION_DEG,
            "max_raised_leg_extension_ratio": MAX_RAISED_LEG_EXTENSION_RATIO,
            "min_opposite_knee_extension_deg": MIN_OPPOSITE_KNEE_EXTENSION_DEG,
            "min_opposite_leg_extension_ratio": MIN_OPPOSITE_LEG_EXTENSION_RATIO,
            "max_opposite_thigh_from_down_deg": MAX_OPPOSITE_THAI_FROM_DOWN_DEG,
            "min_ankle_elevation_shoulder_widths": MIN_ANKLE_ELEVATION_SHOULDER_WIDTHS,
        },
        "current_configuration_texts": configuration,
        "proposed_relation": None,
        "reason": None,
    }

    if route not in ADMISSIBLE_ROUTES:
        record.update(status="route_abstain", reason="route_outside_pose_bearing_shadow_scope")
        return record
    if projected is None:
        record.update(status="insufficient_evidence", reason=specialist_error or "sam3d_v16_specialist_unavailable")
        return record
    if dwpose_points is None:
        record.update(status="insufficient_evidence", reason=dwpose_error or "dwpose_observation_unavailable")
        return record

    public_pose = str(projected.get("pose") or "uncertain")
    record["sam3d_public_pose"] = public_pose
    if public_pose != PUBLIC_POSE_REQUIRED:
        record.update(status="not_candidate", reason="public_pose_not_standing")
        return record

    metrics = {side: _side_metrics(projected, dwpose_points, side) for side in ("left", "right")}
    results: dict[str, dict[str, Any]] = {}
    qualified: list[str] = []
    for side in ("left", "right"):
        other = "right" if side == "left" else "left"
        ok, gates = _qualifies_candidate(metrics[side], metrics[other])
        results[side] = {"qualifies": ok, "gates": gates, "metrics": metrics[side]}
        if ok:
            qualified.append(side)

    record["side_evaluations"] = results
    if len(qualified) != 1:
        record["reason"] = "no_unique_unilateral_raised_leg_candidate" if not qualified else "multiple_raised_leg_candidates"
        return record

    side = qualified[0]
    proposed = f"{side} knee raised high with thigh nearly horizontal"
    already = _already_has_specific_relation(configuration, side)
    record.update(
        status="candidate_already_present" if already else "candidate_would_change",
        would_change=not already,
        candidate_side=side,
        proposed_relation=proposed,
        reason="direct_unilateral_raised_leg_topology_from_dwpose_observability_plus_sam3d_per_leg_geometry",
    )
    return record


def _input_files(input_dir: Path, only: set[str]) -> list[Path]:
    paths = sorted(input_dir.glob("*.fact_sheet.json"))
    return [p for p in paths if not only or p.name.removesuffix(".fact_sheet.json") in only]


def main() -> int:
    parser = argparse.ArgumentParser(description="Shadow-only unilateral raised-leg specialist.")
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

    paths = _input_files(input_dir, set(args.only))
    if args.only:
        found = {p.name.removesuffix(".fact_sheet.json") for p in paths}
        missing = sorted(set(args.only) - found)
        if missing:
            raise SystemExit("Missing fact-sheet records: " + ", ".join(missing))
    if not paths:
        raise SystemExit(f"No fact sheets found in {input_dir}")

    records: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".fact_sheet.json")
        out_path = output / f"{key}.unilateral_raised_leg_shadow.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        else:
            sheet = _read_json(path)
            projected, specialist_error = sam3d_shadow._load_specialist_projection(sheet)
            points, dwpose_error = _load_dwpose_points(sheet)
            record = evaluate_shadow(
                image_key=key,
                sheet=sheet,
                projected=projected,
                dwpose_points=points,
                specialist_error=specialist_error,
                dwpose_error=dwpose_error,
            )
            _write_json(out_path, record)
        records.append(record)
        if str(record.get("status") or "").startswith("candidate_"):
            print(f"{key}: {record.get('status')} | {record.get('proposed_relation')}")

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    counts = Counter(str(r.get("status") or "unknown") for r in records)
    candidates = [r for r in records if str(r.get("status") or "").startswith("candidate_")]
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "candidate_count": len(candidates),
        "thresholds": {
            "max_thigh_horizontal_offset_deg": MAX_THAI_HORIZONTAL_OFFSET_DEG,
            "max_raised_knee_flexion_deg": MAX_RAISED_KNEE_FLEXION_DEG,
            "max_raised_leg_extension_ratio": MAX_RAISED_LEG_EXTENSION_RATIO,
            "min_opposite_knee_extension_deg": MIN_OPPOSITE_KNEE_EXTENSION_DEG,
            "min_opposite_leg_extension_ratio": MIN_OPPOSITE_LEG_EXTENSION_RATIO,
            "max_opposite_thigh_from_down_deg": MAX_OPPOSITE_THAI_FROM_DOWN_DEG,
            "min_ankle_elevation_shoulder_widths": MIN_ANKLE_ELEVATION_SHOULDER_WIDTHS,
        },
        "candidates": candidates,
    }
    _write_json(output / "unilateral_raised_leg_shadow.index.json", index)
    print(f"Index: {output / 'unilateral_raised_leg_shadow.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
