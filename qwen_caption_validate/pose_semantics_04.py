from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_03 as v03
from .runner import model_slug, resolve_model_id


_STANDING_RE = re.compile(r"\b(?:stands?|standing|stood)\b", re.I)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _analysis_root(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("analysis")
    return nested if isinstance(nested, dict) else value


def _weight_bearing_standing_supported(result: dict[str, Any], analysis_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Recognize an ordinary asymmetric standing stance as a whole-body primitive.

    A standing person commonly keeps one knee nearly straight while the other is
    mildly flexed. Requiring both knees to cross an identical straightness
    threshold is therefore an anatomical ingredient rule, not a useful posture
    classifier. This refinement requires complete bilateral leg chains, both
    thighs descending mostly vertically, an upright torso, one near-straight
    knee, the other not deeply flexed, and independent Analyze standing semantics.
    """
    features = result.get("geometry_features") or {}
    connectivity = features.get("connectivity") or {}
    angles = features.get("angles_deg") or {}
    directions = features.get("directions_deg") or {}

    complete_legs = all(bool((connectivity.get(f"{side}_leg") or {}).get("complete")) for side in ("left", "right"))
    left_knee = _safe_float(angles.get("left_knee"))
    right_knee = _safe_float(angles.get("right_knee"))
    left_thigh = _safe_float(directions.get("left_thigh_from_horizontal"))
    right_thigh = _safe_float(directions.get("right_thigh_from_horizontal"))
    torso_axis = _safe_float(directions.get("torso_axis_from_vertical"))

    analysis = _analysis_root(analysis_payload)
    summary = str(analysis.get("image_summary") or "")
    semantic_standing = bool(_STANDING_RE.search(summary))

    knees_available = left_knee is not None and right_knee is not None
    thighs_available = left_thigh is not None and right_thigh is not None
    asymmetric_stance = bool(
        knees_available
        and max(left_knee, right_knee) >= 160.0
        and min(left_knee, right_knee) >= 140.0
    )
    thighs_descend = bool(thighs_available and min(left_thigh, right_thigh) >= 65.0)
    torso_upright = bool(torso_axis is not None and abs(torso_axis) <= 25.0)

    support: list[str] = []
    if complete_legs:
        support.append("both DWPose hip-knee-ankle chains are complete and in-frame")
    if asymmetric_stance:
        support.append(
            f"one knee is near-straight while the other is only mildly flexed ({left_knee:.1f}°, {right_knee:.1f}°)"
        )
    if thighs_descend:
        support.append(
            f"both thighs descend mostly vertically ({left_thigh:.1f}°, {right_thigh:.1f}° from horizontal)"
        )
    if torso_upright:
        support.append(f"torso axis is upright ({torso_axis:.1f}° from vertical)")
    if semantic_standing:
        support.append("Analyze independently reports standing")

    return all((complete_legs, asymmetric_stance, thighs_descend, torso_upright, semantic_standing)), support


def _promote_standing(result: dict[str, Any], analysis_payload: dict[str, Any]) -> dict[str, Any]:
    posture = result.get("posture") or {}
    if posture.get("status") == "qualified":
        return result

    supported, support = _weight_bearing_standing_supported(result, analysis_payload)
    if not supported:
        return result

    hypotheses = posture.get("hypotheses") or []
    standing = next((item for item in hypotheses if item.get("id") == "posture_standing"), None)
    if not isinstance(standing, dict):
        return result

    existing_support = [str(item) for item in (standing.get("support") or [])]
    for item in support:
        if item not in existing_support:
            existing_support.append(item)
    standing.update(
        support_score=max(0.82, float(standing.get("support_score") or 0.0)),
        confidence_band="strong",
        caption_preferred=True,
        support=existing_support,
    )
    limitations = [
        item for item in (standing.get("limitations") or [])
        if "knee straightness" not in str(item).lower()
    ]
    standing["limitations"] = limitations
    standing["subsumes"] = list(dict.fromkeys([
        *(standing.get("subsumes") or []),
        "mild asymmetric knee flexion used only as stance evidence",
    ]))

    result["posture"] = {
        "status": "qualified",
        "label": "standing",
        "primitive_id": "posture_standing",
        "support_score": standing["support_score"],
        "confidence_band": "strong",
        "support": standing["support"],
        "limitations": standing["limitations"],
        "subsumes": standing["subsumes"],
        "hypotheses": hypotheses,
        "authority": "top_down_weight_bearing_stance_plus_analyze_semantics",
    }
    result.setdefault("preferred_pose", {})["posture"] = "standing"
    result["human_summary"] = base._human_summary(
        result["posture"],
        result.get("torso_orientation") or {},
        result.get("gestures") or [],
        result.get("head_and_gaze") or [],
        result.get("framing") or {},
    )
    return result


def _install() -> None:
    v03._install()


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    _install()
    result = v03.build_pose_semantics(dwpose, fused_payload, analysis_payload)
    result = _promote_standing(result, analysis_payload)
    result["schema_version"] = "pose-semantics-0.4"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "Ordinary asymmetric standing stances are recognized top-down when both complete legs descend vertically, one knee is near-straight, the other only mildly flexed, the torso is upright, and Analyze independently reports standing.",
        "Mild knee asymmetry used to establish standing is subsumed by the whole-body posture primitive rather than emitted as limb-detail prose.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-04",
        description="Pose semantics v0.4: top-down asymmetric weight-bearing standing refinement.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    _install()
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.5" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    analysis_dir = run_dir / slug
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.4" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion"), (dwpose_dir, "DWPose"), (analysis_dir, "Analyze")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_paths = sorted(fusion_dir.glob("*.fused_v2_3.json"))
    if args.only:
        needles = tuple(args.only)
        fusion_paths = [path for path in fusion_paths if any(needle in path.name for needle in needles)]

    records: list[dict[str, Any]] = []
    for fusion_path in fusion_paths:
        key = fusion_path.name.removesuffix(".fused_v2_3.json")
        out_json = output_dir / f"{key}.pose_semantics.json"
        out_txt = output_dir / f"{key}.pose_semantics.txt"
        if out_json.exists() and out_txt.exists() and not args.overwrite:
            result = _read(out_json)
            records.append({"image_key": key, "status": "reused", "human_summary": result.get("human_summary")})
            continue

        dw_path = dwpose_dir / f"{key}.dwpose.json"
        analysis_path = analysis_dir / f"{key}.analysis.json"
        if not dw_path.is_file() or not analysis_path.is_file():
            records.append({"image_key": key, "status": "missing_source"})
            continue

        result = build_pose_semantics(_read(dw_path), _read(fusion_path), _read(analysis_path))
        result.update({
            "image_key": key,
            "source_paths": {
                "fusion": str(fusion_path),
                "dwpose": str(dw_path),
                "analysis": str(analysis_path),
            },
        })
        _write(out_json, result)
        out_txt.write_text(str(result.get("human_summary") or "") + "\n", encoding="utf-8")
        records.append({
            "image_key": key,
            "status": "written",
            "posture": (result.get("preferred_pose") or {}).get("posture"),
            "human_summary": result.get("human_summary"),
        })

    index = {
        "schema_version": "pose-semantics-0.4-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "records": records,
    }
    _write(output_dir / "pose_semantics.index.json", index)
    print(f"Pose semantics v0.4: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
