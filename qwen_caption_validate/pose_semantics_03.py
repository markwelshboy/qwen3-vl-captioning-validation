from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_02 as v02
from .runner import model_slug, resolve_model_id


_BASE_ARM_GEOMETRY = base._arm_geometry_gestures


def _arm_geometry_gestures(
    features: dict[str, Any],
    fusion: dict[str, Any],
    interactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep arm direction diagnostic unless a higher-level semantic source agrees.

    A complete DWPose shoulder/elbow/wrist chain establishes projected 2-D arm
    direction, but by itself does not establish the human-level gesture "arm
    hanging at the side". Bilateral object interactions consume both arm slots,
    preventing contradictory fallback gestures such as "both hands holding a
    phone" plus "right arm hanging at the side".
    """
    gestures = _BASE_ARM_GEOMETRY(features, fusion, interactions)

    occupied_sides: set[str] = set()
    bilateral_consumes_both = False
    for item in interactions:
        details = item.get("details") or {}
        side = details.get("actor_side")
        if side in {"left", "right"}:
            occupied_sides.add(str(side))
        if details.get("class") == "bilateral_object_interaction":
            bilateral_consumes_both = True
    if bilateral_consumes_both:
        occupied_sides.update({"left", "right"})

    out: list[dict[str, Any]] = []
    for gesture in gestures:
        details = gesture.get("details") or {}
        side = details.get("actor_side")
        gesture_class = details.get("class")
        if side in occupied_sides and gesture_class in {"arm_down", "arm_raised"}:
            continue

        if gesture_class in {"arm_down", "arm_raised"}:
            support = [str(entry) for entry in (gesture.get("support") or [])]
            semantic_agreement = any("governed arm semantics agree" in entry.lower() for entry in support)
            if not semantic_agreement:
                old_score = float(gesture.get("support_score") or 0.0)
                gesture["support_score"] = round(min(old_score, 0.45), 3)
                gesture["confidence_band"] = "weak"
                gesture["caption_preferred"] = False
                gesture.setdefault("limitations", []).append(
                    "DWPose-only projected arm direction is diagnostic; human-level arm gesture requires independent semantic corroboration"
                )
        out.append(gesture)
    return out


def _install() -> None:
    v02._install()
    base._arm_geometry_gestures = _arm_geometry_gestures


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    _install()
    result = base.build_pose_semantics(dwpose, fused_payload, analysis_payload)
    result["schema_version"] = "pose-semantics-0.3"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "Bilateral object interactions consume both arm slots before fallback arm-geometry gestures are considered.",
        "DWPose-only projected arm direction remains diagnostic/WEAK and is not caption-preferred without independent governed semantic agreement.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-03",
        description="Pose semantics v0.3: semantic gesture consumption and conservative arm-direction fallback.",
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
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.3" / slug)).expanduser().resolve()

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
        "schema_version": "pose-semantics-0.3-run",
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
    print(f"Pose semantics v0.3: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
