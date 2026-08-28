from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import pose_semantics as base
from .runner import model_slug, resolve_model_id


_BASE_VISIBLE = base._visible
_BASE_INTERACTIONS = base._interaction_gestures
_BASE_FRAMING = base._framing_summary


def _visible_in_frame(point: np.ndarray | None) -> bool:
    """Pose-semantics visibility means observed inside the image, not extrapolated."""
    if not _BASE_VISIBLE(point):
        return False
    assert point is not None
    return bool(float(point[0]) <= 1.000001 and float(point[1]) <= 1.000001)


def _interaction_gestures(features: dict[str, Any], fusion: dict[str, Any]) -> list[dict[str, Any]]:
    gestures = _BASE_INTERACTIONS(features, fusion)

    # Preserve explicit bilateral/plural actors that v0.1 normalized down to
    # singular "hand" when anatomical side was intentionally unavailable.
    bilateral: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(fusion.get("qualified_interactions") or []):
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        if state.get("selection_usable") is False:
            continue
        actor = str(item.get("actor_part") or "").lower().replace("_", " ")
        kind = str(item.get("type") or "").lower()
        target = str(item.get("target") or "").lower().replace("_", " ").strip()
        if not target or not re.search(r"\b(?:both\s+hands|hands)\b", actor):
            continue
        if kind not in {"hold", "holding", "grip", "grasp", "carry", "carrying"}:
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        verb = "holding" if kind in {"hold", "holding", "grip", "grasp"} else "carrying"
        score = min(1.0, 0.62 + min(0.30, confidence * 0.30))
        bilateral[(target, verb)] = base._primitive(
            f"gesture_explicit_both_hands_{index}",
            f"both hands {verb} {target}",
            score,
            support=[f"qualified interaction explicitly uses plural/bilateral hand actor ({kind}, confidence {confidence:.2f})"],
            subsumes=["singular generic hand interaction", "component arm/hand geometry"],
            details={"class": "bilateral_object_interaction", "target": target, "verb": verb},
        )

    if bilateral:
        targets = set(bilateral)
        filtered: list[dict[str, Any]] = []
        for gesture in gestures:
            details = gesture.get("details") or {}
            if details.get("class") == "object_interaction":
                key = (str(details.get("target") or ""), str(details.get("verb") or "holding"))
                if key in targets:
                    continue
            filtered.append(gesture)
        gestures = list(bilateral.values()) + filtered

    # Body-target anatomical side is not independently established by a generic
    # contact correspondence. Keep source-side text only as diagnostic metadata;
    # human-facing support says simply "hip".
    for gesture in gestures:
        details = gesture.get("details") or {}
        if details.get("class") != "hand_on_hip":
            continue
        normalized: list[str] = []
        for entry in gesture.get("support") or []:
            normalized.append(re.sub(r"->\s+(?:left|right)[ _-]+hip\b", "-> hip", str(entry), flags=re.I))
        gesture["support"] = normalized

    return gestures


def _framing_summary(analysis: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    result = _BASE_FRAMING(analysis, features)
    framing = analysis.get("framing") or {}
    source_scale = str(framing.get("shot_scale") or "").lower()
    extent = str(framing.get("subject_extent") or "")
    pose_extent = str(features.get("pose_extent_hint") or "")

    close_pair = source_scale in {"close_up", "medium_close_up"} and pose_extent in {
        "close_or_medium_close", "face_or_partial_body"
    }
    if close_pair:
        old_label = result.get("label")
        result["label"] = "close-up" if source_scale == "close_up" else "medium close-up"
        extent_claims_long_crop = bool(
            re.search(
                r"\b(?:thighs?|knees?|calf|calves|ankles?|feet|foot|full[ -]?body)\b",
                extent,
                re.I,
            )
        )
        result["arbitration"] = {
            "status": "resolved_conflict" if extent_claims_long_crop else "sources_agree",
            "preferred_sources": ["source_shot_scale", "dwpose_extent_hint"],
            "suppressed_extent_label": old_label if old_label != result["label"] else None,
            "reason": (
                "generic subject_extent conflicts with both the VLM shot-scale label and independent DWPose visible-joint extent"
                if extent_claims_long_crop
                else "VLM shot-scale and DWPose extent independently agree on a close crop"
            ),
        }
    return result


def _install() -> None:
    base._visible = _visible_in_frame
    base._interaction_gestures = _interaction_gestures
    base._framing_summary = _framing_summary


def build_pose_semantics(dwpose: dict[str, Any], fused_payload: dict[str, Any], analysis_payload: dict[str, Any]) -> dict[str, Any]:
    _install()
    result = base.build_pose_semantics(dwpose, fused_payload, analysis_payload)
    result["schema_version"] = "pose-semantics-0.2"
    result["status"] = "experimental_report_only"
    result["refinements"] = [
        "DWPose joints count as observed only when their normalized coordinates lie inside the image frame.",
        "Close-crop framing uses agreement between source shot_scale and DWPose visible-joint extent to veto contradictory long-crop subject_extent prose.",
        "Explicit plural/bilateral hand actors are preserved as both-hands object interactions.",
        "Body-contact target anatomical side remains neutral in human-facing gesture support unless independently qualified elsewhere.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-02",
        description="Pose semantics v0.2: in-frame DWPose visibility plus framing/bilateral interaction arbitration.",
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
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.2" / slug)).expanduser().resolve()

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
            "source_paths": {"fusion": str(fusion_path), "dwpose": str(dw_path), "analysis": str(analysis_path)},
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
        "schema_version": "pose-semantics-0.2-run",
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
    print(f"Pose semantics v0.2: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
