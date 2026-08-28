from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_05 as v05
from .runner import model_slug, resolve_model_id


_SUPPORT_RELATIONS = {"supported_by", "resting_on", "leaning_on", "braced_on"}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _gestalt_root(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("gestalt")
    return nested if isinstance(nested, dict) else payload


def _gestalt_support_edges(gestalt: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for item in gestalt.get("support_configuration") or []:
        if not isinstance(item, dict):
            continue
        relation = str(item.get("relation") or "").lower()
        if relation not in _SUPPORT_RELATIONS:
            continue
        confidence = v05._safe_float(item.get("confidence"), 0.0)
        evidence_status = str(item.get("evidence_status") or "unknown").lower()
        if confidence < 0.70 or evidence_status == "unknown":
            continue
        side, part = v05._body_entity(item.get("body_part"))
        target_class, target = v05._support_target(item.get("target"))
        if part is None or target is None:
            continue
        authority = f"pose_gestalt_{evidence_status}_support"
        v05._append_edge(edges, {
            "actor_side": side,
            "actor_part": part,
            "target_class": target_class,
            "target": target,
            "relation": "support",
            "confidence": round(confidence, 3),
            "authority": authority,
            "source": "pose_gestalt_v1.support_configuration",
            "source_text": dict(item),
        })
    return edges


def _rebuild_support_chains(graph: dict[str, Any]) -> None:
    chains: list[dict[str, Any]] = []
    for head in graph.get("head_hand_edges") or []:
        side = head.get("actor_side")
        if side not in {"left", "right"}:
            continue
        for edge in graph.get("body_support_edges") or []:
            if edge.get("actor_side") != side:
                continue
            if edge.get("actor_part") not in {"elbow", "forearm", "wrist", "arm"}:
                continue
            if edge.get("target_class") != "surface" or edge.get("relation") != "support":
                continue
            head_score = v05._safe_float(head.get("confidence"), 0.0)
            edge_score = v05._safe_float(edge.get("confidence"), 0.0)
            score = min(head_score, edge_score)
            # Contextual gestalt support is allowed, but the complete chain must
            # still be strongly supported before it becomes caption-facing.
            if score < 0.75:
                continue
            chains.append({
                "type": "head_hand_arm_surface_support_chain",
                "side": side,
                "surface": edge.get("target"),
                "support_part": edge.get("actor_part"),
                "support_score": round(score, 3),
                "confidence_band": "strong" if score >= 0.80 else "moderate",
                "caption_preferred": True,
                "authority": "head_hand_support_plus_surface_support_including_top_down_gestalt",
                "support": [
                    f"head/chin is supported by the {side} hand",
                    f"{side} {edge.get('actor_part')} is supported by {edge.get('target')} ({edge.get('authority')})",
                ],
            })
    graph["support_chains"] = chains


def _augment_support_graph(result: dict[str, Any], gestalt: dict[str, Any]) -> None:
    graph = result.get("support_graph") or {}
    graph.setdefault("body_support_edges", [])
    added = _gestalt_support_edges(gestalt)
    for edge in added:
        v05._append_edge(graph["body_support_edges"], edge)
    graph["pose_gestalt_edges_added"] = added
    _rebuild_support_chains(graph)
    result["support_graph"] = graph


def _dedupe_gestures(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for item in items:
        label = str(item.get("label") or "")
        previous = by_label.get(label)
        if previous is None or v05._safe_float(item.get("support_score"), 0.0) > v05._safe_float(previous.get("support_score"), 0.0):
            by_label[label] = item
    return sorted(by_label.values(), key=lambda item: v05._safe_float(item.get("support_score"), 0.0), reverse=True)


def _refresh_support_gestures(result: dict[str, Any]) -> None:
    graph = result.get("support_graph") or {}
    result = v05._apply_support_gestures(result, graph)
    result["gestures"] = _dedupe_gestures(result.get("gestures") or [])
    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in result["gestures"] if item.get("caption_preferred")
    ]


def _probe_contextual_posture(result: dict[str, Any], gestalt: dict[str, Any]) -> dict[str, Any]:
    posture = str(gestalt.get("posture") or "unknown").lower()
    basis = str(gestalt.get("posture_basis") or "unknown").lower()
    confidence = v05._safe_float(gestalt.get("posture_confidence"), 0.0)
    current = result.get("posture") or {}
    current_label = current.get("label") if current.get("status") == "qualified" else None
    graph = result.get("support_graph") or {}

    support_edges = [
        edge for edge in graph.get("body_support_edges") or []
        if edge.get("relation") == "support" and edge.get("target_class") in {"surface", "seat"}
        and v05._safe_float(edge.get("confidence"), 0.0) >= 0.70
    ]
    direct_seat_edges = [
        edge for edge in support_edges
        if edge.get("target_class") == "seat" and edge.get("actor_part") in {"hip", "thigh", "torso"}
    ]
    support_chains = list(graph.get("support_chains") or [])
    contradiction = current_label not in {None, posture} if posture != "unknown" else False

    probe = {
        "posture": posture,
        "posture_basis": basis,
        "posture_confidence": round(confidence, 3),
        "semantic_pose_summary": gestalt.get("semantic_pose_summary"),
        "support_configuration": gestalt.get("support_configuration") or [],
        "evidence": gestalt.get("evidence") or [],
        "counterevidence": gestalt.get("counterevidence") or [],
        "contradicted_by_existing_geometric_posture": contradiction,
        "caption_preferred": False,
        "promotion_reason": None,
    }

    # v0.6 deliberately promotes only the contextual class that the bottom-up
    # pipeline is structurally weak at: seated in cropped portraits. Other probe
    # postures remain diagnostic until independently tested.
    if posture == "seated" and not contradiction:
        corroborated_support = bool(support_edges or support_chains or direct_seat_edges)
        if confidence >= 0.82 and basis in {"contextual", "mixed", "geometric"} and corroborated_support:
            probe["caption_preferred"] = True
            probe["promotion_reason"] = "strong top-down seated gestalt corroborated by support graph"
            if current_label is None:
                result["posture"] = {
                    "status": "qualified",
                    "label": "seated",
                    "primitive_id": "posture_seated_top_down_gestalt",
                    "support_score": round(confidence, 3),
                    "confidence_band": "strong" if confidence >= 0.85 else "moderate",
                    "support": [
                        "top-down pose gestalt identifies seated posture",
                        "support graph independently contains body-to-surface/seat support",
                    ],
                    "limitations": [
                        "lower-body joints may be cropped; seated authority is contextual/support-based rather than full lower-body geometry"
                    ],
                    "subsumes": ["component support evidence used only to establish seated posture"],
                    "hypotheses": current.get("hypotheses") or [],
                    "authority": "top_down_pose_gestalt_plus_support_graph",
                }
                result.setdefault("preferred_pose", {})["posture"] = "seated"
    result["pose_gestalt_probe"] = probe
    return result


def _semantic_economy(result: dict[str, Any]) -> None:
    """Once seated is qualified, suppress redundant same-surface support word soup."""
    posture = result.get("posture") or {}
    if posture.get("status") != "qualified" or posture.get("label") != "seated":
        return
    has_supported_lean = any(
        (item.get("details") or {}).get("class") == "supported_lean" and item.get("caption_preferred")
        for item in result.get("gestures") or []
    )
    support_gestures = [
        item for item in result.get("gestures") or []
        if (item.get("details") or {}).get("class") == "surface_support" and item.get("caption_preferred")
    ]
    if has_supported_lean:
        for item in support_gestures:
            item["caption_preferred"] = False
            item.setdefault("limitations", []).append("subsumed by higher-level supported-lean gesture")
    elif len(support_gestures) > 1:
        # Keep one strongest generic surface-support phrase rather than enumerating
        # hand + hand + arm evidence that all serves the same contextual posture.
        best = max(support_gestures, key=lambda item: v05._safe_float(item.get("support_score"), 0.0))
        for item in support_gestures:
            if item is best:
                continue
            item["caption_preferred"] = False
            item.setdefault("limitations", []).append("redundant same-surface support evidence subsumed by seated posture")
    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in result.get("gestures") or [] if item.get("caption_preferred")
    ]


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = v05.build_pose_semantics(dwpose, fused_payload, analysis_payload)
    gestalt = _gestalt_root(gestalt_payload)
    if gestalt:
        _augment_support_graph(result, gestalt)
        _refresh_support_gestures(result)
        result = _probe_contextual_posture(result, gestalt)
        _semantic_economy(result)
        result["human_summary"] = base._human_summary(
            result.get("posture") or {},
            result.get("torso_orientation") or {},
            result.get("gestures") or [],
            result.get("head_and_gaze") or [],
            result.get("framing") or {},
        )
    else:
        result["pose_gestalt_probe"] = {
            "status": "not_supplied",
            "caption_preferred": False,
        }
    result["schema_version"] = "pose-semantics-0.6"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "A separate top-down pose-gestalt probe can contribute structured support hypotheses without replacing the cached bottom-up Analyze/Fusion evidence.",
        "Strong seated gestalt is promoted only when it is not contradicted by qualified geometric posture and a support graph independently corroborates body-to-surface/seat support.",
        "The probe's natural semantic_pose_summary is retained for comparison but is not copied directly into caption-facing output.",
        "Once seated/support semantics are qualified, redundant same-surface hand/arm evidence is compressed rather than emitted as ingredient-list prose.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-06",
        description="Pose semantics v0.6: combine cached bottom-up evidence with a separate top-down pose gestalt probe.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--gestalt-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.5" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    analysis_dir = run_dir / slug
    gestalt_dir = (args.gestalt_dir or (run_dir / "pose-gestalt-v1" / slug)).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.6" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion"), (dwpose_dir, "DWPose"), (analysis_dir, "Analyze")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2
    if not gestalt_dir.is_dir():
        print(f"Pose gestalt directory not found: {gestalt_dir}", file=sys.stderr)
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
        gestalt_path = gestalt_dir / f"{key}.pose_gestalt.json"
        if not dw_path.is_file() or not analysis_path.is_file() or not gestalt_path.is_file():
            records.append({"image_key": key, "status": "missing_source"})
            continue

        result = build_pose_semantics(
            _read(dw_path),
            _read(fusion_path),
            _read(analysis_path),
            _read(gestalt_path),
        )
        result.update({
            "image_key": key,
            "source_paths": {
                "fusion": str(fusion_path),
                "dwpose": str(dw_path),
                "analysis": str(analysis_path),
                "pose_gestalt": str(gestalt_path),
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
        "schema_version": "pose-semantics-0.6-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "analysis_dir": str(analysis_dir),
        "gestalt_dir": str(gestalt_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "records": records,
    }
    _write(output_dir / "pose_semantics.index.json", index)
    print(f"Pose semantics v0.6: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
