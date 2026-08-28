from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_05 as v05
from . import pose_semantics_06 as v06
from .runner import model_slug, resolve_model_id


_GESTALT_SOURCE = "pose_gestalt_v1.support_configuration"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _edge_is_support(edge: dict[str, Any]) -> bool:
    return (
        edge.get("relation") == "support"
        and edge.get("target_class") in {"surface", "seat"}
        and v05._safe_float(edge.get("confidence"), 0.0) >= 0.70
    )


def _probe_observed_edge(edge: dict[str, Any]) -> bool:
    return edge.get("source") == _GESTALT_SOURCE and edge.get("authority") == "pose_gestalt_observed_support"


def _bottom_up_edge(edge: dict[str, Any]) -> bool:
    return edge.get("source") != _GESTALT_SOURCE


def _cross_source_chain(graph: dict[str, Any], chain: dict[str, Any]) -> bool:
    side = chain.get("side")
    support_part = chain.get("support_part")
    surface = chain.get("surface")
    independent_head = any(
        head.get("actor_side") == side
        and not str(head.get("authority") or "").startswith("pose_gestalt_")
        and v05._safe_float(head.get("confidence"), 0.0) >= 0.70
        for head in graph.get("head_hand_edges") or []
    )
    if not independent_head:
        return False

    matching_probe_edge = any(
        edge.get("source") == _GESTALT_SOURCE
        and edge.get("actor_side") == side
        and edge.get("actor_part") == support_part
        and edge.get("target_class") == "surface"
        and edge.get("target") == surface
        and _edge_is_support(edge)
        for edge in graph.get("body_support_edges") or []
    )
    return matching_probe_edge


def _corroboration_audit(result: dict[str, Any]) -> dict[str, Any]:
    graph = result.get("support_graph") or {}
    edges = [edge for edge in (graph.get("body_support_edges") or []) if isinstance(edge, dict) and _edge_is_support(edge)]
    bottom_up = [edge for edge in edges if _bottom_up_edge(edge)]
    probe_observed = [edge for edge in edges if _probe_observed_edge(edge)]
    cross_source = [
        chain for chain in (graph.get("support_chains") or [])
        if isinstance(chain, dict) and _cross_source_chain(graph, chain)
    ]

    valid = bool(bottom_up or probe_observed or cross_source)
    return {
        "valid": valid,
        "bottom_up_support_count": len(bottom_up),
        "probe_observed_support_count": len(probe_observed),
        "cross_source_support_chain_count": len(cross_source),
        "policy": (
            "seated gestalt requires bottom-up support, probe-observed support, or a cross-source support chain; "
            "probe-contextual support alone cannot self-confirm posture"
        ),
    }


def _restore_baseline_posture(result: dict[str, Any], baseline: dict[str, Any]) -> None:
    result["posture"] = baseline.get("posture") or {}
    result.setdefault("preferred_pose", {})["posture"] = (baseline.get("preferred_pose") or {}).get("posture")
    probe = result.get("pose_gestalt_probe") or {}
    probe["caption_preferred"] = False
    probe["promotion_reason"] = "withheld: top-down posture lacked non-circular support corroboration"
    result["pose_gestalt_probe"] = probe


def _source_target_text(edge: dict[str, Any]) -> str:
    source_text = edge.get("source_text") or {}
    if isinstance(source_text, dict):
        value = source_text.get("target") or source_text.get("text") or ""
    else:
        value = source_text
    return str(value or "").lower()


def _surface_phrase_from_edge(edge: dict[str, Any]) -> str:
    target = str(edge.get("target") or "surface").lower()
    if target not in {"surface", "unknown", ""}:
        return target

    text = _source_target_text(edge)
    has_table = bool(re.search(r"\btable(?:top)?\b", text))
    has_desk = bool(re.search(r"\bdesk(?:top)?\b", text))
    has_counter = bool(re.search(r"\bcounter(?:top)?\b", text))
    if has_table and has_desk:
        return "table or desk"
    if has_table:
        return "table"
    if has_desk:
        return "desk"
    if has_counter:
        return "counter"
    return "surface"


def _matching_chain_edge(graph: dict[str, Any], chain: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for edge in graph.get("body_support_edges") or []:
        if not isinstance(edge, dict):
            continue
        if edge.get("actor_side") != chain.get("side"):
            continue
        if edge.get("actor_part") != chain.get("support_part"):
            continue
        if edge.get("target_class") != "surface" or not _edge_is_support(edge):
            continue
        candidates.append(edge)
    if not candidates:
        return None
    return max(candidates, key=lambda edge: v05._safe_float(edge.get("confidence"), 0.0))


def _enrich_supported_lean_labels(result: dict[str, Any]) -> None:
    graph = result.get("support_graph") or {}
    chains = graph.get("support_chains") or []
    for gesture in result.get("gestures") or []:
        details = gesture.get("details") or {}
        if details.get("class") != "supported_lean":
            continue
        side = details.get("actor_side")
        chain = next((item for item in chains if isinstance(item, dict) and item.get("side") == side), None)
        if not isinstance(chain, dict):
            continue
        edge = _matching_chain_edge(graph, chain)
        if edge is None:
            continue
        phrase = _surface_phrase_from_edge(edge)
        if phrase == "surface":
            continue
        gesture["label"] = f"leaning on the {side} arm at a {phrase}, with the chin resting on the hand"
        details["surface_phrase"] = phrase
        gesture["details"] = details
        chain["surface_phrase"] = phrase


def _authority_rank(item: dict[str, Any]) -> int:
    support = " ".join(str(value) for value in (item.get("support") or [])).lower()
    if "pose_gestalt_observed_support" in support:
        return 5
    if "governed_fusion_body_surface_relation" in support or "governed_fusion_body_part_support" in support:
        return 4
    if "analyze_observed" in support:
        return 3
    if "pose_gestalt_contextual_support" in support:
        return 1
    return 2


def _surface_support_rank(item: dict[str, Any]) -> tuple[int, int, int, float]:
    details = item.get("details") or {}
    part = str(details.get("part") or "")
    surface = str(details.get("surface") or "surface")
    part_rank = {
        "elbow": 6,
        "forearm": 6,
        "arm": 5,
        "hand": 3,
        "wrist": 2,
    }.get(part, 0)
    named_surface = 1 if surface not in {"", "surface", "unknown"} else 0
    return (
        _authority_rank(item),
        part_rank,
        named_surface,
        v05._safe_float(item.get("support_score"), 0.0),
    )


def _semantic_economy_v07(result: dict[str, Any]) -> None:
    posture = result.get("posture") or {}
    seated = posture.get("status") == "qualified" and posture.get("label") == "seated"
    gestures = result.get("gestures") or []
    if not seated:
        return

    has_supported_lean = any(
        (item.get("details") or {}).get("class") == "supported_lean" and item.get("caption_preferred")
        for item in gestures
    )
    surface_supports = [
        item for item in gestures
        if (item.get("details") or {}).get("class") == "surface_support"
        and v05._safe_float(item.get("support_score"), 0.0) >= 0.70
    ]

    if has_supported_lean:
        for item in surface_supports:
            item["caption_preferred"] = False
            limitations = item.setdefault("limitations", [])
            if "subsumed by higher-level supported-lean gesture" not in limitations:
                limitations.append("subsumed by higher-level supported-lean gesture")
    elif surface_supports:
        best = max(surface_supports, key=_surface_support_rank)
        for item in surface_supports:
            item["caption_preferred"] = item is best
            limitations = item.setdefault("limitations", [])
            if item is best:
                item["limitations"] = [
                    value for value in limitations
                    if "redundant same-surface support evidence" not in str(value)
                ]
            elif "redundant same-surface support evidence subsumed by seated posture" not in limitations:
                limitations.append("redundant same-surface support evidence subsumed by seated posture")

    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in gestures if item.get("caption_preferred")
    ]


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Keep an untouched v0.5 posture baseline so a top-down promotion can be
    # rolled back if v0.7 discovers that its apparent corroboration was circular.
    baseline = v05.build_pose_semantics(dwpose, fused_payload, analysis_payload)
    result = v06.build_pose_semantics(dwpose, fused_payload, analysis_payload, gestalt_payload)

    probe = result.get("pose_gestalt_probe") or {}
    if probe.get("caption_preferred") and probe.get("posture") == "seated":
        audit = _corroboration_audit(result)
        result["pose_gestalt_corroboration"] = audit
        if not audit["valid"]:
            _restore_baseline_posture(result, baseline)
    else:
        result["pose_gestalt_corroboration"] = _corroboration_audit(result)

    _enrich_supported_lean_labels(result)
    _semantic_economy_v07(result)
    result["human_summary"] = base._human_summary(
        result.get("posture") or {},
        result.get("torso_orientation") or {},
        result.get("gestures") or [],
        result.get("head_and_gaze") or [],
        result.get("framing") or {},
    )
    result["schema_version"] = "pose-semantics-0.7"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "Probe-contextual body-to-seat completion cannot self-confirm the probe's own seated posture.",
        "Seated gestalt requires bottom-up support, probe-observed support, or a cross-source support chain.",
        "When seated support evidence is redundant, the most informative observed forearm/elbow/named-surface primitive outranks generic hand-on-surface prose.",
        "Supported-lean labels preserve table/desk/counter specificity from the structured support source when the canonical support class is only 'surface'.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-07",
        description="Pose semantics v0.7: non-circular gestalt corroboration and semantic support ranking.",
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
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.7" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion"), (dwpose_dir, "DWPose"), (analysis_dir, "Analyze"), (gestalt_dir, "Pose gestalt")):
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
        "schema_version": "pose-semantics-0.7-run",
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
    print(f"Pose semantics v0.7: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
