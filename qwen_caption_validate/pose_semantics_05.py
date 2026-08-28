from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_04 as v04
from .runner import model_slug, resolve_model_id


_SEATED_RE = re.compile(r"\b(?:sit(?:s|ting)?|sat|seated)\b", re.I)
_SURFACE_RE = re.compile(r"\b(tabletop|table|desktop|desk|countertop|counter|worktop|bar|ledge|surface)\b", re.I)
_SEAT_RE = re.compile(r"\b(chair|seat|bench|stool|sofa|couch)\b", re.I)
_BODY_RE = re.compile(r"\b(?:(left|right)[ _-]+)?(elbow|forearm|wrist|hand|arm|hip|pelvis|thigh|torso|body)\b", re.I)
_SUPPORT_WORD_RE = re.compile(r"\b(?:rest(?:s|ed|ing)?|support(?:s|ed|ing)?|lean(?:s|ed|ing)?|contact(?:s|ed|ing)?|touch(?:es|ed|ing)?|against|on)\b", re.I)


def _analysis_root(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("analysis")
    return nested if isinstance(nested, dict) else value


def _fusion_root(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("fusion")
    return nested if isinstance(nested, dict) else value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _body_entity(value: Any, side_hint: Any = None) -> tuple[str | None, str | None]:
    text = str(value or "").lower().replace("_", " ")
    match = _BODY_RE.search(text)
    if not match:
        return None, None
    side = match.group(1)
    if side not in {"left", "right"} and side_hint in {"left", "right"}:
        side = str(side_hint)
    part = match.group(2).lower()
    if part == "body":
        part = "torso"
    if part == "pelvis":
        part = "hip"
    return side, part


def _support_target(value: Any) -> tuple[str | None, str | None]:
    text = str(value or "").lower().replace("_", " ")
    seat = _SEAT_RE.search(text)
    if seat:
        return "seat", seat.group(1).lower()
    surface = _SURFACE_RE.search(text)
    if surface:
        raw = surface.group(1).lower()
        aliases = {
            "tabletop": "table",
            "desktop": "desk",
            "countertop": "counter",
            "worktop": "counter",
        }
        return "surface", aliases.get(raw, raw)
    return None, None


def _append_edge(edges: list[dict[str, Any]], edge: dict[str, Any]) -> None:
    key = (edge.get("actor_side"), edge.get("actor_part"), edge.get("target_class"), edge.get("target"), edge.get("relation"))
    for existing in edges:
        other = (existing.get("actor_side"), existing.get("actor_part"), existing.get("target_class"), existing.get("target"), existing.get("relation"))
        if key == other:
            if _safe_float(edge.get("confidence")) > _safe_float(existing.get("confidence")):
                existing.update(edge)
            return
    edges.append(edge)


def _edges_from_interactions(items: Any, *, source: str, governed: bool) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return edges
    for item in items:
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        if governed and state.get("selection_usable") is False:
            continue
        evidence_status = str(item.get("evidence_status") or "observed").lower()
        if not governed and evidence_status not in {"observed", ""}:
            continue
        confidence = _safe_float(item.get("confidence"), 0.0)
        if not governed and confidence < 0.70:
            continue
        relation = str(item.get("type") or "").lower()
        if relation not in {"support", "contact", "unknown"}:
            continue
        side_hint = state.get("qualified_actor_anatomical_side") or item.get("actor_anatomical_side")
        side, part = _body_entity(item.get("actor_part"), side_hint)
        target_class, target = _support_target(item.get("target"))
        if part is None or target is None:
            continue
        if item.get("actor_ownership") not in {None, "target"} and not governed:
            continue
        _append_edge(edges, {
            "actor_side": side,
            "actor_part": part,
            "target_class": target_class,
            "target": target,
            "relation": "support" if relation == "support" else "contact",
            "confidence": round(max(confidence, 0.75 if governed else confidence), 3),
            "authority": "governed_fusion_body_surface_relation" if governed else "analyze_observed_body_surface_relation",
            "source": source,
            "source_text": {
                "actor_part": item.get("actor_part"),
                "target": item.get("target"),
                "notes": item.get("notes"),
            },
        })
    return edges


def _edges_from_body_parts(items: Any, *, source: str, governed: bool) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return edges
    for item in items:
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        if governed and state.get("selection_usable") is False:
            continue
        if not governed and item.get("ownership") not in {None, "target"}:
            continue
        confidence = _safe_float(item.get("confidence"), 0.0)
        if not governed and confidence and confidence < 0.70:
            continue
        side_hint = state.get("qualified_anatomical_side") or item.get("anatomical_side")
        side, part = _body_entity(item.get("part"), side_hint)
        if part is None:
            continue
        for field in ("support", "contact", "geometry"):
            text = str(item.get(field) or "")
            if not text or not _SUPPORT_WORD_RE.search(text):
                continue
            target_class, target = _support_target(text)
            if target is None:
                continue
            _append_edge(edges, {
                "actor_side": side,
                "actor_part": part,
                "target_class": target_class,
                "target": target,
                "relation": "support" if field == "support" or re.search(r"\b(?:rest|support|lean)\w*\b", text, re.I) else "contact",
                "confidence": round(max(confidence, 0.75 if governed or confidence == 0.0 else confidence), 3),
                "authority": "governed_fusion_body_part_support" if governed else "analyze_observed_body_part_support",
                "source": source,
                "source_text": {"part": item.get("part"), "field": field, "text": text},
            })
    return edges


def _collect_text(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            out.extend(_collect_text(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(_collect_text(child))
    return out


def _scene_objects(analysis: dict[str, Any]) -> list[str]:
    text = " ".join(_collect_text(analysis.get("non_target_entities") or []))
    found: list[str] = []
    for pattern in (_SURFACE_RE, _SEAT_RE):
        for match in pattern.finditer(text):
            value = match.group(1).lower()
            if value not in found:
                found.append(value)
    return found


def _support_graph(result: dict[str, Any], fused_payload: dict[str, Any], analysis_payload: dict[str, Any]) -> dict[str, Any]:
    analysis = _analysis_root(analysis_payload)
    fusion = _fusion_root(fused_payload)
    edges: list[dict[str, Any]] = []
    sources = [
        _edges_from_interactions(fusion.get("qualified_interactions"), source="fusion.qualified_interactions", governed=True),
        _edges_from_body_parts(fusion.get("qualified_body_parts"), source="fusion.qualified_body_parts", governed=True),
        _edges_from_interactions(((analysis.get("target_subject") or {}).get("interactions")), source="analysis.target_subject.interactions", governed=False),
        _edges_from_body_parts(((analysis.get("target_subject") or {}).get("visible_body_parts")), source="analysis.target_subject.visible_body_parts", governed=False),
    ]
    for group in sources:
        for edge in group:
            _append_edge(edges, edge)

    head_supports: list[dict[str, Any]] = []
    for gesture in result.get("gestures") or []:
        details = gesture.get("details") or {}
        if details.get("class") != "head_support" or not gesture.get("caption_preferred"):
            continue
        side = details.get("actor_side")
        if side not in {"left", "right"}:
            continue
        head_supports.append({
            "actor_side": side,
            "relation": "head_supported_by_hand",
            "confidence": _safe_float(gesture.get("support_score"), 0.0),
            "authority": "pose_semantics_head_support",
            "gesture_id": gesture.get("id"),
        })

    chains: list[dict[str, Any]] = []
    for head in head_supports:
        side = head["actor_side"]
        for edge in edges:
            if edge.get("actor_side") != side:
                continue
            if edge.get("actor_part") not in {"elbow", "forearm", "wrist", "arm"}:
                continue
            if edge.get("target_class") != "surface":
                continue
            score = min(_safe_float(head.get("confidence")), _safe_float(edge.get("confidence")))
            if score < 0.60:
                continue
            chains.append({
                "type": "head_hand_arm_surface_support_chain",
                "side": side,
                "surface": edge.get("target"),
                "support_part": edge.get("actor_part"),
                "support_score": round(score, 3),
                "confidence_band": "strong" if score >= 0.80 else "moderate",
                "caption_preferred": True,
                "authority": "head_hand_support_plus_independent_arm_surface_support",
                "support": [
                    f"head/chin is supported by the {side} hand",
                    f"{side} {edge.get('actor_part')} is independently supported by {edge.get('target')}",
                ],
            })

    return {
        "schema_version": "support-graph-0.1",
        "surface_objects": _scene_objects(analysis),
        "body_support_edges": edges,
        "head_hand_edges": head_supports,
        "support_chains": chains,
        "policy": {
            "surface_presence_alone_does_not_establish_support": True,
            "support_chain_does_not_by_itself_establish_seated": True,
        },
    }


def _support_primitives(graph: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    consumed_edges: set[tuple[Any, ...]] = set()
    for chain in graph.get("support_chains") or []:
        side = str(chain.get("side"))
        surface = str(chain.get("surface") or "surface")
        score = _safe_float(chain.get("support_score"), 0.0)
        out.append(base._primitive(
            f"gesture_supported_lean_{side}_{surface}",
            f"leaning on the {side} arm at a {surface}, with the chin resting on the hand",
            score,
            support=list(chain.get("support") or []),
            subsumes=["head-on-hand contact", "elbow/forearm support detail", "component arm geometry"],
            details={"class": "supported_lean", "actor_side": side, "surface": surface},
        ))
        consumed_edges.add((side, chain.get("support_part"), surface))

    for edge in graph.get("body_support_edges") or []:
        key = (edge.get("actor_side"), edge.get("actor_part"), edge.get("target"))
        if key in consumed_edges or edge.get("target_class") != "surface":
            continue
        if edge.get("relation") != "support":
            continue
        part = str(edge.get("actor_part") or "body part")
        side = edge.get("actor_side")
        side_text = f"{side} " if side in {"left", "right"} else ""
        target = str(edge.get("target") or "surface")
        score = _safe_float(edge.get("confidence"), 0.0)
        if part not in {"elbow", "forearm", "wrist", "hand", "arm"} or score < 0.70:
            continue
        out.append(base._primitive(
            f"gesture_surface_support_{side or 'unknown'}_{part}_{target}",
            f"{side_text}{part} resting on the {target}",
            min(0.90, score),
            support=[str(edge.get("authority"))],
            subsumes=[f"{side_text}{part} contact/support component prose"],
            details={"class": "surface_support", "actor_side": side, "part": part, "surface": target},
        ))
    return out


def _apply_support_gestures(result: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    support_gestures = _support_primitives(graph)
    if not support_gestures:
        return result

    chain_sides = {
        (item.get("details") or {}).get("actor_side")
        for item in support_gestures
        if (item.get("details") or {}).get("class") == "supported_lean"
    }
    kept: list[dict[str, Any]] = []
    for item in result.get("gestures") or []:
        details = item.get("details") or {}
        if details.get("class") == "head_support" and details.get("actor_side") in chain_sides:
            continue
        kept.append(item)
    result["gestures"] = sorted(
        [*support_gestures, *kept],
        key=lambda item: _safe_float(item.get("support_score"), 0.0),
        reverse=True,
    )
    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in result["gestures"] if item.get("caption_preferred")
    ]
    return result


def _contextual_seated(result: dict[str, Any], graph: dict[str, Any], analysis_payload: dict[str, Any]) -> dict[str, Any]:
    analysis = _analysis_root(analysis_payload)
    summary = str(analysis.get("image_summary") or "")
    target_text = " ".join(_collect_text(analysis.get("target_subject") or {}))
    direct_seated = bool(_SEATED_RE.search(" ".join((summary, target_text))))

    seat_support = [
        edge for edge in graph.get("body_support_edges") or []
        if edge.get("target_class") == "seat" and edge.get("actor_part") in {"hip", "thigh", "torso"}
        and edge.get("relation") == "support"
    ]
    arm_surface_support = [
        edge for edge in graph.get("body_support_edges") or []
        if edge.get("target_class") == "surface" and edge.get("actor_part") in {"elbow", "forearm", "wrist", "arm"}
        and edge.get("relation") == "support"
    ]
    support_chains = list(graph.get("support_chains") or [])
    scene_objects = set(graph.get("surface_objects") or [])

    connectivity = (result.get("geometry_features") or {}).get("connectivity") or {}
    complete_legs = sum(bool((connectivity.get(f"{side}_leg") or {}).get("complete")) for side in ("left", "right"))
    framing = str((result.get("framing") or {}).get("label") or "")
    close_crop = framing in {"close-up", "medium close-up", "medium / waist-up"}

    score = 0.0
    support: list[str] = []
    if direct_seated:
        score += 0.55
        support.append("Analyze directly describes the target as seated/sitting")
    if seat_support:
        score += 0.70
        support.append("target pelvis/thigh/torso has an observed support relation to a seat/chair/bench")
    if arm_surface_support:
        score += 0.15
        support.append("an arm/forearm/elbow has an observed load-bearing support relation to a table/desk/surface")
    if support_chains:
        score += 0.10
        support.append("head-hand support continues through the same arm to a supporting surface")
    if scene_objects & {"chair", "seat", "bench", "stool", "sofa", "couch", "table", "desk"}:
        score += 0.05
        support.append("scene contains seating or table/desk furniture")
    if complete_legs == 0 and close_crop:
        score += 0.05
        support.append("lower-body geometric corroboration is unavailable because the crop is close")

    posture = result.get("posture") or {}
    current_label = posture.get("label") if posture.get("status") == "qualified" else None
    contradiction = current_label not in {None, "seated"}
    score = min(1.0, score)
    contextual = {
        "label": "seated" if score >= 0.40 else None,
        "support_score": round(score, 3),
        "confidence_band": "strong" if score >= 0.80 else "moderate" if score >= 0.65 else "weak" if score >= 0.40 else "withheld",
        "caption_preferred": bool(score >= 0.70 and not contradiction),
        "authority": "contextual_support_configuration",
        "support": support,
        "geometric_lower_body_status": "available" if complete_legs else "unavailable",
        "contradicted_by_existing_geometric_posture": contradiction,
        "policy": "contextual seated requires direct seated semantics plus support/context, or direct body-to-seat support; a table lean alone is not enough",
    }
    result["contextual_posture"] = contextual

    if contextual["caption_preferred"] and current_label is None:
        result["posture"] = {
            "status": "qualified",
            "label": "seated",
            "primitive_id": "posture_seated_contextual",
            "support_score": contextual["support_score"],
            "confidence_band": contextual["confidence_band"],
            "support": support,
            "limitations": ["lower-body geometry is not required for this contextual support classification"],
            "subsumes": ["component support/contact evidence used to establish seated context"],
            "hypotheses": posture.get("hypotheses") or [],
            "authority": "contextual_support_configuration",
        }
        result.setdefault("preferred_pose", {})["posture"] = "seated"
    return result


def _rebuild_summary(result: dict[str, Any]) -> None:
    result["human_summary"] = base._human_summary(
        result.get("posture") or {},
        result.get("torso_orientation") or {},
        result.get("gestures") or [],
        result.get("head_and_gaze") or [],
        result.get("framing") or {},
    )


def build_pose_semantics(dwpose: dict[str, Any], fused_payload: dict[str, Any], analysis_payload: dict[str, Any]) -> dict[str, Any]:
    result = v04.build_pose_semantics(dwpose, fused_payload, analysis_payload)
    graph = _support_graph(result, fused_payload, analysis_payload)
    result["support_graph"] = graph
    result = _apply_support_gestures(result, graph)
    result = _contextual_seated(result, graph, analysis_payload)
    _rebuild_summary(result)
    result["schema_version"] = "pose-semantics-0.5"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "Body-to-surface support is represented as an explicit support graph instead of isolated contact prose.",
        "A same-side head-on-hand plus forearm/elbow/wrist-to-surface chain collapses to a supported-lean gesture and consumes the lower-level contact ingredients.",
        "Contextual seated posture may be promoted without visible hips/knees only when direct seated semantics are corroborated by support/context, or when direct body-to-seat support exists.",
        "Missing lower-body geometry means unavailable corroboration, not a veto on contextual seated posture.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-05",
        description="Pose semantics v0.5: support graph, supported-lean compression, and contextual seated posture.",
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
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.5" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    analysis_dir = run_dir / slug
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.5" / slug)).expanduser().resolve()

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
        "schema_version": "pose-semantics-0.5-run",
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
    print(f"Pose semantics v0.5: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
