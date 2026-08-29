from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_05 as v05
from . import pose_semantics_07 as v07
from . import pose_semantics_08 as v08
from . import pose_semantics_09 as v09
from .runner import model_slug, resolve_model_id


_RECLINE_SCENE_RE = re.compile(
    r"\b(?:bed|bedding|bedspread|mattress|pillow|blanket|sheet|duvet|couch|sofa|"
    r"recliner|fabric|textile|soft surface)\b",
    re.I,
)
_REST_SUPPORT_RE = re.compile(
    r"\b(?:support(?:s|ed|ing)?|rest(?:s|ed|ing)?|lying|lies|lay|against)\b",
    re.I,
)
_BODY_SUPPORT_ACTOR_RE = re.compile(
    r"\b(?:head|neck|shoulders?|upper torso|torso|back|upper back|body|hips?|pelvis)\b",
    re.I,
)
_EXPLICIT_SUBJECT_RE = re.compile(r"\b(?:subject|target|torso|head|back|body|hips?|pelvis)\b", re.I)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _analysis_root(value: dict[str, Any]) -> dict[str, Any]:
    return v05._analysis_root(value)


def _strict_reclining_support_evidence(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only body-to-bed/soft-support relations that are explicit in Analyze.

    v0.9 intentionally broadened reclining beyond the seated-only support ontology,
    but its audit allowed an unrelated word such as ``fabric`` elsewhere in the
    scene to make a generic phrase such as ``standing on both feet`` look like
    reclining support.  v0.10 requires the soft/bed-like support to occur in the
    same body-support relation.  The image summary is never used to manufacture
    that relation.
    """
    evidence: list[dict[str, Any]] = []
    subject = analysis.get("target_subject") or {}

    for item in subject.get("interactions") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("evidence_status") or "observed").lower() not in {"observed", ""}:
            continue
        conf = v05._safe_float(item.get("confidence"), 0.0)
        if conf < 0.70:
            continue
        actor = str(item.get("actor_part") or "").replace("_", " ")
        target = str(item.get("target") or "")
        relation = str(item.get("type") or "").lower()
        notes = str(item.get("notes") or "")
        if not _BODY_SUPPORT_ACTOR_RE.search(actor):
            continue
        if relation not in {"support", "contact"}:
            continue
        relation_text = f"{target} {notes}"
        if not _RECLINE_SCENE_RE.search(relation_text):
            continue
        if relation != "support" and not _REST_SUPPORT_RE.search(notes):
            continue
        evidence.append({
            "source": "analysis.target_subject.interactions",
            "actor_part": actor,
            "target": target,
            "relation": relation,
            "notes": notes,
            "confidence": conf,
        })

    for item in subject.get("visible_body_parts") or []:
        if not isinstance(item, dict):
            continue
        if item.get("ownership") not in {None, "target"}:
            continue
        conf = v05._safe_float(item.get("confidence"), 0.0)
        if conf < 0.70:
            continue
        actor = str(item.get("part") or "").replace("_", " ")
        if not _BODY_SUPPORT_ACTOR_RE.search(actor):
            continue
        support_text = " ".join(str(item.get(field) or "") for field in ("support", "contact"))
        if not _RECLINE_SCENE_RE.search(support_text):
            continue
        if not _REST_SUPPORT_RE.search(support_text):
            continue
        evidence.append({
            "source": "analysis.target_subject.visible_body_parts",
            "actor_part": actor,
            "target": support_text,
            "confidence": conf,
        })

    for item in analysis.get("non_target_entities") or []:
        if not isinstance(item, dict):
            continue
        conf = v05._safe_float(item.get("confidence"), 0.0)
        if conf < 0.70:
            continue
        description = str(item.get("description") or "")
        relation = " ".join(str(item.get(field) or "") for field in ("contact", "support"))
        if not _RECLINE_SCENE_RE.search(description):
            continue
        if not _REST_SUPPORT_RE.search(relation):
            continue
        # The relation must explicitly refer to the target/body.  A fabric item
        # merely "resting on floor" is support for the fabric, not the person.
        if not _EXPLICIT_SUBJECT_RE.search(relation):
            continue
        evidence.append({
            "source": "analysis.non_target_entities",
            "actor_part": "target body",
            "target": description,
            "relation": relation,
            "confidence": conf,
        })

    return evidence


def _restore_v08_posture(result: dict[str, Any], baseline: dict[str, Any]) -> None:
    result["posture"] = baseline.get("posture") or {}
    result.setdefault("preferred_pose", {})["posture"] = (baseline.get("preferred_pose") or {}).get("posture")
    result["posture_candidate"] = baseline.get("posture_candidate")
    result["vetoed_posture_candidate"] = None
    probe = result.get("pose_gestalt_probe") or {}
    baseline_probe = baseline.get("pose_gestalt_probe") or {}
    probe["caption_preferred"] = bool(baseline_probe.get("caption_preferred"))
    probe["promotion_reason"] = baseline_probe.get("promotion_reason") or "withheld: strict reclining support verification failed"
    result["pose_gestalt_probe"] = probe


def _tighten_reclining_verification(
    result: dict[str, Any],
    baseline_v08: dict[str, Any],
    analysis_payload: dict[str, Any],
) -> None:
    analysis = _analysis_root(analysis_payload)
    strict = _strict_reclining_support_evidence(analysis)
    audit = result.get("pose_gestalt_corroboration") or {}
    route = str(audit.get("route") or "")

    old_audit = dict(audit)
    result["pose_gestalt_corroboration_v09"] = old_audit

    if route == "reclining_analyze_gestalt_plus_bedlike_support" and not strict:
        _restore_v08_posture(result, baseline_v08)
        audit = {
            **old_audit,
            "valid": False,
            "route": "reclining_strict_support_failed",
            "qualified_posture": (result.get("preferred_pose") or {}).get("posture"),
            "reclining_support_count": 0,
            "reclining_support": [],
            "v10_reclining_support_policy": (
                "bed/soft-support evidence must occur in the same explicit body-support relation; "
                "unrelated scene fabric and generic 'on' language cannot corroborate reclining"
            ),
        }
    else:
        audit = {
            **old_audit,
            "reclining_support_count": len(strict),
            "reclining_support": strict,
            "v10_reclining_support_policy": (
                "bed/soft-support evidence must occur in the same explicit body-support relation; "
                "unrelated scene fabric and generic 'on' language cannot corroborate reclining"
            ),
        }
    result["pose_gestalt_corroboration"] = audit


def _independent_surface_edges(result: dict[str, Any]) -> list[dict[str, Any]]:
    graph = result.get("support_graph") or {}
    out: list[dict[str, Any]] = []
    for edge in graph.get("body_support_edges") or []:
        if not isinstance(edge, dict):
            continue
        if edge.get("source") == v07._GESTALT_SOURCE:
            continue
        if edge.get("relation") != "support" or edge.get("target_class") != "surface":
            continue
        if v05._safe_float(edge.get("confidence"), 0.0) < 0.70:
            continue
        out.append(edge)
    return out


def _surface_name_compatible(a: Any, b: Any) -> bool:
    left = str(a or "surface").lower()
    right = str(b or "surface").lower()
    if left in {"", "surface", "unknown"} or right in {"", "surface", "unknown"}:
        return True
    return left == right


def _matching_independent_surface_edge(
    edges: list[dict[str, Any]],
    *,
    side: Any,
    part: Any,
    surface: Any,
    supported_lean: bool = False,
) -> dict[str, Any] | None:
    part = str(part or "").lower()
    allowed_parts = {part}
    if supported_lean:
        allowed_parts = {"elbow", "forearm", "wrist", "arm"}
    candidates: list[dict[str, Any]] = []
    for edge in edges:
        edge_side = edge.get("actor_side")
        if side in {"left", "right"} and edge_side in {"left", "right"} and edge_side != side:
            continue
        if str(edge.get("actor_part") or "").lower() not in allowed_parts:
            continue
        if not _surface_name_compatible(surface, edge.get("target")):
            continue
        candidates.append(edge)
    if not candidates:
        return None
    return max(candidates, key=lambda item: v05._safe_float(item.get("confidence"), 0.0))


def _harden_probe_surface_gestures(result: dict[str, Any]) -> None:
    """Do not caption a gestalt-only hand/arm-to-surface relation.

    DWPose visibility proves that the limb exists; it does not prove contact or
    load transfer.  A probe-derived support gesture therefore needs a matching
    non-gestalt support edge before it may reach caption-facing prose.
    """
    independent = _independent_surface_edges(result)
    graph = result.get("support_graph") or {}
    chains = graph.get("support_chains") or []

    for gesture in result.get("gestures") or []:
        if not gesture.get("caption_preferred"):
            continue
        support_text = " ".join(str(value) for value in (gesture.get("support") or [])).lower()
        if "pose_gestalt_" not in support_text:
            continue
        details = gesture.get("details") or {}
        kind = details.get("class")
        match = None
        if kind == "surface_support":
            match = _matching_independent_surface_edge(
                independent,
                side=details.get("actor_side"),
                part=details.get("part"),
                surface=details.get("surface"),
            )
        elif kind == "supported_lean":
            side = details.get("actor_side")
            chain = next((item for item in chains if isinstance(item, dict) and item.get("side") == side), None)
            match = _matching_independent_surface_edge(
                independent,
                side=side,
                part=(chain or {}).get("support_part"),
                surface=(chain or {}).get("surface") or details.get("surface"),
                supported_lean=True,
            )
        else:
            continue

        if match is not None:
            gesture.setdefault("support", []).append("v10_independent_surface_support_corroboration")
            continue

        gesture["caption_preferred"] = False
        limitations = gesture.setdefault("limitations", [])
        note = (
            "gestalt-only surface support is diagnostic, not caption-usable: limb visibility does not independently verify contact/support"
        )
        if note not in limitations:
            limitations.append(note)


def _surface_support_economy(result: dict[str, Any]) -> None:
    """Keep at most one caption-facing component support clause per surface.

    Analyze and Fusion can expose the same source fact as hand + arm and again
    through corrected laterality.  Repeating all of those clauses is anatomical
    word soup rather than useful pose semantics.
    """
    gestures = result.get("gestures") or []
    groups: dict[str, list[dict[str, Any]]] = {}
    for gesture in gestures:
        if not gesture.get("caption_preferred"):
            continue
        details = gesture.get("details") or {}
        if details.get("class") != "surface_support":
            continue
        surface = str(details.get("surface") or "surface").lower()
        groups.setdefault(surface, []).append(gesture)

    for surface, items in groups.items():
        if len(items) <= 1:
            continue
        best = max(items, key=v07._surface_support_rank)
        for item in items:
            if item is best:
                continue
            item["caption_preferred"] = False
            limitations = item.setdefault("limitations", [])
            note = f"redundant component support on the same {surface} subsumed by one higher-value support primitive"
            if note not in limitations:
                limitations.append(note)


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_v08 = v08.build_pose_semantics(dwpose, fused_payload, analysis_payload, gestalt_payload)
    result = v09.build_pose_semantics(dwpose, fused_payload, analysis_payload, gestalt_payload)

    _tighten_reclining_verification(result, baseline_v08, analysis_payload)
    _harden_probe_surface_gestures(result)
    _surface_support_economy(result)

    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in (result.get("gestures") or []) if item.get("caption_preferred")
    ]
    result["human_summary"] = base._human_summary(
        result.get("posture") or {},
        result.get("torso_orientation") or {},
        result.get("gestures") or [],
        result.get("head_and_gaze") or [],
        result.get("framing") or {},
    )
    result["schema_version"] = "pose-semantics-0.10"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "Reclining support must occur in the same explicit body-to-bed/soft-support relation; unrelated fabric elsewhere in the scene cannot create reclining evidence.",
        "Probe-only hand/arm surface support is withheld from caption prose unless a non-gestalt support edge independently corroborates the relation; DWPose visibility alone is insufficient.",
        "At most one component surface-support primitive per target surface remains caption-preferred, preventing duplicate hand/arm/laterality clauses from flooding the pose summary.",
        "Posture verification rules and the pose-gestalt prompt are unchanged from v0.9.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-10",
        description="Pose semantics v0.10: v0.9 posture freeze plus support-evidence and gesture-economy hardening.",
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
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.10" / slug)).expanduser().resolve()

    for path, label in (
        (fusion_dir, "Fusion"),
        (dwpose_dir, "DWPose"),
        (analysis_dir, "Analyze"),
        (gestalt_dir, "Pose gestalt"),
    ):
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
            "posture_candidate": (result.get("posture_candidate") or {}).get("label") if isinstance(result.get("posture_candidate"), dict) else None,
            "verification_route": (result.get("pose_gestalt_corroboration") or {}).get("route"),
            "human_summary": result.get("human_summary"),
        })

    index = {
        "schema_version": "pose-semantics-0.10-run",
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
    print(f"Pose semantics v0.10: {output_dir}")
    for record in records:
        suffix = f" [candidate={record.get('posture_candidate')}]" if record.get("posture_candidate") else ""
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}{suffix}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
