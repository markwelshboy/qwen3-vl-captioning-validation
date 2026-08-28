from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

from .laterality_geometry import _read, _write
from .runner import model_slug, resolve_model_id


_BODY_TOKEN_RE = re.compile(
    r"\b(?:(?P<side>left|right)[ _-]+)?(?P<part>"
    r"chin|face|head|shoulder|upper[ _-]+arm|forearm|wrist|hand|fingers?|"
    r"torso|body|chest|abdomen|hip|pelvis|thigh|knee|lower[ _-]+leg|calf|shin|leg|ankle|feet|foot"
    r")\b",
    re.IGNORECASE,
)
_RELATION_GEOMETRY_RE = re.compile(
    r"\b(?:rest(?:s|ed|ing)?|support(?:s|ed|ing)?|touch(?:es|ed|ing)?|contact(?:s|ed|ing)?|"
    r"press(?:es|ed|ing)?|lean(?:s|ed|ing)?|hold(?:s|ing)?|grip(?:s|ped|ping)?|against|near)\b",
    re.IGNORECASE,
)
_BILATERAL_PARTS = {
    "arm", "upper arm", "forearm", "wrist", "hand", "shoulder", "hip", "thigh", "knee", "lower leg", "leg", "ankle", "foot"
}


def _norm_part(value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "chin": "head",
        "face": "head",
        "chest": "torso",
        "abdomen": "torso",
        "body": "torso",
        "pelvis": "hip",
        "finger": "hand",
        "fingers": "hand",
        "feet": "foot",
        "calf": "lower leg",
        "shin": "lower leg",
    }
    return aliases.get(text, text)


def _entities_from_text(value: Any) -> list[tuple[str | None, str]]:
    text = str(value or "")
    out: list[tuple[str | None, str]] = []
    for match in _BODY_TOKEN_RE.finditer(text):
        side = match.group("side").lower() if match.group("side") else None
        out.append((side, _norm_part(match.group("part"))))
    return out


def _entity_from_text(value: Any) -> tuple[str | None, str] | None:
    values = _entities_from_text(value)
    return values[0] if values else None


def _visible_landmarks(dw: dict[str, Any]) -> set[str]:
    target = ((dw.get("derived") or {}).get("target") or {})
    return {str(value) for value in (target.get("visible_body_landmarks") or [])}


def _connectivity(dw: dict[str, Any]) -> dict[str, Any]:
    return (((dw.get("derived") or {}).get("target") or {}).get("connectivity") or {})


def _same_side_segment_visible(visible: set[str], side: str, names: tuple[str, ...], minimum: int) -> bool:
    return sum(f"{side}_{name}" in visible for name in names) >= minimum


def _hand_observed(dw: dict[str, Any], side: str | None) -> bool:
    visible = _visible_landmarks(dw)
    conn = _connectivity(dw)
    sides = (side,) if side in {"left", "right"} else ("left", "right")
    for candidate_side in sides:
        arm = conn.get(f"{candidate_side}_arm") or {}
        if f"{candidate_side}_wrist" in visible and int(arm.get("visible_count") or 0) >= 2:
            return True
    for item in ((dw.get("derived") or {}).get("hand_candidates") or []):
        if not item.get("supported_by_nearby_visible_target_wrist"):
            continue
        candidate_side = item.get("nearest_visible_target_wrist")
        if side is None or candidate_side == side:
            return True
    return False


def _body_entity_observed(dw: dict[str, Any], side: str | None, part: str) -> bool:
    visible = _visible_landmarks(dw)
    sides = (side,) if side in {"left", "right"} else ("left", "right")

    if part == "head":
        head_marks = {"nose", "neck", "right_eye", "left_eye", "right_ear", "left_ear"}
        return len(visible & head_marks) >= 2
    if part == "torso":
        return "neck" in visible and any(name in visible for name in ("left_shoulder", "right_shoulder"))
    if part == "hand":
        return _hand_observed(dw, side)

    for candidate_side in sides:
        if part == "shoulder" and f"{candidate_side}_shoulder" in visible:
            return True
        if part == "upper arm" and _same_side_segment_visible(visible, candidate_side, ("shoulder", "elbow"), 2):
            return True
        if part == "forearm" and _same_side_segment_visible(visible, candidate_side, ("elbow", "wrist"), 2):
            return True
        if part == "wrist" and f"{candidate_side}_wrist" in visible:
            return True
        if part == "hip" and f"{candidate_side}_hip" in visible:
            return True
        if part == "thigh" and _same_side_segment_visible(visible, candidate_side, ("hip", "knee"), 2):
            return True
        if part == "knee" and f"{candidate_side}_knee" in visible:
            return True
        if part == "lower leg" and _same_side_segment_visible(visible, candidate_side, ("knee", "ankle"), 2):
            return True
        if part == "leg" and _same_side_segment_visible(visible, candidate_side, ("hip", "knee", "ankle"), 2):
            return True
        if part == "ankle" and f"{candidate_side}_ankle" in visible:
            return True
        if part == "foot" and f"{candidate_side}_ankle" in visible:
            return True
        if part == "arm":
            arm = _connectivity(dw).get(f"{candidate_side}_arm") or {}
            if int(arm.get("visible_count") or 0) >= 2:
                return True
    return False


def _qualified_actor_entity(item: dict[str, Any]) -> tuple[str | None, str] | None:
    entity = _entity_from_text(item.get("part"))
    if entity is None:
        return None
    side, part = entity
    state = item.get("fusion_v2") or {}
    if part in _BILATERAL_PARTS:
        if not state.get("laterality_selection_usable"):
            return None
        qualified = state.get("qualified_anatomical_side") or item.get("anatomical_side")
        side = str(qualified) if qualified in {"left", "right"} else side
        if side not in {"left", "right"}:
            return None
    return side, part


def _interaction_actor_entity(item: dict[str, Any]) -> tuple[str | None, str] | None:
    entity = _entity_from_text(item.get("actor_part"))
    if entity is None:
        return None
    side, part = entity
    state = item.get("fusion_v2") or {}
    if part in _BILATERAL_PARTS:
        if not state.get("laterality_selection_usable"):
            return None
        qualified = state.get("qualified_actor_anatomical_side")
        side = str(qualified) if qualified in {"left", "right"} else side
        if side not in {"left", "right"}:
            return None
    return side, part


def _relation_target(clause: str, actor: tuple[str | None, str] | None) -> tuple[str | None, str] | None:
    entities = _entities_from_text(clause)
    if not entities:
        return None
    if actor is None:
        return entities[-1]
    actor_side, actor_part = actor
    different = [
        entity for entity in entities
        if entity[1] != actor_part or (actor_side in {"left", "right"} and entity[0] in {"left", "right"} and entity[0] != actor_side)
    ]
    return different[-1] if different else None


def _strip_relation_clauses(value: Any, actor: tuple[str | None, str] | None, dw: dict[str, Any], *, geometry: bool) -> tuple[Any, list[str]]:
    if not isinstance(value, str) or not value.strip():
        return value, []
    clauses = [piece.strip() for piece in re.split(r"[,;]", value) if piece.strip()]
    kept: list[str] = []
    blocked: list[str] = []
    actor_ok = actor is not None and _body_entity_observed(dw, actor[0], actor[1])

    for clause in clauses:
        if geometry and not _RELATION_GEOMETRY_RE.search(clause):
            kept.append(clause)
            continue
        target = _relation_target(clause, actor)
        if target is None:
            kept.append(clause)
            continue
        target_ok = _body_entity_observed(dw, target[0], target[1])
        if not actor_ok or not target_ok:
            blocked.append(clause)
            continue
        kept.append(clause)

    return (", ".join(kept) or None), blocked


def _guard_self_contacts(fusion: dict[str, Any], dw: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema_version": "self-contact-support-audit-1.0",
        "blocked_body_fields": [],
        "blocked_interactions": [],
        "policy": {
            "body_to_body_contact_requires_observed_actor_entity": True,
            "body_to_body_contact_requires_observed_target_entity": True,
            "non_body_targets_are_unchanged": True,
            "dwpose_absence_vetoes_contact_support_not_body_visibility": True,
        },
    }

    for index, item in enumerate(fusion.get("qualified_body_parts") or []):
        if not isinstance(item, dict):
            continue
        actor = _qualified_actor_entity(item)
        for field in ("geometry", "contact", "support"):
            before = item.get(field)
            if not isinstance(before, str):
                continue
            after, blocked = _strip_relation_clauses(before, actor, dw, geometry=(field == "geometry"))
            if blocked:
                item[field] = after
                audit["blocked_body_fields"].append(
                    {
                        "index": index,
                        "part": item.get("part"),
                        "field": field,
                        "source": before,
                        "blocked_clauses": blocked,
                        "reason": "body_relation_lacks_independently_observed_actor_or_target_segment",
                    }
                )

    for index, item in enumerate(fusion.get("qualified_interactions") or []):
        if not isinstance(item, dict):
            continue
        target = _entity_from_text(item.get("target"))
        if target is None:
            continue
        actor = _interaction_actor_entity(item)
        actor_ok = actor is not None and _body_entity_observed(dw, actor[0], actor[1])
        target_ok = _body_entity_observed(dw, target[0], target[1])
        if actor_ok and target_ok:
            continue
        state = item.setdefault("fusion_v2", {})
        state["selection_usable"] = False
        state.setdefault("reasons", []).append(
            "Fusion-v2.3.5 withholds body-to-body interaction because actor and target body segments are not both independently observed"
        )
        audit["blocked_interactions"].append(
            {
                "index": index,
                "type": item.get("type"),
                "actor_part": item.get("actor_part"),
                "target": item.get("target"),
                "actor_observed": actor_ok,
                "target_observed": target_ok,
            }
        )
    return audit


def _safe_confidence(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _orientation_consistency(fusion: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    orientation = fusion.get("orientation_semantics") or {}
    shoulder = (fusion.get("sam3d_geometry_audit") or {}).get("shoulder_depth_rotation") or {}
    head_visibility = ((fusion.get("sam3d_geometry_audit") or {}).get("landmark_visibility") or {}).get("head") or {}
    gaze = ((analysis.get("target_subject") or {}).get("gaze") or {})

    shoulder_deg = _safe_confidence(shoulder.get("magnitude_deg"))
    torso = orientation.get("torso_yaw") or {}
    head = orientation.get("head_yaw") or {}
    strong_upper_torso = shoulder.get("authority") == "qualified_component_geometry" and shoulder_deg >= 50.0
    weak_frontal_torso = torso.get("direction") == "frontal" and torso.get("magnitude") in {"none", "slight"}
    head_camera_frontal = head.get("direction") == "frontal" and head.get("magnitude") in {"none", "slight", "moderate"}
    head_visible = head_visibility.get("visibility") == "visible" and _safe_confidence(head_visibility.get("confidence")) >= 0.75
    gaze_camera = gaze.get("target") in {"camera_lens", "near_camera"}

    audit: dict[str, Any] = {
        "schema_version": "orientation-consistency-audit-1.0",
        "qualified_shoulder_depth_deg": round(shoulder_deg, 3) if shoulder_deg else None,
        "strong_upper_torso_depth": strong_upper_torso,
        "suppressed_semantic_torso_yaw": False,
        "head_torso_relation": None,
    }

    if strong_upper_torso and weak_frontal_torso:
        audit["suppressed_semantic_torso_yaw"] = True
        audit["source_torso_yaw"] = copy.deepcopy(torso)
        fusion.setdefault("orientation_semantics", {})["torso_yaw"] = {
            "direction": "unknown",
            "magnitude": "unknown",
            "confidence": 0.0,
        }

    if strong_upper_torso:
        fusion["qualified_upper_torso_depth_relation"] = {
            "magnitude": "strong",
            "relation": "upper torso strongly turned in depth, near side-on rather than square-on to the camera",
            "authority": "qualified_visible_shoulder_depth_rotation",
            "source_magnitude_deg": round(shoulder_deg, 3),
        }

    if strong_upper_torso and head_camera_frontal and head_visible and gaze_camera:
        relation = {
            "magnitude": "strong",
            "relation": "head turned substantially toward the camera relative to the strongly depth-turned upper torso",
            "camera_relation": "toward_camera",
            "authority": "qualified_shoulder_depth_plus_visible_camera_frontal_head_and_camera_gaze",
        }
        fusion["qualified_head_torso_relation"] = relation
        audit["head_torso_relation"] = copy.deepcopy(relation)

    return audit


def refine_contact_orientation(payload: dict[str, Any], dw: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    fusion = out.get("fusion") if isinstance(out.get("fusion"), dict) else out
    if not isinstance(fusion, dict):
        return out
    fusion["self_contact_support_audit"] = _guard_self_contacts(fusion, dw)
    fusion["orientation_consistency_audit"] = _orientation_consistency(fusion, analysis)
    fusion["schema_version"] = "analysis-fusion-2.3.5"
    fusion.setdefault("selection_policy", {})["self_contact_support"] = (
        "Analyze body-to-body contact/support is caption-usable only when both actor and target body entities are independently observed by DWPose; missing segment evidence vetoes the relation, not body visibility."
    )
    fusion["selection_policy"]["orientation_consistency"] = (
        "Very-high qualified shoulder depth overrides weak frontal torso-yaw semantics; a camera-frontal visible head with camera gaze may establish a strong head-turn-toward-camera relation relative to that depth-turned torso."
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-contact-orientation-refine-235",
        description="Fusion 2.3.5: veto unsupported self-contact/support and reconcile strong torso depth with head/camera orientation.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.4" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    analysis_dir = run_dir / slug
    output_dir = (args.output_dir or (run_dir / "fusion-v2.3.5" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion-v2.3.4"), (dwpose_dir, "DWPose"), (analysis_dir, "Analyze")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = missing = blocked_interactions = blocked_fields = head_relations = 0
    records: list[dict[str, Any]] = []
    for fusion_path in sorted(fusion_dir.glob("*.fused_v2_3.json")):
        key = fusion_path.name.removesuffix(".fused_v2_3.json")
        out_path = output_dir / fusion_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        dw_path = dwpose_dir / f"{key}.dwpose.json"
        analysis_path = analysis_dir / f"{key}.analysis.json"
        if not dw_path.is_file() or not analysis_path.is_file():
            missing += 1
            records.append({"image_key": key, "status": "missing_source"})
            continue
        analysis_record = _read(analysis_path)
        analysis = analysis_record.get("analysis") if isinstance(analysis_record, dict) else None
        if not isinstance(analysis, dict):
            missing += 1
            records.append({"image_key": key, "status": "invalid_analysis"})
            continue

        refined = refine_contact_orientation(_read(fusion_path), _read(dw_path), analysis)
        _write(out_path, refined)
        written += 1
        fusion = refined.get("fusion") or {}
        contact = fusion.get("self_contact_support_audit") or {}
        orient = fusion.get("orientation_consistency_audit") or {}
        bi = len(contact.get("blocked_interactions") or [])
        bf = len(contact.get("blocked_body_fields") or [])
        hr = int(bool(orient.get("head_torso_relation")))
        blocked_interactions += bi
        blocked_fields += bf
        head_relations += hr
        records.append(
            {
                "image_key": key,
                "status": "written",
                "blocked_interactions": bi,
                "blocked_body_fields": bf,
                "suppressed_semantic_torso_yaw": bool(orient.get("suppressed_semantic_torso_yaw")),
                "head_torso_relation_qualified": bool(hr),
            }
        )

    index = {
        "schema_version": "analysis-fusion-2.3.5-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "source_fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "written": written,
        "skipped_existing": skipped,
        "missing_sources": missing,
        "blocked_self_contact_interactions": blocked_interactions,
        "blocked_self_contact_body_fields": blocked_fields,
        "qualified_head_torso_relations": head_relations,
        "records": records,
    }
    _write(output_dir / "contact_orientation_refine_235.index.json", index)
    print(f"Fusion-v2.3.5 output: {output_dir}")
    print(
        f"Written: {written}; reused: {skipped}; missing: {missing}; "
        f"blocked interactions: {blocked_interactions}; blocked body fields: {blocked_fields}; "
        f"head/torso relations: {head_relations}"
    )
    return 0 if written or skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
