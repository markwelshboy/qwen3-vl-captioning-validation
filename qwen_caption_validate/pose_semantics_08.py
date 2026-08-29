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
from .runner import model_slug, resolve_model_id


_STANDING_RE = re.compile(r"\b(?:stands?|standing|stood)\b", re.I)
_SQUAT_RE = re.compile(r"\b(?:squat(?:s|ted|ting)?|crouch(?:es|ed|ing|ed)?)\b", re.I)
_RECLINE_RE = re.compile(r"\b(?:reclin(?:es|ed|ing)|reclined|lying|lies|lay)\b", re.I)
_SEAT_SCENE_RE = re.compile(
    r"\b(?:airplane seat|aircraft seat|car seat|vehicle seat|seat(?:back)?|headrest|backrest|"
    r"chair|stool|bench|sofa|couch)\b",
    re.I,
)
_POSTURE_BEARING_ACTOR_RE = re.compile(r"\b(?:buttocks?|hips?|pelvis|torso|back|upper back|body|head)\b", re.I)
_SUPPORT_TEXT_RE = re.compile(r"\b(?:support(?:s|ed|ing)?|rest(?:s|ed|ing)?|seat(?:ed)?|against|weight)\b", re.I)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _analysis_root(value: dict[str, Any]) -> dict[str, Any]:
    return v05._analysis_root(value)


def _probe_root(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get("gestalt")
    return nested if isinstance(nested, dict) else value


def _normalize_probe_posture(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"squat", "squatting", "crouch", "crouched", "crouching"}:
        return "squatting"
    if text in {"recline", "reclined", "reclining", "lying", "lying_or_reclining"}:
        return "reclining"
    if text in {"stand", "standing"}:
        return "standing"
    if text in {"sit", "sitting", "seated"}:
        return "seated"
    return text or None


def _hypothesis(posture: dict[str, Any], label: str) -> dict[str, Any]:
    for item in posture.get("hypotheses") or []:
        if isinstance(item, dict) and item.get("label") == label:
            return item
    return {}


def _dwpose_target(dwpose: dict[str, Any]) -> dict[str, Any]:
    return ((dwpose.get("derived") or {}).get("target") or {})


def _visible_joints(dwpose: dict[str, Any]) -> set[str]:
    return set(_dwpose_target(dwpose).get("visible_body_landmarks") or [])


def _segment_visible(dwpose: dict[str, Any], side: Any, part: Any) -> bool:
    if side not in {"left", "right"}:
        return False
    part = str(part or "").lower()
    required = {
        "elbow": {f"{side}_elbow"},
        "forearm": {f"{side}_elbow", f"{side}_wrist"},
        "wrist": {f"{side}_wrist"},
        "hand": {f"{side}_wrist"},
        "arm": {f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"},
    }.get(part)
    return bool(required) and required.issubset(_visible_joints(dwpose))


def _bilateral_hip_knee_visible(dwpose: dict[str, Any]) -> bool:
    connectivity = _dwpose_target(dwpose).get("connectivity") or {}
    return all(
        int((connectivity.get(f"{side}_leg") or {}).get("visible_count") or 0) >= 2
        for side in ("left", "right")
    )


def _squat_geometry(dwpose: dict[str, Any]) -> dict[str, Any]:
    features = base._geometry_features(base._target_person(dwpose), dwpose)
    angles = features.get("angles_deg") or {}
    directions = features.get("directions_deg") or {}
    connectivity = features.get("connectivity") or {}

    knee_angles: list[float] = []
    thigh_angles: list[float] = []
    for side in ("left", "right"):
        leg = connectivity.get(f"{side}_leg") or {}
        if leg.get("complete") and angles.get(f"{side}_knee") is not None:
            knee_angles.append(float(angles[f"{side}_knee"]))
        if int(leg.get("visible_count") or 0) >= 2 and directions.get(f"{side}_thigh_from_horizontal") is not None:
            thigh_angles.append(float(directions[f"{side}_thigh_from_horizontal"]))

    deeply_flexed = [value for value in knee_angles if value <= 130.0]
    horizontal_thighs = [value for value in thigh_angles if value <= 40.0]
    return {
        "compatible": bool(deeply_flexed and horizontal_thighs),
        "complete_knee_angles_deg": knee_angles,
        "thigh_angles_from_horizontal_deg": thigh_angles,
        "deeply_flexed_knee_count": len(deeply_flexed),
        "horizontal_thigh_count": len(horizontal_thighs),
    }


def _analyze_posture_bearing_support(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    subject = analysis.get("target_subject") or {}

    for item in subject.get("interactions") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("evidence_status") or "observed").lower() not in {"observed", ""}:
            continue
        if v05._safe_float(item.get("confidence"), 0.0) < 0.70:
            continue
        actor = str(item.get("actor_part") or "")
        target = str(item.get("target") or "")
        relation = str(item.get("type") or "")
        notes = str(item.get("notes") or "")
        if (
            relation in {"support", "contact"}
            and _POSTURE_BEARING_ACTOR_RE.search(actor)
            and _SEAT_SCENE_RE.search(target)
            and (relation == "support" or _SUPPORT_TEXT_RE.search(notes))
        ):
            evidence.append({
                "source": "analysis.target_subject.interactions",
                "actor_part": actor,
                "target": target,
                "confidence": v05._safe_float(item.get("confidence"), 0.0),
            })

    for item in subject.get("visible_body_parts") or []:
        if not isinstance(item, dict):
            continue
        if item.get("ownership") not in {None, "target"}:
            continue
        if v05._safe_float(item.get("confidence"), 0.0) < 0.70:
            continue
        part = str(item.get("part") or "")
        support_text = " ".join(str(item.get(field) or "") for field in ("support", "contact"))
        if _POSTURE_BEARING_ACTOR_RE.search(part) and _SEAT_SCENE_RE.search(support_text) and _SUPPORT_TEXT_RE.search(support_text):
            evidence.append({
                "source": "analysis.target_subject.visible_body_parts",
                "actor_part": part,
                "target": support_text,
                "confidence": v05._safe_float(item.get("confidence"), 0.0),
            })

    for item in analysis.get("non_target_entities") or []:
        if not isinstance(item, dict):
            continue
        if v05._safe_float(item.get("confidence"), 0.0) < 0.70:
            continue
        description = str(item.get("description") or "")
        relation_text = " ".join(str(item.get(field) or "") for field in ("contact", "support"))
        if _SEAT_SCENE_RE.search(description) and _SUPPORT_TEXT_RE.search(relation_text) and re.search(r"\b(?:subject|target|back|body|weight|seat(?:ed)?)\b", relation_text, re.I):
            evidence.append({
                "source": "analysis.non_target_entities",
                "actor_part": "target body",
                "target": description,
                "relation": relation_text,
                "confidence": v05._safe_float(item.get("confidence"), 0.0),
            })

    return evidence


def _analyze_seated_scene_agreement(analysis: dict[str, Any]) -> bool:
    summary = str(analysis.get("image_summary") or "")
    return bool(v05._SEATED_RE.search(summary) and _SEAT_SCENE_RE.search(summary))


def _verified_activity_support(result: dict[str, Any], dwpose: dict[str, Any]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    graph = result.get("support_graph") or {}
    for edge in graph.get("body_support_edges") or []:
        if not isinstance(edge, dict):
            continue
        if edge.get("relation") != "support" or edge.get("target_class") != "surface":
            continue
        if v05._safe_float(edge.get("confidence"), 0.0) < 0.70:
            continue
        if edge.get("source") != v07._GESTALT_SOURCE:
            verified.append({"route": "bottom_up", "edge": edge})
            continue
        visible = _segment_visible(dwpose, edge.get("actor_side"), edge.get("actor_part"))
        edge["actor_visibility_validated"] = visible
        edge["evidence_status_policy"] = "pose-gestalt observed/contextual status is advisory; DWPose actor visibility is checked independently"
        if visible:
            verified.append({"route": "probe_actor_visibility_verified", "edge": edge})
    return verified


def _annotate_support_roles(result: dict[str, Any], dwpose: dict[str, Any]) -> None:
    graph = result.get("support_graph") or {}
    for edge in graph.get("body_support_edges") or []:
        if not isinstance(edge, dict):
            continue
        actor = str(edge.get("actor_part") or "")
        target_class = str(edge.get("target_class") or "")
        source = str(edge.get("source") or "")
        if target_class == "seat" and actor in {"hip", "thigh", "torso"}:
            edge["support_role"] = "posture_bearing_candidate" if source == v07._GESTALT_SOURCE else "posture_bearing"
        elif target_class == "surface" and actor in {"elbow", "forearm", "wrist", "hand", "arm"}:
            edge["support_role"] = "activity_or_lean_support"
        else:
            edge["support_role"] = "other_support"
        if source == v07._GESTALT_SOURCE and actor in {"elbow", "forearm", "wrist", "hand", "arm"}:
            edge["actor_visibility_validated"] = _segment_visible(dwpose, edge.get("actor_side"), actor)
            edge["evidence_status_policy"] = "pose-gestalt evidence_status is advisory, never independent authority"
    graph.setdefault("policy", {})["support_ontology_v08"] = {
        "posture_bearing": "pelvis/torso/body support by a seat/backrest/bed-like structure may establish posture when independently recorded outside the gestalt probe",
        "activity_or_lean_support": "hand/forearm/elbow support on a table/desk/counter establishes leaning/resting, not seated posture by itself",
        "probe_evidence_status": "advisory_only",
    }
    result["support_graph"] = graph


def _demote_unverified_probe_support_gestures(result: dict[str, Any], dwpose: dict[str, Any]) -> None:
    graph = result.get("support_graph") or {}
    chains = graph.get("support_chains") or []
    for gesture in result.get("gestures") or []:
        details = gesture.get("details") or {}
        kind = details.get("class")
        support_text = " ".join(str(value) for value in gesture.get("support") or []).lower()
        if "pose_gestalt_" not in support_text:
            continue
        valid = True
        if kind == "surface_support":
            valid = _segment_visible(dwpose, details.get("actor_side"), details.get("part"))
        elif kind == "supported_lean":
            side = details.get("actor_side")
            chain = next((item for item in chains if isinstance(item, dict) and item.get("side") == side), None)
            valid = bool(chain) and _segment_visible(dwpose, side, (chain or {}).get("support_part"))
        if valid:
            continue
        gesture["caption_preferred"] = False
        limitations = gesture.setdefault("limitations", [])
        note = "pose-gestalt support is not caption-usable because the claimed supporting segment is not independently visible in DWPose"
        if note not in limitations:
            limitations.append(note)


def _restore_baseline_posture(result: dict[str, Any], baseline: dict[str, Any]) -> None:
    result["posture"] = baseline.get("posture") or {}
    result.setdefault("preferred_pose", {})["posture"] = (baseline.get("preferred_pose") or {}).get("posture")


def _qualify_posture(result: dict[str, Any], baseline: dict[str, Any], label: str, score: float, authority: str, support: list[str], limitations: list[str] | None = None) -> None:
    result["posture"] = {
        "status": "qualified",
        "label": label,
        "primitive_id": f"posture_{label}_v08",
        "support_score": round(max(0.0, min(1.0, score)), 3),
        "confidence_band": "strong" if score >= 0.80 else "moderate",
        "caption_preferred": True,
        "support": support,
        "limitations": limitations or [],
        "subsumes": ["component geometry/support evidence used only to establish the whole-pose primitive"],
        "hypotheses": (baseline.get("posture") or {}).get("hypotheses") or [],
        "authority": authority,
    }
    result.setdefault("preferred_pose", {})["posture"] = label


def _verify_posture(
    result: dict[str, Any],
    baseline: dict[str, Any],
    dwpose: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None,
) -> None:
    analysis = _analysis_root(analysis_payload)
    gestalt = _probe_root(gestalt_payload)
    probe_posture = _normalize_probe_posture(gestalt.get("posture"))
    probe_conf = v05._safe_float(gestalt.get("posture_confidence"), 0.0)
    probe_basis = str(gestalt.get("posture_basis") or "unknown").lower()
    summary = str(analysis.get("image_summary") or "")
    baseline_posture = baseline.get("posture") or {}
    baseline_label = baseline_posture.get("label") if baseline_posture.get("status") == "qualified" else None

    _annotate_support_roles(result, dwpose)
    activity_support = _verified_activity_support(result, dwpose)
    posture_bearing = _analyze_posture_bearing_support(analysis)
    seat_scene_agreement = _analyze_seated_scene_agreement(analysis)
    squat_geometry = _squat_geometry(dwpose)
    both_hip_knee = _bilateral_hip_knee_visible(dwpose)
    standing_hyp = _hypothesis(baseline_posture, "standing")

    conflict = bool(baseline_label and probe_posture and baseline_label != probe_posture)
    route = "withheld"
    valid = False
    reasons: list[str] = []

    if baseline_label:
        valid = True
        route = "baseline_qualified"
        reasons.append(f"bottom-up pose semantics already qualifies {baseline_label}")
    elif probe_posture == "seated" and probe_conf >= 0.82 and not conflict:
        analyze_says_seated = bool(v05._SEATED_RE.search(summary))
        if posture_bearing:
            valid = True
            route = "seated_posture_bearing_support"
            reasons.append("Analyze independently records posture-bearing support by a seat/backrest structure")
        elif analyze_says_seated and seat_scene_agreement:
            valid = True
            route = "seated_analyze_gestalt_seat_scene_agreement"
            reasons.append("Analyze independently says seated and names a seat-specific scene/support structure")
        elif analyze_says_seated and activity_support:
            valid = True
            route = "seated_analyze_gestalt_plus_verified_activity_support"
            reasons.append("Analyze independently says seated and a surface-support segment is independently visible/corroborated")
        else:
            reasons.append("arm/table support or gestalt context alone cannot establish seated posture")
    elif probe_posture == "standing" and probe_conf >= 0.85 and not conflict:
        analyze_says_standing = bool(_STANDING_RE.search(summary))
        geometry_support = both_hip_knee and v05._safe_float(standing_hyp.get("support_score"), 0.0) >= 0.40
        if analyze_says_standing and geometry_support:
            valid = True
            route = "standing_analyze_gestalt_plus_bilateral_hip_knee"
            reasons.append("Analyze and gestalt agree on standing, with both observed hip-to-knee chains and compatible upright geometry")
        else:
            reasons.append("standing gestalt lacks Analyze agreement plus useful bilateral lower-body geometry")
    elif probe_posture == "squatting" and probe_conf >= 0.90 and probe_basis in {"geometric", "mixed"} and not conflict:
        if _SQUAT_RE.search(summary) and squat_geometry["compatible"]:
            valid = True
            route = "squatting_analyze_gestalt_plus_flexed_leg_geometry"
            reasons.append("Analyze and geometric gestalt agree on squatting/crouching and DWPose shows flexed-knee plus horizontal-thigh compatibility")
        else:
            reasons.append("squatting gestalt lacks independent Analyze agreement and compatible flexed-leg geometry")
    elif probe_posture == "reclining" and probe_conf >= 0.85 and not conflict:
        if _RECLINE_RE.search(summary) and posture_bearing:
            valid = True
            route = "reclining_analyze_gestalt_plus_posture_bearing_support"
            reasons.append("Analyze and gestalt agree on reclining with independent body/back support")
        else:
            reasons.append("reclining gestalt lacks independent semantic agreement plus posture-bearing support")
    elif probe_posture:
        reasons.append("top-down posture did not meet posture-specific verification requirements")

    current = result.get("posture") or {}
    if current.get("status") == "qualified" and current.get("label") != baseline_label and not valid:
        _restore_baseline_posture(result, baseline)

    promoted = False
    if valid and baseline_label is None and probe_posture:
        if probe_posture == "seated":
            _qualify_posture(
                result, baseline, "seated", probe_conf,
                f"v08_{route}", reasons,
                ["lower-body geometry may be unavailable; authority comes from independently verified context/support rather than reconstructed anatomy"],
            )
            promoted = True
        elif probe_posture == "standing":
            _qualify_posture(result, baseline, "standing", probe_conf, f"v08_{route}", reasons)
            promoted = True
        elif probe_posture == "squatting":
            _qualify_posture(
                result, baseline, "squatting", probe_conf, f"v08_{route}", reasons,
                ["DWPose flexion geometry is used as compatibility evidence; the whole-pose label comes from corroborated top-down recognition"],
            )
            promoted = True
        elif probe_posture == "reclining":
            _qualify_posture(result, baseline, "reclining", probe_conf, f"v08_{route}", reasons)
            promoted = True

    candidate = None
    if baseline_label is None and probe_posture and not promoted and probe_conf >= 0.82:
        candidate = {
            "label": probe_posture,
            "status": "candidate",
            "model_confidence": round(probe_conf, 3),
            "support_score": 0.60 if not conflict else 0.40,
            "confidence_band": "moderate" if not conflict else "weak",
            "caption_preferred": False,
            "review_recommended": True,
            "authority": "top_down_gestalt_hypothesis_not_independently_verified",
            "support": reasons,
        }
    result["posture_candidate"] = candidate

    probe = result.get("pose_gestalt_probe") or {}
    probe["posture"] = probe_posture
    probe["posture_confidence"] = round(probe_conf, 3)
    probe["caption_preferred"] = bool(promoted or (baseline_label and baseline_label == probe_posture))
    if promoted:
        probe["promotion_reason"] = route
    elif baseline_label and baseline_label == probe_posture:
        probe["promotion_reason"] = "agrees with already-qualified bottom-up posture"
    elif probe_posture:
        probe["promotion_reason"] = "withheld from caption: posture remains a top-down candidate pending posture-specific verification"
    result["pose_gestalt_probe"] = probe

    old_audit = result.get("pose_gestalt_corroboration")
    if old_audit is not None:
        result["pose_gestalt_corroboration_v07"] = old_audit
    result["pose_gestalt_corroboration"] = {
        "valid": bool(valid),
        "route": route,
        "probe_posture": probe_posture,
        "probe_confidence": round(probe_conf, 3),
        "probe_basis": probe_basis,
        "baseline_qualified_posture": baseline_label,
        "conflict_with_baseline": conflict,
        "analyze_summary_mentions_seated": bool(v05._SEATED_RE.search(summary)),
        "analyze_summary_mentions_standing": bool(_STANDING_RE.search(summary)),
        "analyze_summary_mentions_squat_or_crouch": bool(_SQUAT_RE.search(summary)),
        "analyze_summary_mentions_reclining": bool(_RECLINE_RE.search(summary)),
        "analyze_seat_scene_agreement": seat_scene_agreement,
        "posture_bearing_support_count": len(posture_bearing),
        "posture_bearing_support": posture_bearing,
        "verified_activity_support_count": len(activity_support),
        "verified_activity_support_routes": [item.get("route") for item in activity_support],
        "bilateral_hip_knee_visible": both_hip_knee,
        "squat_geometry": squat_geometry,
        "policy": (
            "top-down gestalt proposes a recognizable posture; posture-specific independent Analyze/DWPose/support evidence verifies it. "
            "Probe evidence_status is advisory. Arm/table support establishes a lean/activity relation but cannot by itself establish seated posture."
        ),
    }


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = v05.build_pose_semantics(dwpose, fused_payload, analysis_payload)
    result = v07.build_pose_semantics(dwpose, fused_payload, analysis_payload, gestalt_payload)

    _verify_posture(result, baseline, dwpose, analysis_payload, gestalt_payload)
    _demote_unverified_probe_support_gestures(result, dwpose)
    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in result.get("gestures") or [] if item.get("caption_preferred")
    ]
    result["human_summary"] = base._human_summary(
        result.get("posture") or {},
        result.get("torso_orientation") or {},
        result.get("gestures") or [],
        result.get("head_and_gaze") or [],
        result.get("framing") or {},
    )
    result["schema_version"] = "pose-semantics-0.8"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "Top-down pose gestalt is a hypothesis channel, not an evidence authority; its observed/contextual labels are advisory until independently verified.",
        "Support is split into posture-bearing support and activity/lean support. Arm-on-table support cannot by itself establish seated posture.",
        "Standing can be promoted when gestalt and Analyze agree and bilateral observed hip-to-knee geometry is compatible.",
        "Squatting/crouching is a first-class whole-pose primitive when top-down geometric recognition, Analyze semantics, and flexed-leg DWPose compatibility agree.",
        "Unverified high-confidence gestalt postures are retained as AMBER-style posture_candidate records rather than silently discarded or captioned as facts.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-08",
        description="Pose semantics v0.8: posture-specific top-down verification and support ontology.",
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
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.8" / slug)).expanduser().resolve()

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
            "posture_candidate": (result.get("posture_candidate") or {}).get("label") if isinstance(result.get("posture_candidate"), dict) else None,
            "verification_route": (result.get("pose_gestalt_corroboration") or {}).get("route"),
            "human_summary": result.get("human_summary"),
        })

    index = {
        "schema_version": "pose-semantics-0.8-run",
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
    print(f"Pose semantics v0.8: {output_dir}")
    for record in records:
        suffix = f" [candidate={record.get('posture_candidate')}]" if record.get("posture_candidate") else ""
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}{suffix}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
