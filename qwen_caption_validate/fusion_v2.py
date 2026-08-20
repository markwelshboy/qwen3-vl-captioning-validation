from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .analysis_v2_normalize import normalize_analysis_v2
from .pose_evidence import build_pose_evidence
from .runner import model_slug, resolve_model_id, validate_analysis


HANDISH_TOKENS = ("finger", "hand", "wrist", "forearm", "arm")
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_V2_SCHEMA = PACKAGE_ROOT / "schemas" / "analysis_v2.schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-fusion-v2",
        description=(
            "Fuse Analyze-v2 semantic observations with cached DWPose evidence. "
            "The stage is deliberately audit-first: it preserves raw semantic observations, "
            "adds deterministic geometry, and qualifies ownership, laterality, framing, and geometry independently."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing Analyze-v2 validation run directory.")
    parser.add_argument("--model", default="32b-fp8", help="Analysis model alias or Hugging Face model ID.")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: <run_dir>/fusion-v2/<model-slug>).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _is_handish(text: Any) -> bool:
    value = str(text or "").lower()
    return any(token in value for token in HANDISH_TOKENS)


def _normalize_undirected_angle(value: Any) -> dict[str, Any]:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return {"raw_deg": None, "normalized_deg": None, "abs_deg": None, "status": "unavailable"}
    normalized = ((raw + 90.0) % 180.0) - 90.0
    abs_deg = abs(normalized)
    status = "sanity_review" if abs_deg > 70.0 else "usable"
    return {
        "raw_deg": round(raw, 3),
        "normalized_deg": round(normalized, 3),
        "abs_deg": round(abs_deg, 3),
        "status": status,
    }


def _supported_hand_candidates(pose: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in (pose.get("hand_candidates") or [])
        if item.get("supported_by_nearby_visible_target_wrist")
        and item.get("nearest_visible_target_wrist") in {"left", "right"}
    ]


def _supported_hand_sides(pose: dict[str, Any]) -> set[str]:
    return {
        str(item.get("nearest_visible_target_wrist"))
        for item in _supported_hand_candidates(pose)
    }


def _strong_target_hand_support(pose: dict[str, Any]) -> list[dict[str, Any]]:
    """Return hand-root associations that also land on a complete target arm chain."""
    return [
        item
        for item in _supported_hand_candidates(pose)
        if bool(item.get("target_arm_chain_complete"))
    ]


def _mirror_sensitive(analysis: dict[str, Any]) -> bool:
    framing = analysis.get("framing") or {}
    archetype = str(framing.get("photographic_archetype") or "").lower()
    summary = str(analysis.get("image_summary") or "").lower()
    return "mirror" in archetype or "mirror selfie" in summary


def _qualify_body_parts(analysis: dict[str, Any], pose: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    subject = analysis.get("target_subject") or {}
    parts = subject.get("visible_body_parts") or []
    supported_sides = _supported_hand_sides(pose)
    strong_hand_support = _strong_target_hand_support(pose)
    strong_supported_sides = {
        str(item.get("nearest_visible_target_wrist")) for item in strong_hand_support
    }
    mirror_sensitive = _mirror_sensitive(analysis)
    qualified: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index, raw in enumerate(parts):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        part = str(item.get("part") or "unknown")
        side = str(item.get("anatomical_side") or "unknown")
        ownership = str(item.get("ownership") or "unknown")
        connectivity = str(item.get("connectivity_to_target_chain") or "unknown")
        visibility = str(item.get("visibility") or "unknown")
        handish = _is_handish(part)

        selection_usable = True
        qualified_ownership = ownership
        qualified_side = side
        laterality_selection_usable = side in {"left", "right"}
        reasons: list[str] = []
        laterality_reasons: list[str] = []

        if handish:
            same_side_support = side in supported_sides if side in {"left", "right"} else False
            any_support = bool(supported_sides)
            same_side_strong = side in strong_supported_sides if side in {"left", "right"} else False
            any_strong = bool(strong_hand_support)
            visible_chain = connectivity == "connected_visible"
            occluded_chain = connectivity == "connected_but_occluded"
            disconnected = connectivity in {"disconnected_in_crop", "unknown"}
            isolated_finger_fragment = visibility == "fragment" and part.lower() == "fingers"

            if ownership == "target":
                if visible_chain:
                    reasons.append("Analyze-v2 reports a visible target limb chain")
                elif occluded_chain and (same_side_support or (side not in {"left", "right"} and any_support)):
                    reasons.append("short semantic occlusion gap is supported by DWPose hand-root/wrist association")
                elif same_side_support or (side not in {"left", "right"} and any_support):
                    reasons.append("DWPose hand root is near a visible target wrist")
                elif any_strong and isolated_finger_fragment:
                    reasons.append(
                        "target ownership is supported by a hand-root association to a complete target arm chain; anatomical side remains unresolved"
                    )
                else:
                    qualified_ownership = "unknown"
                    selection_usable = False
                    reasons.append("target ownership lacks visible-chain or hand-root/wrist support")
                    warnings.append(
                        f"body part {index} ({part}) claimed target ownership without qualified visible-chain or hand-root support; ownership downgraded"
                    )
            elif ownership == "unknown":
                selection_usable = False
                reasons.append("ownership is explicitly unresolved")

            if disconnected:
                selection_usable = False
                if ownership == "target":
                    qualified_ownership = "unknown"
                reasons.append("body-part fragment is disconnected or unresolved in the visible crop")

            if isolated_finger_fragment:
                if qualified_ownership == "target" and any_strong and not disconnected:
                    selection_usable = True
                    reasons.append(
                        "isolated fingers do not establish the chain by themselves, but deterministic hand-root evidence supports target ownership/action"
                    )
                else:
                    selection_usable = False
                    reasons.append("isolated finger fragment cannot establish an unseen hand/arm chain")

            # Laterality is deliberately independent of target ownership/action.
            # DWPose side is used as a conflict detector on direct images only.
            if side in {"left", "right"}:
                if mirror_sensitive:
                    laterality_selection_usable = False
                    laterality_reasons.append("mirror/reflection geometry makes detector-side validation non-authoritative")
                elif supported_sides and side not in supported_sides:
                    qualified_side = "unknown"
                    laterality_selection_usable = False
                    laterality_reasons.append(
                        f"Analyze-v2 side={side} conflicts with DWPose hand-root association to {sorted(supported_sides)}"
                    )
                    warnings.append(
                        f"body part {index} ({part}) anatomical side conflicts with DWPose hand-root/wrist association; laterality downgraded"
                    )
                elif same_side_strong:
                    laterality_reasons.append("Analyze-v2 side agrees with DWPose hand-root/wrist association on a complete target arm chain")
                elif same_side_support:
                    laterality_reasons.append("Analyze-v2 side agrees with DWPose hand-root/wrist association")
            else:
                laterality_selection_usable = False

        item["fusion_v2"] = {
            "qualified_ownership": qualified_ownership,
            "qualified_anatomical_side": qualified_side,
            "selection_usable": selection_usable,
            "laterality_selection_usable": laterality_selection_usable,
            "reasons": reasons,
            "laterality_reasons": laterality_reasons,
        }
        qualified.append(item)

    return qualified, warnings


def _actor_side(actor: str) -> str:
    text = actor.lower()
    has_left = bool(re.search(r"\bleft\b", text))
    has_right = bool(re.search(r"\bright\b", text))
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return "unknown"


def _qualify_interactions(
    analysis: dict[str, Any],
    qualified_parts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    subject = analysis.get("target_subject") or {}
    interactions = subject.get("interactions") or []
    warnings: list[str] = []
    out: list[dict[str, Any]] = []

    qualified_target_handish = [
        part for part in qualified_parts
        if _is_handish(part.get("part"))
        and (part.get("fusion_v2") or {}).get("qualified_ownership") == "target"
        and (part.get("fusion_v2") or {}).get("selection_usable")
    ]

    for index, raw in enumerate(interactions):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        actor = str(item.get("actor_part") or "")
        actor_ownership = str(item.get("actor_ownership") or "unknown")
        evidence_status = str(item.get("evidence_status") or "unknown")
        selection_usable = evidence_status == "observed"
        qualified_actor_ownership = actor_ownership
        actor_side = _actor_side(actor)
        qualified_actor_side = actor_side
        laterality_selection_usable = actor_side in {"left", "right"}
        reasons: list[str] = []
        laterality_reasons: list[str] = []

        if evidence_status != "observed":
            selection_usable = False
            reasons.append("interaction relationship is not directly observed")

        if _is_handish(actor):
            if actor_ownership == "target" and not qualified_target_handish:
                selection_usable = False
                qualified_actor_ownership = "unknown"
                reasons.append("target hand/finger interaction lacks a qualified target-owned handish body-part observation")
                warnings.append(
                    f"interaction {index} ({item.get('type')}) target hand ownership downgraded because no qualified hand/arm chain supports it"
                )
            elif actor_ownership == "target":
                reasons.append("target hand/finger interaction is supported by at least one qualified target-owned handish observation")
            elif actor_ownership == "unknown":
                selection_usable = False
                reasons.append("actor ownership is unresolved")

            if actor_side in {"left", "right"}:
                side_matches = [
                    part for part in qualified_target_handish
                    if str(part.get("anatomical_side") or "unknown") == actor_side
                ]
                if side_matches and any(
                    bool((part.get("fusion_v2") or {}).get("laterality_selection_usable"))
                    for part in side_matches
                ):
                    laterality_reasons.append("interaction side agrees with a qualified target body-part observation")
                else:
                    qualified_actor_side = "unknown"
                    laterality_selection_usable = False
                    laterality_reasons.append(
                        "interaction action/ownership may remain valid, but anatomical side is not independently qualified"
                    )
            else:
                laterality_selection_usable = False

        item["fusion_v2"] = {
            "qualified_actor_ownership": qualified_actor_ownership,
            "qualified_actor_anatomical_side": qualified_actor_side,
            "selection_usable": selection_usable,
            "laterality_selection_usable": laterality_selection_usable,
            "reasons": reasons,
            "laterality_reasons": laterality_reasons,
        }
        out.append(item)
    return out, warnings


_WEAK_EYE_LEVEL_PATTERNS = (
    re.compile(r"eyes?.*(same|approximately same|aligned).*(height|level).*(camera|lens)", re.I),
    re.compile(r"(camera|lens).*(same|approximately same|aligned).*(height|level).*eyes?", re.I),
    re.compile(r"eyes?.*(same|approximately same|aligned).*(vertical level|plane)", re.I),
    re.compile(r"symmetri(c|cal).*fram", re.I),
    re.compile(r"selfie perspective", re.I),
    re.compile(r"head level to camera", re.I),
)


def _camera_evidence_strength(elevation: str, evidence: list[str]) -> tuple[list[str], list[str]]:
    credible: list[str] = []
    weak: list[str] = []
    for item in evidence:
        text = item.strip()
        low = text.lower()
        if any(pattern.search(text) for pattern in _WEAK_EYE_LEVEL_PATTERNS):
            weak.append(text)
            continue
        if low.startswith("no strong ") or low.startswith("no visible "):
            weak.append(text)
            continue
        if elevation == "high" and any(
            token in low for token in ("overhead", "view down", "looking down", "floor plane", "ground plane", "top surface")
        ):
            credible.append(text)
            continue
        if elevation == "low" and any(
            token in low for token in ("view up", "looking up", "beneath chin", "underside", "ceiling plane")
        ):
            credible.append(text)
            continue
        if elevation == "eye_level" and any(token in low for token in ("horizon", "vanishing", "level ground plane")):
            credible.append(text)
            continue
        weak.append(text)
    return credible, weak


def _camera_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    camera = analysis.get("camera") or {}
    elevation = str(camera.get("elevation") or "unknown")
    confidence = float(camera.get("elevation_confidence") or 0.0)
    evidence = [str(x) for x in (camera.get("elevation_evidence") or []) if x]
    counter = [str(x) for x in (camera.get("elevation_counterevidence") or []) if x]
    credible, weak = _camera_evidence_strength(elevation, evidence)

    qualified = elevation != "unknown" and confidence >= 0.80 and bool(credible)
    reasons: list[str] = []
    if elevation == "unknown":
        reasons.append("Analyze-v2 left camera elevation unresolved")
    if confidence < 0.80:
        reasons.append("camera elevation confidence below 0.80 audit threshold")
    if elevation != "unknown" and not credible:
        reasons.append("camera elevation lacks qualified geometric evidence")
    if weak:
        reasons.append("one or more camera-elevation evidence strings are weak/non-geometric and cannot establish authority")
    if counter:
        reasons.append("counterevidence is present and should remain visible to downstream policy")

    return {
        "elevation": elevation,
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "qualified_geometric_evidence": credible,
        "weak_or_non_geometric_evidence": weak,
        "counterevidence": counter,
        "qualified_semantic_evidence": qualified,
        "selection_usable": False,
        "authority": "report_only_until_camera_axis_is_validated_on_regression_set",
        "reasons": reasons,
    }


def _scene_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    scene = analysis.get("scene") or {}
    structure = scene.get("background_structure") or {}
    specular = str(structure.get("specular_reflective") or "unknown")
    structural = str(structure.get("structural_complexity") or "unknown")
    texture = str(structure.get("texture_complexity") or "unknown")
    strong_lines = str(structure.get("strong_lines_or_angles") or "unknown")
    return {
        "environment_type": scene.get("environment_type"),
        "texture_complexity": texture,
        "structural_complexity": structural,
        "specular_reflective": specular,
        "strong_lines_or_angles": strong_lines,
        "reflections_present": structure.get("reflections_present"),
        "structural_or_specular_burden_flag": bool(
            specular in {"medium", "high"}
            or structural == "high"
            or strong_lines == "high"
            or structure.get("reflections_present") is True
        ),
        "selection_usable": False,
        "authority": "report_only_until_analyze_v2_scene_axes_are_validated",
    }


def _visible_semantic_parts(analysis: dict[str, Any]) -> set[str]:
    parts = ((analysis.get("target_subject") or {}).get("visible_body_parts") or [])
    out: set[str] = set()
    for item in parts:
        if not isinstance(item, dict):
            continue
        part = str(item.get("part") or "").strip().lower()
        if part:
            out.add(part)
        for subpart in item.get("visible_subparts") or []:
            text = str(subpart).strip().lower()
            if text:
                out.add(text)
    return out


def _framing_audit(analysis: dict[str, Any], deterministic_geometry: dict[str, Any]) -> dict[str, Any]:
    framing = analysis.get("framing") or {}
    semantic_scale = str(framing.get("shot_scale") or "unknown")
    pose_extent = str(deterministic_geometry.get("pose_extent_hint") or "unknown")
    connectivity = deterministic_geometry.get("connectivity") or {}
    semantic_parts = _visible_semantic_parts(analysis)

    complete_legs = sum(
        bool((connectivity.get(name) or {}).get("complete"))
        for name in ("left_leg", "right_leg")
    )
    feet_reported = any(
        token == "foot" or "shoe" in token or "ankle" in token
        for token in semantic_parts
    )

    qualified_scale = semantic_scale
    override = False
    conflict = False
    reasons: list[str] = []

    full_length_measurement = pose_extent == "full_length" and complete_legs >= 1
    full_length_semantics = feet_reported
    if full_length_measurement and full_length_semantics and semantic_scale != "full_length":
        qualified_scale = "full_length"
        override = True
        conflict = True
        reasons.append(
            "DWPose reports full-length extent with a complete leg chain and Analyze-v2 reports visible foot/ankle/shoe evidence"
        )
        reasons.append(
            f"semantic shot_scale={semantic_scale} is internally inconsistent with visible distal-leg/foot evidence"
        )
    elif pose_extent == "full_length" and semantic_scale != "full_length":
        reasons.append(
            "DWPose reports full-length extent, but semantic distal-leg/foot evidence is insufficient for deterministic override"
        )

    return {
        "semantic_framing": framing,
        "deterministic_pose_extent_hint": pose_extent,
        "complete_leg_chains": complete_legs,
        "semantic_distal_leg_or_foot_evidence": feet_reported,
        "qualified_shot_scale": qualified_scale,
        "override_applied": override,
        "conflict": conflict,
        "selection_usable": override,
        "authority": "deterministic_full_length_reconciliation_only",
        "reasons": reasons,
    }


def _projected_body_axis_audit(analysis: dict[str, Any], deterministic_geometry: dict[str, Any]) -> dict[str, Any]:
    orientation = ((analysis.get("target_subject") or {}).get("orientation") or {})
    semantic = orientation.get("image_plane_body_axis") or {}
    semantic_direction = str(semantic.get("direction") or "unknown")
    semantic_magnitude = str(semantic.get("magnitude") or "unknown")
    semantic_confidence = float(semantic.get("confidence") or 0.0)

    torso_axis = deterministic_geometry.get("torso_axis_from_vertical") or {}
    shoulder = deterministic_geometry.get("shoulder_line") or {}
    torso_abs = torso_axis.get("abs_deg")
    shoulder_abs = shoulder.get("abs_deg")

    conflict = False
    review_required = False
    projected_signal = "none"
    reasons: list[str] = []

    if torso_axis.get("status") == "usable" and torso_abs is not None and float(torso_abs) >= 15.0:
        projected_signal = "strong_torso_axis_cant"
        review_required = True
        if semantic_direction == "upright" and semantic_magnitude in {"none", "slight"} and semantic_confidence >= 0.75:
            conflict = True
            reasons.append(
                "semantic image-plane body axis is upright while DWPose projected torso axis exceeds 15 degrees"
            )
    elif shoulder.get("status") == "usable" and shoulder_abs is not None and float(shoulder_abs) >= 15.0:
        projected_signal = "strong_shoulder_cant_only"
        review_required = True
        reasons.append(
            "DWPose reports strong projected shoulder-line cant; this is not sufficient to infer torso yaw or recline"
        )
        if semantic_direction == "upright" and semantic_magnitude in {"none", "slight"} and semantic_confidence >= 0.75:
            reasons.append(
                "semantic body-axis description is very neutral despite strong projected shoulder asymmetry; manual/3-D review is warranted"
            )

    return {
        "semantic": semantic,
        "deterministic_torso_axis_from_vertical": torso_axis,
        "deterministic_shoulder_line": shoulder,
        "projected_signal": projected_signal,
        "conflict": conflict,
        "review_required": review_required,
        "selection_usable": False,
        "authority": "report_only_projected_2d_geometry",
        "reasons": reasons,
    }


def fuse_analysis_v2(analysis: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    qualified_parts, part_warnings = _qualify_body_parts(analysis, pose)
    qualified_interactions, interaction_warnings = _qualify_interactions(analysis, qualified_parts)
    geom = pose.get("target_2d_geometry") or {}

    deterministic_geometry = {
        "pose_extent_hint": geom.get("pose_extent_hint"),
        "shoulder_line": _normalize_undirected_angle(geom.get("shoulder_line_angle_from_horizontal_deg")),
        "hip_line": _normalize_undirected_angle(geom.get("hip_line_angle_from_horizontal_deg")),
        "torso_axis_from_vertical": _normalize_undirected_angle(geom.get("torso_axis_angle_from_vertical_deg")),
        "connectivity": geom.get("connectivity") or {},
        "hand_candidates": pose.get("hand_candidates") or [],
    }

    warnings = part_warnings + interaction_warnings
    body_axis_audit = _projected_body_axis_audit(analysis, deterministic_geometry)
    framing_audit = _framing_audit(analysis, deterministic_geometry)
    if body_axis_audit.get("conflict"):
        warnings.append("semantic image-plane body axis conflicts with deterministic projected torso-axis evidence")
    elif body_axis_audit.get("review_required"):
        warnings.append("projected 2-D body geometry is strong enough to warrant review but cannot establish 3-D torso orientation")
    if framing_audit.get("override_applied"):
        warnings.append("semantic framing was reconciled to full_length using DWPose extent plus distal-leg/foot evidence")

    torso_axis = deterministic_geometry["torso_axis_from_vertical"]
    if torso_axis.get("status") == "usable" and float(torso_axis.get("abs_deg") or 0.0) >= 15.0:
        warnings.append(
            "DWPose reports a strong projected torso-axis cant; keep this distinct from semantic torso yaw/recline"
        )

    return {
        "schema_version": "analysis-fusion-2.2",
        "analysis_schema_version": analysis.get("schema_version"),
        "report_only_image_summary": analysis.get("image_summary"),
        "framing_audit": framing_audit,
        "camera_audit": _camera_audit(analysis),
        "orientation_semantics": ((analysis.get("target_subject") or {}).get("orientation") or {}),
        "projected_body_axis_audit": body_axis_audit,
        "deterministic_geometry": deterministic_geometry,
        "qualified_body_parts": qualified_parts,
        "qualified_interactions": qualified_interactions,
        "scene_audit": _scene_audit(analysis),
        "non_target_entities": analysis.get("non_target_entities") or [],
        "embedded_depictions": analysis.get("embedded_depictions") or [],
        "nuisance_regions": analysis.get("nuisance_regions") or [],
        "uncertainties": analysis.get("uncertainties") or [],
        "fusion_warnings": warnings,
        "caption_authority": {
            "image_summary": "report_only_not_caption_authoritative",
            "qualified_body_parts": "authoritative_subject_to_per_item_selection_usable",
            "qualified_interactions": "authoritative_subject_to_action_and_laterality_flags",
            "framing": "use framing_audit.qualified_shot_scale when override_applied, otherwise semantic framing",
            "camera": "report_only",
            "projected_body_axis": "report_only",
            "scene_structural_specular": "report_only",
        },
        "selection_policy": {
            "camera_axes": "report_only",
            "scene_structural_specular_axes": "report_only",
            "unsupported_hand_ownership": "not_selection_authoritative",
            "hand_laterality_conflicts": "laterality_not_selection_authoritative_but_action_may_survive",
            "deterministic_full_length_framing": "qualified_reconciliation_available",
            "deterministic_image_plane_geometry": "auditable_secondary_evidence",
            "note": "Fusion-v2.2 remains audit-first. V8.1 weights are unchanged; new axes must pass regression validation before portfolio integration.",
        },
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    model_dir = run_dir / slug
    if not model_dir.is_dir():
        raise SystemExit(f"Analysis model directory does not exist: {model_dir}")

    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    if not dwpose_dir.is_dir():
        raise SystemExit(f"DWPose directory does not exist: {dwpose_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "fusion-v2" / slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = _read_json(ANALYSIS_V2_SCHEMA)

    analysis_paths = sorted(model_dir.glob("*.analysis.json"))
    written = 0
    skipped = 0
    invalid = 0
    missing_dwpose = 0
    normalized_count = 0
    index: list[dict[str, Any]] = []

    for analysis_path in analysis_paths:
        key = analysis_path.name.removesuffix(".analysis.json")
        out_path = output_dir / f"{key}.fused_v2.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        result = _read_json(analysis_path)
        analysis = result.get("analysis")
        if not isinstance(analysis, dict) or analysis.get("schema_version") != "2.0":
            invalid += 1
            index.append({"image_key": key, "status": "skipped_non_v2_or_invalid_analysis"})
            continue

        normalized, normalization_actions = normalize_analysis_v2(analysis)
        schema_errors = validate_analysis(normalized, schema)
        if schema_errors:
            invalid += 1
            index.append(
                {
                    "image_key": key,
                    "status": "skipped_schema_invalid_after_normalization",
                    "normalization_actions": normalization_actions,
                    "schema_errors": schema_errors,
                }
            )
            continue
        if normalization_actions:
            normalized_count += 1

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        if not dwpose_path.exists():
            missing_dwpose += 1
            index.append({"image_key": key, "status": "missing_dwpose"})
            continue

        pose = build_pose_evidence(_read_json(dwpose_path))
        fused = fuse_analysis_v2(normalized, pose)
        payload = {
            "image": result.get("image"),
            "model": result.get("model"),
            "analysis_path": str(analysis_path),
            "dwpose_path": str(dwpose_path),
            "analysis_schema_valid_after_normalization": True,
            "analysis_normalization_actions": normalization_actions,
            "fusion": fused,
        }
        _write_json(out_path, payload)
        written += 1
        index.append(
            {
                "image_key": key,
                "status": "written",
                "normalization_actions": normalization_actions,
                "fusion_warnings": fused.get("fusion_warnings") or [],
                "camera": fused.get("camera_audit"),
                "framing": fused.get("framing_audit"),
                "body_axis": fused.get("projected_body_axis_audit"),
                "scene": fused.get("scene_audit"),
            }
        )

    summary = {
        "schema_version": "analysis-fusion-2.2-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "analysis_model_slug": slug,
        "dwpose_dir": str(dwpose_dir),
        "output_dir": str(output_dir),
        "written": written,
        "skipped_existing": skipped,
        "normalized_records": normalized_count,
        "invalid_or_non_v2_analysis": invalid,
        "missing_dwpose": missing_dwpose,
        "records": index,
    }
    _write_json(output_dir / "fusion_v2.index.json", summary)

    print(f"Fusion-v2 output: {output_dir}")
    print(
        f"Written: {written}; reused: {skipped}; normalized: {normalized_count}; "
        f"invalid/non-v2: {invalid}; missing DWPose: {missing_dwpose}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
