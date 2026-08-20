from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .pose_evidence import build_pose_evidence
from .runner import model_slug, resolve_model_id


HANDISH_TOKENS = ("finger", "hand", "wrist", "forearm", "arm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-fusion-v2",
        description=(
            "Fuse Analyze-v2 semantic observations with cached DWPose evidence. "
            "The stage is deliberately audit-first: it preserves raw semantic observations, "
            "adds deterministic geometry, and downgrades unsupported ownership/action claims."
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


def _supported_hand_sides(pose: dict[str, Any]) -> set[str]:
    return {
        str(item.get("nearest_visible_target_wrist"))
        for item in (pose.get("hand_candidates") or [])
        if item.get("supported_by_nearby_visible_target_wrist")
        and item.get("nearest_visible_target_wrist") in {"left", "right"}
    }


def _qualify_body_parts(analysis: dict[str, Any], pose: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    subject = analysis.get("target_subject") or {}
    parts = subject.get("visible_body_parts") or []
    supported_sides = _supported_hand_sides(pose)
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
        handish = _is_handish(part)

        selection_usable = True
        qualified_ownership = ownership
        reasons: list[str] = []

        if handish:
            deterministic_support = side in supported_sides if side in {"left", "right"} else bool(supported_sides)
            visible_chain = connectivity == "connected_visible"
            occluded_chain = connectivity == "connected_but_occluded"
            disconnected = connectivity in {"disconnected_in_crop", "unknown"}

            if ownership == "target":
                if visible_chain:
                    reasons.append("Analyze-v2 reports a visible target limb chain")
                elif occluded_chain and deterministic_support:
                    reasons.append("short semantic occlusion gap is supported by DWPose hand-root/wrist association")
                elif deterministic_support:
                    reasons.append("DWPose hand root is near a visible target wrist")
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

            if str(item.get("visibility") or "") == "fragment" and part.lower() == "fingers":
                selection_usable = False
                reasons.append("isolated finger fragment cannot establish an unseen hand/arm chain")

        item["fusion_v2"] = {
            "qualified_ownership": qualified_ownership,
            "selection_usable": selection_usable,
            "reasons": reasons,
        }
        qualified.append(item)

    return qualified, warnings


def _qualify_interactions(
    analysis: dict[str, Any],
    qualified_parts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    subject = analysis.get("target_subject") or {}
    interactions = subject.get("interactions") or []
    warnings: list[str] = []
    out: list[dict[str, Any]] = []

    # Analyze-v2 intentionally keeps this conservative. Until body-part IDs are
    # mandatory, a hand interaction may use any compatible qualified target handish
    # observation as supporting context, but disconnected/unknown fragments can never
    # make the interaction selection-authoritative.
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
        reasons: list[str] = []

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
            elif actor_ownership == "unknown":
                selection_usable = False
                reasons.append("actor ownership is unresolved")

        item["fusion_v2"] = {
            "qualified_actor_ownership": qualified_actor_ownership,
            "selection_usable": selection_usable,
            "reasons": reasons,
        }
        out.append(item)
    return out, warnings


def _camera_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    camera = analysis.get("camera") or {}
    elevation = str(camera.get("elevation") or "unknown")
    confidence = float(camera.get("elevation_confidence") or 0.0)
    evidence = [str(x) for x in (camera.get("elevation_evidence") or []) if x]
    counter = [str(x) for x in (camera.get("elevation_counterevidence") or []) if x]

    qualified = elevation != "unknown" and confidence >= 0.80 and bool(evidence)
    reasons: list[str] = []
    if elevation == "unknown":
        reasons.append("Analyze-v2 left camera elevation unresolved")
    if confidence < 0.80:
        reasons.append("camera elevation confidence below 0.80 audit threshold")
    if elevation != "unknown" and not evidence:
        reasons.append("camera elevation lacks explicit visual evidence")
    if counter:
        reasons.append("counterevidence is present and should remain visible to downstream policy")

    return {
        "elevation": elevation,
        "confidence": round(confidence, 3),
        "evidence": evidence,
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
    torso_axis = deterministic_geometry["torso_axis_from_vertical"]
    if torso_axis.get("status") == "usable" and float(torso_axis.get("abs_deg") or 0.0) >= 15.0:
        warnings.append(
            "DWPose reports a strong image-plane torso-axis cant; keep this distinct from semantic torso yaw/recline"
        )

    return {
        "schema_version": "analysis-fusion-2.0",
        "analysis_schema_version": analysis.get("schema_version"),
        "image_summary": analysis.get("image_summary"),
        "framing": analysis.get("framing") or {},
        "camera_audit": _camera_audit(analysis),
        "orientation_semantics": ((analysis.get("target_subject") or {}).get("orientation") or {}),
        "deterministic_geometry": deterministic_geometry,
        "qualified_body_parts": qualified_parts,
        "qualified_interactions": qualified_interactions,
        "scene_audit": _scene_audit(analysis),
        "non_target_entities": analysis.get("non_target_entities") or [],
        "embedded_depictions": analysis.get("embedded_depictions") or [],
        "nuisance_regions": analysis.get("nuisance_regions") or [],
        "uncertainties": analysis.get("uncertainties") or [],
        "fusion_warnings": warnings,
        "selection_policy": {
            "camera_axes": "report_only",
            "scene_structural_specular_axes": "report_only",
            "unsupported_hand_ownership": "not_selection_authoritative",
            "deterministic_image_plane_geometry": "auditable_secondary_evidence",
            "note": "Fusion-v2 is intentionally audit-first. Do not feed new camera/scene axes into V8.1 selection weights until regression validation passes.",
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

    analysis_paths = sorted(model_dir.glob("*.analysis.json"))
    written = 0
    skipped = 0
    invalid = 0
    missing_dwpose = 0
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

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        if not dwpose_path.exists():
            missing_dwpose += 1
            index.append({"image_key": key, "status": "missing_dwpose"})
            continue

        pose = build_pose_evidence(_read_json(dwpose_path))
        fused = fuse_analysis_v2(analysis, pose)
        payload = {
            "image": result.get("image"),
            "model": result.get("model"),
            "analysis_path": str(analysis_path),
            "dwpose_path": str(dwpose_path),
            "fusion": fused,
        }
        _write_json(out_path, payload)
        written += 1
        index.append({
            "image_key": key,
            "status": "written",
            "fusion_warnings": fused.get("fusion_warnings") or [],
            "camera": fused.get("camera_audit"),
            "scene": fused.get("scene_audit"),
        })

    summary = {
        "schema_version": "analysis-fusion-2.0-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "analysis_model_slug": slug,
        "dwpose_dir": str(dwpose_dir),
        "output_dir": str(output_dir),
        "written": written,
        "skipped_existing": skipped,
        "invalid_or_non_v2_analysis": invalid,
        "missing_dwpose": missing_dwpose,
        "records": index,
    }
    _write_json(output_dir / "fusion_v2.index.json", summary)

    print(f"Fusion-v2 output: {output_dir}")
    print(f"Written: {written}; reused: {skipped}; invalid/non-v2: {invalid}; missing DWPose: {missing_dwpose}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
