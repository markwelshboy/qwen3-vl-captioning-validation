from __future__ import annotations

import re
from typing import Any


_SIDE_WORD_RE = re.compile(r"\b(?:left|right)\b\s*", re.IGNORECASE)


def _redact_laterality_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = _SIDE_WORD_RE.sub("", value)
    return re.sub(r"\s{2,}", " ", text).strip()


def _axis(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        confidence = float(value.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.50:
        return None
    return {
        "direction": value.get("direction"),
        "magnitude": value.get("magnitude"),
        "confidence": round(confidence, 3),
    }


def _depth_band(value: Any) -> str | None:
    try:
        magnitude = float(value)
    except (TypeError, ValueError):
        return None
    if magnitude < 15.0:
        return "low"
    if magnitude < 30.0:
        return "moderate"
    if magnitude < 50.0:
        return "high"
    return "very_high"


def _compact_body_part(item: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any] | None:
    fusion = item.get("fusion_v2") or {}
    if not fusion.get("selection_usable"):
        audit["blocked"].append(
            {
                "path": "fusion.qualified_body_parts",
                "reason": "body_part_not_selection_usable",
                "part": item.get("part"),
            }
        )
        return None

    side = fusion.get("qualified_anatomical_side") or item.get("anatomical_side") or "unknown"
    laterality_ok = bool(fusion.get("laterality_selection_usable"))
    if side not in {"left", "right", "midline"}:
        side = "unknown"
    if side in {"left", "right"} and not laterality_ok:
        audit["blocked"].append(
            {
                "path": "fusion.qualified_body_parts[].anatomical_side",
                "reason": "laterality_not_qualified",
                "part": item.get("part"),
                "reported_side": side,
            }
        )
        side = "unknown"

    part = str(item.get("part") or "unknown")
    geometry = item.get("geometry")
    contact = item.get("contact")
    support = item.get("support")
    if not laterality_ok:
        part = _redact_laterality_text(part)
        geometry = _redact_laterality_text(geometry)
        contact = _redact_laterality_text(contact)
        support = _redact_laterality_text(support)

    return {
        "part": part,
        "anatomical_side": side,
        "ownership": fusion.get("qualified_ownership") or item.get("ownership") or "unknown",
        "visibility": item.get("visibility"),
        "visible_subparts": item.get("visible_subparts") or [],
        "connectivity": item.get("connectivity_to_target_chain"),
        "geometry": geometry,
        "contact": contact,
        "support": support,
        "foreshortening": item.get("foreshortening"),
        "image_location": item.get("image_location"),
        "confidence": item.get("confidence"),
        "laterality_qualified": laterality_ok,
    }


def _compact_interaction(item: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any] | None:
    fusion = item.get("fusion_v2") or {}
    if not fusion.get("selection_usable"):
        audit["blocked"].append(
            {
                "path": "fusion.qualified_interactions",
                "reason": "interaction_not_selection_usable",
                "type": item.get("type"),
                "actor_part": item.get("actor_part"),
            }
        )
        return None

    laterality_ok = bool(fusion.get("laterality_selection_usable"))
    actor_part = str(item.get("actor_part") or "unknown")
    notes = item.get("notes")
    if not laterality_ok:
        actor_part = _redact_laterality_text(actor_part)
        notes = _redact_laterality_text(notes)

    return {
        "type": item.get("type"),
        "actor_part": actor_part,
        "actor_ownership": fusion.get("qualified_actor_ownership") or item.get("actor_ownership") or "unknown",
        "actor_anatomical_side": (
            fusion.get("qualified_actor_anatomical_side")
            if laterality_ok
            else "unknown"
        ),
        "target": item.get("target"),
        "evidence_status": item.get("evidence_status"),
        "confidence": item.get("confidence"),
        "notes": notes,
        "laterality_qualified": laterality_ok,
    }


def _visibility_constraints(analysis: dict[str, Any], sam3d_audit: dict[str, Any]) -> dict[str, list[str]]:
    visibility = sam3d_audit.get("landmark_visibility") or {}
    if not visibility:
        visibility = ((analysis.get("target_subject") or {}).get("geometry_landmark_visibility") or {})

    out: dict[str, list[str]] = {
        "visible": [],
        "partial": [],
        "not_visible": [],
        "unknown": [],
    }
    if not isinstance(visibility, dict):
        return out

    for name, raw in visibility.items():
        if not isinstance(raw, dict):
            continue
        state = str(raw.get("visibility") or "unknown")
        if state not in out:
            state = "unknown"
        out[state].append(str(name))
    for values in out.values():
        values.sort()
    return out


def _qualified_3d_geometry(sam3d_audit: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    provenance = sam3d_audit.get("target_provenance") or {}
    provenance_risk = provenance.get("context_risk") == "requires_review"

    if provenance_risk:
        audit["blocked"].append(
            {
                "path": "fusion.sam3d_geometry_audit",
                "reason": "target_provenance_requires_review",
            }
        )
        return out

    component_specs = (
        ("shoulder_girdle_depth_rotation", "shoulder_depth_rotation"),
        ("pelvis_depth_rotation", "hip_depth_rotation"),
    )
    for output_name, source_name in component_specs:
        source = sam3d_audit.get(source_name) or {}
        authority = source.get("authority")
        band = _depth_band(source.get("magnitude_deg"))
        if authority == "qualified_component_geometry" and band is not None:
            out[output_name] = {
                "magnitude_band": band,
                "direction": "unsigned",
                "authority": "qualified_component_geometry",
            }
            audit["allowed"].append(f"fusion.sam3d_geometry_audit.{source_name}")
        elif source:
            audit["blocked"].append(
                {
                    "path": f"fusion.sam3d_geometry_audit.{source_name}",
                    "reason": str(authority or "unqualified_component_geometry"),
                }
            )

    torso = sam3d_audit.get("torso_depth_rotation") or {}
    torso_band = _depth_band(torso.get("magnitude_deg"))
    if torso.get("authority") == "qualified_3d_geometry" and torso_band is not None:
        out["combined_torso_depth_rotation"] = {
            "magnitude_band": torso_band,
            "direction": "unsigned",
            "authority": "qualified_3d_geometry",
        }
        audit["allowed"].append("fusion.sam3d_geometry_audit.torso_depth_rotation")
    elif torso:
        audit["blocked"].append(
            {
                "path": "fusion.sam3d_geometry_audit.torso_depth_rotation",
                "reason": str(torso.get("authority") or "unqualified_torso_geometry"),
            }
        )

    if sam3d_audit.get("signed_depth_diagnostics") is not None:
        audit["blocked"].append(
            {
                "path": "fusion.sam3d_geometry_audit.signed_depth_diagnostics",
                "reason": "signed_depth_direction_not_validated",
            }
        )
    return out


def _compact_nuisance_regions(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        important = (
            item.get("frame_coverage") == "large"
            or item.get("texture_complexity") == "high"
            or item.get("structural_complexity") == "high"
            or item.get("specular_reflective") == "high"
            or bool(item.get("entropy_focus_candidate"))
        )
        if important:
            out.append(
                {
                    "description": item.get("description"),
                    "image_location": item.get("image_location"),
                    "frame_coverage": item.get("frame_coverage"),
                }
            )
    return out


def build_caption_evidence(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project Fusion-v2.3 into a caption-safe evidence object.

    The returned evidence deliberately does not expose raw SAM3D reconstruction,
    signed depth diagnostics, report-only camera claims, or body/interactions that
    failed Fusion qualification. The second return value is an audit trail for
    human inspection and must not be passed to the caption model as evidence.
    """
    fusion = fused_payload.get("fusion") if isinstance(fused_payload.get("fusion"), dict) else fused_payload
    fusion = fusion if isinstance(fusion, dict) else {}
    audit: dict[str, Any] = {
        "schema_version": "caption-evidence-firewall-audit-1.0",
        "source_fusion_schema": fusion.get("schema_version"),
        "allowed": [],
        "blocked": [],
        "notes": [],
    }

    framing_audit = fusion.get("framing_audit") or {}
    semantic_framing = framing_audit.get("semantic_framing") or analysis.get("framing") or {}
    shot_scale = semantic_framing.get("shot_scale")
    if framing_audit.get("override_applied"):
        shot_scale = framing_audit.get("qualified_shot_scale") or shot_scale
        audit["allowed"].append("fusion.framing_audit.qualified_shot_scale")

    projected_axis = fusion.get("projected_body_axis_audit") or {}
    orientation_source = fusion.get("orientation_semantics") or ((analysis.get("target_subject") or {}).get("orientation") or {})
    orientation: dict[str, Any] = {}
    for name in ("torso_yaw", "torso_pitch", "torso_roll", "head_yaw", "head_pitch", "head_roll"):
        value = _axis(orientation_source.get(name))
        if value is not None:
            orientation[name] = value
    if projected_axis.get("conflict"):
        audit["blocked"].append(
            {
                "path": "fusion.orientation_semantics.image_plane_body_axis",
                "reason": "conflicts_with_deterministic_projected_geometry",
            }
        )
    else:
        value = _axis(orientation_source.get("image_plane_body_axis"))
        if value is not None:
            orientation["image_plane_body_axis"] = value

    audit["blocked"].extend(
        [
            {"path": "analysis.image_summary", "reason": "report_only_not_caption_authoritative"},
            {"path": "fusion.camera_audit", "reason": "camera_axis_report_only"},
            {"path": "fusion.projected_body_axis_audit", "reason": "projected_2d_geometry_report_only"},
            {"path": "fusion.scene_audit.structural_axes", "reason": "scene_structural_axes_report_only"},
        ]
    )

    safe_parts = [
        compact
        for raw in (fusion.get("qualified_body_parts") or [])
        if isinstance(raw, dict)
        for compact in [_compact_body_part(raw, audit)]
        if compact is not None
    ]
    safe_interactions = [
        compact
        for raw in (fusion.get("qualified_interactions") or [])
        if isinstance(raw, dict)
        for compact in [_compact_interaction(raw, audit)]
        if compact is not None
    ]

    target = analysis.get("target_subject") or {}
    scene = analysis.get("scene") or {}
    sam3d_audit = fusion.get("sam3d_geometry_audit") or {}
    visibility = _visibility_constraints(analysis, sam3d_audit)
    qualified_3d = _qualified_3d_geometry(sam3d_audit, audit)

    non_target_entities: list[dict[str, Any]] = []
    for raw in fusion.get("non_target_entities") or analysis.get("non_target_entities") or []:
        if not isinstance(raw, dict):
            continue
        non_target_entities.append(
            {
                "description": raw.get("description"),
                "image_location": raw.get("image_location"),
                "geometry": raw.get("geometry"),
                "confidence": raw.get("confidence"),
            }
        )

    embedded_depictions: list[dict[str, Any]] = []
    for raw in fusion.get("embedded_depictions") or analysis.get("embedded_depictions") or []:
        if isinstance(raw, dict):
            embedded_depictions.append(
                {
                    "description": raw.get("description"),
                    "type": raw.get("type"),
                    "image_location": raw.get("image_location"),
                    "confidence": raw.get("confidence"),
                }
            )

    evidence = {
        "schema_version": "caption-evidence-1.0",
        "source_fusion_schema": fusion.get("schema_version"),
        "framing": {
            "shot_scale": shot_scale,
            "subject_extent": semantic_framing.get("subject_extent"),
            "subject_frame_coverage": semantic_framing.get("subject_frame_coverage"),
            "photographic_archetype": semantic_framing.get("photographic_archetype"),
        },
        "semantic_orientation": orientation,
        "gaze": target.get("gaze"),
        "expression_state": target.get("expression_state") or [],
        "visibility_constraints": visibility,
        "visible_subject_parts": safe_parts,
        "qualified_interactions": safe_interactions,
        "qualified_3d_geometry": qualified_3d,
        "scene": {
            "environment_type": scene.get("environment_type"),
            "environment_confidence": scene.get("environment_confidence"),
            "illumination": scene.get("illumination"),
        },
        "non_target_entities": non_target_entities,
        "embedded_depictions": embedded_depictions,
        "important_nuisance_regions": _compact_nuisance_regions(
            fusion.get("nuisance_regions") or analysis.get("nuisance_regions") or []
        ),
        "uncertainties": fusion.get("uncertainties") or analysis.get("uncertainties") or [],
        "evidence_policy": {
            "not_visible_is_hard_boundary": True,
            "sam3d_direction_is_never_exposed": True,
            "unqualified_laterality_is_redacted": True,
            "report_only_camera_and_projected_geometry_are_withheld": True,
            "raw_image_summary_is_withheld": True,
        },
        "coverage_limitations": [
            "Analyze-v2.1 has no dedicated structured transient-appearance/clothing object. Clothing may therefore be absent unless it appears in qualified visible-part descriptors or other structured entities.",
            "The evidence view is intentionally conservative; omitted facts must not be invented by text-only Compose.",
        ],
    }
    audit["allowed"].extend(
        [
            "fusion.framing_audit",
            "fusion.orientation_semantics",
            "fusion.qualified_body_parts[selection_usable]",
            "fusion.qualified_interactions[selection_usable]",
            "analysis.target_subject.gaze",
            "analysis.target_subject.expression_state",
            "analysis.scene.environment_type",
            "analysis.scene.illumination",
            "analysis.nuisance_regions[important]",
        ]
    )
    return evidence, audit
