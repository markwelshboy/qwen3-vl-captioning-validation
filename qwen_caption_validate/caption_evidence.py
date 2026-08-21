from __future__ import annotations

import re
from typing import Any


_ANATOMICAL_SIDE_RE = re.compile(
    r"(?<![A-Za-z0-9])anatomical[\s_-]*(?:left|right)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SIDE_WORD_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:left|right)(?=$|[^A-Za-z0-9])[\s_-]*",
    re.IGNORECASE,
)
_PLURAL_HAND_RE = re.compile(r"\bboth\s+hands\b|\bhands\b", re.IGNORECASE)


def _redact_laterality_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = _ANATOMICAL_SIDE_RE.sub("side-unspecified", value)
    text = _SIDE_WORD_RE.sub("", text)
    text = re.sub(r"\banatomical\b(?=\s*(?:$|[,;:.]))", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[_-]{2,}", "-", text)
    text = re.sub(r"\s+([,;:.])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(" _-")


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


def _frame_location(value: Any) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    return {
        "reference": "image_frame_only",
        "description": str(value),
    }


def _compact_gaze(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    direction = str(value.get("image_direction") or "unknown")
    direction_map = {
        "image_left": "left_side_of_image_frame",
        "image_center": "center_of_image_frame",
        "image_right": "right_side_of_image_frame",
        "unknown": "unknown",
    }
    return {
        "target": value.get("target"),
        "frame_direction": direction_map.get(direction, "unknown"),
    }


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
    visible_subparts = [str(value) for value in (item.get("visible_subparts") or [])]
    geometry = item.get("geometry")
    contact = item.get("contact")
    support = item.get("support")
    if not laterality_ok:
        part = _redact_laterality_text(part)
        visible_subparts = [_redact_laterality_text(value) for value in visible_subparts]
        geometry = _redact_laterality_text(geometry)
        contact = _redact_laterality_text(contact)
        support = _redact_laterality_text(support)

    # Image-space location is deliberately withheld for target anatomy. In the
    # first firewall experiment 4B sometimes converted frame-right into
    # anatomical right. Target-part position is rarely worth that ambiguity.
    if item.get("image_location"):
        audit["blocked"].append(
            {
                "path": "fusion.qualified_body_parts[].image_location",
                "reason": "target_frame_location_can_be_confused_with_anatomical_laterality",
                "part": item.get("part"),
            }
        )

    return {
        "part": part,
        "anatomical_side": side,
        "ownership": fusion.get("qualified_ownership") or item.get("ownership") or "unknown",
        "visibility": item.get("visibility"),
        "visible_subparts": visible_subparts,
        "connectivity": item.get("connectivity_to_target_chain"),
        "geometry": geometry,
        "contact": contact,
        "support": support,
        "foreshortening": item.get("foreshortening"),
        "confidence": item.get("confidence"),
        "laterality_qualified": laterality_ok,
    }


def _is_explicit_hand_observation(item: dict[str, Any]) -> bool:
    text = " ".join(
        [str(item.get("part") or ""), *[str(v) for v in (item.get("visible_subparts") or [])]]
    ).lower()
    return "hand" in text or "finger" in text


def _qualified_distinct_hand_sides(items: list[dict[str, Any]]) -> set[str]:
    sides: set[str] = set()
    for item in items:
        fusion = item.get("fusion_v2") or {}
        if not fusion.get("selection_usable"):
            continue
        if (fusion.get("qualified_ownership") or item.get("ownership")) != "target":
            continue
        if not fusion.get("laterality_selection_usable"):
            continue
        if not _is_explicit_hand_observation(item):
            continue
        side = fusion.get("qualified_anatomical_side") or item.get("anatomical_side")
        if side in {"left", "right"}:
            sides.add(str(side))
    return sides


def _compact_interaction(
    item: dict[str, Any],
    audit: dict[str, Any],
    qualified_hand_sides: set[str],
) -> dict[str, Any] | None:
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

    actor_part_raw = str(item.get("actor_part") or "unknown")
    notes_raw = str(item.get("notes") or "")
    if _PLURAL_HAND_RE.search(actor_part_raw) or _PLURAL_HAND_RE.search(notes_raw):
        if qualified_hand_sides != {"left", "right"}:
            audit["blocked"].append(
                {
                    "path": "fusion.qualified_interactions",
                    "reason": "plural_hand_interaction_lacks_two_distinct_qualified_hands",
                    "type": item.get("type"),
                    "actor_part": item.get("actor_part"),
                    "qualified_hand_sides": sorted(qualified_hand_sides),
                }
            )
            return None

    laterality_ok = bool(fusion.get("laterality_selection_usable"))
    actor_part = actor_part_raw
    notes: Any = item.get("notes")
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


def _required_claims(qualified_3d: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    descriptions = {
        "shoulder_girdle_depth_rotation": "Mention the visible shoulder depth staggering/rotation in side-neutral language.",
        "pelvis_depth_rotation": "Mention the visible pelvis/hip depth rotation in side-neutral language.",
        "combined_torso_depth_rotation": "Mention the qualified visible torso depth rotation in side-neutral language.",
    }
    for name, value in qualified_3d.items():
        if not isinstance(value, dict):
            continue
        band = value.get("magnitude_band")
        if band not in {"high", "very_high"}:
            continue
        claims.append(
            {
                "id": name,
                "priority": "required",
                "magnitude_band": band,
                "instruction": descriptions.get(name),
                "constraints": [
                    "unsigned",
                    "do_not_name_anatomical_side_from_this_claim",
                    "do_not_use_numeric_angle",
                ],
            }
        )
    return claims


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
                    "frame_location": _frame_location(item.get("image_location")),
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
    signed depth diagnostics, report-only camera claims, raw uncertainty prose,
    target-part frame locations, or body/interactions that failed qualification.
    The second return value is an audit trail for human inspection and must not be
    passed to the caption model as evidence.
    """
    fusion = fused_payload.get("fusion") if isinstance(fused_payload.get("fusion"), dict) else fused_payload
    fusion = fusion if isinstance(fusion, dict) else {}
    audit: dict[str, Any] = {
        "schema_version": "caption-evidence-firewall-audit-1.1",
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
            if value.get("direction") in {"anatomical_left", "anatomical_right"}:
                audit["blocked"].append(
                    {
                        "path": f"fusion.orientation_semantics.{name}.direction",
                        "reason": "anatomical_direction_not_independently_qualified",
                    }
                )
                value["direction"] = "side_unspecified"
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
            {"path": "fusion.uncertainties", "reason": "free_form_uncertainty_text_not_caption_authoritative"},
            {"path": "analysis.uncertainties", "reason": "free_form_uncertainty_text_not_caption_authoritative"},
        ]
    )

    raw_parts = [
        raw for raw in (fusion.get("qualified_body_parts") or []) if isinstance(raw, dict)
    ]
    safe_parts = [
        compact
        for raw in raw_parts
        for compact in [_compact_body_part(raw, audit)]
        if compact is not None
    ]
    qualified_hand_sides = _qualified_distinct_hand_sides(raw_parts)
    safe_interactions = [
        compact
        for raw in (fusion.get("qualified_interactions") or [])
        if isinstance(raw, dict)
        for compact in [_compact_interaction(raw, audit, qualified_hand_sides)]
        if compact is not None
    ]

    target = analysis.get("target_subject") or {}
    scene = analysis.get("scene") or {}
    sam3d_audit = fusion.get("sam3d_geometry_audit") or {}
    visibility = _visibility_constraints(analysis, sam3d_audit)
    qualified_3d = _qualified_3d_geometry(sam3d_audit, audit)
    required_claims = _required_claims(qualified_3d)

    non_target_entities: list[dict[str, Any]] = []
    for raw in fusion.get("non_target_entities") or analysis.get("non_target_entities") or []:
        if not isinstance(raw, dict):
            continue
        non_target_entities.append(
            {
                "description": raw.get("description"),
                "frame_location": _frame_location(raw.get("image_location")),
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
                    "frame_location": _frame_location(raw.get("image_location")),
                    "confidence": raw.get("confidence"),
                }
            )

    evidence = {
        "schema_version": "caption-evidence-1.1",
        "source_fusion_schema": fusion.get("schema_version"),
        "framing": {
            "shot_scale": shot_scale,
            "subject_extent": semantic_framing.get("subject_extent"),
            "subject_frame_coverage": semantic_framing.get("subject_frame_coverage"),
            "photographic_archetype": semantic_framing.get("photographic_archetype"),
        },
        "semantic_orientation": orientation,
        "gaze": _compact_gaze(target.get("gaze")),
        "expression_state": target.get("expression_state") or [],
        "visibility_constraints": visibility,
        "visible_subject_parts": safe_parts,
        "qualified_interactions": safe_interactions,
        "qualified_3d_geometry": qualified_3d,
        "required_claims": required_claims,
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
        "evidence_policy": {
            "not_visible_is_hard_boundary": True,
            "sam3d_direction_is_never_exposed": True,
            "unqualified_laterality_is_redacted": True,
            "unqualified_semantic_anatomical_direction_is_redacted": True,
            "target_body_frame_locations_are_withheld": True,
            "remaining_frame_locations_are_explicitly_non_anatomical": True,
            "plural_hand_interactions_require_two_distinct_qualified_hands": True,
            "high_3d_geometry_is_promoted_to_required_claims": True,
            "raw_uncertainties_are_withheld": True,
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
            "fusion.qualified_interactions[selection_usable_and_cardinality_safe]",
            "analysis.target_subject.gaze[target+frame_direction_only]",
            "analysis.target_subject.expression_state",
            "analysis.scene.environment_type",
            "analysis.scene.illumination",
            "analysis.nuisance_regions[important]",
        ]
    )
    if required_claims:
        audit["notes"].append(
            "High/very-high qualified 3-D geometry is promoted to required_claims for Compose and checked by the post-generation linter."
        )
    return evidence, audit
