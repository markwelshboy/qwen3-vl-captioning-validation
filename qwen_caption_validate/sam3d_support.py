from __future__ import annotations

import json
from typing import Any


LANDMARKS = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
HUMAN_CONTEXT_TOKENS = (
    "person",
    "people",
    "human",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "face",
    "head",
    "hand",
    "arm",
    "body",
    "portrait",
    "photo",
    "photograph",
    "painting",
    "poster",
    "reflection",
)


def _visibility_map(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    subject = analysis.get("target_subject") or {}
    raw = subject.get("geometry_landmark_visibility") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        name: value
        for name, value in raw.items()
        if name in LANDMARKS and isinstance(value, dict)
    }


def _landmark_status(visibility: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    raw = visibility.get(name) or {}
    state = str(raw.get("visibility") or "unknown")
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "landmark": name,
        "visibility": state if state in {"visible", "partial", "not_visible", "unknown"} else "unknown",
        "confidence": round(confidence, 3),
        "evidence": raw.get("evidence"),
    }


def _pair_support(visibility: dict[str, dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    items = [_landmark_status(visibility, left), _landmark_status(visibility, right)]
    states = [item["visibility"] for item in items]
    confidences = [float(item["confidence"]) for item in items]

    if all(state == "visible" for state in states) and min(confidences) >= 0.75:
        support_state = "observed_supported"
        authority = "qualified_component_geometry"
        reason = "both required anatomical regions are directly visible with >=0.75 confidence"
    elif any(state == "not_visible" and confidence >= 0.75 for state, confidence in zip(states, confidences)):
        support_state = "prior_reconstructed"
        authority = "reconstructed_prior_only"
        reason = "at least one required anatomical region is explicitly not visible; returned 3-D geometry necessarily uses body-prior completion"
    elif all(state in {"visible", "partial"} for state in states) and min(confidences) >= 0.50:
        support_state = "partially_supported"
        authority = "report_only_partial_image_support"
        reason = "both regions have some visible evidence, but at least one is partial or below full-support confidence"
    else:
        support_state = "unknown"
        authority = "report_only_visibility_unresolved"
        reason = "visibility support for the required anatomical regions is unresolved"

    return {
        "state": support_state,
        "authority": authority,
        "landmarks": items,
        "reason": reason,
    }


def _metric(metrics: dict[str, Any], new_key: str, old_key: str | None = None) -> Any:
    if new_key in metrics:
        return metrics.get(new_key)
    if old_key:
        return metrics.get(old_key)
    return None


def _may_describe_human_context(item: Any) -> bool:
    if isinstance(item, str):
        text = item.lower()
    else:
        try:
            text = json.dumps(item, ensure_ascii=False).lower()
        except TypeError:
            text = str(item).lower()
    return any(token in text for token in HUMAN_CONTEXT_TOKENS)


def _target_provenance_audit(analysis: dict[str, Any], sam3d_record: dict[str, Any]) -> dict[str, Any]:
    embedded = analysis.get("embedded_depictions") or []
    non_target = analysis.get("non_target_entities") or []
    embedded_human_like = [item for item in embedded if _may_describe_human_context(item)] if isinstance(embedded, list) else []
    non_target_human_like = [item for item in non_target if _may_describe_human_context(item)] if isinstance(non_target, list) else []
    has_context_risk = bool(embedded_human_like or non_target_human_like)
    bbox = sam3d_record.get("bbox") or {}

    return {
        "sam3d_bbox_source": bbox.get("source"),
        "embedded_depiction_count": len(embedded) if isinstance(embedded, list) else None,
        "non_target_entity_count": len(non_target) if isinstance(non_target, list) else None,
        "human_like_embedded_depiction_count": len(embedded_human_like),
        "human_like_non_target_entity_count": len(non_target_human_like),
        "context_risk": "requires_review" if has_context_risk else "no_semantic_multi_subject_risk_detected",
        "authority": (
            "preselected_bbox_requires_target_provenance_review"
            if has_context_risk
            else "preselected_bbox_no_semantic_conflict_detected"
        ),
        "note": (
            "SAM 3D Body reconstructs the supplied bbox; it does not prove that the bbox belongs to the intended identity. "
            "Human-like embedded depictions/non-target people therefore remain an upstream target-selection concern."
        ),
    }


def qualify_sam3d_geometry(analysis: dict[str, Any], sam3d_record: dict[str, Any]) -> dict[str, Any]:
    """Qualify SAM 3D Body metrics by whether their landmark inputs are visible.

    SAM3D reconstructs a complete parametric body even when anatomy lies outside
    the crop or behind occlusion. A returned joint is therefore not equivalent
    to an observed joint. Analyze-v2.1 supplies the semantic visibility support
    used here; DWPose is deliberately not allowed to promote visibility because
    the partial-body regression showed that 2-D pose can also complete distal
    landmarks outside the visible crop.
    """
    visibility = _visibility_map(analysis)
    metrics = sam3d_record.get("metrics") or {}

    shoulders = _pair_support(visibility, "left_shoulder", "right_shoulder")
    hips = _pair_support(visibility, "left_hip", "right_hip")

    shoulder_value = _metric(metrics, "shoulder_out_of_image_plane_deg")
    hip_value = _metric(metrics, "hip_out_of_image_plane_deg")
    combined_value = _metric(metrics, "torso_depth_rotation_proxy_deg")
    torso_axis_value = _metric(
        metrics,
        "torso_axis_out_of_image_plane_deg",
        "torso_depth_tilt_deg",
    )

    if shoulders["state"] == "observed_supported" and hips["state"] == "observed_supported":
        aggregate_state = "observed_supported"
        aggregate_authority = "qualified_3d_geometry"
        aggregate_reason = "both shoulder and hip depth axes are constrained by directly visible landmark regions"
    elif "prior_reconstructed" in {shoulders["state"], hips["state"]}:
        if "observed_supported" in {shoulders["state"], hips["state"]}:
            aggregate_state = "partially_supported"
            aggregate_authority = "report_only_partial_image_support"
            aggregate_reason = "one torso axis is image-supported while the other depends on reconstructed invisible anatomy"
        else:
            aggregate_state = "prior_reconstructed"
            aggregate_authority = "reconstructed_prior_only"
            aggregate_reason = "the combined torso metric depends materially on anatomy explicitly absent from the crop"
    elif "partially_supported" in {shoulders["state"], hips["state"]}:
        aggregate_state = "partially_supported"
        aggregate_authority = "report_only_partial_image_support"
        aggregate_reason = "the combined torso metric has incomplete landmark visibility support"
    else:
        aggregate_state = "unknown"
        aggregate_authority = "report_only_visibility_unresolved"
        aggregate_reason = "landmark visibility support is unavailable or unresolved"

    visibility_available = bool(visibility)
    if not visibility_available:
        aggregate_state = "unknown"
        aggregate_authority = "report_only_requires_analyze_v2_1_visibility"
        aggregate_reason = "Analyze-v2.1 geometry_landmark_visibility is absent; legacy analyses cannot grant SAM3D geometry authority"

    provenance = _target_provenance_audit(analysis, sam3d_record)
    if provenance["context_risk"] == "requires_review" and aggregate_authority == "qualified_3d_geometry":
        aggregate_authority = "qualified_geometry_pending_target_provenance"
        aggregate_reason += "; semantic context contains another person or embedded human depiction, so target-bbox provenance still requires review"

    return {
        "schema_version": "sam3d-support-audit-0.1",
        "sam3d_schema_version": sam3d_record.get("schema_version"),
        "landmark_visibility_available": visibility_available,
        "landmark_visibility": {
            name: _landmark_status(visibility, name) for name in LANDMARKS
        },
        "target_provenance": provenance,
        "shoulder_depth_rotation": {
            "magnitude_deg": shoulder_value,
            "support": shoulders,
            "authority": shoulders["authority"] if visibility_available else "report_only_requires_analyze_v2_1_visibility",
        },
        "hip_depth_rotation": {
            "magnitude_deg": hip_value,
            "support": hips,
            "authority": hips["authority"] if visibility_available else "report_only_requires_analyze_v2_1_visibility",
        },
        "torso_depth_rotation": {
            "magnitude_deg": combined_value,
            "support_state": aggregate_state,
            "authority": aggregate_authority,
            "reason": aggregate_reason,
            "direction": "unsigned",
            "caption_usable": aggregate_authority == "qualified_3d_geometry",
            "selection_usable": False,
        },
        "torso_axis_out_of_image_plane": {
            "magnitude_deg": torso_axis_value,
            "support_state": aggregate_state,
            "authority": aggregate_authority,
            "reason": (
                "camera-relative hip-midpoint to shoulder-midpoint depth tilt; this is not world-relative recline/gravity"
            ),
            "selection_usable": False,
        },
        "signed_depth_diagnostics": metrics.get("signed_depth_fraction_diagnostics"),
        "method_validation": {
            "bbox_robustness": "passed_four_crop_regression_2026_08_20",
            "validated_property": "unsigned torso depth-rotation ordering remained stable across DWPose bbox padding 0.10/0.20/0.35 and full-image input",
            "selection_integration": "not_yet_enabled",
        },
        "limitations": [
            "SAM 3D Body reconstructs complete anatomy and can return plausible geometry for body regions outside the crop.",
            "Visibility support comes from Analyze-v2.1 and is semantic evidence, not a segmentation mask.",
            "Unsigned depth magnitude has been validated on the current regression set; signed rotation direction remains diagnostic only.",
            "Target bbox provenance is not automatically proven by SAM3D itself.",
        ],
    }
