from __future__ import annotations

import copy
from typing import Any

from .caption_projection_150 import build_caption_projection as _build_150
from .caption_projection_150 import lint_caption as _lint_150


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def _apply_structural_semantic_economy(
    evidence: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    """Remove component pose evidence once Pose Semantics is the caption authority.

    Projection 1.5.0 installed the semantic posture/gesture layer but left the
    component body-part and interaction arrays visible to Compose.  A text model
    can therefore ignore the semantic-economy instruction and re-serialize knee,
    foot, arm, contact, and support details that were only useful for verification.

    Projection 1.5.1 makes that boundary structural: semantic pose primitives stay
    caption-facing, while component pose/support bookkeeping is quarantined to the
    projection audit.  Independent orientation/head/framing/scene/appearance fields
    remain available because they are not subsumed by whole-body posture.
    """
    projection = _projection_root(audit)
    pose = evidence.get("pose_orientation")
    if not isinstance(pose, dict):
        return
    semantic_pose = pose.get("semantic_pose")
    if not isinstance(semantic_pose, dict) or semantic_pose.get("authority") != "pose-semantics-0.10":
        return

    removed_fields: dict[str, int] = {}
    for field in ("visible_subject_parts", "qualified_interactions", "gesture_semantics"):
        value = pose.get(field)
        if isinstance(value, list):
            removed_fields[field] = len(value)
            pose[field] = []
        elif value is not None:
            removed_fields[field] = 1
            pose.pop(field, None)

    removed_claim_ids: list[str] = []
    kept_claims: list[dict[str, Any]] = []
    for item in evidence.get("required_claims") or []:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("id") or "")
        component_claim = (
            claim_id.startswith("salient_interaction_")
            or claim_id == "chin_rest_on_hand_gesture"
        )
        if component_claim:
            removed_claim_ids.append(claim_id)
            continue
        kept_claims.append(copy.deepcopy(item))
    evidence["required_claims"] = kept_claims

    integration = projection.setdefault("pose_semantics_integration", {})
    integration["structural_semantic_economy"] = {
        "enabled": True,
        "component_pose_fields_removed": removed_fields,
        "component_required_claims_removed": removed_claim_ids,
        "caption_pose_authority": "pose_orientation.semantic_pose",
        "independent_context_retained": [
            "semantic_orientation",
            "upper_torso_depth_relation",
            "head_torso_relation",
            "framing_camera",
            "appearance",
            "environment_lighting",
            "scene/non-target context",
        ],
        "policy": (
            "component body-part/contact/support evidence used to establish a semantic posture or gesture "
            "is audit-only once Pose Semantics v0.10 is supplied; Compose receives semantic pose primitives "
            "plus independent orientation, framing, appearance, scene, and lighting evidence"
        ),
    }
    projection.setdefault("blocked", []).append(
        {
            "path": "caption-evidence.pose_orientation.component_pose_evidence",
            "reason": "semantic_pose_primitives_structurally_subsume_component_pose_support_bookkeeping",
            "removed_fields": removed_fields,
            "removed_required_claim_ids": removed_claim_ids,
        }
    )


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    pose_semantics: dict[str, Any] | None = None,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_150(
        fused_payload,
        analysis,
        pose_semantics=pose_semantics,
        caption_policy=caption_policy,
    )
    evidence["projection_revision"] = "1.5.1"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.5.1"

    if pose_semantics is not None:
        _apply_structural_semantic_economy(evidence, audit)

    projection.setdefault("notes", []).append(
        "Projection 1.5.1 makes semantic economy structural: when Pose Semantics v0.10 is supplied, "
        "component visible-body-part, qualified-interaction, and legacy gesture evidence is withheld from Compose."
    )
    return evidence, audit


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_150(caption, evidence))
    result["schema_version"] = "caption-authority-lint-1.5.1"
    return result
