from __future__ import annotations

import copy
import re
from typing import Any

from . import caption_projection_156 as v156


_DEPTH_COMPONENT_KEYS = {
    "shoulder_girdle_depth_rotation",
    "pelvis_depth_rotation",
    "combined_torso_depth_rotation",
}
_DEPTH_CLAIM_IDS = {
    *_DEPTH_COMPONENT_KEYS,
    "signed_shoulder_nearer_relation",
    "signed_torso_depth_direction",
    "upper_torso_side_on_relation",
}
_LOW_LEVEL_DEPTH_RE = re.compile(
    r"\b(?:shoulder\s+girdle|shoulders?|pelvis|hips?|torso|upper\s+body)\b[^.!?]{0,80}"
    r"\b(?:depth\s+(?:rotation|stagger(?:ing)?)|stagger(?:ed|ing)?\s+in\s+depth)\b|"
    r"\b(?:depth\s+(?:rotation|stagger(?:ing)?)|stagger(?:ed|ing)?\s+in\s+depth)\b[^.!?]{0,80}"
    r"\b(?:shoulder\s+girdle|shoulders?|pelvis|hips?|torso|upper\s+body)\b",
    re.I,
)
_FRONTAL_BODY_RE = re.compile(
    r"\b(?:body|torso|upper\s+body)\b[^.!?]{0,45}\b(?:near[- ]frontal|frontal|square[- ]on)\b|"
    r"\b(?:near[- ]frontal|frontal|square[- ]on)\b[^.!?]{0,45}\b(?:body|torso|upper\s+body)\b",
    re.I,
)


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def _remove_claims(evidence: dict[str, Any], ids: set[str]) -> list[str]:
    removed: list[str] = []
    kept: list[dict[str, Any]] = []
    for item in evidence.get("required_claims") or []:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("id") or "")
        if claim_id in ids:
            removed.append(claim_id)
            continue
        kept.append(copy.deepcopy(item))
    evidence["required_claims"] = kept
    return removed


def _apply_structural_economy_157(evidence: dict[str, Any], audit: dict[str, Any]) -> None:
    """Quarantine component geometry and positive visibility once semantics exist.

    Replay of 1.5.6 exposed two remaining side channels into Compose:
      * qualified_3d_geometry still exposed unsigned shoulder/pelvis/torso depth
        components even after Subject Geometry supplied the governed body yaw;
      * visibility_constraints.visible/partial gave Compose an attractive body-part
        checklist even though Pose Semantics had already removed visible_subject_parts.

    Neither is needed for caption generation.  Negative visibility remains a hard
    boundary for lint; positive/partial/unknown visibility stays audit-only.
    """
    projection = _projection_root(audit)
    pose = evidence.get("pose_orientation") or {}
    integration = projection.get("subject_geometry_semantics_integration") or {}
    fact_source = integration.get("fact_source") or {}
    body_fact = fact_source.get("body_orientation")

    economy = projection.setdefault("projection_157_structural_economy", {})

    removed_geometry: dict[str, Any] = {}
    removed_claims: list[str] = []
    if isinstance(body_fact, dict):
        q3d = pose.get("qualified_3d_geometry")
        if isinstance(q3d, dict):
            for key in list(q3d):
                if key in _DEPTH_COMPONENT_KEYS:
                    removed_geometry[key] = copy.deepcopy(q3d.pop(key))
        if "upper_torso_depth_relation" in pose:
            removed_geometry["upper_torso_depth_relation"] = copy.deepcopy(pose.pop("upper_torso_depth_relation"))
        removed_claims = _remove_claims(evidence, _DEPTH_CLAIM_IDS)

    visibility = evidence.get("visibility_constraints")
    removed_positive_visibility: dict[str, Any] = {}
    if isinstance(visibility, dict):
        for key in ("visible", "partial", "unknown"):
            if key in visibility:
                removed_positive_visibility[key] = copy.deepcopy(visibility.pop(key))
        visibility.setdefault("not_visible", [])

    economy.update(
        {
            "body_fact_present": isinstance(body_fact, dict),
            "component_depth_geometry_audit_only": removed_geometry,
            "component_depth_required_claims_removed": removed_claims,
            "positive_visibility_audit_only": removed_positive_visibility,
            "caption_visibility_policy": "not_visible_only",
            "policy": (
                "Governed Subject Geometry body yaw subsumes shoulder/pelvis/torso depth ingredients. "
                "Pose Semantics/Subject Geometry also subsume positive anatomy visibility checklists; only hard not-visible constraints remain caption-facing."
            ),
        }
    )
    projection.setdefault("blocked", []).append(
        {
            "path": "caption-evidence.pose_orientation.qualified_3d_geometry[body-yaw components]",
            "reason": "subject_geometry_body_fact_semantically_subsumes_component_depth_rotation",
            "removed_keys": sorted(removed_geometry),
        }
    )
    projection.setdefault("blocked", []).append(
        {
            "path": "caption-evidence.visibility_constraints[visible|partial|unknown]",
            "reason": "positive_visibility_is_verification_bookkeeping_not_caption_content",
            "removed_categories": sorted(removed_positive_visibility),
        }
    )


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    pose_semantics: dict[str, Any] | None = None,
    subject_geometry_semantics: dict[str, Any] | None = None,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = v156.build_caption_projection(
        fused_payload,
        analysis,
        pose_semantics=pose_semantics,
        subject_geometry_semantics=subject_geometry_semantics,
        caption_policy=caption_policy,
    )
    evidence["projection_revision"] = "1.5.7"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.5.7"
    _apply_structural_economy_157(evidence, audit)
    projection.setdefault("notes", []).append(
        "Projection 1.5.7 structurally removes body-yaw component depth geometry/claims once Subject Geometry is authoritative and exposes only hard not-visible visibility constraints to Compose."
    )
    return evidence, audit


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(v156.lint_caption(caption, evidence))
    violations = list(result.get("violations") or [])

    pose = evidence.get("pose_orientation") or {}
    subject_orientation = pose.get("subject_geometry_orientation") or {}
    body_orientation = subject_orientation.get("body_orientation")

    if not isinstance(body_orientation, dict):
        match = _FRONTAL_BODY_RE.search(caption)
        if match:
            violations.append(
                {
                    "type": "suppressed_frontal_body_orientation_resurfaced",
                    "text": match.group(0),
                    "description": "frontal/near-frontal body yaw was deliberately suppressed as low-information and must remain audit-only",
                }
            )

    depth_match = _LOW_LEVEL_DEPTH_RE.search(caption)
    if depth_match:
        violations.append(
            {
                "type": "component_depth_geometry_resurfaced",
                "text": depth_match.group(0),
                "description": "shoulder/pelvis/torso depth-rotation ingredients are audit-only once governed Subject Geometry is available",
            }
        )

    result["schema_version"] = "caption-authority-lint-1.5.7"
    result["violations"] = v156.v155._dedupe_findings(violations)
    result["violation_count"] = len(result["violations"])
    result["passed"] = not result["violations"]
    return result
