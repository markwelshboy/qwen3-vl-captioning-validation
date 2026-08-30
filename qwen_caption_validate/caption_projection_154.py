from __future__ import annotations

import copy
from typing import Any

from .caption_projection_153 import build_caption_projection as _build_153
from .caption_projection_153 import lint_caption as _lint_153


_REDUNDANT_BODY_CLAIMS = {
    "signed_shoulder_nearer_relation",
    "signed_torso_depth_direction",
}


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def _semantics_root(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get("subject_geometry_semantics")
    return value if isinstance(value, dict) else payload


def _body_is_fact(root: dict[str, Any]) -> bool:
    body = root.get("body_orientation")
    return isinstance(body, dict) and body.get("status") == "FACT" and isinstance(body.get("value"), dict)


def _remove_required_claims(evidence: dict[str, Any], ids: set[str]) -> list[str]:
    removed: list[str] = []
    kept: list[Any] = []
    for item in evidence.get("required_claims") or []:
        if isinstance(item, dict) and str(item.get("id") or "") in ids:
            removed.append(str(item.get("id")))
            continue
        kept.append(copy.deepcopy(item))
    evidence["required_claims"] = kept
    return removed


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    pose_semantics: dict[str, Any] | None = None,
    subject_geometry_semantics: dict[str, Any] | None = None,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Projection 1.5.4: finish Subject Geometry semantic economy.

    Projection 1.5.3 correctly replaced legacy torso/head orientation fields with
    governed Subject Geometry Semantics, but replay on Caption02-02/PoseBlind02
    exposed two bookkeeping leaks:

    * the older required claims ``signed_shoulder_nearer_relation`` and
      ``signed_torso_depth_direction`` could survive beside the new categorical
      body orientation, recreating the redundant shoulder/depth prose we meant
      to remove;
    * Subject Geometry v0.2 stores source disagreements under
      ``cross_source_validation.conflicts`` while 1.5.3 looked for the obsolete
      root-level ``cross_source_conflicts`` key.

    v1.5.4 fixes only those two issues. Camera-subject geometry remains
    audit-only and all 1.5.3 authority rules are otherwise unchanged.
    """
    evidence, audit = _build_153(
        fused_payload,
        analysis,
        pose_semantics=pose_semantics,
        subject_geometry_semantics=subject_geometry_semantics,
        caption_policy=caption_policy,
    )
    evidence["projection_revision"] = "1.5.4"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.5.4"

    root = _semantics_root(subject_geometry_semantics)
    integration = projection.get("subject_geometry_semantics_integration")
    if isinstance(integration, dict) and root:
        nested_conflicts = ((root.get("cross_source_validation") or {}).get("conflicts") or [])
        legacy_conflicts = root.get("cross_source_conflicts") or []
        conflicts = nested_conflicts if nested_conflicts else legacy_conflicts
        integration["cross_source_conflicts_audit_only"] = copy.deepcopy(conflicts)

        if _body_is_fact(root):
            removed = _remove_required_claims(evidence, _REDUNDANT_BODY_CLAIMS)
            if removed:
                prior = [
                    str(value)
                    for value in (integration.get("legacy_required_claims_removed") or [])
                    if value
                ]
                for claim_id in removed:
                    if claim_id not in prior:
                        prior.append(claim_id)
                integration["legacy_required_claims_removed"] = prior
                integration["semantic_economy_required_claims_removed"] = removed

        integration["projection_revision"] = "1.5.4"
        integration["policy"] = (
            "FACT Subject Geometry body orientation supersedes caption-facing shoulder-nearer and torso-depth ingredient claims; "
            "source measurements and cross-source disagreements remain audit-only. Camera-subject geometry remains audit-only."
        )

    projection.setdefault("notes", []).append(
        "Projection 1.5.4 removes residual shoulder-nearer/torso-depth required claims when a FACT Subject Geometry body orientation is present and preserves v0.2 nested cross-source conflicts in audit."
    )
    return evidence, audit


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_153(caption, evidence))
    result["schema_version"] = "caption-authority-lint-1.5.4"
    return result
