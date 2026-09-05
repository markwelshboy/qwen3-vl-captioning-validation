from __future__ import annotations

from copy import deepcopy
from typing import Any


ADAPTER_VERSION = "semantic-v3-pose-adapter-0.1"

_POSE_MAP = {
    "sitting": "seated",
    "seated": "seated",
    "standing": "standing",
    "reclined": "reclining",
    "reclining": "reclining",
    "lying": "lying",
    "crouching": "crouching",
    "squatting": "squatting",
    "kneeling": "kneeling",
    "walking": "walking",
    "uncertain": "unknown",
    "unknown": "unknown",
    None: "unknown",
}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_pose(value: Any) -> str:
    key = str(value).strip().lower() if value is not None else None
    return _POSE_MAP.get(key, "unknown")


def adapt_pose_v016(record: dict[str, Any]) -> dict[str, Any]:
    """Project the frozen relational-pose-profile-0.16 record into Fusion authority.

    Fusion consumes governed/public posture, reconstruction provenance, semantic-recovery
    policy, modifiers, and governed physical relations. It intentionally does not derive
    semantics again from raw DWPose/SAM3D arrays.
    """
    profile = _dict(record.get("profile"))
    projected = _dict(profile.get("sam3d_projected_pose"))
    assertion = _dict(projected.get("assertion_authority"))
    recovery = _dict(projected.get("semantic_recovery"))
    physical = _dict(projected.get("physical_governance"))
    modifiers = _dict(projected.get("posture_modifier_diagnostic"))
    relations = _dict(profile.get("relations"))

    public_pose = _normalize_pose(projected.get("pose"))
    best_candidate = _normalize_pose(projected.get("best_candidate_pose"))
    governed_best_score = physical.get("governed_best_score")
    if governed_best_score is None:
        scores = _dict(projected.get("posture_scores"))
        governed_best_score = scores.get(projected.get("best_candidate_pose"))

    if public_pose != "unknown":
        authority_state = "asserted"
    elif assertion.get("withheld_reason") or recovery.get("needed") is True:
        authority_state = "withheld"
    else:
        authority_state = "unknown"

    adapted_relations: dict[str, Any] = {}
    for name in ("head_supported_by_hand", "head_supported_by_fist", "hands_on_hips"):
        relation = _dict(relations.get(name))
        if not relation:
            continue
        geometry_match = relation.get("geometry_match")
        if geometry_match is True:
            value: bool | None = True
        elif geometry_match is False:
            value = False
        else:
            value = None
        adapted_relations[name] = {
            "value": value,
            "support_class": relation.get("support_class"),
            "crop_support": relation.get("crop_support"),
            "rejection_reason": relation.get("rejection_reason"),
            "side": relation.get("side"),
            "v14_proximal_chain_guard": deepcopy(relation.get("v14_proximal_chain_guard")),
            "support_topology_guard": deepcopy(relation.get("support_topology_guard")),
        }

    return {
        "schema_version": ADAPTER_VERSION,
        "source_schema_version": profile.get("schema_version"),
        "public": {
            "value": public_pose,
            "authority_state": authority_state,
            "crop_support": projected.get("crop_support"),
            "support_class": projected.get("support_class"),
            "selected_path": assertion.get("selected_path"),
            "selected_path_authority": assertion.get("selected_path_authority"),
            "withheld_reason": assertion.get("withheld_reason"),
            "authority_semantics": assertion.get("authority_semantics"),
        },
        "reconstruction": {
            "best_candidate": best_candidate,
            "score": governed_best_score,
            "winner_margin": projected.get("winner_margin"),
            "support_class": projected.get("support_class"),
        },
        "semantic_recovery": {
            "needed": bool(recovery.get("needed")),
            "candidate": _normalize_pose(recovery.get("candidate_pose")),
            "reason": recovery.get("reason"),
            "candidate_score": recovery.get("candidate_score"),
            "winner_margin": recovery.get("winner_margin"),
            "recommended_fusion_action": recovery.get("recommended_fusion_action"),
        },
        "modifiers": {
            "pose_family": _normalize_pose(modifiers.get("pose_family_for_modifier")),
            "lean_severity": modifiers.get("lean_severity"),
            "lean_direction": modifiers.get("lean_direction"),
            "shoulder_line_tilt_severity": modifiers.get("shoulder_line_tilt_severity"),
            "suggested_modifiers": deepcopy(modifiers.get("suggested_modifiers") or []),
            "suggested_compound_pose_modifier": modifiers.get("suggested_compound_pose_modifier"),
        },
        "relations": adapted_relations,
        "policy": deepcopy(profile.get("policy")),
    }
