from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import semantic_v3_fusion as base


FUSION_VERSION = "semantic-fusion-3.0.1"
RUN_VERSION = "semantic-fusion-3.0.1-run"
_BASE_FUSE = base.fuse_semantic_v3


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fix_posture_confidence(fused: dict[str, Any]) -> dict[str, Any]:
    posture = _dict(_dict(fused.get("canonical")).get("posture"))
    pose_public = _dict(_dict(fused.get("pose_adapter")).get("public"))
    if not posture:
        return fused

    pose_joint_authority = pose_public.get("selected_path_authority")
    if pose_joint_authority is None:
        pose_joint_authority = pose_public.get("crop_support")
    posture["pose_joint_authority"] = pose_joint_authority
    posture["pose_joint_authority_semantics"] = "joint/crop corroboration authority; not probabilistic confidence"

    if posture.get("status") == "asserted":
        semantic_value = str(posture.get("semantic_value") or "unknown")
        semantic_confidence = float(posture.get("semantic_confidence") or 0.0)
        semantic_assessment = str(posture.get("semantic_assessment") or "unknown")
        if semantic_value == posture.get("value") and semantic_assessment in {"supported", "plausible"}:
            posture["confidence"] = semantic_confidence
            posture["confidence_basis"] = "Analyze semantic confidence corroborated by Pose v0.16 public assertion"
        else:
            posture["confidence"] = None
            posture["confidence_basis"] = "Pose v0.16 publishes authority, not calibrated probability; no aligned semantic confidence available"
    elif posture.get("status") == "recovered":
        posture["confidence_basis"] = "Analyze semantic confidence under Pose-authorized semantic recovery"
    else:
        posture["confidence_basis"] = "withheld/unknown"
    return fused


def _govern_pose_modifiers(fused: dict[str, Any]) -> dict[str, Any]:
    canonical = _dict(fused.get("canonical"))
    pose = _dict(fused.get("pose_adapter"))
    public = _dict(pose.get("public"))
    modifiers = deepcopy(_dict(pose.get("modifiers")))

    if public.get("authority_state") == "asserted" and public.get("value") not in {None, "unknown"}:
        modifiers["authority_state"] = "asserted"
        modifiers["provenance_only"] = False
        canonical["pose_modifiers"] = modifiers
        return fused

    canonical["pose_modifiers"] = {
        "authority_state": "withheld",
        "pose_family": None,
        "lean_severity": None,
        "lean_direction": None,
        "shoulder_line_tilt_severity": None,
        "suggested_modifiers": [],
        "suggested_compound_pose_modifier": None,
        "reconstruction_pose_family": modifiers.get("pose_family"),
        "provenance_only": True,
        "reason": "Public Pose posture was withheld; reconstruction-derived modifiers cannot re-enter canonical truth.",
    }
    return fused


def _downgrade_rejected_hand_specificity(fused: dict[str, Any]) -> dict[str, Any]:
    canonical = _dict(fused.get("canonical"))
    conflicts = _list(fused.get("conflicts"))
    rejected_parts: set[str] = set()
    for conflict in conflicts:
        if not isinstance(conflict, dict) or conflict.get("type") != "semantic_hand_head_chain_vs_pose_topology":
            continue
        original = _dict(conflict.get("semantic_original"))
        part = str(original.get("actor_part") or "")
        if part:
            rejected_parts.add(part)

    if not rejected_parts:
        return fused

    adjustments = _list(fused.setdefault("authority_adjustments", []))
    for item in _list(canonical.get("interactions")):
        if not isinstance(item, dict):
            continue
        if str(item.get("actor_part") or "") not in rejected_parts:
            continue
        if item.get("actor_ownership") != "unknown":
            continue
        original_part = item.get("actor_part")
        original_interpretation = item.get("interpretation")
        item["actor_part"] = "distal_hand_or_finger_fragment"
        item["interpretation"] = "distal hand/finger fragment near head or face; proximal palm/wrist chain not established"
        item["limitations"] = list(item.get("limitations") or []) + [
            "Fusion downgraded whole-hand specificity because governed Pose supports distal proximity but rejects the proximal palm/wrist chain."
        ]
        adjustments.append({
            "type": "downgrade_anatomical_specificity_after_proximal_chain_rejection",
            "from_part": original_part,
            "to_part": item["actor_part"],
            "original_interpretation": original_interpretation,
        })

    for item in _list(canonical.get("ownership_assessments")):
        if not isinstance(item, dict):
            continue
        if str(item.get("part") or "") not in rejected_parts:
            continue
        if item.get("ownership") != "unknown":
            continue
        original_part = item.get("part")
        item["part"] = "distal_hand_or_finger_fragment"
        item["limitations"] = list(item.get("limitations") or []) + [
            "Whole-hand identity withheld; only distal hand/finger-fragment specificity remains canonical."
        ]
        adjustments.append({
            "type": "downgrade_ownership_part_specificity_after_proximal_chain_rejection",
            "from_part": original_part,
            "to_part": item["part"],
        })
    return fused


def _govern_unresolved_support_descriptions(fused: dict[str, Any]) -> dict[str, Any]:
    canonical = _dict(fused.get("canonical"))
    adjustments = _list(fused.setdefault("authority_adjustments", []))
    for item in _list(canonical.get("support_context")):
        if not isinstance(item, dict):
            continue
        if item.get("target_status") != "unresolved" or item.get("target_ref") is not None:
            continue
        description = item.get("target_description")
        if description is None:
            continue
        item["target_description"] = None
        item["limitations"] = list(item.get("limitations") or []) + [
            "Fusion withheld the ungrounded support-target description; broad support relation remains unresolved."
        ]
        adjustments.append({
            "type": "withhold_unresolved_support_target_description",
            "original_target_description": description,
        })
    return fused


def _has_semantic_hand_head_relation(canonical: dict[str, Any]) -> bool:
    for item in _list(canonical.get("interactions")):
        if not isinstance(item, dict):
            continue
        actor = str(item.get("actor_part") or "").lower()
        if not any(token in actor for token in ("hand", "finger", "fist")):
            continue
        target = " ".join(
            str(item.get(key) or "").lower()
            for key in ("target_ref", "target_text", "interpretation")
        )
        if any(token in target for token in ("target_subject", "head", "face", "chin", "neck")):
            return True
    return False


def _gate_negative_physical_relations(fused: dict[str, Any]) -> dict[str, Any]:
    canonical = _dict(fused.get("canonical"))
    physical = _dict(canonical.get("physical_relations"))
    relation = _dict(physical.get("head_supported_by_hand"))
    if relation.get("value") is not False:
        return fused
    if _has_semantic_hand_head_relation(canonical):
        return fused

    original = deepcopy(relation)
    physical["head_supported_by_hand"] = {
        "value": None,
        "authority": "not_asserted",
        "reason": "No semantic hand/head relation required resolution; negative Pose diagnostic remains provenance only.",
        "support_class": None,
    }
    _list(fused.setdefault("authority_adjustments", [])).append({
        "type": "withhold_unneeded_negative_physical_relation",
        "relation": "head_supported_by_hand",
        "pose_diagnostic": original,
    })
    return fused


def fuse_semantic_v3(
    *,
    image_key: str,
    extract_wrapper: dict[str, Any],
    analyze_artifact: dict[str, Any],
    gestalt_artifact: dict[str, Any],
    pose_record: dict[str, Any],
) -> dict[str, Any]:
    fused = _BASE_FUSE(
        image_key=image_key,
        extract_wrapper=extract_wrapper,
        analyze_artifact=analyze_artifact,
        gestalt_artifact=gestalt_artifact,
        pose_record=pose_record,
    )
    fused["schema_version"] = FUSION_VERSION
    fused["authority_adjustments"] = []
    _fix_posture_confidence(fused)
    _govern_pose_modifiers(fused)
    _downgrade_rejected_hand_specificity(fused)
    _govern_unresolved_support_descriptions(fused)
    _gate_negative_physical_relations(fused)
    fused.setdefault("policy", {})["canonical_confidence"] = (
        "Pose joint/crop authority is never serialized as probability/confidence. "
        "Aligned Analyze confidence may be retained separately from Pose assertion authority."
    )
    fused["policy"]["withheld_modifier_policy"] = (
        "When public Pose posture is withheld, reconstruction-derived pose family/modifiers remain provenance only."
    )
    fused["policy"]["specificity_downgrade"] = (
        "A rejected proximal hand chain can reduce canonical anatomy from whole-hand specificity to a distal hand/finger fragment without inventing ownership."
    )
    fused["policy"]["negative_relation_policy"] = (
        "A negative Pose relation diagnostic is canonical only when it resolves a relevant semantic relation; otherwise it remains provenance."
    )
    return fused


def main() -> int:
    base.FUSION_VERSION = FUSION_VERSION
    base.RUN_VERSION = RUN_VERSION
    base.fuse_semantic_v3 = fuse_semantic_v3
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
