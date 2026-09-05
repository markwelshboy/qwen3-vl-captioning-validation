from __future__ import annotations

import re
from typing import Any

from . import semantic_v3_fusion as base
from . import semantic_v3_fusion_v301 as v301


FUSION_VERSION = "semantic-fusion-3.0.2"
RUN_VERSION = "semantic-fusion-3.0.2-run"
_BASE_FUSE = v301.fuse_semantic_v3


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _remove_self_denied_evidence(fused: dict[str, Any]) -> dict[str, Any]:
    """Do not let canonical evidence assert an observation its own limitations deny.

    Analyze can reasonably classify an action contextually while still emitting one
    over-literal evidence phrase.  For example, "lips touching cup" must not survive
    canonicalization when the same item says "no direct visual of lips touching cup".
    This pass removes only the directly contradicted evidence phrase; it does not
    change the action label, confidence, or remaining evidence.
    """
    canonical = _dict(fused.get("canonical"))
    adjustments = _list(fused.setdefault("authority_adjustments", []))

    for collection_name in ("actions", "interactions"):
        for index, item in enumerate(_list(canonical.get(collection_name))):
            if not isinstance(item, dict):
                continue
            denied: list[str] = []
            for limitation in _list(item.get("limitations")):
                if not isinstance(limitation, str):
                    continue
                match = re.search(r"no direct visual of\s+([^,;]+)", limitation, flags=re.IGNORECASE)
                if match:
                    denied.append(match.group(1).strip().lower())
            if not denied:
                continue

            kept: list[Any] = []
            removed: list[str] = []
            for evidence in _list(item.get("evidence")):
                if not isinstance(evidence, str):
                    kept.append(evidence)
                    continue
                lower = evidence.strip().lower()
                if any(phrase and phrase in lower for phrase in denied):
                    removed.append(evidence)
                else:
                    kept.append(evidence)
            if removed:
                item["evidence"] = kept
                adjustments.append({
                    "type": "remove_evidence_denied_by_item_limitation",
                    "collection": collection_name,
                    "index": index,
                    "removed_evidence": removed,
                    "denied_phrases": denied,
                })
    return fused


def _scrub_rejected_proximal_chain_evidence(fused: dict[str, Any]) -> dict[str, Any]:
    """Keep a specificity downgrade from retaining evidence for the rejected whole chain."""
    canonical = _dict(fused.get("canonical"))
    adjustments = _list(fused.setdefault("authority_adjustments", []))
    conflicts = _list(fused.get("conflicts"))
    has_rejected_chain = any(
        isinstance(item, dict) and item.get("type") == "semantic_hand_head_chain_vs_pose_topology"
        for item in conflicts
    )
    if not has_rejected_chain:
        return fused

    rejected_tokens = (
        "hand_under",
        "thumb",
        "wrist",
        "palm",
        "forearm",
        "upper_arm",
        "extended arm",
        "connected_visible",
        "visible_subparts include",
    )

    # Posing can remain contextual from gaze/expression, but the rejected whole-hand
    # description must not be one of the canonical reasons for it.
    for index, item in enumerate(_list(canonical.get("actions"))):
        if not isinstance(item, dict):
            continue
        removed: list[str] = []
        kept: list[Any] = []
        for evidence in _list(item.get("evidence")):
            if isinstance(evidence, str):
                lower = evidence.lower()
                if "hand gesture" in lower and any(term in lower for term in ("chin", "head", "face")):
                    removed.append(evidence)
                    continue
            kept.append(evidence)
        if removed:
            item["evidence"] = kept
            item["limitations"] = list(item.get("limitations") or []) + [
                "Fusion removed whole-hand gesture evidence after governed Pose rejected the proximal palm/wrist chain."
            ]
            adjustments.append({
                "type": "remove_rejected_whole_hand_action_evidence",
                "index": index,
                "removed_evidence": removed,
            })

    for index, item in enumerate(_list(canonical.get("interactions"))):
        if not isinstance(item, dict) or item.get("actor_part") != "distal_hand_or_finger_fragment":
            continue
        removed: list[str] = []
        kept: list[Any] = []
        for evidence in _list(item.get("evidence")):
            if isinstance(evidence, str) and any(token in evidence.lower() for token in rejected_tokens):
                removed.append(evidence)
            else:
                kept.append(evidence)
        if removed:
            item["evidence"] = kept

        cleaned_limits: list[Any] = []
        for limitation in _list(item.get("limitations")):
            if not isinstance(limitation, str):
                cleaned_limits.append(limitation)
                continue
            if "ownership of hand is candidate but not contradicted" in limitation.lower():
                prefix = limitation.split(";", 1)[0].strip()
                if prefix:
                    cleaned_limits.append(prefix)
                continue
            cleaned_limits.append(limitation)
        item["limitations"] = cleaned_limits

        if removed:
            adjustments.append({
                "type": "remove_rejected_proximal_chain_interaction_evidence",
                "index": index,
                "removed_evidence": removed,
            })

    # Once whole-part target ownership is explicitly withheld, inherited semantic
    # evidence asserting a connected target arm is no longer canonical evidence.
    for index, item in enumerate(_list(canonical.get("ownership_assessments"))):
        if not isinstance(item, dict):
            continue
        if item.get("part") != "distal_hand_or_finger_fragment" or item.get("ownership") != "unknown":
            continue
        removed = [value for value in _list(item.get("evidence")) if isinstance(value, str)]
        if removed:
            item["evidence"] = []
            item["limitations"] = [
                value
                for value in _list(item.get("limitations"))
                if not (isinstance(value, str) and ("watch" in value.lower() or "wrist accessory" in value.lower()))
            ]
            item["limitations"] = list(item.get("limitations") or []) + [
                "No positive target-ownership evidence remains canonical after the proximal-chain rejection."
            ]
            adjustments.append({
                "type": "remove_rejected_whole_part_ownership_evidence",
                "index": index,
                "removed_evidence": removed,
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
    _scrub_rejected_proximal_chain_evidence(fused)
    _remove_self_denied_evidence(fused)
    fused.setdefault("policy", {})["canonical_evidence_consistency"] = (
        "Canonical evidence cannot retain a whole/proximal anatomical chain that Fusion has withheld, "
        "and an evidence phrase is removed when the same item explicitly says that observation was not directly visible."
    )
    return fused


def main() -> int:
    base.FUSION_VERSION = FUSION_VERSION
    base.RUN_VERSION = RUN_VERSION
    base.fuse_semantic_v3 = fuse_semantic_v3
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
