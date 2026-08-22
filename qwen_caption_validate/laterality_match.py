from __future__ import annotations

import re
from typing import Any

import numpy as np

from .laterality_geometry import ARM_RE, DISTAL_RE, LEG_RE, MHR70, SIDE_RE, _connectivity, _sam_vote

def _expected_x(value: Any) -> float | None:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    left, right = "left" in text, "right" in text
    if left and right:
        return None
    if left:
        return 0.35 if "center" in text or "centre" in text else 0.22
    if right:
        return 0.65 if "center" in text or "centre" in text else 0.78
    return 0.50 if "center" in text or "centre" in text else None


def _entity_text(item: dict[str, Any]) -> str:
    return " ".join([str(item.get("part") or ""), *[str(value) for value in item.get("visible_subparts") or []]])


def _family(item: dict[str, Any]) -> str | None:
    text = _entity_text(item)
    if ARM_RE.search(text):
        return "arm"
    if LEG_RE.search(text):
        return "leg"
    return None


def _distal_arm(item: dict[str, Any]) -> bool:
    detail = " ".join((_entity_text(item), str(item.get("geometry") or ""), str(item.get("contact") or ""), str(item.get("support") or "")))
    return bool(DISTAL_RE.search(detail))


def _match_hand(item: dict[str, Any], entities: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    qualified = [entity for entity in entities if entity.get("qualified_side")]
    if not qualified:
        return None, "no_qualified_observed_hand_entity"
    if len(qualified) == 1:
        return qualified[0], "single_qualified_observed_hand_entity"
    expected = _expected_x(item.get("image_location"))
    if expected is None:
        return None, "multiple_hand_entities_without_frame_disambiguation"
    ranked = sorted((abs(float(entity["root_xy"][0]) - expected), entity) for entity in qualified)
    if ranked[1][0] - ranked[0][0] < 0.10:
        return None, "multiple_hand_entities_frame_match_ambiguous"
    return ranked[0][1], "hand_entity_frame_location_match"


def _match_chain(item: dict[str, Any], family: str, dw: dict[str, Any], points: dict[str, np.ndarray], sam2d: np.ndarray | None) -> tuple[str | None, dict[str, Any]]:
    expected = _expected_x(item.get("image_location"))
    if expected is None:
        return None, {"reason": "no_frame_location_for_entity_matching"}
    text = " ".join((_entity_text(item), str(item.get("geometry") or ""))).lower()
    joints = ["shoulder", "elbow"] if family == "arm" else ["hip", "knee", "ankle"]
    if family == "arm" and any(token in text for token in ("forearm", "wrist", "hand", "finger")):
        joints.append("wrist")
    choices = []
    for side in ("left", "right"):
        names = [f"{side}_{joint}" for joint in joints if f"{side}_{joint}" in points]
        if not names:
            continue
        centroid_x = float(np.mean([points[name][0] for name in names]))
        votes = [_sam_vote(name, dw, points, sam2d) for name in names if name in MHR70]
        statuses = [vote.get("status") for vote in votes]
        sam_status = "conflicts" if "conflicts" in statuses else ("agrees" if "agrees" in statuses else "unresolved")
        chain = (_connectivity(dw).get(f"{side}_{family}") or {})
        choices.append({
            "side": side,
            "score": abs(centroid_x - expected),
            "observed_joints": names,
            "observed_count": len(names),
            "complete": bool(chain.get("complete")),
            "sam3d_status": sam_status,
            "sam3d_votes": votes,
        })
    choices.sort(key=lambda choice: choice["score"])
    if not choices or choices[0]["score"] > 0.32:
        return None, {"reason": "no_observed_chain_matches_semantic_frame_location", "choices": choices}
    if len(choices) > 1 and choices[1]["score"] - choices[0]["score"] < 0.10:
        return None, {"reason": "observed_chain_frame_location_ambiguous", "choices": choices}
    best = choices[0]
    if best["sam3d_status"] == "conflicts":
        return None, {"reason": "sam3d_conflicts_with_dwpose_joint_labels", "choices": choices}
    if best["observed_count"] < 2 and not best["complete"]:
        return None, {"reason": "insufficient_observed_chain_support", "choices": choices}
    authority = "dwpose_sam_correlated" if best["sam3d_status"] == "agrees" else ("dwpose_complete_chain" if best["complete"] else "dwpose_observed_chain")
    return best["side"], {"reason": "semantic_entity_matched_to_observed_dwpose_chain", "authority": authority, "match": best}


def _raw_side(value: Any) -> str | None:
    match = SIDE_RE.search(str(value or "").replace("_", " "))
    return match.group(1).lower() if match else None


def _side_name(value: Any, side: str | None) -> Any:
    if not isinstance(value, str):
        return value
    text = value.replace("_", " ")
    text = SIDE_RE.sub(side or "", text, count=1)
    return re.sub(r"\s+", " ", text).strip(" _-")


