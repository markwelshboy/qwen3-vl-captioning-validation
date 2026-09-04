from __future__ import annotations

"""x3p3 governance v0.3.

This layer composes governance v0.2 and adds one additional topology-only
safeguard: a broad arm/leg label cannot claim target-connected visibility when
its own visible subparts contain only distal anatomy and no proximal anchor.

The rule is intentionally conservative and one-way. It may downgrade ownership
and connectedness, but never upgrades or invents visual evidence.
"""

import copy
import re
from typing import Any

from .extract_v3_normalize_x3p3 import normalize_x3p3_wire as _normalize_v02

NORMALIZER_VERSION = "x3p3-governance-0.3"


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _side_from_text(value: Any) -> str:
    normalized = _norm(value)
    left = bool(re.search(r"(^|_)left($|_)", normalized))
    right = bool(re.search(r"(^|_)right($|_)", normalized))
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return "unknown"


def _side_compatible(value: Any, side: str) -> bool:
    candidate = _side_from_text(value)
    return side == "unknown" or candidate == "unknown" or candidate == side


def _limb_region(value: Any) -> str | None:
    text = _norm(value)
    if any(term in text for term in ("finger", "hand", "palm", "wrist")):
        return "hand"
    if any(term in text for term in ("forearm", "lower_arm", "upper_arm")) or text.endswith("_arm") or text == "arm":
        return "arm"
    if any(term in text for term in ("toe", "foot", "ankle")):
        return "foot"
    if any(term in text for term in ("thigh", "knee", "shin", "calf", "lower_leg", "upper_leg")) or text.endswith("_leg") or text == "leg":
        return "leg"
    return None


def _region_covered_by_limb(region: str | None, limb: str) -> bool:
    if limb == "arm":
        return region in {"arm", "hand"}
    if limb == "leg":
        return region in {"leg", "foot"}
    return False


def _has_external_proximal_anchor(
    body_parts: list[Any],
    *,
    current: dict[str, Any],
    limb: str,
    side: str,
) -> bool:
    required = {"shoulder", "upper_arm"} if limb == "arm" else {"hip", "thigh", "upper_leg"}
    for item in body_parts:
        if item is current or not isinstance(item, dict):
            continue
        if item.get("o") != "target":
            continue
        if item.get("k") not in {"connected_visible", "connected_but_occluded"}:
            continue
        item_side = str(item.get("a") or "unknown")
        if side in {"left", "right"} and item_side not in {side, "unknown"}:
            continue
        part = _norm(item.get("p"))
        subparts = {_norm(value) for value in (item.get("s") or [])}
        if any(term in part for term in required) or bool(subparts & required):
            return True
    return False


def _proximal_gap(body_part: dict[str, Any], body_parts: list[Any]) -> tuple[str, str] | None:
    """Return (limb_family, fragment_part) for a topology-only proximal gap."""

    if body_part.get("o") != "target" or body_part.get("k") != "connected_visible":
        return None

    part = _norm(body_part.get("p"))
    side = str(body_part.get("a") or "unknown")
    subparts = {_norm(value) for value in (body_part.get("s") or [])}

    armish = _limb_region(part) == "arm"
    if armish:
        distal = bool(subparts & {"forearm", "lower_arm", "hand", "wrist", "palm", "fingers"}) or part in {"forearm", "lower_arm"}
        proximal = bool(subparts & {"shoulder", "upper_arm"}) or part == "upper_arm"
        if distal and not proximal and not _has_external_proximal_anchor(
            body_parts, current=body_part, limb="arm", side=side
        ):
            return "arm", "arm_fragment"

    legish = _limb_region(part) == "leg"
    if legish:
        distal = bool(subparts & {"knee", "shin", "calf", "lower_leg", "foot", "ankle", "toes"}) or part in {"lower_leg", "shin", "calf"}
        proximal = bool(subparts & {"hip", "thigh", "upper_leg"}) or part in {"thigh", "upper_leg"}
        if distal and not proximal and not _has_external_proximal_anchor(
            body_parts, current=body_part, limb="leg", side=side
        ):
            return "leg", "leg_fragment"

    return None


def _actor_fragment_part(actor_part: Any, limb: str) -> str:
    region = _limb_region(actor_part)
    text = _norm(actor_part)
    if limb == "arm":
        if "finger" in text:
            return "fingers"
        if region == "hand":
            return "hand_fragment"
        return "arm_fragment"
    if limb == "leg":
        if region == "foot":
            return "foot_fragment"
        return "leg_fragment"
    return "human_fragment"


def _appearance_uncertainty(appearance: dict[str, Any], *, limb: str) -> str:
    category = str(appearance.get("c") or "appearance")
    descriptors = [str(value) for value in (appearance.get("d") or []) if str(value).strip()]
    location = str(appearance.get("l") or "ambiguous fragment")
    detail = f" ({', '.join(descriptors)})" if descriptors else ""
    return (
        f"target appearance candidate on ambiguous {limb} fragment: "
        f"{category}{detail} at {location}; target ownership unresolved"
    )


def normalize_x3p3_wire(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out, prior = _normalize_v02(data)
    actions = list(prior.get("actions") or [])
    uncertainties = out.get("u") if isinstance(out.get("u"), list) else []

    subject = out.get("s") if isinstance(out.get("s"), dict) else {}
    body_parts = subject.get("bp") if isinstance(subject.get("bp"), list) else []
    fragments = subject.get("hf") if isinstance(subject.get("hf"), list) else []

    downgraded: list[dict[str, str]] = []
    kept_body_parts: list[Any] = []
    for index, body_part in enumerate(body_parts):
        if not isinstance(body_part, dict):
            kept_body_parts.append(body_part)
            continue

        gap = _proximal_gap(body_part, body_parts)
        if gap is None:
            kept_body_parts.append(body_part)
            continue

        limb, fragment_part = gap
        side = str(body_part.get("a") or "unknown")
        normalized_fragment = {
            "p": fragment_part,
            "n": None,
            "a": body_part.get("a", "unknown"),
            "o": "unknown",
            "k": "unknown",
            "g": list(body_part.get("g") or []),
            "c": list(body_part.get("c") or []),
            "l": body_part.get("l", "unknown"),
            "q": body_part.get("q", "m"),
        }
        fragments.append(normalized_fragment)
        downgraded.append({"limb": limb, "side": side, "fragment_part": fragment_part})
        actions.append(
            {
                "rule": "unanchored_limb_chain_to_fragment",
                "path": f"s.bp.{index}",
                "original": copy.deepcopy(body_part),
                "normalized_fragment": copy.deepcopy(normalized_fragment),
                "reason": "connected_visible limb contains distal subparts without a visible proximal anchor",
            }
        )

    if downgraded:
        subject["bp"] = kept_body_parts
        subject["hf"] = fragments

        interactions = subject.get("ix") if isinstance(subject.get("ix"), list) else []
        for index, interaction in enumerate(interactions):
            if not isinstance(interaction, dict) or interaction.get("o") != "target":
                continue
            actor_region = _limb_region(interaction.get("p"))
            actor_side = _side_from_text(interaction.get("p"))
            matched = next(
                (
                    item
                    for item in downgraded
                    if _region_covered_by_limb(actor_region, item["limb"])
                    and (item["side"] == "unknown" or actor_side == "unknown" or actor_side == item["side"])
                ),
                None,
            )
            if matched is None:
                continue
            original_actor = interaction.get("p")
            interaction["o"] = "unknown"
            interaction["p"] = _actor_fragment_part(original_actor, matched["limb"])
            actions.append(
                {
                    "rule": "interaction_actor_follows_unanchored_limb_fragment",
                    "path": f"s.ix.{index}",
                    "ownership_from": "target",
                    "ownership_to": "unknown",
                    "actor_part_from": original_actor,
                    "actor_part_to": interaction.get("p"),
                }
            )

        for field, kind in (("ac", "accessory"), ("mk", "marking")):
            values = subject.get(field) if isinstance(subject.get(field), list) else []
            kept: list[Any] = []
            for index, appearance in enumerate(values):
                if not isinstance(appearance, dict):
                    kept.append(appearance)
                    continue
                region = _limb_region(appearance.get("l"))
                location_side = _side_from_text(appearance.get("l"))
                matched = next(
                    (
                        item
                        for item in downgraded
                        if _region_covered_by_limb(region, item["limb"])
                        and (item["side"] == "unknown" or location_side == "unknown" or location_side == item["side"])
                    ),
                    None,
                )
                if matched is None:
                    kept.append(appearance)
                    continue
                uncertainty = _appearance_uncertainty(appearance, limb=matched["limb"])
                if uncertainty not in uncertainties:
                    uncertainties.append(uncertainty)
                actions.append(
                    {
                        "rule": f"target_{kind}_follows_unanchored_limb_fragment",
                        "path": f"s.{field}.{index}",
                        "original": copy.deepcopy(appearance),
                        "uncertainty": uncertainty,
                    }
                )
            subject[field] = kept

    out["u"] = uncertainties
    return out, {
        "version": NORMALIZER_VERSION,
        "action_count": len(actions),
        "actions": actions,
    }
