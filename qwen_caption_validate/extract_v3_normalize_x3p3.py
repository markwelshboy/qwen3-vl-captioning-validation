from __future__ import annotations

"""Mechanical governance normalization for x3p3 Extract wire records.

The normalizer runs after xgrammar has produced JSON and before Pydantic
structural validation. It may only repair/downgrade claims that are
unambiguously unsafe from the record's own topology. It never adds image
semantics, and callers must retain the original raw response for provenance.
"""

import copy
import re
from typing import Any

NORMALIZER_VERSION = "x3p3-governance-0.2"


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _distal_family(part: Any) -> str | None:
    value = _norm(part)
    if "finger" in value or "hand" in value or value == "wrist" or value.endswith("_wrist"):
        return "hand"
    if "toe" in value or "foot" in value or value == "ankle" or value.endswith("_ankle"):
        return "foot"
    return None


def _fragment_part(part: Any) -> str:
    value = _norm(part)
    if "finger" in value:
        return "fingers"
    if "hand" in value or "wrist" in value:
        return "hand_fragment"
    if "toe" in value or "foot" in value or "ankle" in value:
        return "foot_fragment"
    return "human_fragment"


def _side_from_text(value: Any) -> str:
    normalized = _norm(value)
    left = bool(re.search(r"(^|_)left($|_)", normalized))
    right = bool(re.search(r"(^|_)right($|_)", normalized))
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return "unknown"


def _same_side_or_unknown(location: Any, side: str) -> bool:
    location_side = _side_from_text(location)
    return side == "unknown" or location_side == "unknown" or location_side == side


def _parent_matches(item: dict[str, Any], *, family: str, side: str) -> bool:
    if item.get("o") != "target":
        return False
    if item.get("k") not in {"connected_visible", "connected_but_occluded"}:
        return False

    parent_side = str(item.get("a") or "unknown")
    if side in {"left", "right"} and parent_side not in {side, "unknown"}:
        return False

    part = _norm(item.get("p"))
    subparts = {_norm(value) for value in (item.get("s") or [])}
    if family == "hand":
        parent_terms = ("arm", "forearm", "upper_arm", "lower_arm")
        return any(term in part for term in parent_terms) or bool(
            subparts & {"arm", "forearm", "upper_arm", "lower_arm"}
        )
    if family == "foot":
        parent_terms = ("leg", "lower_leg", "shin", "calf")
        return any(term in part for term in parent_terms) or bool(
            subparts & {"leg", "lower_leg", "shin", "calf"}
        )
    return False


def _filter_completion_cues(cues: list[Any], *, family: str) -> tuple[list[Any], list[Any]]:
    """Drop cues that reconstruct a parent/whole region after topology downgrade."""

    blocked = (
        {"palm", "wrist", "forearm", "upper_arm", "lower_arm"}
        if family == "hand"
        else {"ankle", "shin", "calf", "lower_leg", "upper_leg"}
    )
    kept: list[Any] = []
    removed: list[Any] = []
    for cue in cues:
        normalized = _norm(cue)
        if any(term in normalized for term in blocked):
            removed.append(cue)
        else:
            kept.append(cue)
    return kept, removed


def _actor_matches_downgrade(actor_part: Any, downgraded: dict[str, str]) -> bool:
    actor = _norm(actor_part)
    family = downgraded["family"]
    side = downgraded["side"]
    if _distal_family(actor) != family:
        return False
    return _same_side_or_unknown(actor, side)


def _appearance_matches_downgrade(
    appearance: dict[str, Any],
    downgraded: list[dict[str, str]],
) -> dict[str, str] | None:
    location = _norm(appearance.get("l"))
    family = _distal_family(location)
    if family is None:
        return None

    same_family = [item for item in downgraded if item["family"] == family]
    for item in same_family:
        if _same_side_or_unknown(location, item["side"]):
            location_side = _side_from_text(location)
            if item["side"] == "unknown" or location_side == item["side"]:
                return item

    if len(same_family) == 1 and _same_side_or_unknown(location, same_family[0]["side"]):
        return same_family[0]
    return None


def _appearance_uncertainty(
    appearance: dict[str, Any],
    downgraded: dict[str, str],
    *,
    kind: str,
) -> str:
    category = str(appearance.get("c") or kind)
    descriptors = [str(value) for value in (appearance.get("d") or []) if str(value).strip()]
    location = str(appearance.get("l") or "ambiguous fragment")
    detail = f" ({', '.join(descriptors)})" if descriptors else ""
    return (
        f"{kind} candidate on ambiguous {downgraded['family']} fragment: "
        f"{category}{detail} at {location}; target ownership unresolved"
    )


def _marking_matches_downgrade(
    marking: dict[str, Any],
    downgraded: list[dict[str, str]],
) -> dict[str, str] | None:
    return _appearance_matches_downgrade(marking, downgraded)


def _marking_uncertainty(marking: dict[str, Any], downgraded: dict[str, str]) -> str:
    return _appearance_uncertainty(marking, downgraded, kind="marking")


def _specific_distal_subpart(appearance: dict[str, Any]) -> tuple[str, str] | None:
    """Return a precision-critical distal subpart and side, if explicitly named.

    The rule intentionally covers only subparts where a broad visible ``hand`` or
    ``foot`` is insufficient evidence for a high-confidence marking claim.
    """

    pieces = [appearance.get("l"), *(appearance.get("d") or [])]
    text = _norm(" ".join(str(value) for value in pieces if value is not None))
    side = _side_from_text(text)

    if "palm" in text:
        return "palm", side
    if any(term in text for term in ("index_finger", "middle_finger", "ring_finger", "pinky", "little_finger", "thumb")):
        return "fingers", side
    if "wrist" in text:
        return "wrist", side
    if any(term in text for term in ("big_toe", "little_toe", "toe")):
        return "toes", side
    if "ankle" in text:
        return "ankle", side
    return None


def _body_part_explicitly_supports_subpart(
    body_part: dict[str, Any],
    *,
    required: str,
    side: str,
) -> bool:
    if body_part.get("o") != "target":
        return False
    body_side = str(body_part.get("a") or "unknown")
    if side in {"left", "right"} and body_side not in {side, "unknown"}:
        return False

    part = _norm(body_part.get("p"))
    subparts = {_norm(value) for value in (body_part.get("s") or [])}
    if required == "palm":
        return "palm" in part or "palm" in subparts
    if required == "fingers":
        return "finger" in part or "fingers" in subparts or any("finger" in value for value in subparts)
    if required == "wrist":
        return "wrist" in part or "wrist" in subparts
    if required == "toes":
        return "toe" in part or "toes" in subparts or any("toe" in value for value in subparts)
    if required == "ankle":
        return "ankle" in part or "ankle" in subparts
    return False


def _marking_has_specific_subpart_support(
    marking: dict[str, Any],
    body_parts: list[Any],
) -> tuple[bool, str | None, str]:
    requirement = _specific_distal_subpart(marking)
    if requirement is None:
        return True, None, "unknown"

    required, side = requirement
    supported = any(
        isinstance(body_part, dict)
        and _body_part_explicitly_supports_subpart(body_part, required=required, side=side)
        for body_part in body_parts
    )
    return supported, required, side


def normalize_x3p3_wire(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a normalized deep copy plus a complete mechanical action report."""

    out = copy.deepcopy(data)
    actions: list[dict[str, Any]] = []
    uncertainties = out.get("u") if isinstance(out.get("u"), list) else []

    entities = out.get("e") if isinstance(out.get("e"), list) else []
    known_entity_ids = {
        str(item.get("i"))
        for item in entities
        if isinstance(item, dict) and isinstance(item.get("i"), str)
    }

    # Support is non-authoritative. If Qwen emits a dangling eN but also a
    # usable textual description, clear only the broken ref and preserve text.
    hypotheses = out.get("h") if isinstance(out.get("h"), dict) else {}
    supports = hypotheses.get("sup") if isinstance(hypotheses.get("sup"), list) else []
    for index, support in enumerate(supports):
        if not isinstance(support, dict):
            continue
        ref = support.get("t")
        description = support.get("d")
        if (
            isinstance(ref, str)
            and ref.startswith("e")
            and ref not in known_entity_ids
            and isinstance(description, str)
            and description.strip()
        ):
            support["t"] = None
            actions.append(
                {
                    "rule": "support_dangling_ref_to_description",
                    "path": f"h.sup.{index}.t",
                    "from": ref,
                    "to": None,
                    "preserved_description": description,
                }
            )

    subject = out.get("s") if isinstance(out.get("s"), dict) else {}
    body_parts = subject.get("bp") if isinstance(subject.get("bp"), list) else []
    fragments = subject.get("hf") if isinstance(subject.get("hf"), list) else []

    downgraded: list[dict[str, str]] = []
    kept_body_parts: list[Any] = []
    for index, body_part in enumerate(body_parts):
        if not isinstance(body_part, dict):
            kept_body_parts.append(body_part)
            continue

        family = _distal_family(body_part.get("p"))
        side = str(body_part.get("a") or "unknown")
        claims_target_chain = (
            family is not None
            and body_part.get("o") == "target"
            and body_part.get("k") == "connected_visible"
        )
        anchored = False
        if claims_target_chain:
            anchored = any(
                other is not body_part
                and isinstance(other, dict)
                and _parent_matches(other, family=family, side=side)
                for other in body_parts
            )

        if not claims_target_chain or anchored:
            kept_body_parts.append(body_part)
            continue

        original_geometry = list(body_part.get("g") or [])
        safe_geometry, removed_geometry = _filter_completion_cues(
            original_geometry, family=family or "human"
        )
        normalized_fragment = {
            "p": _fragment_part(body_part.get("p")),
            "n": None,
            "a": body_part.get("a", "unknown"),
            "o": "unknown",
            "k": "unknown",
            "g": safe_geometry,
            "c": list(body_part.get("c") or []),
            "l": body_part.get("l", "unknown"),
            "q": body_part.get("q", "m"),
        }
        fragments.append(normalized_fragment)
        downgraded_item = {
            "family": family or "human",
            "side": side,
            "original_part": str(body_part.get("p") or "unknown"),
            "fragment_part": str(normalized_fragment["p"]),
        }
        downgraded.append(downgraded_item)
        actions.append(
            {
                "rule": "unanchored_distal_target_part_to_fragment",
                "path": f"s.bp.{index}",
                "original": copy.deepcopy(body_part),
                "normalized_fragment": copy.deepcopy(normalized_fragment),
                "removed_completion_cues": removed_geometry,
                "reason": "no visible parent limb chain in s.bp",
            }
        )

    if downgraded:
        subject["bp"] = kept_body_parts
        subject["hf"] = fragments

        interactions = subject.get("ix") if isinstance(subject.get("ix"), list) else []
        for index, interaction in enumerate(interactions):
            if not isinstance(interaction, dict) or interaction.get("o") != "target":
                continue
            matched = next(
                (
                    item
                    for item in downgraded
                    if _actor_matches_downgrade(interaction.get("p"), item)
                ),
                None,
            )
            if matched is None:
                continue
            original_actor_part = interaction.get("p")
            interaction["o"] = "unknown"
            interaction["p"] = matched["fragment_part"]
            original_cues = list(interaction.get("c") or [])
            safe_cues, removed_cues = _filter_completion_cues(
                original_cues, family=matched["family"]
            )
            interaction["c"] = safe_cues
            actions.append(
                {
                    "rule": "interaction_actor_ownership_follows_fragment",
                    "path": f"s.ix.{index}",
                    "ownership_from": "target",
                    "ownership_to": "unknown",
                    "actor_part_from": original_actor_part,
                    "actor_part_to": interaction.get("p"),
                    "removed_completion_cues": removed_cues,
                }
            )

        accessories = subject.get("ac") if isinstance(subject.get("ac"), list) else []
        kept_accessories: list[Any] = []
        for index, accessory in enumerate(accessories):
            if not isinstance(accessory, dict):
                kept_accessories.append(accessory)
                continue
            matched = _appearance_matches_downgrade(accessory, downgraded)
            if matched is None:
                kept_accessories.append(accessory)
                continue

            uncertainty = _appearance_uncertainty(accessory, matched, kind="accessory")
            if uncertainty not in uncertainties:
                uncertainties.append(uncertainty)
            actions.append(
                {
                    "rule": "target_accessory_to_ambiguous_fragment_uncertainty",
                    "path": f"s.ac.{index}",
                    "original": copy.deepcopy(accessory),
                    "uncertainty": uncertainty,
                }
            )
        subject["ac"] = kept_accessories

    # Markings are deliberately precision-first. First quarantine markings tied
    # to an ownership-downgraded fragment. Then require explicit visibility for
    # precision-critical distal subparts such as palm/wrist/specific fingers.
    markings = subject.get("mk") if isinstance(subject.get("mk"), list) else []
    kept_markings: list[Any] = []
    for index, marking in enumerate(markings):
        if not isinstance(marking, dict):
            kept_markings.append(marking)
            continue

        matched = _marking_matches_downgrade(marking, downgraded) if downgraded else None
        if matched is not None:
            uncertainty = _marking_uncertainty(marking, matched)
            if uncertainty not in uncertainties:
                uncertainties.append(uncertainty)
            actions.append(
                {
                    "rule": "target_marking_to_ambiguous_fragment_uncertainty",
                    "path": f"s.mk.{index}",
                    "original": copy.deepcopy(marking),
                    "uncertainty": uncertainty,
                }
            )
            continue

        supported, required_subpart, side = _marking_has_specific_subpart_support(
            marking, kept_body_parts
        )
        if not supported and required_subpart is not None:
            category = str(marking.get("c") or "marking")
            descriptors = [str(value) for value in (marking.get("d") or []) if str(value).strip()]
            detail = f" ({', '.join(descriptors)})" if descriptors else ""
            location = str(marking.get("l") or "unknown")
            uncertainty = (
                f"marking candidate lacks explicit visible-subpart support: "
                f"{category}{detail} at {location}; required {side + ' ' if side != 'unknown' else ''}"
                f"{required_subpart} visibility not established"
            )
            if uncertainty not in uncertainties:
                uncertainties.append(uncertainty)
            actions.append(
                {
                    "rule": "target_marking_requires_visible_subpart",
                    "path": f"s.mk.{index}",
                    "original": copy.deepcopy(marking),
                    "required_subpart": required_subpart,
                    "required_side": side,
                    "uncertainty": uncertainty,
                }
            )
            continue

        kept_markings.append(marking)

    subject["mk"] = kept_markings
    out["u"] = uncertainties

    report = {
        "version": NORMALIZER_VERSION,
        "action_count": len(actions),
        "actions": actions,
    }
    return out, report
