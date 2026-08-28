from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_135 import build_caption_projection as _build_135
from .caption_projection_135 import lint_caption as _lint_135

_SUPPORT_RE = re.compile(r"\bsupport(?:s|ed|ing)?\b", re.IGNORECASE)
_GROUND_SUPPORT_RE = re.compile(r"\b(?:weight|floor|ground|standing)\b", re.IGNORECASE)
_PROXIMAL_ARM_RE = re.compile(r"\b(?:forearm|upper\s+arm|arm|elbow)\b", re.IGNORECASE)
_HAND_RE = re.compile(r"\b(?:hand|fingers?)\b", re.IGNORECASE)
_NEGATIVE_APPEARANCE_RE = re.compile(
    r"\b(?:wear(?:s|ing)?\s+)?(?:no\s+visible|without\s+visible)\s+"
    r"(?:clothing|clothes|garments?|accessories)(?:\s+(?:or|and)\s+(?:clothing|clothes|garments?|accessories))?\b",
    re.IGNORECASE,
)
_TORSO_ANGLED_DEPTH_RE = re.compile(
    r"\b(?:torso|upper\s+body|body)\b[^.!?]{0,80}?\b(?:angled|turned|rotated)\s+in\s+depth\b|"
    r"\b(?:angled|turned|rotated)\s+in\s+depth\b[^.!?]{0,80}?\b(?:torso|upper\s+body|body)\b",
    re.IGNORECASE,
)
_SUMMARY_APPAREL_RE = re.compile(
    r"\b(?:(?:black|white|gray|grey|blue|green|red|yellow|pink|orange|purple|brown|beige|tan|cream|"
    r"teal|navy|dark|light|floral|patterned|striped|high[- ]waisted|halter)\s+){0,4}"
    r"(?:halter\s+top|swimsuit|swimwear|bathing\s+suit|one[- ]piece(?:\s+swimsuit)?|bikini|bodysuit|bottoms?)\b",
    re.IGNORECASE,
)

_UNSIGNED_DEPTH_IDS = {
    "shoulder_girdle_depth_rotation",
    "pelvis_depth_rotation",
    "combined_torso_depth_rotation",
}

_SETTING_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("kitchen setting", re.compile(r"\bkitchen\b", re.I), "kitchen"),
    ("park setting", re.compile(r"\bpark\b", re.I), "park"),
    ("beach setting", re.compile(r"\bbeach\b|\bshore(?:line)?\b|\bcoast(?:line)?\b", re.I), "beach"),
    ("forest or wooded setting", re.compile(r"\bforest\b|\bwood(?:ed|land)?\b", re.I), "forest"),
    ("garden setting", re.compile(r"\bgarden\b", re.I), "garden"),
    ("street setting", re.compile(r"\bstreet\b|\bsidewalk\b|\bpavement\b", re.I), "street"),
    ("office setting", re.compile(r"\boffice\b", re.I), "office"),
    ("studio setting", re.compile(r"\bstudio\b", re.I), "studio"),
    ("restaurant or cafe setting", re.compile(r"\brestaurant\b|\bcaf[eé]\b|\bcoffee\s+shop\b", re.I), "cafe"),
    ("airport or transit setting", re.compile(r"\bairport\b|\bterminal\b|\btransit\b|\bstation\b", re.I), "airport"),
    ("airplane cabin", re.compile(r"\bairplane\b|\baircraft\b|\bplane\s+cabin\b", re.I), "airplane"),
    ("vehicle interior", re.compile(r"\bcar\s+interior\b|\bvehicle\s+interior\b|\bbus\s+interior\b", re.I), "vehicle"),
)

_CONCRETE_SCENE_OBJECT_RE = re.compile(
    r"\b(?:backpacks?|bags?|boxes?|carts?|trolleys?|suitcases?|luggage|lamps?|fixtures?|paintings?|"
    r"pictures?|signs?|bottles?|cups?|mugs?|chairs?|benches?|tables?|phones?|smartphones?|vehicles?|"
    r"cars?|bicycles?|bikes?|umbrellas?|books?|monitors?|laptops?|keyboards?|garments?)\b",
    re.IGNORECASE,
)


def _pose(evidence: dict[str, Any]) -> dict[str, Any]:
    value = evidence.get("pose_orientation")
    return value if isinstance(value, dict) else evidence


def _support_target(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ")
    match = _SUPPORT_RE.search(text)
    if not match:
        return ""
    tail = text[match.end():]
    tail = re.split(r"\b(?:via|through|with|using|by)\b", tail, maxsplit=1)[0]
    tail = re.sub(r"\b(?:the|a|an|her|his|their)\b", " ", tail)
    words = [word for word in re.findall(r"[a-z]+", tail) if len(word) >= 3]
    return " ".join(words[:3])


def _part_rank(value: Any) -> int:
    text = str(value or "").lower().replace("_", " ")
    if _HAND_RE.search(text):
        return 4
    if "wrist" in text:
        return 3
    if "forearm" in text:
        return 2
    if _PROXIMAL_ARM_RE.search(text):
        return 1
    return 0


def _coalesce_pose_support(evidence: dict[str, Any], projection: dict[str, Any]) -> None:
    """Keep a support relation at its most direct visible actor instead of restating the chain."""
    parts = [item for item in (_pose(evidence).get("visible_subject_parts") or []) if isinstance(item, dict)]
    direct: dict[tuple[str, str], int] = {}
    for index, item in enumerate(parts):
        support = str(item.get("support") or "")
        if not support or not _SUPPORT_RE.search(support) or _GROUND_SUPPORT_RE.search(support):
            continue
        target = _support_target(support)
        if not target:
            continue
        side = str(item.get("anatomical_side") or "unknown").lower()
        key = (side, target)
        rank = _part_rank(item.get("part"))
        current = direct.get(key)
        if current is None or rank > _part_rank(parts[current].get("part")):
            direct[key] = index

    for index, item in enumerate(parts):
        support = str(item.get("support") or "")
        if not support or not _SUPPORT_RE.search(support) or _GROUND_SUPPORT_RE.search(support):
            continue
        target = _support_target(support)
        side = str(item.get("anatomical_side") or "unknown").lower()
        best = direct.get((side, target))
        if best is None or best == index:
            continue
        if _part_rank(item.get("part")) >= _part_rank(parts[best].get("part")):
            continue
        before_support = item.get("support")
        before_contact = item.get("contact")
        item["support"] = None
        if isinstance(before_contact, str) and re.search(r"\bvia\s+(?:the\s+)?hand\b", before_contact, re.I):
            item["contact"] = None
        projection.setdefault("blocked", []).append(
            {
                "path": f"caption-evidence-1.3.pose_orientation.visible_subject_parts[{index}]",
                "reason": "proximal_support_chain_subsumed_by_more_direct_visible_actor",
                "source_support": before_support,
                "source_contact": before_contact,
                "retained_actor_part": parts[best].get("part"),
            }
        )


def _support_claims(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for item in _pose(evidence).get("visible_subject_parts") or []:
        if not isinstance(item, dict):
            continue
        support = str(item.get("support") or "").strip()
        if not support or not _SUPPORT_RE.search(support) or _GROUND_SUPPORT_RE.search(support):
            continue
        target = _support_target(support)
        if not target:
            continue
        side = str(item.get("anatomical_side") or "unknown").lower()
        label = str(item.get("part") or "body part").replace("_", " ")
        claim = {
            "priority": "required",
            "description": f"{label}: {support}",
            "support_text": support,
            "anatomical_side": side if side in {"left", "right"} else "unknown",
            "actor_part": label,
            "semantic_target": target,
        }
        key = (claim["anatomical_side"], target)
        if key not in best or _part_rank(label) > _part_rank(best[key].get("actor_part")):
            best[key] = claim

    claims: list[dict[str, Any]] = []
    for value in best.values():
        value = dict(value)
        value["id"] = f"support_relation_{len(claims) + 1}"
        claims.append(value)
    return claims


def _coalesce_depth_claims(claims: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    ids = {str(item.get("id") or "") for item in claims if isinstance(item, dict)}
    drop: set[str] = set()
    if "signed_torso_depth_direction" in ids:
        drop.update(_UNSIGNED_DEPTH_IDS)
    elif "signed_shoulder_nearer_relation" in ids:
        drop.add("shoulder_girdle_depth_rotation")
        if "combined_torso_depth_rotation" in ids:
            drop.add("pelvis_depth_rotation")
    elif "combined_torso_depth_rotation" in ids:
        drop.update({"shoulder_girdle_depth_rotation", "pelvis_depth_rotation"})
    return [item for item in claims if str(item.get("id") or "") not in drop], sorted(drop)


def _coalesce_3d_payload(evidence: dict[str, Any]) -> None:
    pose = _pose(evidence)
    geometry = pose.get("qualified_3d_geometry")
    if not isinstance(geometry, dict):
        return
    ids = {str(item.get("id") or "") for item in (evidence.get("required_claims") or []) if isinstance(item, dict)}
    if "signed_torso_depth_direction" in ids:
        pose["qualified_3d_geometry"] = {}
        return
    if "combined_torso_depth_rotation" in geometry:
        pose["qualified_3d_geometry"] = {
            "combined_torso_depth_rotation": geometry["combined_torso_depth_rotation"]
        }
        return
    if "signed_shoulder_nearer_relation" in ids:
        geometry.pop("shoulder_girdle_depth_rotation", None)


def _scene_source_texts(evidence: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    env = evidence.get("environment_lighting") or {}
    for item in env.get("important_background_or_nuisance_regions") or []:
        if isinstance(item, dict):
            texts.append(str(item.get("description") or ""))
    scene = env.get("scene") or {}
    background = scene.get("background_structure") if isinstance(scene, dict) else None
    if isinstance(background, dict):
        texts.append(str(background.get("notes") or ""))
    for item in evidence.get("non_target_entities") or []:
        if isinstance(item, dict):
            texts.append(str(item.get("description") or ""))
    return [text for text in texts if text.strip()]


def _scene_gestalt_claims(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    texts = _scene_source_texts(evidence)
    for description, pattern, keyword in _SETTING_PATTERNS:
        if any(pattern.search(text) for text in texts):
            return [
                {
                    "id": "scene_gestalt_1",
                    "description": description,
                    "keywords": [keyword],
                    "minimum_keyword_matches": 1,
                    "attribution": "scene_or_background_not_trigger_identity",
                    "semantic_compression_allowed": True,
                }
            ]
    return []


def _singular_scene_keyword(value: str) -> str:
    word = value.lower()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if re.search(r"(?:x|z|ch|sh|ss)es$", word) and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _concrete_region_claims(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    env = evidence.get("environment_lighting") or {}
    for item in env.get("important_background_or_nuisance_regions") or []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        keywords: list[str] = []
        for match in _CONCRETE_SCENE_OBJECT_RE.finditer(description):
            keyword = _singular_scene_keyword(match.group(0))
            if keyword not in keywords:
                keywords.append(keyword)
        if not keywords:
            continue
        claims.append(
            {
                "id": f"scene_object_region_{len(claims) + 1}",
                "description": description,
                "keywords": keywords,
                "minimum_keyword_matches": min(2, len(keywords)),
                "attribution": "scene_or_background_not_trigger_identity",
                "semantic_compression_allowed": True,
            }
        )
    return claims


def _scene_claims(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    # Keep the review pressure bounded: at most one broad setting concept and one
    # concrete object-region obligation. This preserves yellow-bag/boxes-type
    # separation without turning floor/wall/background texture into a prose quota.
    gestalt = _scene_gestalt_claims(evidence)
    concrete = _concrete_region_claims(evidence)
    return [*gestalt[:1], *concrete[:1]]


def _preferred_scene_entities(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, item in enumerate(evidence.get("non_target_entities") or []):
        if not isinstance(item, dict):
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.75 or not str(item.get("description") or "").strip():
            continue
        ranked.append((confidence, -index, copy.deepcopy(item)))
    ranked.sort(reverse=True, key=lambda value: (value[0], value[1]))
    preferred = [item for _, _, item in ranked[:3]]

    seen = {str(item.get("description") or "").lower() for item in preferred}
    env = evidence.get("environment_lighting") or {}
    for item in env.get("important_background_or_nuisance_regions") or []:
        if len(preferred) >= 3 or not isinstance(item, dict):
            break
        description = str(item.get("description") or "").strip()
        if not description or description.lower() in seen or not _CONCRETE_SCENE_OBJECT_RE.search(description):
            continue
        preferred.append({"description": description, "source": "important_scene_region"})
        seen.add(description.lower())
    return preferred


def _extract_summary_apparel(analysis: dict[str, Any]) -> list[str]:
    summary = str(analysis.get("image_summary") or "")
    out: list[str] = []
    seen: set[str] = set()
    for match in _SUMMARY_APPAREL_RE.finditer(summary):
        phrase = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:-")
        key = phrase.lower()
        if phrase and key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


def _enrich_transient_appearance(evidence: dict[str, Any], analysis: dict[str, Any], projection: dict[str, Any]) -> None:
    transient = evidence.setdefault("transient_appearance", {})
    descriptors = transient.setdefault("descriptors", [])
    existing = {str(value).lower() for value in descriptors}
    added: list[str] = []
    for phrase in _extract_summary_apparel(analysis):
        if phrase.lower() in existing:
            continue
        descriptors.append(phrase)
        existing.add(phrase.lower())
        added.append(phrase)
    if added:
        projection.setdefault("allowed", []).append(
            {
                "path": "analysis.image_summary[appearance-only quarantine]",
                "reason": "extended_transient_apparel_whitelist",
                "descriptors": added,
            }
        )


def _qualify_side_neutral_standing(evidence: dict[str, Any], projection: dict[str, Any]) -> None:
    pose = _pose(evidence)
    posture = pose.get("whole_body_posture")
    if not isinstance(posture, dict):
        return
    allowed = [str(value) for value in (posture.get("allowed") or [])]
    if "standing" in allowed:
        return
    parts = [item for item in (pose.get("visible_subject_parts") or []) if isinstance(item, dict)]
    weight_bearing_legs = [
        item
        for item in parts
        if re.search(r"\bleg\b", str(item.get("part") or ""), re.I)
        and str(item.get("visibility") or "").lower() in {"full", "visible"}
        and re.search(r"\bweight[- ]bearing\b", str(item.get("support") or ""), re.I)
    ]
    grounded_feet = any(
        re.search(r"\bfeet?\b", str(item.get("part") or ""), re.I)
        and str(item.get("visibility") or "").lower() in {"partial", "full", "visible"}
        and re.search(r"\b(?:floor|ground)\b", " ".join(str(item.get(field) or "") for field in ("contact", "support")), re.I)
        for item in parts
    )
    torso_on_feet = any(
        re.search(r"\btorso\b", str(item.get("part") or ""), re.I)
        and re.search(r"\bon\s+feet\b", str(item.get("support") or ""), re.I)
        for item in parts
    )
    if len(weight_bearing_legs) < 2 or not grounded_feet or not torso_on_feet:
        return
    posture["allowed"] = sorted(set(allowed) | {"standing"})
    posture.setdefault("evidence", []).append(
        "two full visible weight-bearing leg observations plus visible feet on floor and torso supported on feet"
    )
    projection.setdefault("allowed", []).append(
        {
            "path": "caption-evidence-1.3.pose_orientation.whole_body_posture",
            "reason": "side_neutral_full_leg_and_grounded_feet_support_qualifies_standing",
        }
    )


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_135(fused_payload, analysis, caption_policy=caption_policy)
    evidence["projection_revision"] = "1.4.0"
    projection = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    if isinstance(projection, dict):
        projection["schema_version"] = "caption-projection-audit-1.4.0"

    projection_target = projection if isinstance(projection, dict) else audit
    _enrich_transient_appearance(evidence, analysis, projection_target)
    _qualify_side_neutral_standing(evidence, projection_target)
    _coalesce_pose_support(evidence, projection_target)

    existing = [
        copy.deepcopy(item)
        for item in (evidence.get("required_claims") or [])
        if isinstance(item, dict) and not str(item.get("id") or "").startswith("support_relation_")
    ]
    existing.extend(_support_claims(evidence))
    coalesced, dropped = _coalesce_depth_claims(existing)
    evidence["required_claims"] = coalesced
    _coalesce_3d_payload(evidence)

    previous_scene = copy.deepcopy(evidence.get("required_scene_claims") or [])
    evidence["required_scene_claims"] = _scene_claims(evidence)
    env = evidence.setdefault("environment_lighting", {})
    env["preferred_scene_entities"] = _preferred_scene_entities(evidence)
    env["scene_detail_policy"] = "prefer_distinctive_entities_and_semantic_gestalt_over_generic_surface_inventory"
    hard = evidence.setdefault("hard_constraints", {})
    hard["important_scene_regions_must_be_captioned"] = False
    hard["required_scene_claims_must_be_captioned"] = True

    if isinstance(projection, dict):
        if dropped:
            projection.setdefault("blocked", []).append(
                {
                    "path": "caption-evidence-1.3.required_claims",
                    "reason": "redundant_depth_claims_subsumed_by_stronger_semantic_depth_relation",
                    "claim_ids": dropped,
                }
            )
        if previous_scene != evidence["required_scene_claims"]:
            projection.setdefault("blocked", []).append(
                {
                    "path": "caption-evidence-1.3.required_scene_claims",
                    "reason": "generic_nuisance_inventory_demoted_but_concrete_object_regions_preserved",
                    "previous_claims": previous_scene,
                    "replacement_claims": copy.deepcopy(evidence["required_scene_claims"]),
                }
            )
        if env["preferred_scene_entities"]:
            projection.setdefault("allowed", []).append(
                {
                    "path": "caption-evidence-1.3.environment_lighting.preferred_scene_entities",
                    "reason": "high_confidence_or_concrete_scene_entities_prioritized_over_generic_surfaces",
                    "count": len(env["preferred_scene_entities"]),
                }
            )
        projection.setdefault("notes", []).append(
            "Projection 1.4.0 optimizes semantic coverage under compression: overlapping support/depth evidence may be expressed once; generic surfaces are not prose quotas; concrete object regions remain protected from omission."
        )
    return evidence, audit


def _orientation_violation_is_anatomy_bridge(caption: str, violation: dict[str, Any]) -> bool:
    if violation.get("type") != "orientation_side_invented_from_side_neutral_relation":
        return False
    text = str(violation.get("text") or "").strip()
    if not re.search(r"\b(?:left|right)\s*$", text, re.I):
        return False
    return bool(re.search(re.escape(text) + r"\s+(?:hand|wrist|forearm|arm|elbow|shoulder|hip|pelvis|thigh|knee|ankle|leg|foot|feet)\b", caption, re.I))


def _support_claim_is_covered(caption: str, claim: dict[str, Any], evidence: dict[str, Any]) -> bool:
    targets: set[str] = set()
    semantic_target = str(claim.get("semantic_target") or "").lower()
    if semantic_target:
        targets.update(word for word in re.findall(r"[a-z]+", semantic_target) if len(word) >= 3)
    actor = str(claim.get("actor_part") or "").lower().replace("_", " ")
    for item in _pose(evidence).get("visible_subject_parts") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("part") or "").lower().replace("_", " ")
        if actor and label != actor:
            continue
        related = " ".join(str(item.get(field) or "") for field in ("geometry", "contact", "support")).lower()
        for match in re.finditer(r"\b(?:contact\s+with|resting\s+on|supports?|supporting)\s+(?:the\s+)?([a-z]+)", related):
            if len(match.group(1)) >= 3:
                targets.add(match.group(1))
    if not targets:
        return False
    for match in _SUPPORT_RE.finditer(caption):
        window = caption[max(0, match.start() - 45):match.end() + 55].lower()
        if any(re.search(rf"\b{re.escape(target)}\b", window) for target in targets):
            return True
    return False


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_135(caption, evidence))
    violations: list[dict[str, Any]] = []
    for item in result.get("violations") or []:
        if _orientation_violation_is_anatomy_bridge(caption, item):
            continue
        if item.get("type") == "contradicts_signed_torso_depth" and _TORSO_ANGLED_DEPTH_RE.search(caption):
            continue
        violations.append(item)

    if _NEGATIVE_APPEARANCE_RE.search(caption):
        violations.append(
            {
                "type": "unsupported_negative_appearance_claim",
                "text": _NEGATIVE_APPEARANCE_RE.search(caption).group(0),
            }
        )

    claims = {
        str(item.get("id") or ""): item
        for item in (evidence.get("required_claims") or [])
        if isinstance(item, dict)
    }
    warnings: list[dict[str, Any]] = []
    for item in result.get("warnings") or []:
        claim_id = str(item.get("claim_id") or "")
        claim = claims.get(claim_id)
        if (
            item.get("type") == "required_claim_not_detected"
            and claim_id.startswith("support_relation_")
            and isinstance(claim, dict)
            and _support_claim_is_covered(caption, claim, evidence)
        ):
            continue
        warnings.append(item)

    result["schema_version"] = "caption-authority-lint-1.4.0"
    result["violations"] = violations
    result["warnings"] = warnings
    result["violation_count"] = len(violations)
    result["warning_count"] = len(warnings)
    result["passed"] = not violations
    return result
