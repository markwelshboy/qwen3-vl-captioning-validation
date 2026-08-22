from __future__ import annotations

import copy
import re
from typing import Any

from .caption_evidence import build_caption_evidence


DEFAULT_PROTECTED_TRAITS = [
    "natural hair color",
    "baseline hair length",
    "eye color",
    "facial structure",
    "apparent age",
    "skin tone",
]

_TRANSIENT_APPEARANCE_TOKENS = (
    "shirt",
    "t-shirt",
    "tee",
    "tank top",
    "blouse",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "robe",
    "dress",
    "skirt",
    "trouser",
    "pants",
    "jeans",
    "shorts",
    "suit",
    "tie",
    "scarf",
    "hat",
    "cap",
    "headband",
    "sunglasses",
    "glasses",
    "eyeglasses",
    "mask",
    "watch",
    "smartwatch",
    "bracelet",
    "necklace",
    "earring",
    "jewelry",
    "ring",
    "sock",
    "shoe",
    "boot",
    "sandal",
    "glove",
)
_HAIR_STATE_RE = re.compile(
    r"(?:hair.{0,30}(?:tied(?: back)?|pulled back|ponytail|bun|wet|windblown|windswept|braid(?:ed)?|covered|tucked)|"
    r"(?:tied(?: back)?|pulled back|ponytail|bun|wet|windblown|windswept|braid(?:ed)?|covered|tucked).{0,30}hair)",
    re.IGNORECASE,
)
_HANDISH_RE = re.compile(r"\b(?:hand|hands|finger|fingers|fingertip|fingertips)\b", re.IGNORECASE)

_COLOR = (
    r"(?:dark\s+(?:gray|grey|blue|green|brown|red)|light\s+(?:gray|grey|blue|green|brown|red)|"
    r"black|white|gray|grey|dark|light|blue|purple|red|green|yellow|pink|orange|navy|teal|"
    r"maroon|brown|beige|tan|cream)"
)
_ITEM = (
    r"(?:shirt|t-?shirt|tee|tank\s+top|blouse|sweater|hoodie|jacket|coat|robe|dress|skirt|trousers?|"
    r"pants|jeans|shorts|suit|tie|scarf|hat|cap|headband|sunglasses|eyeglasses|glasses|mask|"
    r"smartwatch|watch|bracelet|necklace|earrings?|jewelry|ring|socks?|shoes?|boots?|sandals?|gloves?)"
)
_COLORED_ITEM_RE = re.compile(rf"\b{_COLOR}(?:\s+[A-Za-z][A-Za-z0-9'-]*){{0,1}}\s+{_ITEM}\b", re.IGNORECASE)
_BARE_ITEM_RE = re.compile(rf"\b{_ITEM}\b", re.IGNORECASE)
_SHIRTLESS_RE = re.compile(r"\bshirtless\b", re.IGNORECASE)
_SIDE_UNSPECIFIED_RE = re.compile(r"\bside[-_ ]unspecified\b", re.IGNORECASE)
_POSTURE_PATTERNS = {
    "standing": re.compile(r"\b(?:stands?|standing|stood)\b", re.IGNORECASE),
    "seated": re.compile(r"\b(?:sits?|sitting|seated|sat)\b", re.IGNORECASE),
    "lying": re.compile(r"\b(?:lies|lying|lie)\b", re.IGNORECASE),
    "reclined": re.compile(r"\b(?:reclines?|reclined|reclining)\b", re.IGNORECASE),
}
_SCENE_BACKGROUND_KEYS = (
    "texture_complexity",
    "structural_complexity",
    "specular_reflective",
    "repeated_geometry",
    "strong_lines_or_angles",
    "reflections_present",
    "notes",
)
_SCENE_CLAIM_STOPWORDS = {
    "a",
    "an",
    "and",
    "area",
    "background",
    "behind",
    "clutter",
    "frame",
    "image",
    "including",
    "large",
    "near",
    "of",
    "region",
    "subject",
    "the",
    "visible",
    "with",
}


def _fusion_root(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("fusion") if isinstance(payload.get("fusion"), dict) else payload
    return value if isinstance(value, dict) else {}


def _strip_hand_clause(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    pieces = re.split(r"([,;])", value)
    kept: list[str] = []
    for index in range(0, len(pieces), 2):
        clause = pieces[index].strip()
        if clause and not _HANDISH_RE.search(clause):
            kept.append(clause)
    text = ", ".join(kept).strip()
    return text or None


def _has_supported_hand_root(fusion: dict[str, Any], side: str) -> bool:
    deterministic = fusion.get("deterministic_geometry") or {}
    for item in deterministic.get("hand_candidates") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("supported_by_nearby_visible_target_wrist"):
            continue
        if item.get("nearest_visible_target_wrist") == side:
            return True
    return False


def _is_hand_observation(item: dict[str, Any]) -> bool:
    text = " ".join(
        [str(item.get("part") or ""), *[str(value) for value in (item.get("visible_subparts") or [])]]
    ).lower()
    return "hand" in text or "finger" in text


def _normalized_observation_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(?:anatomical\s+)?(?:left|right)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:_-")


def _hand_observation_signature(item: dict[str, Any]) -> tuple[Any, ...]:
    part = _normalized_observation_text(item.get("part"))
    if "hand" in part or "finger" in part:
        part = "hand"
    subparts = tuple(sorted(_normalized_observation_text(value) for value in (item.get("visible_subparts") or [])))
    return (
        part,
        str(item.get("visibility") or ""),
        subparts,
        _normalized_observation_text(item.get("geometry")),
        _normalized_observation_text(item.get("contact")),
        _normalized_observation_text(item.get("support")),
        _normalized_observation_text(item.get("image_location")),
    )


def _dedupe_hand_observations(fusion: dict[str, Any], audit: dict[str, Any]) -> None:
    """Collapse a semantic duplicate only when deterministic hand-root evidence resolves it."""
    parts = [item for item in (fusion.get("qualified_body_parts") or []) if isinstance(item, dict)]
    if not parts:
        return

    deterministic = fusion.get("deterministic_geometry") or {}
    supported_root_sides = {
        str(item.get("nearest_visible_target_wrist"))
        for item in (deterministic.get("hand_candidates") or [])
        if isinstance(item, dict)
        and item.get("supported_by_nearby_visible_target_wrist")
        and item.get("nearest_visible_target_wrist") in {"left", "right"}
    }
    if not supported_root_sides:
        return

    qualified_by_signature: dict[tuple[Any, ...], set[str]] = {}
    for item in parts:
        if not _is_hand_observation(item):
            continue
        state = item.get("fusion_v2") or {}
        if not state.get("selection_usable") or not state.get("laterality_selection_usable"):
            continue
        side = state.get("qualified_anatomical_side") or item.get("anatomical_side")
        if side not in supported_root_sides:
            continue
        qualified_by_signature.setdefault(_hand_observation_signature(item), set()).add(str(side))

    if not qualified_by_signature:
        return

    kept: list[dict[str, Any]] = []
    for item in parts:
        if not _is_hand_observation(item):
            kept.append(item)
            continue
        state = item.get("fusion_v2") or {}
        reasons = " ".join(str(value) for value in (state.get("laterality_reasons") or []))
        signature = _hand_observation_signature(item)
        matching_sides = qualified_by_signature.get(signature) or set()
        is_conflicted_duplicate = (
            bool(state.get("selection_usable"))
            and not bool(state.get("laterality_selection_usable"))
            and bool(matching_sides)
            and "conflicts with dwpose hand-root association" in reasons.lower()
        )
        if not is_conflicted_duplicate:
            kept.append(item)
            continue
        audit["blocked"].append(
            {
                "path": "fusion.qualified_body_parts",
                "reason": "duplicate_hand_observation_collapsed_to_qualified_deterministic_root",
                "reported_anatomical_side": item.get("anatomical_side"),
                "qualified_root_sides": sorted(matching_sides),
            }
        )

    fusion["qualified_body_parts"] = kept


def _sanitize_distal_arm_claims(
    fused_payload: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Remove unsupported distal claims and collapse deterministic duplicate hand observations."""
    payload = copy.deepcopy(fused_payload)
    fusion = _fusion_root(payload)
    _dedupe_hand_observations(fusion, audit)

    deterministic = fusion.get("deterministic_geometry") or {}
    connectivity = deterministic.get("connectivity") or {}

    for item in fusion.get("qualified_body_parts") or []:
        if not isinstance(item, dict):
            continue
        part = str(item.get("part") or "").lower()
        side = str(item.get("anatomical_side") or "unknown").lower()
        if "arm" not in part or side not in {"left", "right"}:
            continue
        chain = connectivity.get(f"{side}_arm") or {}
        visible = {str(value) for value in (chain.get("visible") or [])}
        wrist_name = f"{side}_wrist"
        if wrist_name in visible or _has_supported_hand_root(fusion, side):
            continue

        subparts = [str(value) for value in (item.get("visible_subparts") or [])]
        filtered = [value for value in subparts if not _HANDISH_RE.search(value)]
        changed = filtered != subparts
        item["visible_subparts"] = filtered
        for field in ("geometry", "contact", "support"):
            before = item.get(field)
            after = _strip_hand_clause(before)
            if after != before:
                changed = True
                item[field] = after
        if changed:
            audit["blocked"].append(
                {
                    "path": "fusion.qualified_body_parts",
                    "reason": "distal_hand_claim_withheld_without_deterministic_wrist_or_hand_root_support",
                    "source_anatomical_side": side,
                    "part": item.get("part"),
                    "visible_chain_landmarks": sorted(visible),
                }
            )
    return payload


def _normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" ,.;:-"))


def _extract_transient_phrases(value: Any) -> list[str]:
    """Extract only tightly whitelisted transient appearance phrases from free text."""
    if not isinstance(value, str) or not value.strip():
        return []
    text = value
    found: list[str] = []

    for match in _COLORED_ITEM_RE.finditer(text):
        found.append(_normalize_phrase(match.group(0)))

    for match in _BARE_ITEM_RE.finditer(text):
        bare = _normalize_phrase(match.group(0))
        if any(existing.lower().endswith(bare.lower()) for existing in found):
            continue
        found.append(bare)

    if _SHIRTLESS_RE.search(text):
        found.append("shirtless")

    low = text.lower()
    hair_states = (
        (r"hair.{0,30}pulled back|pulled back.{0,30}hair", "hair pulled back"),
        (r"hair.{0,30}tied back|tied back.{0,30}hair", "hair tied back"),
        (r"\bwet hair\b|hair.{0,20}\bwet\b", "wet hair"),
        (r"\bwindblown hair\b|hair.{0,20}\bwindblown\b", "windblown hair"),
        (r"\bwindswept hair\b|hair.{0,20}\bwindswept\b", "windswept hair"),
        (r"hair.{0,30}\bponytail\b|\bponytail\b.{0,30}hair", "hair in a ponytail"),
        (r"hair.{0,30}\bbun\b|\bbun\b.{0,30}hair", "hair in a bun"),
        (r"hair.{0,30}\bbraid(?:ed)?\b|\bbraid(?:ed)?\b.{0,30}hair", "braided hair"),
    )
    for pattern, normalized in hair_states:
        if re.search(pattern, low, re.IGNORECASE):
            found.append(normalized)

    out: list[str] = []
    seen: set[str] = set()
    for phrase in found:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


def _summary_appearance_text(value: Any) -> str:
    """Return summary text only as input to strict transient-appearance extraction."""
    return value if isinstance(value, str) else ""


def _transient_appearance(
    parts: list[dict[str, Any]],
    analysis: dict[str, Any],
    audit: dict[str, Any],
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    def add(text: str, *, source: str) -> None:
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        values.append(text)
        audit["allowed"].append(
            {
                "path": source,
                "reason": "strict_transient_appearance_whitelist",
                "descriptor": text,
            }
        )

    for item in parts:
        source_values = [
            *(item.get("visible_subparts") or []),
            item.get("geometry"),
            item.get("contact"),
            item.get("support"),
        ]
        for raw in source_values:
            text = str(raw or "").strip()
            if not text:
                continue
            extracted = _extract_transient_phrases(text)
            for descriptor in extracted:
                add(descriptor, source="caption-evidence-1.1.visible_subject_parts")
            low = text.lower()
            if not extracted and any(
                token in low for token in ("hair", "beard", "eye", "face", "nose", "mouth", "ear", "skin", "tattoo")
            ):
                audit["blocked"].append(
                    {
                        "path": "caption_projection.transient_appearance",
                        "reason": "intrinsic_or_identity_like_descriptor_not_caption_authoritative",
                        "descriptor": text,
                    }
                )

    # Analyze-v2.1 did not have a dedicated transient-appearance object. Run its full
    # summary through the strict appearance extractor so mixed sentences such as
    # "shirtless ... in a wooded environment" do not lose the appearance fact. Only
    # extracted whitelist phrases cross this boundary; raw summary prose never reaches
    # Compose, so pose/camera/laterality and background wording cannot become authority.
    summary = _summary_appearance_text(analysis.get("image_summary"))
    for descriptor in _extract_transient_phrases(summary):
        add(descriptor, source="analysis.image_summary[appearance-only quarantine]")

    return values


def _project_orientation(
    orientation: Any,
    audit: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(orientation, dict):
        return {}
    out: dict[str, Any] = {}
    for name, raw in orientation.items():
        if not isinstance(raw, dict):
            continue
        value = dict(raw)
        direction = str(value.get("direction") or "unknown")
        if direction in {"side_unspecified", "anatomical_left", "anatomical_right"}:
            value.pop("direction", None)
            if name.endswith("_yaw"):
                value["relation"] = "turned_from_frontal"
            elif name.endswith("_roll"):
                value["relation"] = "tilted_from_upright"
            else:
                value["relation"] = "deviated_from_neutral"
            audit["blocked"].append(
                {
                    "path": f"caption-evidence-1.1.semantic_orientation.{name}.direction",
                    "reason": "anatomical_side_direction_replaced_with_side_neutral_relation",
                }
            )
        out[name] = value
    return out


def _sanitize_side_neutral_text(value: Any, audit: dict[str, Any], path: str) -> Any:
    if not isinstance(value, str) or not _SIDE_UNSPECIFIED_RE.search(value):
        return value
    text = re.sub(r"\bto\s+side[-_ ]unspecified\b", "to the side", value, flags=re.IGNORECASE)
    text = re.sub(r"\btowards?\s+side[-_ ]unspecified\b", "toward the side", text, flags=re.IGNORECASE)
    text = _SIDE_UNSPECIFIED_RE.sub("the side", text)
    text = re.sub(r"\s+", " ", text).strip()
    audit["blocked"].append(
        {
            "path": path,
            "reason": "side_unspecified_meta_label_replaced_with_natural_side_neutral_text",
        }
    )
    return text or None


def _strip_unauthorized_posture(
    value: Any,
    allowed: set[str],
    audit: dict[str, Any],
    path: str,
) -> Any:
    if not isinstance(value, str):
        return value
    text = value
    original = value
    for posture, pattern in _POSTURE_PATTERNS.items():
        if posture in allowed or not pattern.search(text):
            continue
        clauses = [piece.strip() for piece in re.split(r"[,;]", text) if piece.strip()]
        kept: list[str] = []
        for clause in clauses:
            if not pattern.search(clause):
                kept.append(clause)
                continue
            reduced = pattern.sub("", clause)
            reduced = re.sub(r"^\s*(?:with\s+)", "", reduced, flags=re.IGNORECASE)
            reduced = re.sub(r"\s+", " ", reduced).strip(" ,.;:-")
            if reduced.lower().startswith(("on ground", "on the ground", "on floor", "on the floor", "on elevator floor")):
                reduced = ""
            if reduced:
                kept.append(reduced)
        text = ", ".join(kept)
        audit["blocked"].append(
            {
                "path": path,
                "reason": "unauthorized_whole_body_posture_removed_from_subordinate_pose_text",
                "posture": posture,
                "source_text": original,
            }
        )
    return text.strip() or None


def _pose_parts(
    parts: list[dict[str, Any]],
    allowed_postures: set[str],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in enumerate(parts):
        projected: dict[str, Any] = {
            "part": item.get("part"),
            "anatomical_side": item.get("anatomical_side"),
            "visibility": item.get("visibility"),
            "geometry": item.get("geometry"),
            "contact": item.get("contact"),
            "support": item.get("support"),
            "foreshortening": item.get("foreshortening"),
            "laterality_qualified": bool(item.get("laterality_qualified")),
        }
        for field in ("geometry", "contact", "support"):
            path = f"caption-evidence-1.1.visible_subject_parts[{index}].{field}"
            value = _sanitize_side_neutral_text(projected.get(field), audit, path)
            projected[field] = _strip_unauthorized_posture(value, allowed_postures, audit, path)
        out.append(projected)
    return out


def _qualified_whole_body_posture(parts: list[dict[str, Any]]) -> dict[str, Any]:
    allowed: set[str] = set()
    evidence: list[str] = []

    for item in parts:
        geometry = str(item.get("geometry") or "")
        contact = str(item.get("contact") or "")
        support = str(item.get("support") or "")
        text = " ".join((geometry, contact, support)).lower()

        direct_weight_support = bool(
            re.search(
                r"\b(?:weight-bearing|weight bearing|supporting (?:the )?body weight|supports? (?:the )?body weight)\b",
                text,
            )
        )
        grounded_feet = bool(
            re.search(
                r"\b(?:foot|feet)\b.{0,35}\b(?:ground|floor|planted)\b|"
                r"\bplanted\b.{0,25}\b(?:foot|feet)\b|"
                r"\b(?:ground|floor)\b.{0,35}\b(?:foot|feet)\b",
                text,
            )
        )
        if direct_weight_support and grounded_feet:
            allowed.add("standing")
            evidence.append(f"{item.get('part')}: visible planted/grounded feet supporting body weight")

        if re.search(r"\b(?:seated|sitting)\b", text) or re.search(
            r"\b(?:pelvis|torso|body)\b.{0,35}\b(?:chair|seat|bench)\b",
            text,
        ):
            allowed.add("seated")
            evidence.append(f"{item.get('part')}: explicit seated/support relationship")

        if re.search(r"\b(?:lying|lies|lie)\b", text) and re.search(
            r"\b(?:bed|bedspread|ground|floor|couch|sofa|surface)\b",
            text,
        ):
            allowed.add("lying")
            evidence.append(f"{item.get('part')}: explicit lying/support relationship")

        if re.search(r"\breclin(?:e|es|ed|ing)\b", text) and re.search(
            r"\b(?:bed|bedspread|chair|seat|couch|sofa|surface|supported|resting)\b",
            text,
        ):
            allowed.add("reclined")
            evidence.append(f"{item.get('part')}: explicit reclined/support relationship")

    return {
        "allowed": sorted(allowed),
        "authority": "direct_visible_support_only",
        "evidence": evidence,
    }


def _compact_gaze(gaze: Any, audit: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(gaze, dict):
        return None
    if gaze.get("frame_direction") not in (None, "unknown"):
        audit["blocked"].append(
            {
                "path": "caption-evidence-1.1.gaze.frame_direction",
                "reason": "horizontal_frame_direction_withheld_from_caption_projection",
            }
        )
    target = gaze.get("target")
    return {"target": target} if target else None


def _without_frame_location(items: Any, audit: dict[str, Any], path: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = {key: value for key, value in raw.items() if key not in {"frame_location", "image_location"}}
        if len(item) != len(raw):
            audit["blocked"].append(
                {
                    "path": path,
                    "reason": "horizontal_frame_location_withheld_from_caption_projection",
                }
            )
        out.append(item)
    return out


def _project_scene(base_scene: Any, analysis_scene: Any, audit: dict[str, Any]) -> dict[str, Any]:
    """Keep governed scene facts plus Analyze's structured background description."""
    out = dict(base_scene) if isinstance(base_scene, dict) else {}
    source = analysis_scene if isinstance(analysis_scene, dict) else {}
    raw_background = source.get("background_structure")
    if not isinstance(raw_background, dict):
        return out

    background = {
        key: raw_background.get(key)
        for key in _SCENE_BACKGROUND_KEYS
        if raw_background.get(key) is not None
    }
    if background:
        out["background_structure"] = background
        audit["allowed"].append(
            {
                "path": "analysis.scene.background_structure",
                "reason": "structured_scene_context_is_caption_safe",
            }
        )
    return out


def _scene_claim_keyword(value: str) -> str:
    word = value.lower()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if re.search(r"(?:x|z|ch|sh|ss)es$", word) and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _required_scene_claims(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for index, item in enumerate(regions, start=1):
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        keywords: list[str] = []
        seen: set[str] = set()
        for raw in re.findall(r"[A-Za-z0-9]+", description.lower()):
            if raw in _SCENE_CLAIM_STOPWORDS or len(raw) < 3:
                continue
            word = _scene_claim_keyword(raw)
            if word in _SCENE_CLAIM_STOPWORDS or word in seen:
                continue
            seen.add(word)
            keywords.append(word)
        claims.append(
            {
                "id": f"important_scene_region_{index}",
                "description": description,
                "keywords": keywords,
                "minimum_keyword_matches": min(2, len(keywords)) if keywords else 0,
                "attribution": "scene_or_background_not_trigger_identity",
            }
        )
    return claims


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a task-shaped caption contract from governed Fusion evidence."""
    projection_audit: dict[str, Any] = {
        "schema_version": "caption-projection-audit-1.3.1",
        "allowed": [],
        "blocked": [],
        "notes": [],
    }
    sanitized = _sanitize_distal_arm_claims(fused_payload, projection_audit)
    base, firewall_audit = build_caption_evidence(sanitized, analysis)

    parts = [item for item in (base.get("visible_subject_parts") or []) if isinstance(item, dict)]
    policy = dict(caption_policy or {})
    protected = list(DEFAULT_PROTECTED_TRAITS)
    for value in policy.get("protected_traits") or []:
        text = str(value).strip()
        if text and text.lower() not in {item.lower() for item in protected}:
            protected.append(text)
    policy["protected_traits"] = protected

    for limitation in base.get("coverage_limitations") or []:
        projection_audit["notes"].append(
            {
                "type": "coverage_limitation_withheld_from_compose",
                "text": str(limitation),
            }
        )

    projected_orientation = _project_orientation(base.get("semantic_orientation") or {}, projection_audit)
    posture = _qualified_whole_body_posture(parts)
    allowed_postures = {str(value) for value in posture.get("allowed") or []}
    projected_parts = _pose_parts(parts, allowed_postures, projection_audit)
    scene = _project_scene(base.get("scene") or {}, analysis.get("scene") or {}, projection_audit)
    important_regions = _without_frame_location(
        base.get("important_nuisance_regions") or [],
        projection_audit,
        "caption-evidence-1.1.important_nuisance_regions[].frame_location",
    )
    required_scene_claims = _required_scene_claims(important_regions)

    evidence = {
        "schema_version": "caption-evidence-1.3",
        "projection_revision": "1.3.1",
        "source_caption_evidence_schema": base.get("schema_version"),
        "source_fusion_schema": base.get("source_fusion_schema"),
        "caption_policy": policy,
        "transient_appearance": {
            "descriptors": _transient_appearance(parts, analysis, projection_audit),
            "expression": base.get("expression_state") or [],
        },
        "pose_orientation": {
            "semantic_orientation": projected_orientation,
            "whole_body_posture": posture,
            "gaze": _compact_gaze(base.get("gaze"), projection_audit),
            "visible_subject_parts": projected_parts,
            "qualified_interactions": base.get("qualified_interactions") or [],
            "qualified_3d_geometry": base.get("qualified_3d_geometry") or {},
        },
        "framing_camera": {
            "framing": base.get("framing") or {},
            "camera_relationship": None,
        },
        "environment_lighting": {
            "scene": scene,
            "important_background_or_nuisance_regions": important_regions,
        },
        "required_scene_claims": required_scene_claims,
        "non_target_entities": _without_frame_location(
            base.get("non_target_entities") or [],
            projection_audit,
            "caption-evidence-1.1.non_target_entities[].frame_location",
        ),
        "embedded_depictions": _without_frame_location(
            base.get("embedded_depictions") or [],
            projection_audit,
            "caption-evidence-1.1.embedded_depictions[].frame_location",
        ),
        "required_claims": base.get("required_claims") or [],
        "hard_constraints": {
            "visibility": base.get("visibility_constraints") or {},
            "never_infer_unqualified_anatomical_laterality": True,
            "never_convert_frame_position_to_anatomical_side": True,
            "never_complete_missing_distal_anatomy": True,
            "whole_body_posture_must_be_listed_in_pose_orientation": True,
            "important_scene_regions_must_be_captioned": True,
            "do_not_caption_intrinsic_identity_traits": True,
            "protected_traits": protected,
        },
    }
    audit = {
        "schema_version": "caption-firewall-and-projection-audit-1.3.1",
        "firewall": firewall_audit,
        "projection": projection_audit,
    }
    return evidence, audit
