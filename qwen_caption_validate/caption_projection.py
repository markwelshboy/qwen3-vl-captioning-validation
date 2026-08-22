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
    "bag",
    "backpack",
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
    r"(?:t-?shirt|tee|tank\s+top|blouse|sweater|hoodie|jacket|coat|robe|dress|skirt|trousers?|"
    r"pants|jeans|shorts|suit|tie|scarf|hat|cap|headband|sunglasses|eyeglasses|glasses|mask|"
    r"smartwatch|watch|bracelet|necklace|earrings?|jewelry|ring|bag|backpack|socks?|shoes?|boots?|"
    r"sandals?|gloves?)"
)
_COLORED_ITEM_RE = re.compile(rf"\b{_COLOR}(?:\s+[A-Za-z][A-Za-z0-9'-]*){{0,1}}\s+{_ITEM}\b", re.IGNORECASE)
_BARE_ITEM_RE = re.compile(rf"\b{_ITEM}\b", re.IGNORECASE)
_SHIRTLESS_RE = re.compile(r"\bshirtless\b", re.IGNORECASE)


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


def _sanitize_distal_arm_claims(
    fused_payload: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Remove semantic hand claims when deterministic arm evidence stops before the wrist."""
    payload = copy.deepcopy(fused_payload)
    fusion = _fusion_root(payload)
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
    text = re.sub(r"\s+", " ", value.strip(" ,.;:-"))
    return text


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

    # Analyze-v2.1 did not have a dedicated transient-appearance object. Salvage only
    # whitelisted clothing/accessory/temporary-hair phrases from its summary. The raw
    # summary itself is never exposed to Compose, and pose/camera/laterality prose from
    # it cannot cross this projection boundary.
    summary = analysis.get("image_summary")
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


def _pose_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in parts:
        out.append(
            {
                "part": item.get("part"),
                "anatomical_side": item.get("anatomical_side"),
                "visibility": item.get("visibility"),
                "geometry": item.get("geometry"),
                "contact": item.get("contact"),
                "support": item.get("support"),
                "foreshortening": item.get("foreshortening"),
                "laterality_qualified": bool(item.get("laterality_qualified")),
            }
        )
    return out


def _qualified_whole_body_posture(parts: list[dict[str, Any]]) -> dict[str, Any]:
    allowed: set[str] = set()
    evidence: list[str] = []

    for item in parts:
        geometry = str(item.get("geometry") or "")
        contact = str(item.get("contact") or "")
        support = str(item.get("support") or "")
        text = " ".join((geometry, contact, support)).lower()

        if ("weight-bearing" in text or "weight bearing" in text) and re.search(
            r"\b(?:foot|feet)\b.{0,25}\b(?:ground|floor|planted)\b|\bplanted\b.{0,20}\b(?:foot|feet)\b",
            text,
        ):
            allowed.add("standing")
            evidence.append(f"{item.get('part')}: visible weight-bearing foot/ground support")

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


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a task-shaped caption contract from governed Fusion evidence."""
    projection_audit: dict[str, Any] = {
        "schema_version": "caption-projection-audit-1.3",
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

    evidence = {
        "schema_version": "caption-evidence-1.3",
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
            "visible_subject_parts": _pose_parts(parts),
            "qualified_interactions": base.get("qualified_interactions") or [],
            "qualified_3d_geometry": base.get("qualified_3d_geometry") or {},
        },
        "framing_camera": {
            "framing": base.get("framing") or {},
            "camera_relationship": None,
        },
        "environment_lighting": {
            "scene": base.get("scene") or {},
            "important_background_or_nuisance_regions": _without_frame_location(
                base.get("important_nuisance_regions") or [],
                projection_audit,
                "caption-evidence-1.1.important_nuisance_regions[].frame_location",
            ),
        },
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
            "do_not_caption_intrinsic_identity_traits": True,
            "protected_traits": protected,
        },
    }
    audit = {
        "schema_version": "caption-firewall-and-projection-audit-1.3",
        "firewall": firewall_audit,
        "projection": projection_audit,
    }
    return evidence, audit
