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
    "shoe",
    "boot",
    "sandal",
    "glove",
)
_HAIR_STATE_RE = re.compile(
    r"(?:hair.*(?:tied|ponytail|bun|wet|windblown|windswept|braid|covered|tucked)|"
    r"(?:tied|ponytail|bun|wet|windblown|windswept|braid|covered|tucked).*hair)",
    re.IGNORECASE,
)
_HANDISH_RE = re.compile(r"\b(?:hand|hands|finger|fingers|fingertip|fingertips)\b", re.IGNORECASE)


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
    """Remove semantic hand claims when deterministic arm evidence stops before the wrist.

    Analyze can occasionally complete an arm through a hand that is outside the crop.
    Fusion-v2.3 intentionally remains frozen, so the caption projection applies this
    stricter task-specific governor before constructing caption-safe evidence.
    """
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


def _transient_appearance(parts: list[dict[str, Any]], audit: dict[str, Any]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in parts:
        for raw in item.get("visible_subparts") or []:
            text = str(raw).strip()
            low = text.lower()
            if not text:
                continue
            allowed = any(token in low for token in _TRANSIENT_APPEARANCE_TOKENS) or bool(_HAIR_STATE_RE.search(text))
            if not allowed:
                if any(token in low for token in ("hair", "beard", "eye", "face", "nose", "mouth", "ear", "skin", "tattoo")):
                    audit["blocked"].append(
                        {
                            "path": "caption_projection.transient_appearance",
                            "reason": "intrinsic_or_identity_like_descriptor_not_caption_authoritative",
                            "descriptor": text,
                        }
                    )
                continue
            key = low
            if key not in seen:
                seen.add(key)
                values.append(text)
    return values


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
    """Build a task-shaped caption contract from governed Fusion evidence.

    Caption Evidence 1.1 is a source-oriented firewall. This projection is deliberately
    task-oriented: it exposes only the categories the caption writer needs, while
    retaining hard constraints and required claims separately.
    """
    projection_audit: dict[str, Any] = {
        "schema_version": "caption-projection-audit-1.2",
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

    evidence = {
        "schema_version": "caption-evidence-1.2",
        "source_caption_evidence_schema": base.get("schema_version"),
        "source_fusion_schema": base.get("source_fusion_schema"),
        "caption_policy": policy,
        "transient_appearance": {
            "descriptors": _transient_appearance(parts, projection_audit),
            "expression": base.get("expression_state") or [],
        },
        "pose_orientation": {
            "semantic_orientation": base.get("semantic_orientation") or {},
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
            "do_not_caption_intrinsic_identity_traits": True,
            "protected_traits": protected,
        },
        "coverage_limitations": base.get("coverage_limitations") or [],
    }
    audit = {
        "schema_version": "caption-firewall-and-projection-audit-1.2",
        "firewall": firewall_audit,
        "projection": projection_audit,
    }
    return evidence, audit
