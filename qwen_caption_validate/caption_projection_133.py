from __future__ import annotations

import copy
import re
from typing import Any

from .caption_lint import lint_caption as _base_lint_caption
from .caption_projection import build_caption_projection as _base_build_caption_projection

_SIDE_ARM_RE = re.compile(r"\b(left|right)\s+(arm|forearm|hand|wrist|elbow)\b", re.IGNORECASE)
_SIGNED_NEAR_RE = re.compile(r"\b(left|right)\s+shoulder\b.{0,55}?\b(?:closer|nearer)\b|\b(?:closer|nearer)\b.{0,55}?\b(left|right)\s+shoulder\b", re.IGNORECASE)
_TORSO_ANGLE_RE = re.compile(r"\b(?:torso|upper body|body)\b.{0,70}?\b(?:angled|turned|rotated|depth|not\s+square|not\s+straight)\b|\b(?:angled|turned|rotated|depth|not\s+square|not\s+straight)\b.{0,70}?\b(?:torso|upper body|body)\b", re.IGNORECASE)
_FRONTAL_TORSO_RE = re.compile(r"\b(?:torso|upper body|body)\b.{0,35}?\b(?:frontal|square[- ]on|straight[- ]on)\b", re.IGNORECASE)
_SENTENCE_BRIDGE_SIDE_RE = re.compile(r"[.!?]\s+(?:left|right)\s*$", re.IGNORECASE)


def _fusion_root(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("fusion") if isinstance(payload.get("fusion"), dict) else payload
    return value if isinstance(value, dict) else {}


def _sync_refined_laterality(payload: dict[str, Any]) -> dict[str, Any]:
    """Make raw compatibility fields agree with governed Fusion-v2 laterality.

    Projection 1.3.2 still has a few compatibility paths that read the legacy
    anatomical_side field. After Fusion 2.3.1+ corrects a semantic record, that
    legacy field may intentionally retain Analyze's original side for auditing.
    A caption-facing copy must instead expose the qualified side so downstream
    distal-hand sanitization does not validate the wrong arm chain.
    """
    out = copy.deepcopy(payload)
    fusion = _fusion_root(out)
    for item in fusion.get("qualified_body_parts") or []:
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        side = state.get("qualified_anatomical_side")
        if state.get("laterality_selection_usable") and side in {"left", "right"}:
            item["anatomical_side"] = side
    for item in fusion.get("qualified_interactions") or []:
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        side = state.get("qualified_actor_anatomical_side")
        if state.get("laterality_selection_usable") and side in {"left", "right"}:
            item["actor_anatomical_side"] = side
    return out


def _guard_complementary_side_inference(evidence: dict[str, Any], audit: dict[str, Any]) -> None:
    """Remove a sided same-family target when the interaction actor is unsigned.

    Text models are prone to completing "an arm crossed over the left arm" into
    "the right arm crossed over the left arm". If the actor side itself is not
    qualified, retain the relation but redact the target side as well.
    """
    pose = evidence.get("pose_orientation") or {}
    for index, item in enumerate(pose.get("qualified_interactions") or []):
        if not isinstance(item, dict):
            continue
        actor_side = str(item.get("actor_anatomical_side") or "unknown").lower()
        actor = str(item.get("actor_part") or "")
        target = str(item.get("target") or "")
        if actor_side in {"left", "right"}:
            continue
        if not re.search(r"\b(?:arm|forearm|hand|wrist|elbow)\b", actor, re.I):
            continue
        match = _SIDE_ARM_RE.search(target)
        if not match:
            continue
        target_family = match.group(2).lower()
        if target_family not in {"arm", "forearm", "hand", "wrist", "elbow"}:
            continue
        item["target"] = _SIDE_ARM_RE.sub(lambda m: m.group(2), target, count=1)
        notes = item.get("notes")
        if isinstance(notes, str):
            item["notes"] = _SIDE_ARM_RE.sub(lambda m: m.group(2), notes)
        audit.setdefault("blocked", []).append({
            "path": f"caption-evidence-1.3.qualified_interactions[{index}].target",
            "reason": "unsigned_actor_cannot_license_complementary_side_inference",
            "source_target": target,
        })


def _signed_required_claims(fusion: dict[str, Any]) -> list[dict[str, Any]]:
    signed = fusion.get("signed_depth_authority_audit") or {}
    components = signed.get("components") or {}
    shoulder = components.get("shoulder") or {}
    claims: list[dict[str, Any]] = []
    if shoulder.get("action") == "qualified" and shoulder.get("nearer_anatomical_side") in {"left", "right"}:
        side = str(shoulder["nearer_anatomical_side"])
        claims.append({
            "id": "signed_shoulder_nearer_relation",
            "priority": "required",
            "nearer_anatomical_side": side,
            "magnitude_band": (
                "very_high" if float(shoulder.get("magnitude_deg") or 0) >= 50
                else "high" if float(shoulder.get("magnitude_deg") or 0) >= 30
                else "moderate"
            ),
            "description": f"the {side} shoulder is closer to the camera than the opposite shoulder",
        })
    torso = signed.get("torso_direction") or {}
    if torso.get("action") == "qualified" and torso.get("nearer_anatomical_side") in {"left", "right"}:
        claims.append({
            "id": "signed_torso_depth_direction",
            "priority": "required",
            "nearer_anatomical_side": str(torso["nearer_anatomical_side"]),
            "description": "the torso is angled in depth rather than square-on to the camera",
        })
    return claims


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    synced = _sync_refined_laterality(fused_payload)
    evidence, audit = _base_build_caption_projection(synced, analysis, caption_policy=caption_policy)
    evidence["projection_revision"] = "1.3.3"
    projection = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    if isinstance(projection, dict):
        projection["schema_version"] = "caption-projection-audit-1.3.3"
    _guard_complementary_side_inference(evidence, projection if isinstance(projection, dict) else audit)
    claims = _signed_required_claims(_fusion_root(synced))
    existing_ids = {str(item.get("id")) for item in (evidence.get("required_claims") or []) if isinstance(item, dict)}
    for claim in claims:
        if claim["id"] not in existing_ids:
            evidence.setdefault("required_claims", []).append(claim)
    if claims and isinstance(projection, dict):
        projection.setdefault("allowed", []).append({
            "path": "fusion.signed_depth_authority_audit",
            "reason": "visibility_gated_signed_depth_relations_are_must_cover_caption_facts",
            "claim_ids": [claim["id"] for claim in claims],
        })
    return evidence, audit


def _signed_shoulder_claim_present(caption: str, side: str) -> bool:
    for match in _SIGNED_NEAR_RE.finditer(caption):
        found = (match.group(1) or match.group(2) or "").lower()
        if found == side:
            return True
    return False


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_base_lint_caption(caption, evidence))

    # Projection 1.3.2's orientation regex can cross a sentence boundary and
    # attach the next sentence's "Left shoulder..." to "Head tilted down...".
    kept = []
    for violation in result.get("violations") or []:
        if (
            violation.get("type") == "orientation_side_invented_from_side_neutral_relation"
            and _SENTENCE_BRIDGE_SIDE_RE.search(str(violation.get("text") or ""))
        ):
            continue
        kept.append(violation)
    result["violations"] = kept

    warnings = list(result.get("warnings") or [])
    for claim in evidence.get("required_claims") or []:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or "")
        side = str(claim.get("nearer_anatomical_side") or "").lower()
        if claim_id == "signed_shoulder_nearer_relation" and side in {"left", "right"}:
            if not _signed_shoulder_claim_present(caption, side):
                warnings.append({
                    "type": "required_claim_not_detected",
                    "claim_id": claim_id,
                    "nearer_anatomical_side": side,
                })
            opposite = "right" if side == "left" else "left"
            if _signed_shoulder_claim_present(caption, opposite):
                kept.append({
                    "type": "contradicts_signed_shoulder_depth",
                    "expected_nearer_anatomical_side": side,
                    "reported_nearer_anatomical_side": opposite,
                })
        elif claim_id == "signed_torso_depth_direction":
            if not _TORSO_ANGLE_RE.search(caption):
                warnings.append({"type": "required_claim_not_detected", "claim_id": claim_id})
            if _FRONTAL_TORSO_RE.search(caption):
                kept.append({
                    "type": "contradicts_signed_torso_depth",
                    "expected": "angled_in_depth_not_square_on",
                })

    # De-duplicate warnings because future base linters may learn these claim ids.
    unique_warnings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for warning in warnings:
        key = (warning.get("type"), warning.get("claim_id"), warning.get("nearer_anatomical_side"))
        if key in seen:
            continue
        seen.add(key)
        unique_warnings.append(warning)

    result["schema_version"] = "caption-authority-lint-1.3.3"
    result["violations"] = kept
    result["warnings"] = unique_warnings
    result["violation_count"] = len(kept)
    result["warning_count"] = len(unique_warnings)
    result["passed"] = not kept
    return result
