from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_133 import build_caption_projection as _build_133
from .caption_projection_133 import lint_caption as _lint_133

_SIGNED_NEAR_RE = re.compile(
    r"\b(left|right)\s+shoulder\b.{0,60}?\b(?:closer|nearer)\b|"
    r"\b(?:closer|nearer)\b.{0,60}?\b(left|right)\s+shoulder\b",
    re.IGNORECASE,
)
_NON_SQUARE_TORSO_RE = re.compile(
    r"\b(?:torso|upper body|body)\b.{0,70}?\b(?:not\s+square[- ]on|not\s+straight[- ]on|"
    r"angled\s+in\s+depth|turned\s+in\s+depth|rotated\s+in\s+depth)\b|"
    r"\b(?:not\s+square[- ]on|not\s+straight[- ]on|angled\s+in\s+depth|turned\s+in\s+depth|"
    r"rotated\s+in\s+depth)\b.{0,70}?\b(?:torso|upper body|body)\b",
    re.IGNORECASE,
)
_FRONTAL_TERM_RE = re.compile(r"\b(?:frontal|square[- ]on|straight[- ]on)\b", re.IGNORECASE)
_TORSO_TERM_RE = re.compile(r"\b(?:torso|upper body|body)\b", re.IGNORECASE)
_SUBJECT_LYING_RE = re.compile(
    r"\b(?:BLIND7|subject|person|woman|man|she|he|they)\b[^.!?]{0,70}?\b(?:lies|lying)\b|"
    r"\b(?:lying)\b[^.!?]{0,40}?\b(?:BLIND7|subject|person|woman|man|her|him|them)\b",
    re.IGNORECASE,
)
_SUPPORT_RE = re.compile(r"\bsupport(?:s|ed|ing)?\b", re.IGNORECASE)
_SUPPORT_TARGET_RE = re.compile(r"\bsupport(?:s|ed|ing)?\s+(?:the\s+|her\s+|his\s+|their\s+)?([A-Za-z][A-Za-z -]{0,30})", re.IGNORECASE)
_ROTATION_GEOMETRY_RE = re.compile(r"\b(?:rotat(?:ed|ion|ing)?|turned|angled|not\s+square[- ]on|not\s+straight[- ]on)\b", re.IGNORECASE)
_DEPTH_GEOMETRY_RE = re.compile(r"\b(?:closer|nearer)\b.{0,30}?\b(?:camera|viewer)\b|\b(?:camera|viewer)\b.{0,30}?\b(?:closer|nearer)\b", re.IGNORECASE)


def _pose(evidence: dict[str, Any]) -> dict[str, Any]:
    value = evidence.get("pose_orientation")
    return value if isinstance(value, dict) else evidence


def _authorized_nearer_shoulder_sides(evidence: dict[str, Any]) -> set[str]:
    sides: set[str] = set()
    pose = _pose(evidence)
    for item in pose.get("visible_subject_parts") or []:
        if not isinstance(item, dict):
            continue
        side = str(item.get("anatomical_side") or "unknown").lower()
        if side not in {"left", "right"} or not item.get("laterality_qualified"):
            continue
        label = str(item.get("part") or "").lower()
        geometry = str(item.get("geometry") or "")
        if "shoulder" in label and _DEPTH_GEOMETRY_RE.search(geometry):
            sides.add(side)
    for claim in evidence.get("required_claims") or []:
        if not isinstance(claim, dict) or claim.get("id") != "signed_shoulder_nearer_relation":
            continue
        side = str(claim.get("nearer_anatomical_side") or "").lower()
        if side in {"left", "right"}:
            sides.add(side)
    return sides


def _torso_non_square_authorized(evidence: dict[str, Any]) -> bool:
    for claim in evidence.get("required_claims") or []:
        if isinstance(claim, dict) and claim.get("id") == "signed_torso_depth_direction":
            return True
    for item in _pose(evidence).get("visible_subject_parts") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("part") or "").lower()
        geometry = str(item.get("geometry") or "")
        if ("torso" in label or "body" in label) and _ROTATION_GEOMETRY_RE.search(geometry):
            return True
    return False


def _positive_frontal_torso_claim(caption: str) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+", caption):
        if not _TORSO_TERM_RE.search(sentence):
            continue
        for match in _FRONTAL_TERM_RE.finditer(sentence):
            prefix = sentence[max(0, match.start() - 36):match.start()].lower()
            if re.search(r"(?:\bnot\b|\brather\s+than\b|\binstead\s+of\b|\bas\s+opposed\s+to\b)(?:\s+being)?\s*$", prefix):
                continue
            return True
    return False


def _support_claims(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _pose(evidence).get("visible_subject_parts") or []:
        if not isinstance(item, dict):
            continue
        support = str(item.get("support") or "").strip()
        if not support or not _SUPPORT_RE.search(support):
            continue
        # Ground/weight-bearing support is already governed by whole-body posture
        # and need not become another prose quota. Preserve specific local support
        # relations such as a hand supporting the chin.
        if re.search(r"\b(?:weight|floor|ground|standing)\b", support, re.IGNORECASE):
            continue
        side = str(item.get("anatomical_side") or "unknown").lower()
        label = str(item.get("part") or "body part").replace("_", " ")
        description = f"{label}: {support}"
        signature = description.lower()
        if signature in seen:
            continue
        seen.add(signature)
        claims.append({
            "id": f"support_relation_{len(claims) + 1}",
            "priority": "required",
            "description": description,
            "support_text": support,
            "anatomical_side": side if side in {"left", "right"} else "unknown",
        })
    return claims


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_133(fused_payload, analysis, caption_policy=caption_policy)
    evidence["projection_revision"] = "1.3.4"
    projection = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    if isinstance(projection, dict):
        projection["schema_version"] = "caption-projection-audit-1.3.4"

    existing = {str(item.get("id")) for item in (evidence.get("required_claims") or []) if isinstance(item, dict)}
    added: list[str] = []
    for claim in _support_claims(evidence):
        if claim["id"] in existing:
            continue
        evidence.setdefault("required_claims", []).append(claim)
        added.append(str(claim["id"]))
    if added and isinstance(projection, dict):
        projection.setdefault("allowed", []).append({
            "path": "caption-evidence-1.3.visible_subject_parts[].support",
            "reason": "specific_local_support_relations_are_must_cover_pose_facts",
            "claim_ids": added,
        })
    return evidence, audit


def _support_claim_present(caption: str, claim: dict[str, Any]) -> bool:
    support_text = str(claim.get("support_text") or "")
    target_match = _SUPPORT_TARGET_RE.search(support_text)
    if not target_match:
        return bool(_SUPPORT_RE.search(caption))
    target_words = [word.lower() for word in re.findall(r"[A-Za-z]+", target_match.group(1)) if len(word) >= 3]
    target = target_words[0] if target_words else ""
    if not target:
        return bool(_SUPPORT_RE.search(caption))
    for match in _SUPPORT_RE.finditer(caption):
        window = caption[max(0, match.start() - 45):match.end() + 45].lower()
        if re.search(rf"\b{re.escape(target)}\b", window):
            return True
    return False


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_133(caption, evidence))
    violations: list[dict[str, Any]] = []

    for violation in result.get("violations") or []:
        if violation.get("type") == "unsupported_whole_body_posture" and violation.get("posture") == "lying":
            if not _SUBJECT_LYING_RE.search(caption):
                continue
        if violation.get("type") == "contradicts_signed_torso_depth":
            if not _positive_frontal_torso_claim(caption):
                continue
        violations.append(violation)

    authorized_near = _authorized_nearer_shoulder_sides(evidence)
    for match in _SIGNED_NEAR_RE.finditer(caption):
        side = (match.group(1) or match.group(2) or "").lower()
        if side in {"left", "right"} and side not in authorized_near:
            violations.append({
                "type": "unqualified_signed_shoulder_depth_relation",
                "reported_nearer_anatomical_side": side,
                "authorized_nearer_anatomical_sides": sorted(authorized_near),
                "text": match.group(0),
            })

    if _NON_SQUARE_TORSO_RE.search(caption) and not _torso_non_square_authorized(evidence):
        violations.append({
            "type": "unqualified_torso_depth_relation",
            "text": _NON_SQUARE_TORSO_RE.search(caption).group(0),
        })

    warnings = list(result.get("warnings") or [])
    for claim in evidence.get("required_claims") or []:
        if not isinstance(claim, dict):
            continue
        if str(claim.get("id") or "").startswith("support_relation_") and not _support_claim_present(caption, claim):
            warnings.append({
                "type": "required_claim_not_detected",
                "claim_id": claim.get("id"),
                "description": claim.get("description"),
            })

    unique_violations: list[dict[str, Any]] = []
    seen_v: set[tuple[Any, ...]] = set()
    for item in violations:
        key = (item.get("type"), item.get("text"), item.get("reported_nearer_anatomical_side"))
        if key in seen_v:
            continue
        seen_v.add(key)
        unique_violations.append(item)

    unique_warnings: list[dict[str, Any]] = []
    seen_w: set[tuple[Any, ...]] = set()
    for item in warnings:
        key = (item.get("type"), item.get("claim_id"), item.get("description"))
        if key in seen_w:
            continue
        seen_w.add(key)
        unique_warnings.append(item)

    result["schema_version"] = "caption-authority-lint-1.3.4"
    result["violations"] = unique_violations
    result["warnings"] = unique_warnings
    result["violation_count"] = len(unique_violations)
    result["warning_count"] = len(unique_warnings)
    result["passed"] = not unique_violations
    return result
