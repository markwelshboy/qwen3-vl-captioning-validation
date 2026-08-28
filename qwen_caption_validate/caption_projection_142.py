from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_141 import build_caption_projection as _build_141
from .caption_projection_141 import lint_caption as _lint_141

_HEAD_FORWARD_RE = re.compile(r"\bhead\b[^.!?]{0,50}?\b(?:faces?|facing)\s+forward\b", re.IGNORECASE)
_HEAD_TOWARD_CAMERA_RE = re.compile(
    r"\b(?:head|face)\b[^.!?]{0,90}?\b(?:turn(?:ed|ing)?|face(?:s|d|ing)?|look(?:s|ed|ing)?)\b[^.!?]{0,50}?\b(?:camera|lens)\b|"
    r"\b(?:turn(?:ed|ing)?|face(?:s|d|ing)?|look(?:s|ed|ing)?)\b[^.!?]{0,50}?\b(?:camera|lens)\b[^.!?]{0,90}?\b(?:head|face)\b",
    re.IGNORECASE,
)
_UPPER_TORSO_DEPTH_RE = re.compile(
    r"\b(?:torso|upper\s+body|body)\b[^.!?]{0,100}?\b(?:side[- ]?on|sideways|turned|angled|rotated|not\s+square[- ]?on)\b|"
    r"\b(?:side[- ]?on|sideways|turned|angled|rotated|not\s+square[- ]?on)\b[^.!?]{0,100}?\b(?:torso|upper\s+body|body)\b",
    re.IGNORECASE,
)


def _fusion_root(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("fusion") if isinstance(payload.get("fusion"), dict) else payload
    return value if isinstance(value, dict) else {}


def _replace_head_geometry(pose: dict[str, Any], relation: str) -> None:
    for item in pose.get("visible_subject_parts") or []:
        if not isinstance(item, dict):
            continue
        if re.search(r"\bhead\b", str(item.get("part") or ""), re.I):
            item["geometry"] = relation


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_141(fused_payload, analysis, caption_policy=caption_policy)
    evidence["projection_revision"] = "1.4.2"
    projection = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    if isinstance(projection, dict):
        projection["schema_version"] = "caption-projection-audit-1.4.2"

    fusion = _fusion_root(fused_payload)
    pose = evidence.setdefault("pose_orientation", {})
    semantic = pose.setdefault("semantic_orientation", {})
    claims = [copy.deepcopy(item) for item in (evidence.get("required_claims") or []) if isinstance(item, dict)]

    upper = fusion.get("qualified_upper_torso_depth_relation")
    if isinstance(upper, dict) and upper.get("authority") == "qualified_visible_shoulder_depth_rotation":
        semantic.pop("torso_yaw", None)
        pose["upper_torso_depth_relation"] = {
            "magnitude": upper.get("magnitude"),
            "relation": upper.get("relation"),
            "authority": upper.get("authority"),
        }
        if not any(item.get("id") == "upper_torso_side_on_relation" for item in claims):
            claims.append(
                {
                    "id": "upper_torso_side_on_relation",
                    "priority": "required",
                    "description": "upper torso strongly turned in depth, near side-on rather than square-on to camera",
                    "instruction": "Describe the upper torso/body as strongly turned in depth or near side-on, not frontal/square-on.",
                }
            )
        if isinstance(projection, dict):
            projection.setdefault("allowed", []).append(
                {
                    "path": "fusion.qualified_upper_torso_depth_relation",
                    "reason": "qualified_visible_shoulder_depth_overrides_weak_frontal_torso_semantics",
                }
            )

    head_relation = fusion.get("qualified_head_torso_relation")
    if isinstance(head_relation, dict) and head_relation.get("camera_relation") == "toward_camera":
        semantic.pop("head_yaw", None)
        relation_text = str(head_relation.get("relation") or "head turned substantially toward the camera relative to the torso")
        pose["head_torso_relation"] = {
            "magnitude": head_relation.get("magnitude"),
            "relation": relation_text,
            "camera_relation": "toward_camera",
            "authority": head_relation.get("authority"),
        }
        _replace_head_geometry(pose, relation_text)
        if not any(item.get("id") == "head_turn_toward_camera_relative_torso" for item in claims):
            claims.append(
                {
                    "id": "head_turn_toward_camera_relative_torso",
                    "priority": "required",
                    "description": "head turned substantially toward camera relative to strongly depth-turned torso",
                    "instruction": "Explicitly describe the head/face as turned toward the camera relative to the torso. Do not say merely that the head faces forward.",
                }
            )
        if isinstance(projection, dict):
            projection.setdefault("blocked", []).append(
                {
                    "path": "caption-evidence-1.3.pose_orientation.semantic_orientation.head_yaw",
                    "reason": "absolute_frontal_head_semantics_are_misleading_without_relative_torso_context",
                }
            )
            projection.setdefault("allowed", []).append(
                {
                    "path": "fusion.qualified_head_torso_relation",
                    "reason": "camera_frontal_visible_head_plus_camera_gaze_relative_to_strong_depth_turned_torso",
                }
            )

    evidence["required_claims"] = claims
    if isinstance(projection, dict):
        projection.setdefault("notes", []).append(
            "Projection 1.4.2 resolves pose consistency before prose: strong visible shoulder depth can suppress weak frontal torso semantics, and a camera-facing head is expressed relative to that torso rather than as ambiguous 'facing forward'."
        )
    return evidence, audit


def _claim_present(evidence: dict[str, Any], claim_id: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("id") == claim_id
        for item in (evidence.get("required_claims") or [])
    )


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_141(caption, evidence))
    violations = list(result.get("violations") or [])
    warnings = list(result.get("warnings") or [])

    if _claim_present(evidence, "head_turn_toward_camera_relative_torso"):
        if _HEAD_FORWARD_RE.search(caption):
            violations.append(
                {
                    "type": "contradicts_head_torso_camera_turn",
                    "text": _HEAD_FORWARD_RE.search(caption).group(0),
                }
            )
        if not _HEAD_TOWARD_CAMERA_RE.search(caption):
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": "head_turn_toward_camera_relative_torso",
                    "description": "head turned toward camera relative to torso",
                }
            )

    if _claim_present(evidence, "upper_torso_side_on_relation") and not _UPPER_TORSO_DEPTH_RE.search(caption):
        warnings.append(
            {
                "type": "required_claim_not_detected",
                "claim_id": "upper_torso_side_on_relation",
                "description": "upper torso strongly turned in depth / near side-on",
            }
        )

    # Deduplicate findings when the inherited linter happened to emit the same semantic warning.
    def dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in items:
            key = (
                str(item.get("type") or ""),
                str(item.get("claim_id") or ""),
                str(item.get("text") or item.get("description") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    violations = dedupe(violations)
    warnings = dedupe(warnings)
    result["schema_version"] = "caption-authority-lint-1.4.2"
    result["violations"] = violations
    result["warnings"] = warnings
    result["violation_count"] = len(violations)
    result["warning_count"] = len(warnings)
    result["passed"] = not violations
    return result
