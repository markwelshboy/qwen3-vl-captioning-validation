from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_143 import build_caption_projection as _build_143
from .caption_projection_143 import lint_caption as _lint_143


_POSTURES = {"standing", "seated", "squatting", "reclining"}
_POSTURE_PATTERNS = {
    "standing": re.compile(r"\b(?:stand|stands|standing|stood)\b", re.I),
    "seated": re.compile(r"\b(?:seat(?:ed)?|sit|sits|sitting|sat)\b", re.I),
    "squatting": re.compile(r"\b(?:squat|squats|squatting|crouch|crouches|crouching|crouched)\b", re.I),
    "reclining": re.compile(r"\b(?:reclin(?:e|es|ed|ing)|lying|lies|lay|laid)\b", re.I),
}


def _pose_semantics_root(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get("pose_semantics") if isinstance(payload.get("pose_semantics"), dict) else payload
    return value if isinstance(value, dict) else {}


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def _preferred_posture(semantics: dict[str, Any]) -> str | None:
    preferred = semantics.get("preferred_pose") or {}
    posture = str(preferred.get("posture") or "").lower().strip()
    return posture if posture in _POSTURES else None


def _preferred_gestures(semantics: dict[str, Any]) -> list[str]:
    preferred = semantics.get("preferred_pose") or {}
    out: list[str] = []
    seen: set[str] = set()
    for value in preferred.get("gestures") or []:
        label = re.sub(r"\s+", " ", str(value or "")).strip(" .;,:")
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def _candidate_audit(semantics: dict[str, Any]) -> dict[str, Any] | None:
    candidate = semantics.get("posture_candidate")
    if not isinstance(candidate, dict):
        return None
    return {
        "label": candidate.get("label"),
        "status": candidate.get("status"),
        "model_confidence": candidate.get("model_confidence"),
        "confidence_band": candidate.get("confidence_band"),
        "review_recommended": candidate.get("review_recommended"),
        "authority": candidate.get("authority"),
    }


def _normalize_claim_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip(" .;,:")


def _install_semantic_pose(
    evidence: dict[str, Any],
    audit: dict[str, Any],
    pose_semantics: dict[str, Any],
) -> None:
    semantics = _pose_semantics_root(pose_semantics)
    projection = _projection_root(audit)
    pose = evidence.setdefault("pose_orientation", {})
    posture = _preferred_posture(semantics)
    gestures = _preferred_gestures(semantics)

    # Pose Semantics becomes the single posture authority when supplied.  This is
    # intentionally fail-closed: a candidate/withheld posture must not leak back
    # into Compose through Projection 1.4.x fallback posture logic.
    old_posture = copy.deepcopy(pose.get("whole_body_posture") or {})
    pose["whole_body_posture"] = {
        "allowed": [posture] if posture else [],
        "authority": "pose_semantics_v0.10_fact" if posture else "pose_semantics_v0.10_withheld",
        "evidence": [
            "Pose Semantics v0.10 preferred_pose.posture is caption-preferred FACT"
        ] if posture else [
            "Pose Semantics v0.10 did not qualify a caption-preferred posture FACT"
        ],
    }
    pose["semantic_pose"] = {
        "schema_version": "caption-semantic-pose-1.0",
        "posture": posture,
        "gestures": gestures,
        "authority": "pose-semantics-0.10",
        "semantic_economy": (
            "semantic posture/gesture primitives subsume component geometry used only to establish them"
        ),
    }

    claims = [
        copy.deepcopy(item)
        for item in evidence.get("required_claims") or []
        if isinstance(item, dict)
        and not str(item.get("id") or "").startswith("whole_body_posture_")
        and str(item.get("id") or "") != "semantic_pose_posture"
        and not str(item.get("id") or "").startswith("semantic_pose_gesture_")
    ]

    if posture:
        claims.append(
            {
                "id": "semantic_pose_posture",
                "priority": "required",
                "description": f"subject is {posture}",
                "posture": posture,
                "authority": "pose-semantics-0.10",
                "instruction": (
                    "State this qualified whole-body posture once in natural language. "
                    "Do not replace it with a checklist of joint angles or support fields that merely established it."
                ),
            }
        )

    existing_descriptions = {_normalize_claim_text(item.get("description")) for item in claims}
    for index, label in enumerate(gestures, start=1):
        normalized = _normalize_claim_text(label)
        if not normalized or normalized in existing_descriptions:
            continue
        claims.append(
            {
                "id": f"semantic_pose_gesture_{index}",
                "priority": "required",
                "description": label,
                "authority": "pose-semantics-0.10",
                "instruction": (
                    "Express this recognizable gesture/support primitive naturally once. "
                    "Do not also serialize lower-level arm/hand geometry that only establishes the same relation."
                ),
            }
        )
        existing_descriptions.add(normalized)

    evidence["required_claims"] = claims

    candidate = _candidate_audit(semantics)
    vetoed = semantics.get("vetoed_posture_candidate")
    projection["pose_semantics_integration"] = {
        "schema_version": semantics.get("schema_version"),
        "qualified_posture_fact": posture,
        "caption_preferred_gestures": gestures,
        "posture_candidate_audit_only": candidate,
        "vetoed_posture_candidate_audit_only": copy.deepcopy(vetoed) if isinstance(vetoed, dict) else None,
        "previous_projection_posture": old_posture,
        "candidate_exposed_to_caption_evidence": False,
        "policy": (
            "preferred_pose is caption-facing; posture_candidate and vetoed_posture_candidate are review/audit only. "
            "When a semantic primitive is qualified, component evidence used only to establish it is semantically subsumed."
        ),
    }
    projection.setdefault("allowed", []).append(
        {
            "path": "pose-semantics-0.10.preferred_pose",
            "reason": "qualified_pose_semantics_fact_and_caption_preferred_gestures_are_primary_caption_pose_authority",
        }
    )
    if candidate:
        projection.setdefault("blocked", []).append(
            {
                "path": "pose-semantics-0.10.posture_candidate",
                "reason": "candidate_is_review_only_and_must_not_reach_compose_as_fact",
                "candidate_label": candidate.get("label"),
            }
        )


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    pose_semantics: dict[str, Any] | None = None,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_143(fused_payload, analysis, caption_policy=caption_policy)
    evidence["projection_revision"] = "1.5.0"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.5.0"

    if pose_semantics is not None:
        _install_semantic_pose(evidence, audit, pose_semantics)
    else:
        projection.setdefault("notes", []).append(
            "Projection 1.5.0 received no Pose Semantics record and therefore preserves Projection 1.4.3 pose behavior."
        )

    projection.setdefault("notes", []).append(
        "Projection 1.5.0 integrates Pose Semantics v0.10 as the primary caption-facing posture/gesture layer. "
        "FACT reaches Compose, CANDIDATE/WITHHELD remains audit-only, and semantic primitives subsume redundant component prose."
    )
    return evidence, audit


def _semantic_gesture_present(caption: str, description: str) -> bool:
    tokens = [
        token
        for token in re.findall(r"[a-z]+", description.lower())
        if token not in {"the", "a", "an", "on", "at", "to", "of", "with", "and", "body", "subject"}
    ]
    if not tokens:
        return True
    # Require the most informative noun plus at least one other semantic token.
    hits = sum(bool(re.search(rf"\b{re.escape(token)}\w*\b", caption, re.I)) for token in tokens)
    return hits >= min(2, len(tokens))


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_143(caption, evidence))
    warnings = list(result.get("warnings") or [])
    semantic = ((evidence.get("pose_orientation") or {}).get("semantic_pose") or {})
    posture = str(semantic.get("posture") or "").lower()

    if posture in _POSTURE_PATTERNS and not _POSTURE_PATTERNS[posture].search(caption):
        warnings.append(
            {
                "type": "required_claim_not_detected",
                "claim_id": "semantic_pose_posture",
                "description": f"subject is {posture}",
            }
        )

    for claim in evidence.get("required_claims") or []:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or "")
        if not claim_id.startswith("semantic_pose_gesture_"):
            continue
        description = str(claim.get("description") or "")
        if description and not _semantic_gesture_present(caption, description):
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": claim_id,
                    "description": description,
                }
            )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in warnings:
        key = (
            str(item.get("type") or ""),
            str(item.get("claim_id") or ""),
            str(item.get("text") or item.get("description") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    result["schema_version"] = "caption-authority-lint-1.5.0"
    result["warnings"] = deduped
    result["warning_count"] = len(deduped)
    result["passed"] = not result.get("violations")
    return result
