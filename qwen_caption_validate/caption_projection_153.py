from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_152 import build_caption_projection as _build_152
from .caption_projection_152 import lint_caption as _lint_152


_BODY_PATTERNS = {
    "slightly_angled": re.compile(r"\b(?:slightly|gently)\s+(?:angled|turned)|\bslight\s+angle\b", re.I),
    "three_quarter": re.compile(r"\bthree[- ]quarter\b|\b(?:body|torso)\b[^.!?]{0,60}\b(?:angled|turned)\b", re.I),
    "side_on": re.compile(r"\bside[- ]?on\b|\bsideways\b", re.I),
    "rear_three_quarter": re.compile(r"\brear\s+three[- ]quarter\b|\bthree[- ]quarter\b[^.!?]{0,40}\b(?:rear|back)\b", re.I),
    "rear": re.compile(r"\b(?:back|rear)\s+(?:to|toward)\s+(?:the\s+)?camera\b|\bfacing\s+away\b", re.I),
}
_HEAD_TOWARD_CAMERA_RE = re.compile(
    r"\b(?:head|face)\b[^.!?]{0,90}?\b(?:turn(?:ed|ing)?|look(?:s|ed|ing)?|face(?:s|d|ing)?)\b[^.!?]{0,60}?\b(?:camera|lens)\b|"
    r"\b(?:turn(?:ed|ing)?|look(?:s|ed|ing)?|face(?:s|d|ing)?)\b[^.!?]{0,60}?\b(?:camera|lens)\b[^.!?]{0,90}?\b(?:head|face)\b",
    re.I,
)
_FACE_PATTERNS = {
    "three_quarter": re.compile(r"\b(?:face|head)\b[^.!?]{0,60}\bthree[- ]quarter\b|\bthree[- ]quarter\b[^.!?]{0,60}\b(?:face|head)\b", re.I),
    "profile": re.compile(r"\bprofile\b", re.I),
    "away_from_camera": re.compile(r"\b(?:face|head)\b[^.!?]{0,60}\b(?:away|back)\b|\blooking\s+away\b", re.I),
}


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def _semantics_root(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get("subject_geometry_semantics")
    return value if isinstance(value, dict) else payload


def _fact_value(root: dict[str, Any], field: str) -> dict[str, Any] | None:
    semantic = root.get(field)
    if not isinstance(semantic, dict) or semantic.get("status") != "FACT":
        return None
    value = semantic.get("value")
    return copy.deepcopy(value) if isinstance(value, dict) else None


def _frame_suffix(side: Any) -> str:
    value = str(side or "").strip().lower()
    return f", facing frame {value}" if value in {"left", "right"} else ""


def _body_description(body: dict[str, Any]) -> str:
    orientation = str(body.get("orientation") or "").strip().lower()
    suffix = _frame_suffix(body.get("faces_frame"))
    descriptions = {
        "frontal": "body is approximately frontal to the camera",
        "slightly_angled": "body is slightly angled to the camera" + suffix,
        "three_quarter": "body is at a three-quarter angle to the camera" + suffix,
        "side_on": "body is nearly side-on to the camera" + suffix,
        "rear_three_quarter": "body is at a rear three-quarter angle to the camera" + suffix,
        "rear": "body faces substantially away from the camera" + suffix,
    }
    return descriptions.get(orientation, "supported body orientation relative to the camera")


def _face_description(face: dict[str, Any]) -> str:
    orientation = str(face.get("orientation") or "").strip().lower()
    descriptions = {
        "toward_camera": "face is oriented toward the camera",
        "three_quarter": "face is at a three-quarter angle to the camera",
        "profile": "face is near profile to the camera",
        "away_from_camera": "face is oriented away from the camera",
    }
    return descriptions.get(orientation, "supported face orientation relative to the camera")


def _remove_claims(evidence: dict[str, Any], ids: set[str]) -> list[str]:
    removed: list[str] = []
    kept: list[dict[str, Any]] = []
    for item in evidence.get("required_claims") or []:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("id") or "")
        if claim_id in ids:
            removed.append(claim_id)
            continue
        kept.append(copy.deepcopy(item))
    evidence["required_claims"] = kept
    return removed


def _append_claim(evidence: dict[str, Any], claim: dict[str, Any]) -> None:
    claims = [copy.deepcopy(item) for item in evidence.get("required_claims") or [] if isinstance(item, dict)]
    claim_id = str(claim.get("id") or "")
    claims = [item for item in claims if str(item.get("id") or "") != claim_id]
    claims.append(claim)
    evidence["required_claims"] = claims


def _install_subject_geometry_orientation(
    evidence: dict[str, Any],
    audit: dict[str, Any],
    subject_geometry_semantics: dict[str, Any],
) -> None:
    root = _semantics_root(subject_geometry_semantics)
    projection = _projection_root(audit)
    pose = evidence.setdefault("pose_orientation", {})

    body = _fact_value(root, "body_orientation")
    face = _fact_value(root, "face_orientation")
    head = _fact_value(root, "head_body_relation")

    # Camera-subject geometry is a legitimate subject-frame FACT in v0.2, but it
    # deliberately stays audit-only here.  Projection cannot reinterpret it as
    # world high/low or selfie/external capture mode.
    camera_semantic = copy.deepcopy(root.get("camera_subject_relation"))

    removed_fields: dict[str, Any] = {}
    semantic_orientation = pose.get("semantic_orientation")
    if not isinstance(semantic_orientation, dict):
        semantic_orientation = {}
        pose["semantic_orientation"] = semantic_orientation

    if body is not None:
        if "torso_yaw" in semantic_orientation:
            removed_fields["semantic_orientation.torso_yaw"] = semantic_orientation.pop("torso_yaw")
        if "upper_torso_depth_relation" in pose:
            removed_fields["upper_torso_depth_relation"] = pose.pop("upper_torso_depth_relation")
    if face is not None:
        if "head_yaw" in semantic_orientation:
            removed_fields["semantic_orientation.head_yaw"] = semantic_orientation.pop("head_yaw")
    if head is not None and "head_torso_relation" in pose:
        removed_fields["head_torso_relation"] = pose.pop("head_torso_relation")

    removed_claims = _remove_claims(
        evidence,
        {
            "upper_torso_side_on_relation" if body is not None else "",
            "head_turn_toward_camera_relative_torso" if head is not None else "",
        },
    )

    caption_orientation: dict[str, Any] = {
        "schema_version": "caption-subject-geometry-orientation-1.0",
        "authority": "subject-geometry-semantics-0.2",
        "semantic_economy": "categorical subject/camera orientation replaces component depth/yaw prose; source angles remain audit-only",
    }
    if body is not None:
        caption_orientation["body_orientation"] = {
            "orientation": body.get("orientation"),
            "faces_frame": body.get("faces_frame"),
        }
    if face is not None:
        caption_orientation["face_orientation"] = {
            "orientation": face.get("orientation"),
        }
    if head is not None:
        caption_orientation["head_body_relation"] = {
            "relation": head.get("relation"),
            "body_orientation": head.get("body_orientation"),
            "body_faces_frame": head.get("body_faces_frame"),
            "face_orientation": head.get("face_orientation"),
        }
    pose["subject_geometry_orientation"] = caption_orientation

    # A qualified compound relation subsumes the separate body/face wording.  It
    # is the human-level semantic unit we ultimately want Compose to verbalize.
    if head is not None and head.get("relation") == "turned_toward_camera" and body is not None:
        description = _body_description(body) + ", with the head turned toward the camera"
        _append_claim(
            evidence,
            {
                "id": "subject_geometry_compound_orientation",
                "priority": "required",
                "description": description,
                "authority": "subject-geometry-semantics-0.2",
                "instruction": (
                    "Express this as one compact pose relation. State the body orientation/frame direction and the head turn toward the camera together; "
                    "do not serialize shoulder-depth measurements, root angles, or separate redundant torso/head clauses."
                ),
            },
        )
    else:
        if body is not None and str(body.get("orientation") or "") != "frontal":
            _append_claim(
                evidence,
                {
                    "id": "subject_geometry_body_orientation",
                    "priority": "required",
                    "description": _body_description(body),
                    "authority": "subject-geometry-semantics-0.2",
                    "instruction": "Describe the supported body-to-camera orientation once in natural language; do not quote degrees or shoulder-depth measurements.",
                },
            )
        if face is not None and str(face.get("orientation") or "") in {"three_quarter", "profile", "away_from_camera"}:
            _append_claim(
                evidence,
                {
                    "id": "subject_geometry_face_orientation",
                    "priority": "required",
                    "description": _face_description(face),
                    "authority": "subject-geometry-semantics-0.2",
                    "instruction": "Describe the supported face orientation naturally once; do not quote reconstructed angles.",
                },
            )

    integration = projection.setdefault("subject_geometry_semantics_integration", {})
    integration.update(
        {
            "schema_version": root.get("schema_version"),
            "fact_source": {
                "body_orientation": copy.deepcopy((root.get("body_orientation") or {}).get("value")) if body is not None else None,
                "face_orientation": copy.deepcopy((root.get("face_orientation") or {}).get("value")) if face is not None else None,
                "head_body_relation": copy.deepcopy((root.get("head_body_relation") or {}).get("value")) if head is not None else None,
            },
            "caption_facing": copy.deepcopy(caption_orientation),
            "legacy_fields_removed": removed_fields,
            "legacy_required_claims_removed": removed_claims,
            "camera_subject_relation_audit_only": camera_semantic,
            "cross_source_conflicts_audit_only": copy.deepcopy(root.get("cross_source_conflicts") or []),
            "candidate_or_withheld_audit_only": {
                field: copy.deepcopy(root.get(field))
                for field in ("body_orientation", "face_orientation", "head_body_relation")
                if isinstance(root.get(field), dict) and root.get(field, {}).get("status") != "FACT"
            },
            "policy": (
                "FACT body/face/head subject geometry is caption-facing as categorical semantics; candidate/withheld states, source angles, cross-source conflicts, "
                "and subject-relative camera geometry remain audit-only. A FACT compound head/body relation semantically subsumes legacy shoulder-depth/head-torso prose."
            ),
        }
    )
    projection.setdefault("allowed", []).append(
        {
            "path": "subject-geometry-semantics-0.2.FACT_orientation",
            "reason": "dwpose-observation-gated calibrated subject geometry is the primary caption-facing body/face orientation authority",
        }
    )
    projection.setdefault("blocked", []).append(
        {
            "path": "subject-geometry-semantics-0.2.camera_subject_relation",
            "reason": "subject-relative camera geometry requires separate capture/posture-aware interpretation before caption exposure",
        }
    )


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    pose_semantics: dict[str, Any] | None = None,
    subject_geometry_semantics: dict[str, Any] | None = None,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_152(
        fused_payload,
        analysis,
        pose_semantics=pose_semantics,
        caption_policy=caption_policy,
    )
    evidence["projection_revision"] = "1.5.3"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.5.3"

    if subject_geometry_semantics is not None:
        _install_subject_geometry_orientation(evidence, audit, subject_geometry_semantics)
    else:
        projection.setdefault("notes", []).append(
            "Projection 1.5.3 received no Subject Geometry Semantics record and therefore preserves Projection 1.5.2 orientation behavior."
        )

    projection.setdefault("notes", []).append(
        "Projection 1.5.3 integrates Subject Geometry Semantics v0.2 FACT body/face/head relations and keeps subject-relative camera geometry audit-only."
    )
    return evidence, audit


def _claim(evidence: dict[str, Any], claim_id: str) -> dict[str, Any] | None:
    for item in evidence.get("required_claims") or []:
        if isinstance(item, dict) and item.get("id") == claim_id:
            return item
    return None


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_152(caption, evidence))
    violations = list(result.get("violations") or [])
    warnings = list(result.get("warnings") or [])

    body_claim = _claim(evidence, "subject_geometry_body_orientation")
    compound_claim = _claim(evidence, "subject_geometry_compound_orientation")
    face_claim = _claim(evidence, "subject_geometry_face_orientation")

    # The inherited linter only knows the older shoulder-depth authority.  New
    # governed subject geometry is an equal-or-stronger authority for this prose.
    if body_claim or compound_claim:
        violations = [item for item in violations if item.get("type") != "unqualified_torso_depth_relation"]

    orientation = ((evidence.get("pose_orientation") or {}).get("subject_geometry_orientation") or {})
    body = orientation.get("body_orientation") or {}
    face = orientation.get("face_orientation") or {}

    if body_claim:
        band = str(body.get("orientation") or "")
        pattern = _BODY_PATTERNS.get(band)
        if pattern is not None and not pattern.search(caption):
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": "subject_geometry_body_orientation",
                    "description": body_claim.get("description"),
                }
            )

    if compound_claim:
        band = str(body.get("orientation") or "")
        pattern = _BODY_PATTERNS.get(band)
        if (pattern is not None and not pattern.search(caption)) or not _HEAD_TOWARD_CAMERA_RE.search(caption):
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": "subject_geometry_compound_orientation",
                    "description": compound_claim.get("description"),
                }
            )

    if face_claim:
        band = str(face.get("orientation") or "")
        pattern = _FACE_PATTERNS.get(band)
        if pattern is not None and not pattern.search(caption):
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": "subject_geometry_face_orientation",
                    "description": face_claim.get("description"),
                }
            )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in warnings:
        key = (
            str(item.get("type") or ""),
            str(item.get("claim_id") or ""),
            str(item.get("description") or item.get("text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    result["schema_version"] = "caption-authority-lint-1.5.3"
    result["violations"] = violations
    result["violation_count"] = len(violations)
    result["warnings"] = deduped
    result["warning_count"] = len(deduped)
    result["passed"] = not violations
    return result
