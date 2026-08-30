from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_154 import build_caption_projection as _build_154
from .caption_projection_154 import lint_caption as _lint_154


_FACE_YAW_MAP = {
    "toward_camera": "near_frontal",
    "three_quarter": "three_quarter",
    "profile": "profile",
    "away_from_camera": "rearward",
}
_FACE_YAW_PATTERNS = {
    "three_quarter": re.compile(r"\b(?:head|face)\b[^.!?]{0,80}\bthree[- ]quarter\b|\bthree[- ]quarter\b[^.!?]{0,80}\b(?:head|face)\b", re.I),
    "profile": re.compile(r"\b(?:head|face)\b[^.!?]{0,80}\bprofile\b|\bprofile\b[^.!?]{0,80}\b(?:head|face)\b", re.I),
    "rearward": re.compile(r"\b(?:head|face)\b[^.!?]{0,80}\b(?:away|rearward|back)\b|\bturned\s+away\b", re.I),
}
_UNSUPPORTED_HEAD_TOWARD_RE = re.compile(
    r"\b(?:head|face)\b[^.!?]{0,90}\bturn(?:ed|ing)?\b[^.!?]{0,70}\b(?:toward|towards)\b[^.!?]{0,30}\b(?:camera|lens)\b|"
    r"\bturn(?:ed|ing)?\b[^.!?]{0,70}\b(?:head|face)\b[^.!?]{0,70}\b(?:toward|towards)\b[^.!?]{0,30}\b(?:camera|lens)\b",
    re.I,
)
_UPRIGHT_POSTURE_RE = re.compile(r"\bupright\s+(?:body\s+)?posture\b", re.I)


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def _remove_claim(evidence: dict[str, Any], claim_id: str) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for item in evidence.get("required_claims") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("id") or "") == claim_id:
            removed.append(copy.deepcopy(item))
        else:
            kept.append(copy.deepcopy(item))
    evidence["required_claims"] = kept
    return removed


def _append_claim(evidence: dict[str, Any], claim: dict[str, Any]) -> None:
    claim_id = str(claim.get("id") or "")
    claims = [
        copy.deepcopy(item)
        for item in evidence.get("required_claims") or []
        if isinstance(item, dict) and str(item.get("id") or "") != claim_id
    ]
    claims.append(copy.deepcopy(claim))
    evidence["required_claims"] = claims


def _neutral_orientation_field(field: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    direction = str(value.get("direction") or "").lower()
    magnitude = str(value.get("magnitude") or "").lower()
    relation = str(value.get("relation") or "").lower()
    if direction == "neutral" and magnitude in {"", "none"}:
        return True
    if relation == "upright_in_image_plane" and magnitude in {"", "none"}:
        return True
    # Projection 1.4.x normalizes image_plane_body_axis down to its semantic
    # payload and may drop the source `relation` string. A zero-magnitude body
    # axis is still the same low-information "upright/no visible tilt" fact, so
    # quarantine it by field identity rather than requiring the lost relation.
    if field == "image_plane_body_axis" and magnitude in {"", "none"}:
        return True
    return False


def _face_yaw_description(band: str) -> str:
    return {
        "three_quarter": "head is turned to a three-quarter angle in yaw relative to the camera",
        "profile": "head is turned near profile in yaw relative to the camera",
        "rearward": "head is turned substantially away from the camera in yaw",
    }.get(band, "supported horizontal head/face yaw relative to the camera")


def _apply_yaw_only_semantic_economy(evidence: dict[str, Any], audit: dict[str, Any]) -> None:
    projection = _projection_root(audit)
    pose = evidence.get("pose_orientation")
    if not isinstance(pose, dict):
        return

    orientation = pose.get("subject_geometry_orientation")
    if not isinstance(orientation, dict):
        return

    integration = projection.setdefault("subject_geometry_semantics_integration", {})
    v155 = projection.setdefault("subject_geometry_semantics_155", {})

    # SAM3D's face proxy is calibrated for horizontal yaw. The old label
    # `toward_camera` sounded like a full 3-D head direction and tempted Compose
    # to erase independently supported head pitch (e.g. a head pitched down at a
    # phone). Rename it at the caption boundary and suppress the near-frontal
    # band as low-information prose.
    source_face = orientation.pop("face_orientation", None)
    face_yaw_band = None
    if isinstance(source_face, dict):
        source_label = str(source_face.get("orientation") or "").strip().lower()
        face_yaw_band = _FACE_YAW_MAP.get(source_label)
        v155["source_face_orientation_audit_only"] = copy.deepcopy(source_face)
        v155["source_face_orientation_interpretation"] = "horizontal_yaw_only"

    head = orientation.get("head_body_relation")
    if isinstance(head, dict):
        source_head_face = str(head.pop("face_orientation", "") or "").strip().lower()
        mapped = _FACE_YAW_MAP.get(source_head_face) or face_yaw_band
        if mapped:
            head["face_yaw_band"] = mapped
        head["relation_scope"] = "compensating_horizontal_yaw_relative_to_body"

    if face_yaw_band and face_yaw_band != "near_frontal":
        orientation["face_yaw_orientation"] = {
            "yaw_band": face_yaw_band,
            "axis": "horizontal_yaw_only",
        }
    elif face_yaw_band == "near_frontal":
        v155["near_frontal_face_yaw_caption_suppressed"] = True

    # A frontal body yaw is useful internally but normally contributes no useful
    # training-caption prose. Keep it in audit while retaining non-frontal bands.
    body = orientation.get("body_orientation")
    if isinstance(body, dict) and str(body.get("orientation") or "").lower() == "frontal":
        v155["frontal_body_orientation_audit_only"] = copy.deepcopy(body)
        orientation.pop("body_orientation", None)

    # Replace the inherited ambiguous face-orientation required claim with an
    # explicitly yaw-only claim. Near-frontal face yaw intentionally has no claim.
    removed_face_claims = _remove_claim(evidence, "subject_geometry_face_orientation")
    if removed_face_claims:
        v155["legacy_face_orientation_claims_removed"] = removed_face_claims
    if face_yaw_band and face_yaw_band != "near_frontal":
        _append_claim(
            evidence,
            {
                "id": "subject_geometry_face_yaw_orientation",
                "priority": "required",
                "description": _face_yaw_description(face_yaw_band),
                "authority": "subject-geometry-semantics-0.2-via-projection-1.5.5",
                "instruction": (
                    "Express only the supported horizontal head/face yaw naturally once. "
                    "Do not infer eye gaze or head pitch from this yaw band."
                ),
            },
        )

    # Compound head/body relations remain useful: they are based on the large
    # difference between body yaw and face yaw. Clarify that this is a relative
    # horizontal turn, not evidence that the eyes are looking at the lens.
    for claim in evidence.get("required_claims") or []:
        if not isinstance(claim, dict) or claim.get("id") != "subject_geometry_compound_orientation":
            continue
        claim["authority"] = "subject-geometry-semantics-0.2-via-projection-1.5.5"
        claim["instruction"] = (
            "Express the body orientation/frame direction plus the compensating horizontal head turn as one compact pose relation. "
            "The head-turn relation is yaw-relative-to-body evidence, not eye-gaze evidence; do not say the subject is looking at the camera unless independent gaze evidence supports it."
        )

    # Neutral Analyze axes are low-information and can create false posture prose
    # such as 'upright posture'. Preserve non-neutral pitch/roll (e.g. head down),
    # but quarantine neutral/none and zero-tilt image-plane entries to audit.
    semantic_orientation = pose.get("semantic_orientation")
    removed_neutral: dict[str, Any] = {}
    if isinstance(semantic_orientation, dict):
        for field in list(semantic_orientation):
            value = semantic_orientation.get(field)
            if _neutral_orientation_field(field, value):
                removed_neutral[field] = copy.deepcopy(value)
                semantic_orientation.pop(field, None)
    if removed_neutral:
        v155["neutral_semantic_orientation_audit_only"] = removed_neutral

    orientation["schema_version"] = "caption-subject-geometry-orientation-1.1"
    orientation["semantic_economy"] = (
        "non-frontal body yaw and salient face yaw are caption-facing; near-frontal face/body yaw, source degrees, and neutral axes are audit-only"
    )
    integration["caption_facing"] = copy.deepcopy(orientation)
    integration["projection_revision"] = "1.5.5"
    integration["face_geometry_scope"] = "horizontal_yaw_only"
    integration["neutral_orientation_fields_audit_only"] = copy.deepcopy(removed_neutral)
    integration["policy"] = (
        "Subject Geometry face orientation is interpreted strictly as horizontal yaw. Near-frontal face yaw and frontal body yaw are low-information audit facts; "
        "non-frontal yaw and qualified compensating head/body yaw remain caption-facing. Neutral Analyze axes are quarantined while non-neutral pitch/roll survives."
    )
    projection.setdefault("blocked", []).append(
        {
            "path": "caption-evidence.pose_orientation.neutral_orientation_and_near_frontal_yaw",
            "reason": "low_information_or_axis_ambiguous_semantics_are_audit_only_in_projection_1.5.5",
            "neutral_fields": sorted(removed_neutral),
            "near_frontal_face_yaw_suppressed": face_yaw_band == "near_frontal",
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
    evidence, audit = _build_154(
        fused_payload,
        analysis,
        pose_semantics=pose_semantics,
        subject_geometry_semantics=subject_geometry_semantics,
        caption_policy=caption_policy,
    )
    evidence["projection_revision"] = "1.5.5"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.5.5"

    if subject_geometry_semantics is not None:
        _apply_yaw_only_semantic_economy(evidence, audit)

    projection.setdefault("notes", []).append(
        "Projection 1.5.5 treats reconstructed face orientation as yaw-only, suppresses near-frontal body/face yaw prose, and quarantines neutral Analyze axes while retaining non-neutral pitch/roll."
    )
    return evidence, audit


def _dedupe_findings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            str(item.get("type") or ""),
            str(item.get("claim_id") or item.get("posture") or ""),
            str(item.get("description") or item.get("text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_154(caption, evidence))
    violations = list(result.get("violations") or [])
    warnings = list(result.get("warnings") or [])

    pose = evidence.get("pose_orientation") or {}
    subject_orientation = pose.get("subject_geometry_orientation") or {}
    head_relation = subject_orientation.get("head_body_relation") or {}
    semantic_pose = pose.get("semantic_pose") or {}
    posture = str(semantic_pose.get("posture") or "").strip().lower()

    if not head_relation and _UNSUPPORTED_HEAD_TOWARD_RE.search(caption):
        violations.append(
            {
                "type": "unsupported_head_turn_toward_camera",
                "description": "caption states a head/face turn toward the camera without a qualified compound head/body yaw relation",
            }
        )

    if not posture and _UPRIGHT_POSTURE_RE.search(caption):
        violations.append(
            {
                "type": "unsupported_whole_body_posture",
                "posture": "upright",
                "description": "upright posture is not licensed when Pose Semantics withholds whole-body posture",
            }
        )

    face_claim = next(
        (
            item
            for item in evidence.get("required_claims") or []
            if isinstance(item, dict) and item.get("id") == "subject_geometry_face_yaw_orientation"
        ),
        None,
    )
    face_yaw = subject_orientation.get("face_yaw_orientation") or {}
    if isinstance(face_claim, dict):
        band = str(face_yaw.get("yaw_band") or "")
        pattern = _FACE_YAW_PATTERNS.get(band)
        if pattern is not None and not pattern.search(caption):
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": "subject_geometry_face_yaw_orientation",
                    "description": face_claim.get("description"),
                }
            )

    result["schema_version"] = "caption-authority-lint-1.5.5"
    result["violations"] = _dedupe_findings(violations)
    result["violation_count"] = len(result["violations"])
    result["warnings"] = _dedupe_findings(warnings)
    result["warning_count"] = len(result["warnings"])
    result["passed"] = not result["violations"]
    return result