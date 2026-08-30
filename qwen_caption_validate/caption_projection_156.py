from __future__ import annotations

import copy
import re
from typing import Any

from . import caption_projection_155 as v155


# Only match explicit BODY/TORSO yaw prose.  Do not match photographic framing
# such as "three-quarter framing showing the upper body".
_UNSUPPORTED_BODY_YAW_RE = re.compile(
    r"\b(?:body|torso|upper\s+body)\b[^.!?]{0,55}\b(?:"
    r"(?:slightly|gently)\s+angled|"
    r"side[- ]?on|"
    r"three[- ]quarter(?:\s+(?:angle|orientation|view))?|"
    r"facing\s+frame\s+(?:left|right)|"
    r"angled\s+(?:toward|towards|to|away\s+from)\s+(?:the\s+)?camera|"
    r"turned\s+(?:toward|towards|away\s+from)\s+(?:the\s+)?camera"
    r")\b",
    re.I,
)


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    pose_semantics: dict[str, Any] | None = None,
    subject_geometry_semantics: dict[str, Any] | None = None,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Projection 1.5.6: enforce the negative space of governed body yaw.

    Projection 1.5.5 correctly suppresses frontal body yaw and near-frontal face
    yaw from caption-facing evidence.  Calibration showed that Compose could still
    invent a non-frontal body relation when *no* governed body orientation was
    present (e.g. "the body is slightly angled toward the camera").

    v1.5.6 keeps the 1.5.5 evidence surface unchanged and makes that absence
    authoritative at lint time: explicit body/torso camera-yaw prose is forbidden
    unless ``subject_geometry_orientation.body_orientation`` is actually present.
    """
    evidence, audit = v155.build_caption_projection(
        fused_payload,
        analysis,
        pose_semantics=pose_semantics,
        subject_geometry_semantics=subject_geometry_semantics,
        caption_policy=caption_policy,
    )
    evidence["projection_revision"] = "1.5.6"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.5.6"
    projection.setdefault("notes", []).append(
        "Projection 1.5.6 treats absence of caption-facing body orientation as a hard negative: Compose may not invent non-frontal body/torso yaw from framing, face direction, or scene context."
    )
    return evidence, audit


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(v155.lint_caption(caption, evidence))
    violations = list(result.get("violations") or [])

    pose = evidence.get("pose_orientation") or {}
    subject_orientation = pose.get("subject_geometry_orientation") or {}
    body_orientation = subject_orientation.get("body_orientation")

    if not isinstance(body_orientation, dict):
        match = _UNSUPPORTED_BODY_YAW_RE.search(caption)
        if match:
            violations.append(
                {
                    "type": "unsupported_body_camera_orientation",
                    "text": match.group(0),
                    "description": (
                        "caption states a non-frontal body/torso camera-yaw relation even though Projection 1.5.6 exposes no governed body orientation"
                    ),
                }
            )

    result["schema_version"] = "caption-authority-lint-1.5.6"
    result["violations"] = v155._dedupe_findings(violations)
    result["violation_count"] = len(result["violations"])
    result["passed"] = not result["violations"]
    return result
