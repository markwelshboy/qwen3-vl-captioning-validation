from __future__ import annotations

import copy
from typing import Any

from .caption_projection_151 import build_caption_projection as _build_151
from .caption_projection_151 import lint_caption as _lint_151


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    pose_semantics: dict[str, Any] | None = None,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_151(
        fused_payload,
        analysis,
        pose_semantics=pose_semantics,
        caption_policy=caption_policy,
    )
    evidence["projection_revision"] = "1.5.2"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.5.2"
    projection.setdefault("notes", []).append(
        "Projection 1.5.2 preserves Projection 1.5.1 caption evidence and normalizes the legacy "
        "reclined/reclining posture vocabulary at lint time."
    )
    return evidence, audit


def _semantic_posture(evidence: dict[str, Any]) -> str | None:
    pose = evidence.get("pose_orientation")
    if not isinstance(pose, dict):
        return None
    semantic = pose.get("semantic_pose")
    if not isinstance(semantic, dict):
        return None
    posture = str(semantic.get("posture") or "").strip().lower()
    return posture or None


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Lint with canonical Pose Semantics posture vocabulary.

    caption_lint historically names the reclining posture family ``reclined`` while
    Pose Semantics v0.10 deliberately exposes the canonical FACT ``reclining``.
    The underlying regex correctly recognizes reclines/reclined/reclining, but the
    legacy allowed-posture comparison therefore reports a false violation.  Treat
    that one vocabulary mismatch as equivalent; do not suppress any other posture
    or authority violation.
    """
    result = copy.deepcopy(_lint_151(caption, evidence))
    semantic_posture = _semantic_posture(evidence)

    violations: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for item in result.get("violations") or []:
        if (
            semantic_posture == "reclining"
            and item.get("type") == "unsupported_whole_body_posture"
            and item.get("posture") == "reclined"
        ):
            normalized.append(
                {
                    "type": "posture_vocabulary_alias_normalized",
                    "legacy_posture": "reclined",
                    "semantic_posture": "reclining",
                }
            )
            continue
        violations.append(item)

    result["schema_version"] = "caption-authority-lint-1.5.2"
    result["violations"] = violations
    result["violation_count"] = len(violations)
    result["passed"] = not violations
    if normalized:
        result["normalized_findings"] = list(result.get("normalized_findings") or []) + normalized
    return result
