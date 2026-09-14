from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v07 as phase4b6


SCHEMA_VERSION = "caption-fact-sheet-0.2.7"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.7"

# This is intentionally a narrow adjudication threshold, not a new general
# pose-classification ontology. DWPose knee angle uses 180 degrees for a
# straight leg; values below this cutoff are treated as deeply flexed.
DEEP_SUPPORT_KNEE_ANGLE_DEG = 120.0

_ADJUDICABLE_MODES = {"pose_allowed", "pose_guided"}
_SEATED_RE = re.compile(r"\b(?:seated|sitting)\b", re.I)
_STANDISH_RE = re.compile(r"\b(?:stand|standing|stands|upright)\b", re.I)
_CROUCH_COMPATIBLE_RE = re.compile(r"\b(?:crouch|crouched|crouching|squat|squatting)\b", re.I)
_OBJECT_SUPPORT_RE = re.compile(
    r"\b(?:against|supported\s+by|resting\s+(?:on|against)|leaning\s+against)\b",
    re.I,
)
_NONCOMMITTAL_POSES = {"", "unknown", "unspecified", "unresolved", "none", "null"}


def _body(sheet: dict[str, Any]) -> dict[str, Any]:
    facts = sheet.get("facts")
    if not isinstance(facts, dict):
        facts = {}
        sheet["facts"] = facts
    body = facts.get("body")
    if not isinstance(body, dict):
        body = {}
        facts["body"] = body
    return body


def _pose_text(body: dict[str, Any]) -> str | None:
    pose = body.get("pose_candidate")
    if not isinstance(pose, dict):
        return None
    text = pose.get("composer_text") or pose.get("normalized_text") or pose.get("text")
    return str(text).strip() if text is not None else None


def _configuration_text(body: dict[str, Any]) -> str:
    configuration = body.get("configuration")
    if not isinstance(configuration, list):
        return ""
    parts: list[str] = []
    for item in configuration:
        if not isinstance(item, dict):
            continue
        text = item.get("composer_text") or item.get("normalized_text") or item.get("text")
        if text:
            parts.append(str(text))
    return "; ".join(parts)


def _is_explicit_seated_context(body: dict[str, Any]) -> tuple[bool, str | None]:
    pose_text = _pose_text(body) or ""
    if not _SEATED_RE.search(pose_text):
        return False, None
    config_text = _configuration_text(body)
    if _OBJECT_SUPPORT_RE.search(config_text):
        return True, "explicit_seated_object_supported_context"
    return True, "explicit_seated_pose"


def _support_knee_angle(
    points: dict[str, tuple[float, float] | None],
    side: str,
) -> float | None:
    if side not in {"left", "right"}:
        return None
    helper = phase4b6.phase4b5.phase4b4._joint_angle
    return helper(
        points.get(f"{side}_hip"),
        points.get(f"{side}_knee"),
        points.get(f"{side}_ankle"),
    )


def _load_points(
    sheet: dict[str, Any],
) -> tuple[dict[str, tuple[float, float] | None] | None, str | None]:
    return phase4b6.phase4b5.phase4b4._load_dwpose_points_for_sheet(sheet)


def _support_record(
    *,
    status: str,
    original_shape: str | None,
    reason: str,
    would_change: bool = False,
    proposed_shape: str | None = None,
    support_side: str | None = None,
    knee_angle_deg: float | None = None,
    support_axis_angle_deg: float | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "shadow_only": True,
        "composer_authoritative": False,
        "would_change": bool(would_change),
        "original_shape": original_shape,
        "proposed_shape": proposed_shape,
        "evidence": {
            "support_side": support_side,
            "support_knee_angle_deg": round(knee_angle_deg, 1) if knee_angle_deg is not None else None,
            "deep_flexion_threshold_deg": DEEP_SUPPORT_KNEE_ANGLE_DEG,
            "support_axis_angle_from_vertical_deg": support_axis_angle_deg,
        },
        "reason": reason,
    }


def _pose_record(
    *,
    status: str,
    original_pose: str | None,
    reason: str,
    would_change: bool = False,
    proposed_pose: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "shadow_only": True,
        "composer_authoritative": False,
        "would_change": bool(would_change),
        "original_pose": original_pose,
        "proposed_pose": proposed_pose,
        "reason": reason,
    }


def _apply_shadow_adjudication(
    sheet: dict[str, Any],
    *,
    points: dict[str, tuple[float, float] | None] | None = None,
) -> dict[str, Any]:
    """Attach bounded support/pose adjudication without changing v07 facts.

    Authority is deliberately narrow:
      * only pose_allowed/pose_guided crops are adjudicable;
      * explicit seated interpretations are protected;
      * only an existing v07 mostly-upright support claim can be challenged;
      * only the already-bound support-side knee is measured;
      * broad pose may be proposed as crouched only for standing or
        noncommittal upstream pose candidates.

    The records are shadow-only. They do not modify support_geometry,
    pose_candidate or configuration and are not composer-authoritative.
    """
    out = copy.deepcopy(sheet)
    body = _body(out)

    policy = out.get("policy") if isinstance(out.get("policy"), dict) else {}
    mode = str(policy.get("mode") or "")
    pose_text = _pose_text(body)

    support = body.get("support_geometry")
    support = support if isinstance(support, dict) else {}
    original_shape = str(support.get("overall_shape")) if support.get("overall_shape") is not None else None
    support_side = str(support.get("support_side")) if support.get("support_side") is not None else None
    support_axis = support.get("support_axis_angle_from_vertical_deg")
    support_axis = float(support_axis) if isinstance(support_axis, (int, float)) else None

    if mode not in _ADJUDICABLE_MODES:
        mode_label = mode or "unknown"
        reason = f"crop_policy_{mode_label}_disallows_broad_pose_adjudication"
        body["support_shape_adjudication"] = _support_record(
            status="abstain",
            original_shape=original_shape,
            reason=reason,
            support_side=support_side,
            support_axis_angle_deg=support_axis,
        )
        body["pose_adjudication"] = _pose_record(
            status="abstain",
            original_pose=pose_text,
            reason=reason,
        )
        return out

    seated, seated_reason = _is_explicit_seated_context(body)
    if seated:
        reason = seated_reason or "explicit_seated_pose"
        body["support_shape_adjudication"] = _support_record(
            status="protected",
            original_shape=original_shape,
            reason=reason,
            support_side=support_side,
            support_axis_angle_deg=support_axis,
        )
        body["pose_adjudication"] = _pose_record(
            status="protected",
            original_pose=pose_text,
            reason=reason,
        )
        return out

    if not support or not support.get("available"):
        reason = "v07_support_geometry_unavailable"
        body["support_shape_adjudication"] = _support_record(
            status="insufficient_evidence",
            original_shape=original_shape,
            reason=reason,
            support_side=support_side,
            support_axis_angle_deg=support_axis,
        )
        body["pose_adjudication"] = _pose_record(
            status="insufficient_evidence",
            original_pose=pose_text,
            reason=reason,
        )
        return out

    if original_shape != "mostly_upright_over_support_leg":
        reason = "v07_support_shape_is_not_upright_claim"
        body["support_shape_adjudication"] = _support_record(
            status="preserved",
            original_shape=original_shape,
            reason=reason,
            support_side=support_side,
            support_axis_angle_deg=support_axis,
        )
        body["pose_adjudication"] = _pose_record(
            status="preserved",
            original_pose=pose_text,
            reason=reason,
        )
        return out

    if support_side not in {"left", "right"}:
        reason = "support_side_unresolved"
        body["support_shape_adjudication"] = _support_record(
            status="insufficient_evidence",
            original_shape=original_shape,
            reason=reason,
            support_side=support_side,
            support_axis_angle_deg=support_axis,
        )
        body["pose_adjudication"] = _pose_record(
            status="insufficient_evidence",
            original_pose=pose_text,
            reason=reason,
        )
        return out

    point_error: str | None = None
    resolved_points = points
    if resolved_points is None:
        resolved_points, point_error = _load_points(out)

    knee_angle = _support_knee_angle(resolved_points or {}, support_side)
    if knee_angle is None:
        reason = point_error or "support_knee_chain_unavailable"
        body["support_shape_adjudication"] = _support_record(
            status="insufficient_evidence",
            original_shape=original_shape,
            reason=reason,
            support_side=support_side,
            support_axis_angle_deg=support_axis,
        )
        body["pose_adjudication"] = _pose_record(
            status="insufficient_evidence",
            original_pose=pose_text,
            reason=reason,
        )
        return out

    if knee_angle >= DEEP_SUPPORT_KNEE_ANGLE_DEG:
        reason = "support_knee_not_deeply_flexed"
        body["support_shape_adjudication"] = _support_record(
            status="preserved",
            original_shape=original_shape,
            reason=reason,
            support_side=support_side,
            knee_angle_deg=knee_angle,
            support_axis_angle_deg=support_axis,
        )
        body["pose_adjudication"] = _pose_record(
            status="preserved",
            original_pose=pose_text,
            reason=reason,
        )
        return out

    body["support_shape_adjudication"] = _support_record(
        status="adjudicated",
        original_shape=original_shape,
        proposed_shape="deeply_flexed_over_support_leg",
        would_change=True,
        reason="deep_support_knee_flexion_vetoes_mostly_upright_support_shape",
        support_side=support_side,
        knee_angle_deg=knee_angle,
        support_axis_angle_deg=support_axis,
    )

    normalized_pose = (pose_text or "").strip().lower()
    if _CROUCH_COMPATIBLE_RE.search(pose_text or ""):
        body["pose_adjudication"] = _pose_record(
            status="preserved",
            original_pose=pose_text,
            reason="upstream_pose_already_compatible_with_deep_support_flexion",
        )
    elif _STANDISH_RE.search(pose_text or "") or normalized_pose in _NONCOMMITTAL_POSES:
        body["pose_adjudication"] = _pose_record(
            status="adjudicated",
            original_pose=pose_text,
            proposed_pose="crouched",
            would_change=True,
            reason="deep_support_knee_flexion_conflicts_with_standing_or_noncommittal_pose",
        )
    else:
        body["pose_adjudication"] = _pose_record(
            status="preserved",
            original_pose=pose_text,
            reason="explicit_nonstanding_pose_is_outside_bounded_adjudication_authority",
        )

    return out


_BASE_APPLY_PHASE4B4 = phase4b6.phase4b5.phase4b4._apply_phase4b4


def _apply_phase4b7_shadow(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B4(sheet)
    out = _apply_shadow_adjudication(out)

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.7-shadow"
    audit["specialist_adjudication_phase"] = "4B.7-shadow"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        specialist_adjudication_is_shadow_only=True,
        composer_authority_is_unchanged=True,
        crop_policy_precedes_pose_adjudication=True,
        explicit_seated_pose_is_protected=True,
        support_knee_flexion_can_only_veto_existing_upright_support_claim=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    # Run after the full v07 support-geometry stack. v07 still owns every
    # authoritative fact; this stage only appends shadow adjudication records.
    phase4b6.phase4b5.phase4b4._apply_phase4b4 = _apply_phase4b7_shadow
    phase4b6.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b6.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b6.main()


if __name__ == "__main__":
    raise SystemExit(main())
