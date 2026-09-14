from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np

from . import fact_sheet_specialist_normalizer_v07 as phase4b6
from . import sam3d_relational_pose_profile_16 as sam3d16

SCHEMA_VERSION = "caption-fact-sheet-0.2.8"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.8"

_ADJUDICABLE_MODES = {"pose_allowed", "pose_guided"}
_SEATED_RE = re.compile(r"\b(?:seated|sitting)\b", re.I)
_POSE_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("crouching", re.compile(r"\b(?:crouch|crouched|crouching)\b", re.I)),
    ("squatting", re.compile(r"\b(?:squat|squatted|squatting)\b", re.I)),
    ("sitting", re.compile(r"\b(?:sit|sits|sitting|seated)\b", re.I)),
    ("standing", re.compile(r"\b(?:stand|stands|standing|upright)\b", re.I)),
    ("reclined", re.compile(r"\b(?:recline|reclined|reclining|lying|lays?|horizontal)\b", re.I)),
)


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


def _pose_family(text: str | None) -> str | None:
    if not text:
        return None
    for family, pattern in _POSE_FAMILY_PATTERNS:
        if pattern.search(text):
            return family
    return None


def _route_mode(sheet: dict[str, Any]) -> str:
    policy = sheet.get("policy") if isinstance(sheet.get("policy"), dict) else {}
    return str(policy.get("mode") or "")


def _load_specialist_projection(sheet: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    sources = sheet.get("sources") if isinstance(sheet.get("sources"), dict) else {}
    policy_text = sources.get("perception_policy")
    if not policy_text:
        return None, "perception_policy_source_missing"

    policy_path = Path(str(policy_text)).expanduser()
    if not policy_path.is_file():
        return None, "perception_policy_source_not_found"

    try:
        read_json = phase4b6.phase4b5.phase4b4._read_json
        policy = read_json(policy_path)
        size = policy.get("image_size")
        policy_sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
        dwpose_text = policy_sources.get("dwpose")
        sam3d_text = policy_sources.get("sam3d_arrays")
        if not isinstance(size, list) or len(size) < 2:
            return None, "image_size_missing"
        if not dwpose_text:
            return None, "dwpose_source_missing"
        if not sam3d_text:
            return None, "sam3d_arrays_source_missing"

        dwpose_path = Path(str(dwpose_text)).expanduser()
        sam3d_path = Path(str(sam3d_text)).expanduser()
        if not dwpose_path.is_file():
            return None, "dwpose_source_not_found"
        if not sam3d_path.is_file():
            return None, "sam3d_arrays_source_not_found"

        with np.load(sam3d_path, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        profile = sam3d16.build_profile(
            arrays,
            read_json(dwpose_path),
            int(size[0]),
            int(size[1]),
        )
        projected = profile.get("sam3d_projected_pose")
        return (projected if isinstance(projected, dict) else None), None
    except Exception as exc:  # pragma: no cover - surfaced in fact-sheet diagnostics
        return None, f"sam3d_v16_specialist_failed:{type(exc).__name__}"


def _specialist_summary(projected: dict[str, Any]) -> dict[str, Any]:
    assertion = projected.get("assertion_authority") if isinstance(projected.get("assertion_authority"), dict) else {}
    modifier = projected.get("posture_modifier_diagnostic") if isinstance(projected.get("posture_modifier_diagnostic"), dict) else {}
    scores = projected.get("posture_score_percent") if isinstance(projected.get("posture_score_percent"), dict) else {}
    return {
        "public_pose": projected.get("pose"),
        "best_candidate_pose": projected.get("best_candidate_pose"),
        "posture_score_percent": copy.deepcopy(scores),
        "joint_authority_percent": projected.get("crop_support_percent"),
        "assertion_path": assertion.get("selected_path"),
        "assertion_path_authority_percent": assertion.get("selected_path_authority_percent"),
        "withheld_reason": assertion.get("withheld_reason"),
        "lean_severity": modifier.get("lean_severity"),
        "lean_direction": modifier.get("lean_direction"),
        "torso_inclination_from_vertical_deg": modifier.get("torso_inclination_from_vertical_deg"),
        "shoulder_line_declination_deg": modifier.get("shoulder_line_declination_deg"),
    }


def _broad_pose_adjudication(
    body: dict[str, Any],
    projected: dict[str, Any],
) -> dict[str, Any]:
    qwen_text = _pose_text(body)
    qwen_family = _pose_family(qwen_text)
    public_pose = str(projected.get("pose") or "uncertain")
    best_pose = str(projected.get("best_candidate_pose") or "uncertain")
    assertion = projected.get("assertion_authority") if isinstance(projected.get("assertion_authority"), dict) else {}
    assertion_path = str(assertion.get("selected_path") or "")

    record = {
        "status": "observed",
        "shadow_only": True,
        "composer_authoritative": False,
        "would_change": False,
        "qwen_pose_text": qwen_text,
        "qwen_pose_family": qwen_family,
        "specialist": _specialist_summary(projected),
        "proposed_pose_family": None,
        "reason": None,
    }

    if public_pose in {"", "uncertain"}:
        record.update(status="abstain", reason="sam3d_v16_public_pose_withheld")
        return record

    if qwen_family == public_pose:
        record.update(status="confirmed", reason="qwen_pose_family_agrees_with_publishable_sam3d_v16_pose")
        return record

    # Deliberately narrow replacement authority. Crouching is the one family in
    # v0.16 with a pose-specific observed hip+knee assertion path. That permits
    # a standing/noncommittal Qwen candidate to be challenged without turning
    # every SAM3D reconstruction into prose authority.
    if (
        public_pose == "crouching"
        and best_pose == "crouching"
        and assertion_path == "observed_crouch_hip_knee_chain"
        and qwen_family in {None, "standing"}
    ):
        record.update(
            status="adjudicated",
            would_change=True,
            proposed_pose_family="crouching",
            reason="observed_sam3d_v16_crouch_path_overrules_standing_or_noncommittal_qwen_pose",
        )
        return record

    record.update(
        status="conflict_no_authority_to_replace",
        reason="specialist_disagrees_but_no_pose_specific_observed_replacement_path_is_authorized",
    )
    return record


def _forward_torso_relation_adjudication(
    body: dict[str, Any],
    projected: dict[str, Any],
) -> dict[str, Any]:
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    forward_items: list[dict[str, Any]] = []
    for item in configuration:
        if not isinstance(item, dict):
            continue
        text = str(item.get("composer_text") or item.get("normalized_text") or item.get("text") or "")
        if phase4b6.FORWARD_TORSO_RE.search(text):
            forward_items.append(item)

    modifier = projected.get("posture_modifier_diagnostic") if isinstance(projected.get("posture_modifier_diagnostic"), dict) else {}
    severity = str(modifier.get("lean_severity") or "unavailable")
    direction = str(modifier.get("lean_direction") or "direction_indeterminate")
    record = {
        "status": "not_applicable",
        "shadow_only": True,
        "composer_authoritative": False,
        "would_change": False,
        "relation_count": len(forward_items),
        "source_texts": [str(x.get("text") or "") for x in forward_items],
        "lean_severity": severity,
        "lean_direction": direction,
        "torso_inclination_from_vertical_deg": modifier.get("torso_inclination_from_vertical_deg"),
        "reason": "no_forward_torso_relation_to_adjudicate",
    }
    if not forward_items:
        return record

    if severity == "upright":
        record.update(
            status="contradicted",
            would_change=True,
            reason="dwpose_observed_torso_axis_is_upright_in_sam3d_v16_modifier_diagnostic",
        )
        return record

    if direction in {"forward", "forward_possible"} and severity in {
        "slight", "moderate", "heavy", "near_horizontal"
    }:
        record.update(
            status="supported",
            reason="sam3d_v16_posture_modifier_supports_forward_torso_relation",
        )
        return record

    record.update(
        status="ambiguous",
        reason="sam3d_v16_modifier_does_not_cleanly_support_or_contradict_forward_torso_relation",
    )
    return record


def _apply_shadow_adjudication(
    sheet: dict[str, Any],
    *,
    projected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(sheet)
    body = _body(out)
    mode = _route_mode(out)
    qwen_text = _pose_text(body)

    if mode not in _ADJUDICABLE_MODES:
        reason = f"route_{mode or 'unknown'}_outside_bounded_broad_pose_authority"
        body["broad_pose_adjudication"] = {
            "status": "abstain", "shadow_only": True, "composer_authoritative": False,
            "would_change": False, "qwen_pose_text": qwen_text, "reason": reason,
        }
        body["torso_relation_adjudication"] = {
            "status": "abstain", "shadow_only": True, "composer_authoritative": False,
            "would_change": False, "reason": reason,
        }
        return out

    if _SEATED_RE.search(qwen_text or ""):
        reason = "explicit_qwen_seated_pose_protected_during_initial_sam3d_adjudication_rollout"
        body["broad_pose_adjudication"] = {
            "status": "protected", "shadow_only": True, "composer_authoritative": False,
            "would_change": False, "qwen_pose_text": qwen_text, "qwen_pose_family": "sitting", "reason": reason,
        }
        body["torso_relation_adjudication"] = {
            "status": "protected", "shadow_only": True, "composer_authoritative": False,
            "would_change": False, "reason": reason,
        }
        return out

    specialist = projected
    error: str | None = None
    if specialist is None:
        specialist, error = _load_specialist_projection(out)
    if specialist is None:
        reason = error or "sam3d_v16_specialist_unavailable"
        body["broad_pose_adjudication"] = {
            "status": "insufficient_evidence", "shadow_only": True, "composer_authoritative": False,
            "would_change": False, "qwen_pose_text": qwen_text, "reason": reason,
        }
        body["torso_relation_adjudication"] = {
            "status": "insufficient_evidence", "shadow_only": True, "composer_authoritative": False,
            "would_change": False, "reason": reason,
        }
        return out

    body["broad_pose_adjudication"] = _broad_pose_adjudication(body, specialist)
    body["torso_relation_adjudication"] = _forward_torso_relation_adjudication(body, specialist)
    return out


_BASE_APPLY_PHASE4B4 = phase4b6.phase4b5.phase4b4._apply_phase4b4


def _apply_phase4b8_shadow(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B4(sheet)
    out = _apply_shadow_adjudication(out)
    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.8-shadow"
    audit["specialist_adjudication_phase"] = "4B.8-shadow"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        sam3d_v16_broad_pose_adjudication_is_shadow_only=True,
        composer_authority_is_unchanged=True,
        crop_route_precedes_broad_pose_adjudication=True,
        explicit_seated_pose_is_protected=True,
        dwpose_local_joint_scalars_do_not_independently_classify_broad_pose=True,
        crouch_replacement_requires_observed_sam3d_v16_hip_knee_assertion_path=True,
        torso_forward_relation_is_checked_separately_from_broad_pose_family=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    # v07 remains the authoritative specialist stack. This successor intentionally
    # bypasses the failed v08 knee-angle experiment and appends only shadow
    # SAM3D-v16 adjudication records.
    phase4b6.phase4b5.phase4b4._apply_phase4b4 = _apply_phase4b8_shadow
    phase4b6.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b6.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b6.main()


if __name__ == "__main__":
    raise SystemExit(main())
