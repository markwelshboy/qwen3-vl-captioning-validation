from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from . import selfie_evidence_shadow_v03 as base

SCHEMA_VERSION = "selfie-evidence-shadow-0.4"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "selfie-evidence-shadow-v0.4"


def _apply_nearer_shoulder_consistency(
    arm: dict[str, Any],
    shoulder: dict[str, Any],
) -> dict[str, Any]:
    out = dict(arm)
    raw_grade = str(out.get("grade") or "none")
    selected = out.get("selected_arm")
    nearer = shoulder.get("anatomical_side_nearer")
    shoulder_status = str(shoulder.get("status") or "")
    clear = bool(
        shoulder_status == "clear"
        and shoulder.get("publishable_candidate")
        and nearer in {"left", "right"}
    )

    if not clear or selected not in {"left", "right"}:
        consistency = "not_available"
        effective_grade = raw_grade
        reason = "clear_nearer_shoulder_not_available_for_internal_cross_check"
    elif selected == nearer:
        consistency = "consistent"
        effective_grade = raw_grade
        reason = "selected_foreground_arm_matches_clear_nearer_shoulder_side"
    else:
        consistency = "conflict"
        effective_grade = "none"
        reason = (
            "selected_foreground_arm_is_opposite_clear_nearer_shoulder; "
            "withhold_foreground_arm_as_selfie_evidence"
        )

    out["raw_grade_before_shoulder_crosscheck"] = raw_grade
    out["grade"] = effective_grade
    out["nearer_shoulder_consistency"] = {
        "status": consistency,
        "selected_arm_anatomical_side_internal": selected,
        "clear_nearer_shoulder_anatomical_side": nearer if clear else None,
        "reason": reason,
    }

    if effective_grade == "none":
        out["composer_text"] = None
        out["reason"] = reason

    return out


def evaluate(
    policy: dict[str, Any],
    framing: dict[str, Any],
    gestalt: dict[str, Any] | None,
    mesh_arm: dict[str, Any] | None,
    arrays: dict[str, Any],
    dwpose: dict[str, Any],
) -> dict[str, Any]:
    geometry = base.base.camera_shadow.build_subject_geometry(arrays, dwpose)
    camera_viewpoint = base.base.camera_shadow._camera_viewpoint(geometry)
    shoulder = base.base.camera_shadow._shoulder_depth(arrays, dwpose)
    portrait = base.base.camera_shadow._portrait_context(framing, policy)

    neutral = base.base._neutral_selfie_semantic(gestalt)
    camera = base.base._camera_evidence(camera_viewpoint)
    arm_raw = base._foreground_arm_evidence(mesh_arm)
    arm = _apply_nearer_shoulder_consistency(arm_raw, shoulder)
    decision = base._decision(neutral, camera, arm, shoulder, portrait)
    decision["decision_rule_version"] = "v04_nearer_shoulder_crosschecked_arm"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": policy.get("image_key"),
        "portrait_context": portrait,
        "evidence": {
            "neutral_semantic": neutral,
            "camera_viewpoint": camera,
            "foreground_arm": arm,
            "nearer_shoulder": {
                **shoulder,
                "role": "supportive_context_and_internal_foreground_arm_crosscheck",
            },
        },
        "decision": decision,
        "invariants": {
            "dedicated_selfie_prompt_is_not_used": True,
            "neutral_gestalt_selfie_language_is_high_precision_but_not_required": True,
            "selfie_requires_at_least_two_independent_primary_cue_families": True,
            "at_least_one_primary_cue_must_be_strong": True,
            "camera_plus_combined_foreground_arm_can_recover_neutral_qwen_miss": True,
            "nearer_shoulder_never_counts_as_primary_selfie_evidence": True,
            "clear_nearer_shoulder_may_veto_opposite_side_foreground_arm_candidate": True,
            "nearer_shoulder_crosscheck_is_internal_and_does_not_anatomically_label_caption_arm": True,
            "mesh_is_reconstruction_and_cannot_own_foreground_arm_alone": True,
            "dwpose_body_and_hand_landmarks_are_direct_observation": True,
            "foreground_arm_caption_text_is_frame_relative_and_side_neutral": True,
            "selfie_label_does_not_imply_arm_holds_camera": True,
            "misses_are_preferred_to_cross_geometry_false_positives": True,
        },
    }


def main() -> int:
    original_argv = list(sys.argv)
    old_evaluate = base.base.evaluate
    old_schema = base.base.SCHEMA_VERSION
    old_output = base.base.DEFAULT_OUTPUT_SUBDIR
    old_mesh_dir = base.base.DEFAULT_MESH_ARM_SUBDIR
    try:
        base.base.evaluate = evaluate
        base.base.SCHEMA_VERSION = SCHEMA_VERSION
        base.base.DEFAULT_MESH_ARM_SUBDIR = base.DEFAULT_MESH_ARM_SUBDIR
        base.base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
        return base.base.main()
    finally:
        base.base.evaluate = old_evaluate
        base.base.SCHEMA_VERSION = old_schema
        base.base.DEFAULT_OUTPUT_SUBDIR = old_output
        base.base.DEFAULT_MESH_ARM_SUBDIR = old_mesh_dir
        sys.argv[:] = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
