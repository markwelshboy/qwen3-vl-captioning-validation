from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from . import selfie_evidence_shadow_v02 as base

SCHEMA_VERSION = "selfie-evidence-shadow-0.3"
DEFAULT_MESH_ARM_SUBDIR = Path("semantic-v3") / "mesh-arm-occupancy-shadow-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "selfie-evidence-shadow-v0.3"

_BASE_DECISION = base._decision


def _foreground_arm_evidence(mesh_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(mesh_record, dict) or mesh_record.get("status") != "ok":
        return {
            "grade": "none",
            "selected_arm": None,
            "reason": "mesh_arm_v02_unavailable",
        }

    arms = mesh_record.get("arms") if isinstance(mesh_record.get("arms"), dict) else {}
    ranked = {"strong": 3, "moderate": 2, "weak": 1, "insufficient": 0, "none": 0}
    candidates: list[tuple[int, float, str, dict[str, Any]]] = []

    for side in ("left", "right"):
        arm = arms.get(side) if isinstance(arms.get(side), dict) else {}
        grade = str(arm.get("combined_foreground_arm_grade") or "insufficient")
        occupancy = arm.get("visible_mesh_area_fraction")
        occupancy = float(occupancy) if isinstance(occupancy, (int, float)) else 0.0
        candidates.append((ranked.get(grade, 0), occupancy, side, arm))

    candidates.sort(reverse=True)
    _, _, side, arm = candidates[0] if candidates else (0, 0.0, "", {})
    grade = str(arm.get("combined_foreground_arm_grade") or "insufficient")
    if grade not in {"strong", "moderate", "weak"}:
        grade = "none"

    path = arm.get("distal_arm_path_proxy") if isinstance(arm.get("distal_arm_path_proxy"), dict) else {}
    region = path.get("distal_frame_region") or arm.get("frame_region")
    mesh_grade = str(arm.get("evidence_grade") or "insufficient")
    path_grade = str(path.get("grade") or "none")

    composer_text = None
    if grade in {"strong", "moderate"} and region:
        straightness = path.get("complete_chain_straightness")
        strong_extension = bool(
            path_grade == "strong"
            and (
                straightness is None
                or (isinstance(straightness, (int, float)) and straightness >= 0.72)
            )
        )
        if strong_extension:
            composer_text = f"an outstretched arm extends into the {str(region).replace('_', '-')} foreground"
        else:
            composer_text = f"an arm extends into the {str(region).replace('_', '-')} foreground"

    return {
        "grade": grade,
        "selected_arm": side or None,
        "frame_region": region,
        "composer_text": composer_text,
        "mesh_occupancy_grade": mesh_grade,
        "visible_mesh_area_fraction": arm.get("visible_mesh_area_fraction"),
        "occupancy_band": arm.get("occupancy_band"),
        "observed_arm_path_grade": path_grade,
        "distal_arm_path_proxy": path,
        "dwpose_hand_support": arm.get("dwpose_hand_support"),
        "dwpose_observation_support": arm.get("dwpose_observation_support"),
        "reason": (
            "combined_projected_mesh_occupancy_and_observed_distal_arm_path"
            if grade != "none"
            else "no_arm_cleared_combined_mesh_and_observed_path_evidence"
        ),
        "authority": "mesh_projection_plus_dwpose_body_and_hand_observation_shadow",
    }


def _decision(
    neutral: dict[str, Any],
    camera: dict[str, Any],
    arm: dict[str, Any],
    shoulder: dict[str, Any],
    portrait: dict[str, Any],
) -> dict[str, Any]:
    out = _BASE_DECISION(neutral, camera, arm, shoulder, portrait)

    # In v0.2 only unprompted Qwen selfie language emitted the high-level
    # "selfie-style capture" phrase. Here the fused evidence decision itself is
    # allowed to own that semantic, so camera+foreground-arm can recover obvious
    # selfie compositions that neutral Qwen missed (e.g. validation case 00026).
    if out.get("publishable_selfie"):
        promoted = [str(v) for v in (out.get("promoted_fact_candidates") or [])]
        if "selfie-style capture" not in promoted:
            promoted.insert(0, "selfie-style capture")
        out["promoted_fact_candidates"] = promoted
        out["selfie_semantic_authority"] = "multi_family_fused_evidence_shadow"
    else:
        out["selfie_semantic_authority"] = None

    out["decision_rule_version"] = "v03_combined_arm_geometry"
    return out


def evaluate(
    policy: dict[str, Any],
    framing: dict[str, Any],
    gestalt: dict[str, Any] | None,
    mesh_arm: dict[str, Any] | None,
    arrays: dict[str, Any],
    dwpose: dict[str, Any],
) -> dict[str, Any]:
    geometry = base.camera_shadow.build_subject_geometry(arrays, dwpose)
    camera_viewpoint = base.camera_shadow._camera_viewpoint(geometry)
    shoulder = base.camera_shadow._shoulder_depth(arrays, dwpose)
    portrait = base.camera_shadow._portrait_context(framing, policy)

    neutral = base._neutral_selfie_semantic(gestalt)
    camera = base._camera_evidence(camera_viewpoint)
    arm = _foreground_arm_evidence(mesh_arm)
    decision = _decision(neutral, camera, arm, shoulder, portrait)

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
                "role": "supportive_context_only",
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
            "mesh_is_reconstruction_and_cannot_own_foreground_arm_alone": True,
            "dwpose_body_and_hand_landmarks_are_direct_observation": True,
            "foreground_arm_caption_text_is_frame_relative_and_side_neutral": True,
            "selfie_label_does_not_imply_arm_holds_camera": True,
            "misses_are_preferred_to_single_cue_false_positives": True,
        },
    }


def main() -> int:
    original_argv = list(sys.argv)
    old_evaluate = base.evaluate
    old_decision = base._decision
    old_schema = base.SCHEMA_VERSION
    old_mesh_dir = base.DEFAULT_MESH_ARM_SUBDIR
    old_output = base.DEFAULT_OUTPUT_SUBDIR
    try:
        base.evaluate = evaluate
        base._decision = _decision
        base.SCHEMA_VERSION = SCHEMA_VERSION
        base.DEFAULT_MESH_ARM_SUBDIR = DEFAULT_MESH_ARM_SUBDIR
        base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
        return base.main()
    finally:
        base.evaluate = old_evaluate
        base._decision = old_decision
        base.SCHEMA_VERSION = old_schema
        base.DEFAULT_MESH_ARM_SUBDIR = old_mesh_dir
        base.DEFAULT_OUTPUT_SUBDIR = old_output
        sys.argv[:] = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
