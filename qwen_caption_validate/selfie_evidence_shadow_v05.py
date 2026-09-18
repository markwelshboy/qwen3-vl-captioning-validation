from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from . import selfie_evidence_shadow_v03 as base

SCHEMA_VERSION = "selfie-evidence-shadow-0.5"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "selfie-evidence-shadow-v0.5"


def _rank(grade: str) -> int:
    return {"none": 0, "insufficient": 0, "weak": 1, "moderate": 2, "strong": 3}.get(str(grade), 0)


def _arm_candidate(side: str, arm: dict[str, Any]) -> dict[str, Any]:
    grade = str(arm.get("combined_foreground_arm_grade") or "insufficient")
    path = arm.get("distal_arm_path_proxy") if isinstance(arm.get("distal_arm_path_proxy"), dict) else {}
    region = path.get("distal_frame_region") or arm.get("frame_region")
    path_grade = str(path.get("grade") or "none")
    mesh_grade = str(arm.get("evidence_grade") or "insufficient")

    composer_text = None
    if grade in {"strong", "moderate"} and region:
        straightness = path.get("complete_chain_straightness")
        outstretched = bool(
            path_grade == "strong"
            and (
                straightness is None
                or (isinstance(straightness, (int, float)) and straightness >= 0.72)
            )
        )
        noun = "an outstretched arm" if outstretched else "an arm"
        composer_text = f"{noun} extends into the {str(region).replace('_', '-')} foreground"

    return {
        "anatomical_side_internal": side,
        "grade": grade if grade in {"strong", "moderate", "weak"} else "none",
        "frame_region": region,
        "composer_text": composer_text,
        "mesh_occupancy_grade": mesh_grade,
        "visible_mesh_area_fraction": arm.get("visible_mesh_area_fraction"),
        "occupancy_band": arm.get("occupancy_band"),
        "observed_arm_path_grade": path_grade,
        "distal_arm_path_proxy": path,
        "dwpose_hand_support": arm.get("dwpose_hand_support"),
        "dwpose_observation_support": arm.get("dwpose_observation_support"),
    }


def _foreground_arm_partition(
    mesh_record: dict[str, Any] | None,
    shoulder: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(mesh_record, dict) or mesh_record.get("status") != "ok":
        return {
            "selfie_arm_evidence": {
                "grade": "none",
                "selected_arm": None,
                "reason": "mesh_arm_v02_unavailable",
            },
            "composition_arm_candidates": [],
        }

    arms = mesh_record.get("arms") if isinstance(mesh_record.get("arms"), dict) else {}
    candidates = [
        _arm_candidate(side, arms.get(side) if isinstance(arms.get(side), dict) else {})
        for side in ("left", "right")
    ]

    composition = [
        c for c in candidates
        if _rank(c["grade"]) >= _rank("moderate") and c.get("frame_region")
    ]

    nearer = shoulder.get("anatomical_side_nearer")
    shoulder_clear = bool(
        shoulder.get("status") == "clear"
        and shoulder.get("publishable_candidate")
        and nearer in {"left", "right"}
    )

    selected: dict[str, Any] | None = None
    reason: str
    if shoulder_clear:
        same_side = next(
            (c for c in candidates if c["anatomical_side_internal"] == nearer),
            None,
        )
        if same_side and _rank(same_side["grade"]) >= _rank("moderate"):
            selected = same_side
            reason = "foreground_arm_matches_clear_nearer_shoulder_and_clears_evidence_threshold"
        else:
            reason = (
                "no_moderate_or_strong_foreground_arm_candidate_on_clear_nearer_shoulder_side"
            )
    else:
        # Precision-first: an arm can remain a valid composition fact, but without
        # a clear nearer-shoulder anchor it does not count as primary selfie evidence.
        reason = "clear_nearer_shoulder_unavailable_for_selfie_arm_binding"

    if selected is None:
        selfie_arm = {
            "grade": "none",
            "selected_arm": None,
            "frame_region": None,
            "composer_text": None,
            "reason": reason,
            "clear_nearer_shoulder_anatomical_side": nearer if shoulder_clear else None,
        }
    else:
        selfie_arm = {
            "grade": selected["grade"],
            "selected_arm": selected["anatomical_side_internal"],
            "frame_region": selected["frame_region"],
            "composer_text": selected["composer_text"],
            "mesh_occupancy_grade": selected["mesh_occupancy_grade"],
            "visible_mesh_area_fraction": selected["visible_mesh_area_fraction"],
            "occupancy_band": selected["occupancy_band"],
            "observed_arm_path_grade": selected["observed_arm_path_grade"],
            "distal_arm_path_proxy": selected["distal_arm_path_proxy"],
            "dwpose_hand_support": selected["dwpose_hand_support"],
            "dwpose_observation_support": selected["dwpose_observation_support"],
            "reason": reason,
            "clear_nearer_shoulder_anatomical_side": nearer,
            "authority": "nearer_shoulder_bound_mesh_plus_dwpose_foreground_arm_shadow",
        }

    return {
        "selfie_arm_evidence": selfie_arm,
        "composition_arm_candidates": composition,
        "all_arm_candidates": candidates,
        "note": (
            "Foreground-arm composition and selfie-arm evidence are separate. "
            "An opposite-side arm may be a real visible foreground arm without being "
            "the arm that contributes evidence for an arm's-length selfie composition."
        ),
    }


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
    partition = _foreground_arm_partition(mesh_arm, shoulder)
    selfie_arm = partition["selfie_arm_evidence"]

    decision = base._decision(neutral, camera, selfie_arm, shoulder, portrait)
    decision["decision_rule_version"] = "v05_separate_composition_arm_from_selfie_arm"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": policy.get("image_key"),
        "portrait_context": portrait,
        "evidence": {
            "neutral_semantic": neutral,
            "camera_viewpoint": camera,
            "foreground_arm": selfie_arm,
            "foreground_arm_composition": {
                "candidates": partition["composition_arm_candidates"],
                "all_candidates": partition["all_arm_candidates"],
                "role": "visible_composition_facts_not_automatically_selfie_evidence",
            },
            "nearer_shoulder": {
                **shoulder,
                "role": "supportive_context_and_selfie_arm_binding_anchor",
            },
        },
        "decision": decision,
        "invariants": {
            "dedicated_selfie_prompt_is_not_used": True,
            "neutral_gestalt_selfie_language_is_high_precision_but_not_required": True,
            "selfie_requires_at_least_two_independent_primary_cue_families": True,
            "at_least_one_primary_cue_must_be_strong": True,
            "camera_plus_bound_foreground_arm_can_recover_neutral_qwen_miss": True,
            "nearer_shoulder_never_counts_as_primary_selfie_evidence": True,
            "opposite_side_foreground_arm_can_remain_valid_composition_fact": True,
            "only_nearer_shoulder_side_arm_can_count_as_primary_selfie_arm_evidence": True,
            "foreground_arm_caption_text_is_frame_relative_and_side_neutral": True,
            "selfie_arm_binding_does_not_claim_arm_holds_camera": True,
            "misses_are_preferred_to_unanchored_arm_false_positives": True,
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
