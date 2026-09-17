from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import crouched_stance_depth_shadow_v01 as crouch_shadow
from . import fact_sheet_specialist_normalizer_v14 as phase4b13

SCHEMA_VERSION = "caption-fact-sheet-0.2.14"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.14"


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


def _depth_binding(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_relation": "hips_slightly_lowered",
        "classification": evidence.get("depth_classification"),
        "authority": (
            "canonical_crouching_plus_high_authority_bilateral_sam3d_moderate_lower_body_geometry"
        ),
        "sam3d_public_pose": evidence.get("sam3d_public_pose"),
        "sam3d_knee_authority": evidence.get("sam3d_knee_authority"),
        "per_side": copy.deepcopy(evidence.get("per_side")),
        "aggregate": copy.deepcopy(evidence.get("aggregate")),
        "qualifier_gates": copy.deepcopy(evidence.get("qualifier_gates")),
        "thresholds": copy.deepcopy(evidence.get("thresholds")),
    }


def _apply_crouched_stance_depth_authority(
    sheet: dict[str, Any],
    *,
    projected: dict[str, Any] | None = None,
    specialist_error: str | None = None,
) -> dict[str, Any]:
    """Promote only the population-gated moderate crouch-depth modifier.

    This stage is downstream of broad-pose authority.  It cannot create a
    crouching pose.  It may only publish ``hips slightly lowered`` when the
    canonical pose is already crouching and the reviewed bilateral SAM3D gate
    identifies moderate rather than deep lowering.
    """
    out = copy.deepcopy(sheet)
    body = _body(out)

    specialist = projected
    load_error = specialist_error
    if specialist is None:
        specialist, load_error = crouch_shadow.sam3d_shadow._load_specialist_projection(out)

    evidence = crouch_shadow.evaluate_shadow(
        image_key=str(out.get("image_key") or "unknown"),
        sheet=out,
        projected=specialist,
        specialist_error=load_error,
    )

    shadow_status = str(evidence.get("status") or "not_applicable")
    qualifies = shadow_status == "candidate_hips_slightly_lowered"

    # Never carry a stale depth fact forward.  The authoritative fact is
    # reconstructed from the current canonical pose + current specialist data.
    body.pop("crouched_stance_depth", None)

    adjudication: dict[str, Any] = {
        "status": "not_applicable",
        "authoritative_stage": True,
        "composer_authoritative": False,
        "applied": False,
        "canonical_relation": None,
        "depth_classification": evidence.get("depth_classification"),
        "shadow_status": shadow_status,
        "reason": evidence.get("reason"),
        "evidence": copy.deepcopy(evidence),
    }

    if qualifies:
        binding = _depth_binding(evidence)
        body["crouched_stance_depth"] = {
            "text": "hips slightly lowered",
            "composer_text": "hips slightly lowered",
            "normalized_text": "hips slightly lowered",
            "domain": "global_configuration",
            "authority": "specialist_authoritative",
            "promotion_status": "accepted_specialist_crouched_stance_depth_candidate",
            "specialist_owner": "sam3d_v16_crouched_stance_depth_specialist",
            "classification": "moderate_lowering",
            "semantic_relation": "hips_slightly_lowered",
            "crouched_stance_depth_binding": binding,
        }
        adjudication.update(
            status="adjudicated",
            composer_authoritative=True,
            applied=True,
            canonical_relation="hips_slightly_lowered",
            depth_classification="moderate_lowering",
            reason=(
                "population_review_promoted_authoritative_crouching_plus_bilateral_moderate_lowering"
            ),
            broad_pose_created=False,
            support_contact_claim_created=False,
            support_side_claim_created=False,
        )
    elif shadow_status == "candidate_crouched_stance_only":
        adjudication.update(
            status="confirmed_crouched_stance_without_depth_modifier",
            reason=evidence.get("reason"),
        )
    elif shadow_status == "route_abstain":
        adjudication["status"] = "route_abstain"

    body["crouched_stance_depth_adjudication"] = adjudication
    return out


_BASE_APPLY_PHASE4B13 = phase4b13._apply_phase4b13_authoritative


def _apply_phase4b14_authoritative(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B13(sheet)
    out = _apply_crouched_stance_depth_authority(out)

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.14-authoritative"
    audit["specialist_adjudication_phase"] = "4B.14-authoritative"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        crouched_stance_depth_is_downstream_of_authoritative_broad_pose=True,
        crouched_stance_depth_cannot_create_crouching=True,
        hips_slightly_lowered_requires_population_gated_bilateral_moderate_geometry=True,
        crouched_stance_depth_does_not_publish_support_contact=True,
        crouched_stance_depth_does_not_publish_support_side=True,
        prior_broad_pose_torso_support_knee_and_raised_leg_gates_remain_authoritative=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    # Let v14 retain the validated shared writer stack while substituting this
    # successor application function.  The captured base above still executes
    # every Phase-4B.13 specialist before the new depth qualifier.
    phase4b13._apply_phase4b13_authoritative = _apply_phase4b14_authoritative
    phase4b13.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b13.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b13.main()


if __name__ == "__main__":
    raise SystemExit(main())
