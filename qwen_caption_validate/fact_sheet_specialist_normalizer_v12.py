from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v11 as phase4b10

SCHEMA_VERSION = "caption-fact-sheet-0.2.11"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.11"

ANKLE_HEIGHT_ONLY_AUTHORITY = phase4b10.ANKLE_HEIGHT_ONLY_AUTHORITY
DERIVED_KNEE_FROM_WEAK_PLANTED_AUTHORITY = "opposite_of_dwpose_bound_planted_foot_with_observed_leg_chain"
DIRECT_KNEE_AUTHORITIES = {
    "dwpose_bilateral_hip_knee_relative_height",
    "dwpose_bilateral_thigh_angle_from_vertical",
}
STRONG_PLANTED_FROM_KNEE_AUTHORITY = "opposite_of_dwpose_bound_raised_knee_with_observed_leg_chain"


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


def _binding(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("laterality_binding")
    return value if isinstance(value, dict) else {}


def _relation(item: dict[str, Any]) -> str:
    return str(_binding(item).get("semantic_relation") or "")


def _authority(item: dict[str, Any]) -> str:
    return str(_binding(item).get("authority") or "")


def _source_text(item: dict[str, Any]) -> str:
    value = item.get("text")
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())
    return ""


def _apply_support_seed_truth_gate(sheet: dict[str, Any]) -> dict[str, Any]:
    """Withhold asymmetric support topology that is seeded only by ankle height.

    Phase-4B.10 removed explicit `foot_lifted` claims when their only evidence was
    bilateral ankle-height asymmetry.  The 00064 control exposed a second path:
    ankle height can first choose a planted foot, after which the opposite knee
    is lateralized by inference.  That makes the final knee side *look* like an
    independent raised-knee decision even though the whole topology originated
    from the same weak ankle-height seed.

    This stage therefore distinguishes two cases:
      * direct raised-knee geometry -> side may publish, and an opposite planted
        foot may publish when derived from that observed knee chain;
      * ankle-height-planted -> opposite-knee inference -> asymmetric side/contact
        topology is not publishable.  The planted-foot claim is removed, while
        the Qwen semantic relation `one knee raised` is retained without side.

    Raw bindings are preserved inside the adjudication record for audit.
    """
    out = copy.deepcopy(sheet)
    body = _body(out)
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []

    weak_planted = [
        item for item in configuration
        if isinstance(item, dict)
        and _relation(item) == "foot_planted"
        and _authority(item) == ANKLE_HEIGHT_ONLY_AUTHORITY
    ]
    derived_knees = [
        item for item in configuration
        if isinstance(item, dict)
        and _relation(item) == "knee_raised"
        and _authority(item) == DERIVED_KNEE_FROM_WEAK_PLANTED_AUTHORITY
    ]

    if not weak_planted or not derived_knees:
        body["support_topology_adjudication"] = {
            "status": "not_applicable",
            "authoritative_stage": True,
            "composer_authoritative": False,
            "applied": False,
            "reason": "no_weak_ankle_height_seeded_planted_to_raised_knee_topology",
        }
        return out

    weak_planted_ids = {id(item) for item in weak_planted}
    derived_knee_ids = {id(item) for item in derived_knees}
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    delateralized: list[dict[str, Any]] = []

    for item in configuration:
        if not isinstance(item, dict):
            kept.append(copy.deepcopy(item))
            continue
        if id(item) in weak_planted_ids:
            removed.append(copy.deepcopy(item))
            continue
        if id(item) in derived_knee_ids:
            revised = copy.deepcopy(item)
            original_binding = copy.deepcopy(_binding(item))
            source = _source_text(item) or "one knee raised"
            # Keep the semantic relation proposed by Qwen, but revoke the side
            # that was inferred from a weak ankle-height support seed.
            revised["composer_text"] = source
            revised["normalized_text"] = source
            revised["promotion_status"] = "accepted_relation_truth_laterality_withheld"
            revised["specialist_owner"] = "support_topology_truth_gate"
            revised["withheld_laterality_binding"] = original_binding
            revised.pop("laterality_binding", None)
            revised.pop("support_shape_refinement", None)
            delateralized.append({
                "before": copy.deepcopy(item),
                "after": copy.deepcopy(revised),
            })
            kept.append(revised)
            continue
        kept.append(copy.deepcopy(item))

    body["configuration"] = kept

    support = body.get("support_geometry") if isinstance(body.get("support_geometry"), dict) else None
    support_invalidated = False
    if support and str(support.get("elevated_relation") or "") == "knee_raised":
        support["composer_eligible"] = False
        support["publication_status"] = "withheld_weak_ankle_height_support_seed"
        support["note_phase4b11"] = (
            "Support side was selected from bilateral ankle-height asymmetry; the resulting opposite-knee topology is retained for audit only."
        )
        support_invalidated = True

    laterality = body.get("anatomical_laterality") if isinstance(body.get("anatomical_laterality"), dict) else {}
    bindings = laterality.get("relation_bindings") if isinstance(laterality.get("relation_bindings"), list) else []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        authority = str(binding.get("authority") or "")
        relation = str(binding.get("semantic_relation") or "")
        if (
            (relation == "foot_planted" and authority == ANKLE_HEIGHT_ONLY_AUTHORITY)
            or (relation == "knee_raised" and authority == DERIVED_KNEE_FROM_WEAK_PLANTED_AUTHORITY)
        ):
            binding["publication_status"] = "withheld_weak_support_seed"
            binding["semantic_truth_validated"] = False
    if bindings:
        laterality["relation_bindings"] = bindings
        body["anatomical_laterality"] = laterality

    body["support_topology_adjudication"] = {
        "status": "withheld_weak_support_seed",
        "authoritative_stage": True,
        "composer_authoritative": True,
        "applied": True,
        "would_change": True,
        "reason": "ankle_height_seeded_planted_foot_cannot_authorize_opposite_knee_laterality",
        "removed_configuration_relations": removed,
        "delateralized_configuration_relations": delateralized,
        "support_geometry_publication_revoked": support_invalidated,
        "policy": "relation_truth_before_laterality_and_support_topology_publication",
    }
    return out


_BASE_APPLY_PHASE4B10 = phase4b10._apply_phase4b10_authoritative


def _apply_phase4b11_authoritative(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B10(sheet)
    out = _apply_support_seed_truth_gate(out)

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.11-authoritative"
    audit["specialist_adjudication_phase"] = "4B.11-authoritative"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        ankle_height_seed_cannot_authorize_asymmetric_support_topology=True,
        knee_side_derived_from_weak_planted_foot_is_withheld=True,
        qwen_raised_knee_semantics_can_survive_without_untrusted_side=True,
        direct_observed_knee_laterality_remains_publishable=True,
        planted_foot_derived_from_direct_raised_knee_remains_publishable=True,
        broad_pose_torso_and_ground_contact_gates_remain_authoritative=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    phase4b10.phase4b9.phase4b8.phase4b6.phase4b5.phase4b4._apply_phase4b4 = _apply_phase4b11_authoritative
    phase4b10.phase4b9.phase4b8.phase4b6.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b10.phase4b9.phase4b8.phase4b6.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b10.phase4b9.phase4b8.phase4b6.main()


if __name__ == "__main__":
    raise SystemExit(main())
