from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v10 as phase4b9

SCHEMA_VERSION = "caption-fact-sheet-0.2.10"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.10"

# DWPose BODY18 gives us ankle positions, not a reliable visible floor-contact
# test.  A bilateral hip->ankle height difference may help bind an already
# validated relation to a side, but it must not independently validate the
# semantic claim that a foot has left the floor.
ANKLE_HEIGHT_ONLY_AUTHORITY = "dwpose_bilateral_hip_ankle_relative_height"


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


def _semantic_relation(item: dict[str, Any]) -> str:
    return str(_binding(item).get("semantic_relation") or "")


def _binding_authority(item: dict[str, Any]) -> str:
    return str(_binding(item).get("authority") or "")


def _apply_support_contact_truth_gate(sheet: dict[str, Any]) -> dict[str, Any]:
    """Default-deny unsupported asymmetric foot-contact claims.

    The upstream Qwen body acquisition is allowed to *propose* `foot_lifted`
    and `foot_planted`.  Existing v06/v07 code then binds those relations to
    left/right using DWPose ankle geometry.  That is useful for laterality, but
    ankle height alone does not establish loss of floor contact.

    Therefore:
      * every `foot_lifted` relation whose only specialist authority is the
        bilateral ankle-height heuristic is withheld from canonical config;
      * the paired `foot_planted` relation from the same heuristic is withheld
        too, because together they encode an unsupported asymmetric support
        topology;
      * planted-foot relations derived from an independently bound raised-knee
        chain are preserved (e.g. 00064);
      * raw removed relations and support diagnostics remain available for
        provenance/audit.
    """
    out = copy.deepcopy(sheet)
    body = _body(out)
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []

    lifted = [
        item for item in configuration
        if isinstance(item, dict)
        and _semantic_relation(item) == "foot_lifted"
    ]

    if not lifted:
        body["support_contact_adjudication"] = {
            "status": "not_applicable",
            "authoritative_stage": True,
            "composer_authoritative": False,
            "applied": False,
            "reason": "no_asymmetric_foot_lifted_relation_present",
        }
        return out

    # Current DWPose foot-lifted publication paths are all ankle-height-only.
    # If a future specialist adds a genuine ground-contact validator, do not
    # silently suppress it here: only the known weak authority is default-denied.
    weak_lifted = [item for item in lifted if _binding_authority(item) == ANKLE_HEIGHT_ONLY_AUTHORITY]
    if not weak_lifted:
        body["support_contact_adjudication"] = {
            "status": "defer_to_stronger_specialist",
            "authoritative_stage": True,
            "composer_authoritative": False,
            "applied": False,
            "reason": "foot_lifted_relation_has_non_ankle_height_authority",
        }
        return out

    weak_planted = [
        item for item in configuration
        if isinstance(item, dict)
        and _semantic_relation(item) == "foot_planted"
        and _binding_authority(item) == ANKLE_HEIGHT_ONLY_AUTHORITY
    ]

    remove_ids = {id(item) for item in weak_lifted + weak_planted}
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in configuration:
        if isinstance(item, dict) and id(item) in remove_ids:
            removed.append(copy.deepcopy(item))
        else:
            kept.append(copy.deepcopy(item))
    body["configuration"] = kept

    # Preserve the old diagnostic, but explicitly revoke composer/publication
    # eligibility when its support topology was built from the now-withheld
    # foot-lifted pair.
    support = body.get("support_geometry") if isinstance(body.get("support_geometry"), dict) else None
    support_invalidated = False
    if support and str(support.get("elevated_relation") or "") == "foot_lifted":
        support["composer_eligible"] = False
        support["publication_status"] = "withheld_unverified_ground_contact"
        support["note_phase4b10"] = (
            "Ankle-height asymmetry is retained as diagnostic geometry only; it does not establish that a foot is off the floor."
        )
        support_invalidated = True

    laterality = body.get("anatomical_laterality") if isinstance(body.get("anatomical_laterality"), dict) else {}
    bindings = laterality.get("relation_bindings") if isinstance(laterality.get("relation_bindings"), list) else []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        relation = str(binding.get("semantic_relation") or "")
        authority = str(binding.get("authority") or "")
        if authority == ANKLE_HEIGHT_ONLY_AUTHORITY and relation in {"foot_lifted", "foot_planted"}:
            binding["publication_status"] = "withheld_unverified_ground_contact"
            binding["semantic_truth_validated"] = False
    if bindings:
        laterality["relation_bindings"] = bindings
        body["anatomical_laterality"] = laterality

    body["support_contact_adjudication"] = {
        "status": "withheld_unverified_ground_contact",
        "authoritative_stage": True,
        "composer_authoritative": True,
        "applied": bool(removed),
        "would_change": bool(removed),
        "reason": (
            "dwpose_ankle_height_can_bind_relative_side_geometry_but_cannot_validate_loss_of_floor_contact"
        ),
        "removed_configuration_relations": removed,
        "support_geometry_publication_revoked": support_invalidated,
        "policy": "relation_truth_before_laterality_publication",
    }
    return out


_BASE_APPLY_PHASE4B9 = phase4b9._apply_phase4b9_authoritative


def _apply_phase4b10_authoritative(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B9(sheet)
    out = _apply_support_contact_truth_gate(out)

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.10-authoritative"
    audit["specialist_adjudication_phase"] = "4B.10-authoritative"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        qwen_foot_contact_language_is_semantic_proposal_not_truth=True,
        relation_truth_precedes_laterality_publication=True,
        ankle_height_does_not_validate_ground_contact=True,
        ankle_height_only_foot_lifted_relations_are_withheld=True,
        paired_ankle_height_only_planted_relations_are_withheld=True,
        raised_knee_supported_planted_relations_are_preserved=True,
        broad_pose_and_torso_adjudication_remain_authoritative=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    # Reuse the validated v10 broad-pose/torso promotion, then add only the
    # support-contact truth gate before the v07 writer emits the next schema.
    phase4b9.phase4b8.phase4b6.phase4b5.phase4b4._apply_phase4b4 = _apply_phase4b10_authoritative
    phase4b9.phase4b8.phase4b6.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b9.phase4b8.phase4b6.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b9.phase4b8.phase4b6.main()


if __name__ == "__main__":
    raise SystemExit(main())
