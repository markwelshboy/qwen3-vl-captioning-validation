from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v13 as phase4b12
from . import unilateral_raised_leg_shadow_v01 as raised_shadow

SCHEMA_VERSION = "caption-fact-sheet-0.2.13"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.13"

_GENERIC_ONE_KNEE_RAISED_RE = re.compile(
    r"\bone\s+knee\b.{0,24}\braised\b|\braised\b.{0,24}\bone\s+knee\b",
    re.I,
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


def _item_text(item: dict[str, Any]) -> str:
    value = item.get("composer_text") or item.get("normalized_text") or item.get("text")
    return " ".join(str(value).split()) if value is not None else ""


def _specialist_binding(evidence: dict[str, Any], *, source_text: str | None) -> dict[str, Any]:
    side = str(evidence.get("candidate_side") or "")
    side_eval = evidence.get("side_evaluations") if isinstance(evidence.get("side_evaluations"), dict) else {}
    selected = side_eval.get(side) if isinstance(side_eval.get(side), dict) else {}
    metrics = selected.get("metrics") if isinstance(selected.get("metrics"), dict) else {}
    dw = metrics.get("dwpose") if isinstance(metrics.get("dwpose"), dict) else {}
    sam = metrics.get("sam3d") if isinstance(metrics.get("sam3d"), dict) else {}
    return {
        "semantic_relation": "unilateral_raised_knee",
        "canonical_text": evidence.get("proposed_relation"),
        "anatomical_side": side or None,
        "authority": "direct_unilateral_raised_leg_topology_from_dwpose_observability_plus_sam3d_per_leg_geometry",
        "source_text": source_text,
        "dwpose_ankle_elevation_shoulder_widths": dw.get("ankle_elevation_shoulder_widths"),
        "sam3d_knee_flexion_deg": sam.get("knee_flexion_deg"),
        "sam3d_thigh_axis_from_image_down_deg": sam.get("thigh_axis_from_image_down_deg"),
        "sam3d_thigh_horizontal_offset_deg": sam.get("thigh_horizontal_offset_deg"),
        "sam3d_leg_extension_ratio": sam.get("leg_extension_ratio"),
        "thresholds": copy.deepcopy(evidence.get("thresholds")),
    }


def _canonical_item_from_source(item: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    revised = copy.deepcopy(item)
    proposed = str(evidence.get("proposed_relation") or "").strip()
    source_text = _item_text(item) or None
    revised["composer_text"] = proposed
    revised["normalized_text"] = proposed
    revised["promotion_status"] = "accepted_specialist_unilateral_raised_leg_candidate"
    revised["specialist_owner"] = "sam3d_v16_unilateral_raised_leg_specialist"
    revised["unilateral_raised_leg_binding"] = _specialist_binding(evidence, source_text=source_text)
    revised.pop("laterality_binding", None)
    revised.pop("withheld_laterality_binding", None)
    return revised


def _new_canonical_item(evidence: dict[str, Any]) -> dict[str, Any]:
    proposed = str(evidence.get("proposed_relation") or "").strip()
    return {
        "text": proposed,
        "source": "sam3d_v16_unilateral_raised_leg_specialist",
        "domain": "configuration",
        "authority": "specialist_authoritative",
        "promotion_status": "accepted_specialist_unilateral_raised_leg_candidate",
        "composer_text": proposed,
        "normalized_text": proposed,
        "specialist_owner": "sam3d_v16_unilateral_raised_leg_specialist",
        "unilateral_raised_leg_binding": _specialist_binding(evidence, source_text=None),
    }


def _apply_unilateral_raised_leg_authority(
    sheet: dict[str, Any],
    *,
    projected: dict[str, Any] | None = None,
    dwpose_points: dict[str, tuple[float, float] | None] | None = None,
    specialist_error: str | None = None,
    dwpose_error: str | None = None,
) -> dict[str, Any]:
    """Promote only the population-gated unilateral raised-leg relation.

    This stage intentionally describes the distinctive raised leg itself rather
    than publishing the mechanical consequence on the opposite side.  It never
    creates a support-leg, planted-foot, or weight-bearing claim.
    """
    out = copy.deepcopy(sheet)
    body = _body(out)

    specialist = projected
    specialist_load_error = specialist_error
    if specialist is None:
        specialist, specialist_load_error = raised_shadow.sam3d_shadow._load_specialist_projection(out)

    observed = dwpose_points
    observed_load_error = dwpose_error
    if observed is None:
        observed, observed_load_error = raised_shadow._load_dwpose_points(out)

    evidence = raised_shadow.evaluate_shadow(
        image_key=str(out.get("image_key") or "unknown"),
        sheet=out,
        projected=specialist,
        dwpose_points=observed,
        specialist_error=specialist_load_error,
        dwpose_error=observed_load_error,
    )

    shadow_status = str(evidence.get("status") or "not_candidate")
    qualifies = shadow_status in {"candidate_already_present", "candidate_would_change"}
    proposed = str(evidence.get("proposed_relation") or "").strip() or None
    adjudication: dict[str, Any] = {
        "status": "not_applicable",
        "authoritative_stage": True,
        "composer_authoritative": False,
        "applied": False,
        "would_change": False,
        "canonical_relation": proposed if qualifies else None,
        "candidate_side": evidence.get("candidate_side"),
        "shadow_status": shadow_status,
        "reason": evidence.get("reason"),
        "evidence": copy.deepcopy(evidence),
    }

    if not qualifies:
        if shadow_status == "route_abstain":
            adjudication["status"] = "route_abstain"
        elif shadow_status == "insufficient_evidence":
            adjudication["status"] = "insufficient_evidence"
        body["unilateral_raised_leg_adjudication"] = adjudication
        return out

    adjudication["composer_authoritative"] = True
    if shadow_status == "candidate_already_present":
        adjudication.update(
            status="confirmed_existing",
            reason="authoritative_unilateral_raised_leg_relation_already_expressed_in_configuration",
        )
        body["unilateral_raised_leg_adjudication"] = adjudication
        return out

    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    generic_indices = [
        i for i, item in enumerate(configuration)
        if isinstance(item, dict) and _GENERIC_ONE_KNEE_RAISED_RE.search(_item_text(item))
    ]

    before = copy.deepcopy(configuration)
    after: list[Any] = []
    replaced_source: dict[str, Any] | None = None

    if generic_indices:
        first = generic_indices[0]
        generic_set = set(generic_indices)
        for i, item in enumerate(configuration):
            if i == first and isinstance(item, dict):
                replaced_source = copy.deepcopy(item)
                after.append(_canonical_item_from_source(item, evidence))
            elif i in generic_set:
                continue
            else:
                after.append(copy.deepcopy(item))
        action = "replaced_generic_raised_knee_relation"
    else:
        after = copy.deepcopy(configuration)
        after.append(_new_canonical_item(evidence))
        action = "added_authoritative_unilateral_raised_leg_relation"

    body["configuration"] = after
    adjudication.update(
        status="adjudicated",
        applied=True,
        would_change=True,
        action=action,
        configuration_before=before,
        configuration_after=copy.deepcopy(after),
        replaced_source_relation=replaced_source,
        reason="population_gated_unilateral_raised_leg_specialist_authorized_direct_leg_geometry",
        support_leg_claim_created=False,
        planted_foot_claim_created=False,
    )
    body["unilateral_raised_leg_adjudication"] = adjudication
    return out


_BASE_APPLY_PHASE4B12 = phase4b12._apply_phase4b12_authoritative


def _apply_phase4b13_authoritative(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B12(sheet)
    out = _apply_unilateral_raised_leg_authority(out)

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.13-authoritative"
    audit["specialist_adjudication_phase"] = "4B.13-authoritative"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        unilateral_raised_leg_uses_population_gated_direct_leg_topology=True,
        unilateral_raised_leg_publishes_candidate_side_only=True,
        unilateral_raised_leg_does_not_publish_support_leg=True,
        unilateral_raised_leg_does_not_publish_planted_foot=True,
        generic_raised_knee_relation_can_be_replaced_with_specific_geometry=True,
        prior_broad_pose_torso_support_and_bilateral_knee_gates_remain_authoritative=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    # Let v13 keep ownership of the shared writer stack, but substitute its
    # authoritative application function with this successor.  The captured
    # base above still calls the validated Phase-4B.12 implementation first.
    phase4b12._apply_phase4b12_authoritative = _apply_phase4b13_authoritative
    phase4b12.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b12.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b12.main()


if __name__ == "__main__":
    raise SystemExit(main())
