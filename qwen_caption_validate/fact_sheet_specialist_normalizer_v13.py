from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import bilateral_knee_flexion_shadow_v01 as knee_shadow
from . import fact_sheet_specialist_normalizer_v12 as phase4b11

SCHEMA_VERSION = "caption-fact-sheet-0.2.12"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.12"

# Keep the shadow-gated thresholds frozen for the first authoritative rollout.
MAX_DEEP_KNEE_ANGLE_DEG = knee_shadow.MAX_DEEP_KNEE_ANGLE_DEG
MIN_KNEE_REGION_AUTHORITY = knee_shadow.MIN_KNEE_REGION_AUTHORITY

_ONE_KNEE_BENT_RE = re.compile(
    r"\bone\s+knee\b.{0,24}\b(?:bent|flexed)\b",
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
    dw = evidence.get("dwpose") if isinstance(evidence.get("dwpose"), dict) else {}
    sam = evidence.get("sam3d_v16") if isinstance(evidence.get("sam3d_v16"), dict) else {}
    return {
        "semantic_relation": "bilateral_knee_flexion",
        "canonical_text": "both knees bent",
        "authority": "observed_dwpose_bilateral_leg_chains_plus_high_authority_sam3d_v16_knee_flexion",
        "source_text": source_text,
        "dwpose_both_full_chains_observed": dw.get("both_full_chains_observed"),
        "dwpose_left_knee_angle_2d_deg": ((dw.get("left") or {}).get("knee_angle_2d_deg") if isinstance(dw.get("left"), dict) else None),
        "dwpose_right_knee_angle_2d_deg": ((dw.get("right") or {}).get("knee_angle_2d_deg") if isinstance(dw.get("right"), dict) else None),
        "sam3d_left_knee_angle_deg": sam.get("left_knee_angle_deg"),
        "sam3d_right_knee_angle_deg": sam.get("right_knee_angle_deg"),
        "sam3d_knee_region_authority": sam.get("knee_region_authority"),
        "sam3d_knee_region_authority_percent": sam.get("knee_region_authority_percent"),
        "max_deep_knee_angle_deg": MAX_DEEP_KNEE_ANGLE_DEG,
        "min_knee_region_authority": MIN_KNEE_REGION_AUTHORITY,
    }


def _canonical_item_from_source(item: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    revised = copy.deepcopy(item)
    source_text = _item_text(item) or None
    revised["composer_text"] = "both knees bent"
    revised["normalized_text"] = "both knees bent"
    revised["promotion_status"] = "accepted_specialist_bilateral_knee_flexion_candidate"
    revised["specialist_owner"] = "sam3d_v16_bilateral_knee_flexion_specialist"
    revised["bilateral_knee_flexion_binding"] = _specialist_binding(
        evidence,
        source_text=source_text,
    )
    return revised


def _new_canonical_item(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": "both knees bent",
        "source": "sam3d_v16_bilateral_knee_flexion_specialist",
        "domain": "configuration",
        "authority": "specialist_authoritative",
        "promotion_status": "accepted_specialist_bilateral_knee_flexion_candidate",
        "composer_text": "both knees bent",
        "normalized_text": "both knees bent",
        "specialist_owner": "sam3d_v16_bilateral_knee_flexion_specialist",
        "bilateral_knee_flexion_binding": _specialist_binding(evidence, source_text=None),
    }


def _apply_bilateral_knee_flexion_authority(
    sheet: dict[str, Any],
    *,
    projected: dict[str, Any] | None = None,
    dwpose_points: dict[str, tuple[float, float] | None] | None = None,
    specialist_error: str | None = None,
    dwpose_error: str | None = None,
) -> dict[str, Any]:
    """Promote the population-gated bilateral-knee relation.

    Publication requires the exact shadow gate that passed the 87-image audit:
    both full DWPose leg chains observed, both SAM3D-v16 knee angles <=135 deg,
    and knee-region authority >=0.80.  Existing correct bilateral wording is
    preserved verbatim.  A generic singular relation such as "one knee slightly
    bent" is normalized to "both knees bent".  If no knee-bend relation exists,
    the specialist may add the canonical bilateral relation while preserving
    unrelated knee semantics.
    """
    out = copy.deepcopy(sheet)
    body = _body(out)

    specialist = projected
    specialist_load_error = specialist_error
    if specialist is None:
        specialist, specialist_load_error = knee_shadow.sam3d_shadow._load_specialist_projection(out)

    observed = dwpose_points
    observed_load_error = dwpose_error
    if observed is None:
        observed, observed_load_error = knee_shadow._load_dwpose_observation(out)

    evidence = knee_shadow.evaluate_shadow(
        image_key=str(out.get("image_key") or "unknown"),
        sheet=out,
        projected=specialist,
        dwpose_points=observed,
        specialist_error=specialist_load_error,
        dwpose_error=observed_load_error,
    )

    shadow_status = str(evidence.get("status") or "not_candidate")
    qualifies = shadow_status in {"candidate_already_present", "candidate_would_change"}
    adjudication: dict[str, Any] = {
        "status": "not_applicable",
        "authoritative_stage": True,
        "composer_authoritative": False,
        "applied": False,
        "would_change": False,
        "canonical_relation": "both knees bent" if qualifies else None,
        "shadow_status": shadow_status,
        "reason": evidence.get("reason"),
        "evidence": copy.deepcopy(evidence),
    }

    if not qualifies:
        if shadow_status == "route_abstain":
            adjudication["status"] = "route_abstain"
        elif shadow_status == "insufficient_evidence":
            adjudication["status"] = "insufficient_evidence"
        body["bilateral_knee_flexion_adjudication"] = adjudication
        return out

    adjudication["composer_authoritative"] = True
    if shadow_status == "candidate_already_present":
        adjudication.update(
            status="confirmed_existing",
            reason="authoritative_bilateral_knee_flexion_already_expressed_in_configuration",
        )
        body["bilateral_knee_flexion_adjudication"] = adjudication
        return out

    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    singular_indices = [
        i for i, item in enumerate(configuration)
        if isinstance(item, dict) and _ONE_KNEE_BENT_RE.search(_item_text(item))
    ]

    before = copy.deepcopy(configuration)
    after: list[Any] = []
    replaced_source: dict[str, Any] | None = None

    if singular_indices:
        first = singular_indices[0]
        singular_set = set(singular_indices)
        for i, item in enumerate(configuration):
            if i == first and isinstance(item, dict):
                replaced_source = copy.deepcopy(item)
                after.append(_canonical_item_from_source(item, evidence))
            elif i in singular_set:
                # Collapse duplicate generic singular bend statements into the
                # one canonical bilateral relation.
                continue
            else:
                after.append(copy.deepcopy(item))
        action = "replaced_under_specified_singular_knee_bend_relation"
    else:
        after = copy.deepcopy(configuration)
        after.append(_new_canonical_item(evidence))
        action = "added_authoritative_bilateral_knee_bend_relation"

    body["configuration"] = after
    adjudication.update(
        status="adjudicated",
        applied=True,
        would_change=True,
        action=action,
        configuration_before=before,
        configuration_after=copy.deepcopy(after),
        replaced_source_relation=replaced_source,
        reason="population_gated_bilateral_knee_flexion_specialist_authorized_canonical_relation",
    )
    body["bilateral_knee_flexion_adjudication"] = adjudication
    return out


_BASE_APPLY_PHASE4B11 = phase4b11._apply_phase4b11_authoritative


def _apply_phase4b12_authoritative(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B11(sheet)
    out = _apply_bilateral_knee_flexion_authority(out)

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.12-authoritative"
    audit["specialist_adjudication_phase"] = "4B.12-authoritative"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        bilateral_knee_flexion_requires_both_observed_dwpose_leg_chains=True,
        bilateral_knee_flexion_requires_high_sam3d_v16_knee_authority=True,
        bilateral_knee_flexion_reuses_existing_135deg_deep_flexion_boundary=True,
        existing_correct_bilateral_knee_language_is_preserved=True,
        under_specified_singular_knee_bend_can_be_replaced=True,
        support_contact_and_support_topology_truth_gates_remain_authoritative=True,
        broad_pose_and_torso_adjudication_remain_authoritative=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    # Reuse the validated v12 authoritative stack, then append only the
    # population-gated bilateral-knee normalization before the v07 writer emits
    # the next schema.
    phase4b11.phase4b10.phase4b9.phase4b8.phase4b6.phase4b5.phase4b4._apply_phase4b4 = _apply_phase4b12_authoritative
    phase4b11.phase4b10.phase4b9.phase4b8.phase4b6.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b11.phase4b10.phase4b9.phase4b8.phase4b6.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b11.phase4b10.phase4b9.phase4b8.phase4b6.main()


if __name__ == "__main__":
    raise SystemExit(main())
