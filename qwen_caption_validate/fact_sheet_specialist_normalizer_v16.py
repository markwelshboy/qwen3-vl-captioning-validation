from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v01 as phase4b0
from . import fact_sheet_specialist_normalizer_v15 as phase4b14
from . import local_configuration_semantics_shadow_v01 as local_shadow

SCHEMA_VERSION = "caption-fact-sheet-0.2.15"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.1-routed-v0.2"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.15"


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


def _points_from_authoritative_laterality(body: dict[str, Any]) -> dict[str, tuple[float, float] | None] | None:
    """Project the already-authoritative DWPose visibility fact into the shadow binder shape.

    Coordinates are deliberately not reconstructed here.  The validated head-support
    side gate only needs observed shoulder/elbow/wrist presence per anatomical side.
    """
    laterality = (
        body.get("anatomical_laterality")
        if isinstance(body.get("anatomical_laterality"), dict)
        else {}
    )
    if not laterality.get("available"):
        return None
    sides = laterality.get("sides") if isinstance(laterality.get("sides"), dict) else {}
    points: dict[str, tuple[float, float] | None] = {}
    any_observed = False
    for side in ("left", "right"):
        side_record = sides.get(side) if isinstance(sides.get(side), dict) else {}
        visible = {str(v) for v in (side_record.get("visible_joints") or [])}
        for part in ("shoulder", "elbow", "wrist"):
            name = f"{side}_{part}"
            observed = part in visible
            points[name] = (0.0, 0.0) if observed else None
            any_observed = any_observed or observed
    return points if any_observed else None


def _canonical_head_support_item(
    semantics: dict[str, Any],
    binding: dict[str, Any],
    *,
    source_items: list[dict[str, Any]],
) -> dict[str, Any]:
    side = binding.get("anatomical_side") if binding.get("status") == "bound" else None
    noun = "fist" if semantics.get("uses_fist_semantics") else "hand"

    composer_text = str(semantics.get("canonical_side_neutral") or "").strip()
    if side:
        composer_text = f"chin resting on the {side} {noun}"
        if semantics.get("forearm_support_explicit"):
            if binding.get("forearm_side_publishable"):
                composer_text += f", with the {side} forearm beneath/supporting the pose"
            else:
                composer_text += ", with the forearm beneath/supporting the pose"

    return {
        "text": composer_text,
        "source": "routed_body_qwen_plus_dwpose_laterality",
        "domain": "configuration",
        "authority": "route_scoped_candidate_with_deterministic_laterality_binding",
        "promotion_status": "accepted_canonical_head_support_candidate",
        "composer_text": composer_text,
        "normalized_text": composer_text,
        "specialist_owner": "local_configuration_head_support_v01",
        "head_support_binding": {
            "semantic_relation": "chin_supported_by_hand_or_fist",
            "canonical_side_neutral": semantics.get("canonical_side_neutral"),
            "uses_fist_semantics": bool(semantics.get("uses_fist_semantics")),
            "forearm_support_explicit": bool(semantics.get("forearm_support_explicit")),
            "anatomical_side": side,
            "laterality_authority": binding.get("authority"),
            "laterality_reason": binding.get("reason"),
            "forearm_side_publishable": bool(binding.get("forearm_side_publishable")),
            "observed_arm_chains": copy.deepcopy(binding.get("observed_arm_chains")),
            "source_relations": copy.deepcopy(source_items),
        },
        "note": (
            "Visible head-support semantics were consolidated from route-scoped Qwen "
            "relationships. Anatomical hand/fist side is published only when the "
            "authoritative DWPose visibility gate binds a single observed wrist; Qwen "
            "left/right is never trusted."
        ),
    }


def _apply_head_support_authority(sheet: dict[str, Any]) -> dict[str, Any]:
    """Consolidate visible head support and bind only the deterministically supported side.

    This stage cannot create broad pose, support-foot/contact, gaze, head orientation,
    or a forearm side that is not independently observed.
    """
    out = copy.deepcopy(sheet)
    body = _body(out)
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []

    texts: list[str] = []
    item_by_text: dict[str, list[dict[str, Any]]] = {}
    for raw in configuration:
        if not isinstance(raw, dict):
            continue
        text = _item_text(raw)
        if not text:
            continue
        texts.append(text)
        item_by_text.setdefault(text, []).append(raw)

    semantics = local_shadow._head_support_semantics(texts)
    adjudication: dict[str, Any] = {
        "status": "not_applicable",
        "authoritative_stage": True,
        "composer_authoritative": False,
        "applied": False,
        "semantic_relation": None,
        "canonical_side_neutral": semantics.get("canonical_side_neutral"),
        "source_relation_texts": copy.deepcopy(semantics.get("source_relations") or []),
        "forearm_source_relation_texts": copy.deepcopy(semantics.get("forearm_source_relations") or []),
        "reason": "no_visible_head_support_semantic_candidate",
        "broad_pose_created": False,
        "support_contact_created": False,
        "gaze_created": False,
        "head_orientation_created": False,
    }

    if semantics.get("status") != "candidate":
        body["head_support_adjudication"] = adjudication
        return out

    points = _points_from_authoritative_laterality(body)
    binding = local_shadow._head_support_side_binding(points, semantics)

    source_texts = {str(v) for v in (semantics.get("source_relations") or [])}
    source_items = [
        copy.deepcopy(item)
        for item in configuration
        if isinstance(item, dict) and _item_text(item) in source_texts
    ]
    if not source_items:
        adjudication.update(
            status="needs_review",
            reason="head_support_semantic_candidate_has_no_matching_configuration_source",
            laterality_binding=copy.deepcopy(binding),
        )
        body["head_support_adjudication"] = adjudication
        return out

    canonical = _canonical_head_support_item(semantics, binding, source_items=source_items)

    before = copy.deepcopy(configuration)
    after: list[Any] = []
    inserted = False
    for item in configuration:
        if isinstance(item, dict) and _item_text(item) in source_texts:
            if not inserted:
                after.append(canonical)
                inserted = True
            continue
        after.append(copy.deepcopy(item))

    body["configuration"] = after
    adjudication.update(
        status="adjudicated",
        composer_authoritative=True,
        applied=True,
        semantic_relation="chin_supported_by_hand_or_fist",
        canonical_composer_text=canonical.get("composer_text"),
        laterality_binding=copy.deepcopy(binding),
        configuration_before=before,
        configuration_after=copy.deepcopy(after),
        reason="route_scoped_head_support_semantics_consolidated_with_authoritative_dwpose_side_binding",
        qwen_laterality_trusted=False,
        forearm_side_published=bool(binding.get("forearm_side_publishable") and binding.get("status") == "bound"),
    )
    body["head_support_adjudication"] = adjudication
    return out


_BASE_APPLY_PHASE4B14 = phase4b14._apply_phase4b14_authoritative


def _apply_phase4b15_authoritative(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B14(sheet)
    out = _apply_head_support_authority(out)

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.15-authoritative"
    audit["specialist_adjudication_phase"] = "4B.15-authoritative"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        head_support_semantics_require_route_scoped_visible_configuration=True,
        head_support_duplicates_are_consolidated=True,
        qwen_head_support_laterality_is_never_trusted=True,
        head_support_hand_side_requires_single_observed_dwpose_wrist_plus_proximal_arm=True,
        head_support_forearm_side_requires_same_side_elbow_and_wrist=True,
        generic_forearm_held_language_does_not_create_head_support=True,
        head_support_stage_cannot_create_broad_pose=True,
        prior_broad_pose_torso_support_knee_raised_leg_and_crouch_depth_gates_remain_authoritative=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    # The v15 execution stack still owns the validated Phase-4B writer and all
    # previous specialist gates.  Point its Phase-4A/policy inputs at the promoted
    # routing-v0.2 artifacts, then substitute only this successor application.
    old_input = phase4b0.DEFAULT_INPUT_SUBDIR
    old_policy = phase4b0.DEFAULT_POLICY_SUBDIR
    old_apply = phase4b14._apply_phase4b14_authoritative
    old_schema = phase4b14.SCHEMA_VERSION
    old_output = phase4b14.DEFAULT_OUTPUT_SUBDIR
    try:
        phase4b0.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
        phase4b0.DEFAULT_POLICY_SUBDIR = DEFAULT_POLICY_SUBDIR
        phase4b14._apply_phase4b14_authoritative = _apply_phase4b15_authoritative
        phase4b14.SCHEMA_VERSION = SCHEMA_VERSION
        phase4b14.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
        return phase4b14.main()
    finally:
        phase4b0.DEFAULT_INPUT_SUBDIR = old_input
        phase4b0.DEFAULT_POLICY_SUBDIR = old_policy
        phase4b14._apply_phase4b14_authoritative = old_apply
        phase4b14.SCHEMA_VERSION = old_schema
        phase4b14.DEFAULT_OUTPUT_SUBDIR = old_output


if __name__ == "__main__":
    raise SystemExit(main())
