from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v01 as phase4b0
from . import fact_sheet_specialist_normalizer_v05 as phase4b4
from . import fact_sheet_specialist_normalizer_v15 as phase4b14
from . import local_configuration_semantics_shadow_v01 as local_shadow

SCHEMA_VERSION = "caption-fact-sheet-0.2.15"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.1-routed-v0.2"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.15"


# Population-reviewed contradiction gate for direct body-to-face relations.
# In the 87-image census, non-device positive controls were <= 0.72 body
# scales, while the first visually false candidate was 1.533 and the known
# 00002/46/50 failures were 3.488-3.988. The gate is deliberately one-sided:
# geometry may veto an obviously impossible relation but may not create one.
FACE_RELATION_CONTRADICTION_MIN_NORM_BODY = 1.35

_FACE_RELATION_RE = re.compile(
    r"\b(?:hand|fist|palm|fingers?|wrist|forearm)\b.{0,55}\b(?:face|chin)\b"
    r"|\b(?:face|chin)\b.{0,55}\b(?:hand|fist|palm|fingers?|wrist|forearm)\b",
    re.I,
)
_DEVICE_MEDIATED_FACE_RELATION_RE = re.compile(
    r"\b(?:phone|smartphone|mobile|device|camera)\b",
    re.I,
)
_NEGATED_FACE_RELATION_RE = re.compile(
    r"\bno\s+(?:visible\s+)?(?:hand|fist|palm|fingers?|wrist|forearm)\b"
    r".{0,55}\b(?:contact|touch(?:es|ing)?|support(?:s|ing|ed)?)\b"
    r".{0,55}\b(?:face|chin)\b"
    r"|\b(?:hand|fist|palm|fingers?|wrist|forearm)\b.{0,40}"
    r"\b(?:does|do)\s+not\s+(?:touch|contact|support)\b"
    r"|\b(?:hand|fist|palm|fingers?|wrist|forearm)\b.{0,40}"
    r"\b(?:is|are)\s+not\s+(?:touching|contacting|supporting)\b",
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


def _positive_direct_face_relation(text: str) -> bool:
    """Return True only for direct positive body-to-face/chin relation claims."""
    value = " ".join(str(text or "").split())
    if not value or not _FACE_RELATION_RE.search(value):
        return False
    if _DEVICE_MEDIATED_FACE_RELATION_RE.search(value):
        return False
    if _NEGATED_FACE_RELATION_RE.search(value):
        return False
    return True


def _face_relation_distance_norm(
    points: dict[str, tuple[float, float] | None],
) -> tuple[float | None, str | None]:
    """Use the same forearm-aware metric validated by the shadow census."""
    from . import local_relation_geometry_shadow_v01 as relation_shadow

    geometry = relation_shadow._geometry(points)
    value = geometry.get("nearest_upper_limb_to_face_norm_body")
    side = geometry.get("nearest_upper_limb_to_face_side")
    if not isinstance(value, (int, float)):
        return None, None
    return float(value), str(side) if side in {"left", "right"} else None


def _apply_face_relation_truth_gate(
    sheet: dict[str, Any],
    *,
    points: dict[str, tuple[float, float] | None] | None = None,
    geometry_error: str | None = None,
) -> dict[str, Any]:
    """Withhold only directly contradicted body-to-face/chin relations.

    Qwen remains the semantic proposer. DWPose is used only as a negative
    truth gate: if every observed hand/forearm candidate is far from the face,
    the relation is withheld from the composer. Missing geometry is an
    abstention, never counterevidence. Device-mediated relations are outside
    this anatomical relation gate and are preserved.
    """
    out = copy.deepcopy(sheet)
    body = _body(out)
    configuration = (
        body.get("configuration")
        if isinstance(body.get("configuration"), list)
        else []
    )

    candidate_indexes = [
        index
        for index, item in enumerate(configuration)
        if isinstance(item, dict) and _positive_direct_face_relation(_item_text(item))
    ]
    device_context_relations = [
        _item_text(item)
        for item in configuration
        if isinstance(item, dict)
        and _FACE_RELATION_RE.search(_item_text(item))
        and _DEVICE_MEDIATED_FACE_RELATION_RE.search(_item_text(item))
    ]
    adjudication: dict[str, Any] = {
        "status": "not_applicable",
        "authoritative_stage": True,
        "composer_authoritative": False,
        "applied": False,
        "candidate_count": len(candidate_indexes),
        "withheld_count": 0,
        "threshold_norm_body": FACE_RELATION_CONTRADICTION_MIN_NORM_BODY,
        "nearest_upper_limb_to_face_norm_body": None,
        "nearest_upper_limb_side": None,
        "geometry_error": geometry_error,
        "reason": "no_direct_positive_body_to_face_relation_candidate",
        "device_mediated_relations_are_out_of_scope": True,
        "device_context_relations": copy.deepcopy(device_context_relations),
        "missing_geometry_is_not_counterevidence": True,
    }

    if not candidate_indexes:
        body["face_relation_truth_adjudication"] = adjudication
        return out

    # A coexisting device-mediated face relation (for example, a hand holding
    # a phone in front of the face) makes 2-D hand/forearm distance a poor
    # contradiction signal for a generic companion relation. The hand may be
    # partially occluded by the device and DWPose can localize the visible
    # wrist/forearm away from the semantic device-hand configuration. In that
    # case, preserve the Qwen relation rather than converting uncertain
    # geometry into counterevidence.
    if device_context_relations:
        adjudication.update(
            status="preserved_device_context",
            reason="coexisting_device_mediated_face_relation_disables_direct_distance_veto",
        )
        body["face_relation_truth_adjudication"] = adjudication
        return out

    resolved_points = points
    point_error = geometry_error
    if resolved_points is None:
        resolved_points, point_error = phase4b4._load_dwpose_points_for_sheet(out)

    if resolved_points is None:
        adjudication.update(
            status="insufficient_evidence",
            geometry_error=point_error,
            reason=point_error or "dwpose_geometry_unavailable",
        )
        body["face_relation_truth_adjudication"] = adjudication
        return out

    distance_norm, side = _face_relation_distance_norm(resolved_points)
    adjudication.update(
        nearest_upper_limb_to_face_norm_body=distance_norm,
        nearest_upper_limb_side=side,
        geometry_error=point_error,
    )
    if distance_norm is None:
        adjudication.update(
            status="insufficient_evidence",
            reason="visible_upper_limb_to_face_distance_unavailable",
        )
        body["face_relation_truth_adjudication"] = adjudication
        return out

    if distance_norm <= FACE_RELATION_CONTRADICTION_MIN_NORM_BODY:
        adjudication.update(
            status="preserved",
            reason="visible_upper_limb_geometry_does_not_contradict_face_relation",
        )
        body["face_relation_truth_adjudication"] = adjudication
        return out

    updated = copy.deepcopy(configuration)
    withheld: list[dict[str, Any]] = []
    for index in candidate_indexes:
        item = updated[index]
        source_text = _item_text(item)
        previous_composer = item.get("composer_text")
        item["composer_text"] = None
        item["promotion_status"] = "withheld_by_face_relation_geometry_contradiction"
        item["specialist_owner"] = "dwpose_face_relation_truth_gate"
        item["face_relation_truth_gate"] = {
            "semantic_relation": "hand_or_forearm_near_face_or_chin",
            "decision": "withhold",
            "source_text": source_text,
            "previous_composer_text": previous_composer,
            "nearest_upper_limb_to_face_norm_body": round(distance_norm, 3),
            "nearest_upper_limb_side": side,
            "contradiction_threshold_norm_body": (
                FACE_RELATION_CONTRADICTION_MIN_NORM_BODY
            ),
            "authority": "dwpose_visible_upper_limb_to_face_geometry",
            "reason": (
                "all observed hand/forearm geometry is too far from the face "
                "for the proposed direct anatomical relation"
            ),
        }
        withheld.append(copy.deepcopy(item["face_relation_truth_gate"]))

    body["configuration"] = updated
    adjudication.update(
        status="adjudicated",
        composer_authoritative=True,
        applied=True,
        withheld_count=len(withheld),
        withheld_relations=withheld,
        reason="visible_dwpose_geometry_strongly_contradicts_direct_face_relation",
    )
    body["face_relation_truth_adjudication"] = adjudication
    return out

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
    out = _apply_face_relation_truth_gate(out)

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
        direct_face_relation_truth_gate_is_contradiction_only=True,
        direct_face_relation_truth_gate_never_creates_relation=True,
        missing_face_relation_geometry_is_not_counterevidence=True,
        device_mediated_face_relations_are_outside_anatomical_truth_gate=True,
        coexisting_device_face_relation_disables_direct_distance_veto=True,
        prior_broad_pose_torso_support_knee_raised_leg_and_crouch_depth_gates_remain_authoritative=True,
    )
    face_gate = (
        _body(out).get("face_relation_truth_adjudication")
        if isinstance(_body(out).get("face_relation_truth_adjudication"), dict)
        else {}
    )
    if face_gate.get("applied") and face_gate.get("withheld_count"):
        warnings = [str(value) for value in (audit.get("warnings") or []) if value]
        warnings.append("face_relation_geometry_contradiction_withheld")
        audit["warnings"] = sorted(set(warnings))
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
