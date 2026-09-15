from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v09 as phase4b8

SCHEMA_VERSION = "caption-fact-sheet-0.2.9"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.9"

# v07 historically fused a local torso relation with a global support-shape
# clause. Broad pose is now owned independently by the SAM3D-v16 adjudicator,
# so canonical torso relations must remain local.
_GLOBAL_UPRIGHT_SUFFIX_RE = re.compile(
    r"\s+while\s+overall\s+stance\s+remains\s+mostly\s+upright\b[\s.,;:!?-]*$",
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


def _text_for_item(item: dict[str, Any]) -> str:
    for key in ("composer_text", "normalized_text", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_forward_torso_item(item: dict[str, Any]) -> bool:
    for key in ("text", "composer_text", "normalized_text"):
        value = item.get(key)
        if isinstance(value, str) and phase4b8.phase4b6.FORWARD_TORSO_RE.search(value):
            return True
    return False


def _localize_forward_text(text: str) -> str:
    localized = _GLOBAL_UPRIGHT_SUFFIX_RE.sub("", text.strip())
    return " ".join(localized.split())


def _localize_forward_configuration(
    configuration: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out = copy.deepcopy(configuration)
    changes: list[dict[str, Any]] = []
    for item in out:
        if not isinstance(item, dict) or not _is_forward_torso_item(item):
            continue
        before = _text_for_item(item)
        if not before:
            continue
        after = _localize_forward_text(before)
        if not after or after == before:
            continue
        item["composer_text"] = after
        item["normalized_text"] = after
        item["promotion_status"] = "accepted_specialist_local_torso_candidate"
        item["specialist_owner"] = "sam3d_v16_torso_relation_adjudicator"
        changes.append({"before": before, "after": after})
    return out, changes


def _promote_broad_pose(body: dict[str, Any]) -> None:
    adjudication = body.get("broad_pose_adjudication")
    if not isinstance(adjudication, dict):
        return

    adjudication["shadow_only"] = False
    adjudication["authoritative_stage"] = True
    adjudication["applied"] = False

    status = str(adjudication.get("status") or "")
    if status == "confirmed":
        adjudication["composer_authoritative"] = True
        pose = body.get("pose_candidate")
        if isinstance(pose, dict):
            canonical = pose.get("composer_text") or pose.get("normalized_text") or pose.get("text")
            adjudication["canonical_pose_text"] = canonical
        return

    if not (
        status == "adjudicated"
        and adjudication.get("would_change") is True
        and isinstance(adjudication.get("proposed_pose_family"), str)
        and adjudication.get("proposed_pose_family")
    ):
        adjudication["composer_authoritative"] = False
        return

    proposed = str(adjudication["proposed_pose_family"]).strip()
    existing = body.get("pose_candidate")
    original = copy.deepcopy(existing) if isinstance(existing, dict) else None

    if isinstance(existing, dict):
        candidate = copy.deepcopy(existing)
    else:
        candidate = {
            "source": "sam3d_v16",
            "domain": "pose",
        }

    # Phase-4B.9 is the canonical post-specialist fact sheet.  Once v09 has
    # enough authority to replace broad pose, no consumer-facing text field may
    # retain the contradicted source wording.  Preserve the complete original
    # candidate in broad_pose_adjudication_binding below for provenance.
    candidate["text"] = proposed
    candidate["composer_text"] = proposed
    candidate["normalized_text"] = proposed
    candidate["authority"] = "sam3d_v16_observed_pose_specialist"
    candidate["promotion_status"] = "accepted_specialist_adjudicated_candidate"
    candidate["specialist_owner"] = "sam3d_v16_broad_pose_adjudicator"
    candidate["broad_pose_adjudication_binding"] = {
        "source_pose_candidate": original,
        "specialist_public_pose": (adjudication.get("specialist") or {}).get("public_pose")
        if isinstance(adjudication.get("specialist"), dict)
        else None,
        "assertion_path": (adjudication.get("specialist") or {}).get("assertion_path")
        if isinstance(adjudication.get("specialist"), dict)
        else None,
        "reason": adjudication.get("reason"),
    }
    body["pose_candidate"] = candidate

    adjudication["composer_authoritative"] = True
    adjudication["applied"] = True
    adjudication["canonical_pose_text"] = proposed


def _promote_torso_relation(body: dict[str, Any]) -> None:
    adjudication = body.get("torso_relation_adjudication")
    if not isinstance(adjudication, dict):
        return

    adjudication["shadow_only"] = False
    adjudication["authoritative_stage"] = True
    adjudication["applied"] = False

    status = str(adjudication.get("status") or "")
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []

    if status == "contradicted" and adjudication.get("would_change") is True:
        kept: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        for item in configuration:
            if isinstance(item, dict) and _is_forward_torso_item(item):
                removed.append(copy.deepcopy(item))
            else:
                kept.append(copy.deepcopy(item))
        body["configuration"] = kept
        adjudication["composer_authoritative"] = True
        adjudication["applied"] = bool(removed)
        adjudication["removed_configuration_relations"] = removed
        if not removed:
            adjudication["application_warning"] = "contradicted_forward_relation_not_found_during_canonical_mutation"
        return

    if status == "supported":
        localized, changes = _localize_forward_configuration(configuration)
        body["configuration"] = localized
        adjudication["composer_authoritative"] = True
        adjudication["applied"] = bool(changes)
        adjudication["localization_changes"] = changes
        return

    adjudication["composer_authoritative"] = False


def _apply_authoritative_adjudication(
    sheet: dict[str, Any],
    *,
    projected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the v09 decision logic, then promote only the population-gated decisions.

    Replacement authority remains deliberately narrow:
      * broad pose changes only through v09's observed crouch hip+knee path;
      * contradicted forward-torso relations are removed;
      * supported forward-torso relations are retained but kept local.

    Support/elevation relations are intentionally untouched here; they are a
    separate specialist problem and remain available for the next phase.
    """
    out = phase4b8._apply_shadow_adjudication(sheet, projected=projected)
    body = _body(out)
    _promote_broad_pose(body)
    _promote_torso_relation(body)
    return out


_BASE_APPLY_PHASE4B4 = phase4b8._BASE_APPLY_PHASE4B4


def _apply_phase4b9_authoritative(sheet: dict[str, Any]) -> dict[str, Any]:
    out = _BASE_APPLY_PHASE4B4(sheet)
    out = _apply_authoritative_adjudication(out)

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.9-authoritative"
    audit["specialist_adjudication_phase"] = "4B.9-authoritative"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        sam3d_v16_broad_pose_adjudication_is_authoritative=True,
        observed_crouch_replacement_requires_hip_knee_assertion_path=True,
        contradicted_forward_torso_relations_are_suppressed=True,
        supported_forward_torso_relations_remain_local=True,
        local_torso_relation_does_not_encode_global_stance=True,
        canonical_pose_text_fields_follow_authoritative_adjudication=True,
        support_and_elevation_semantics_are_unchanged=True,
        dwpose_local_joint_scalars_do_not_independently_classify_broad_pose=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def main() -> int:
    # Preserve v07 as the authoritative lower specialist stack, but replace the
    # underlying Phase-4B application with this population-gated v09 promotion.
    phase4b8.phase4b6.phase4b5.phase4b4._apply_phase4b4 = _apply_phase4b9_authoritative
    phase4b8.phase4b6.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b8.phase4b6.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b8.phase4b6.main()


if __name__ == "__main__":
    raise SystemExit(main())
