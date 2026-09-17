from __future__ import annotations

from pathlib import Path
from typing import Any

from . import framing_semantics_shadow_v01 as v01

SCHEMA_VERSION = "framing-semantics-shadow-0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "framing-semantics-shadow-v0.2"

# A broad-pose decision and a local-configuration decision answer different
# questions.  Cropping can make standing/seated/etc. irrelevant while a local
# relation such as a fist under the chin remains highly salient.
#
# visible_arm_relationship is deliberately sufficient by itself to justify the
# narrow image-conditioned configuration observer.  This does NOT authorize a
# broad posture claim.  Other isolated weak cues still need the pre-existing
# multi-cue score before they trigger that observer.
DIRECT_LOCAL_CONFIGURATION_CUES = {
    "visible_arm_relationship",
}


def _geometry(policy: dict[str, Any]) -> dict[str, Any]:
    return policy.get("geometry") if isinstance(policy.get("geometry"), dict) else {}


def _local_configuration_gate(policy: dict[str, Any]) -> dict[str, Any]:
    geometry = _geometry(policy)
    score = int(geometry.get("configuration_score") or 0)
    raw_cues = geometry.get("configuration_cues")
    cues = [str(x) for x in raw_cues] if isinstance(raw_cues, list) else []
    direct = [cue for cue in cues if cue in DIRECT_LOCAL_CONFIGURATION_CUES]

    if direct:
        supported = True
        reason = "directly_observed_arm_relationship_justifies_local_configuration_observer"
    elif score >= 2:
        supported = True
        reason = "multiple_local_configuration_cues_justify_local_configuration_observer"
    else:
        supported = False
        reason = "no_direct_arm_relationship_and_insufficient_other_local_configuration_cues"

    return {
        "supported": supported,
        "configuration_score": score,
        "configuration_cues": cues,
        "direct_qualifying_cues": direct,
        "broad_pose_authority_created": False,
        "reason": reason,
        "note": (
            "Local configuration eligibility only permits directly visible body relationships. "
            "It does not authorize standing, seated, crouching, squatting, kneeling, reclining, or lying."
        ),
    }


def _pose_gate_shadow(policy: dict[str, Any], tiers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hips_strong = (tiers.get("hips") or {}).get("state") == "strong"
    knees_strong = (tiers.get("knees") or {}).get("state") == "strong"
    broad = bool(hips_strong and knees_strong)

    geometry = _geometry(policy)
    sam = policy.get("sam3d") if isinstance(policy.get("sam3d"), dict) else {}
    complexity = int(geometry.get("pose_complexity_score") or 0)
    guidance_usable = bool(sam.get("available") and int(sam.get("projected_selected_joint_count") or 0) >= 8)
    local = _local_configuration_gate(policy)

    if broad:
        mode = "pose_guided" if complexity >= 2 and guidance_usable else "pose_allowed"
        mode_reason = "broad_pose_has_bilateral_hip_and_knee_observation"
    elif local["supported"]:
        mode = "configuration"
        mode_reason = "broad_pose_withheld_but_local_configuration_observer_is_supported"
    else:
        mode = "framing_only"
        mode_reason = "broad_pose_withheld_and_no_useful_local_configuration_observer_is_supported"

    legacy_visibility = policy.get("visibility") if isinstance(policy.get("visibility"), dict) else {}
    legacy_broad = bool(legacy_visibility.get("broad_pose_supported"))
    legacy_mode = str((policy.get("policy") or {}).get("mode") or "")

    return {
        "broad_pose_supported": broad,
        "local_configuration_supported": bool(local["supported"]),
        "local_configuration_gate": local,
        "proposed_mode": mode,
        "changed_from_legacy": broad != legacy_broad or mode != legacy_mode,
        "legacy_broad_pose_supported": legacy_broad,
        "legacy_mode": legacy_mode,
        "broad_pose_reason": (
            "requires_both_hips_and_both_knees_as_strong_observed_crop_evidence"
            if broad
            else "withholds_broad_pose_without_bilateral_hip_and_knee_visibility"
        ),
        "reason": mode_reason,
    }


_BASE_EVALUATE = v01.evaluate


def evaluate(policy: dict[str, Any], fact: dict[str, Any] | None = None) -> dict[str, Any]:
    out = _BASE_EVALUATE(policy, fact)
    tiers = out.get("tier_evidence") if isinstance(out.get("tier_evidence"), dict) else v01._tier_evidence(policy)
    gate = _pose_gate_shadow(policy, tiers)

    out["schema_version"] = SCHEMA_VERSION
    out["pose_gate_shadow"] = gate
    invariants = out.get("invariants") if isinstance(out.get("invariants"), dict) else {}
    invariants.update(
        broad_pose_and_local_configuration_are_independent=True,
        visible_arm_relationship_can_trigger_configuration_without_broad_pose=True,
        local_configuration_never_creates_broad_pose_authority=True,
        pure_framing_route_remains_available_when_no_local_relationship_is_useful=True,
    )
    out["invariants"] = invariants
    return out


def main() -> int:
    # Reuse the v0.1 framing/shot-scale implementation and writer, replacing
    # only the routing shadow with the corrected two-axis decision.
    v01.evaluate = evaluate
    v01.SCHEMA_VERSION = SCHEMA_VERSION
    v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
