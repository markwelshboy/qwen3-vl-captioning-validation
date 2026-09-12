from __future__ import annotations

import copy
import re
from typing import Any

from . import fact_sheet_specialist_normalizer_v01 as base

SCHEMA_VERSION = "caption-fact-sheet-0.2.1"

# Deliberately narrow: this catches camera-relative torso yaw/orientation language,
# not in-plane posture such as "torso bent forward" or
# "torso oriented horizontally relative to frame".
TORSO_CAMERA_ORIENTATION_RE = re.compile(
    r"(?:\b(?:torso|body)\b.{0,48}\b(?:turn(?:ed|ing)?|angle(?:d)?|fac(?:e|es|ing)|orient(?:ed|ation)?)\b"
    r".{0,48}\b(?:camera|side|three[- ]quarter|frontal|profile|away|toward)\b"
    r"|\b(?:frontal|three[- ]quarter|profile|side[- ]on)\b.{0,32}\b(?:torso|body)\b)",
    re.I,
)


def _torso_visibility(sheet: dict[str, Any]) -> str | None:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    framing = facts.get("framing") if isinstance(facts.get("framing"), dict) else {}
    body_visibility = framing.get("body_visibility") if isinstance(framing.get("body_visibility"), dict) else {}
    value = body_visibility.get("torso")
    return str(value) if value is not None else None


def _apply_torso_semantic_scope(sheet: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    torso = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}
    if not torso:
        return warnings

    mode = str(((sheet.get("policy") or {}).get("mode")) or "unknown")
    visibility = _torso_visibility(sheet)
    available = bool(torso.get("available"))

    torso["route_mode"] = mode
    torso["observed_torso_visibility"] = visibility
    torso["semantic_bandwidth_owner"] = "caption_perception_policy"

    if not available:
        torso["composer_eligible"] = False
        torso["semantic_scope"] = "resolved_null"
        return warnings

    if mode == "framing_only":
        torso["composer_eligible"] = False
        torso["semantic_scope"] = "diagnostic_only_by_route"
        torso["note_phase4b1"] = (
            "SAM3D torso geometry is retained for diagnostics, but the crop governor denied body-configuration "
            "caption bandwidth for framing_only."
        )
        warnings.append("sam3d_torso_withheld_by_framing_only_route")
    elif visibility == "strong":
        torso["composer_eligible"] = True
        torso["semantic_scope"] = "caption_eligible_observed_torso"
    else:
        torso["composer_eligible"] = False
        torso["semantic_scope"] = "diagnostic_only_insufficient_torso_observation"
        torso["note_phase4b1"] = (
            "SAM3D torso geometry remains diagnostic because deterministic crop evidence does not mark the torso "
            "as strongly observed."
        )
        warnings.append("sam3d_torso_withheld_without_strong_torso_observation")
    return warnings


def _apply_direction_only_head_authority(sheet: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    head = facts.get("head_pose") if isinstance(facts.get("head_pose"), dict) else {}
    if not head:
        return warnings

    for axis_name in ("horizontal", "vertical"):
        axis = head.get(axis_name) if isinstance(head.get(axis_name), dict) else {}
        if not axis:
            continue
        authority = str(axis.get("authority") or "unavailable")
        if authority == "corroborated_direction" and axis.get("publishable"):
            if axis.get("degrees") is not None:
                axis["candidate_degrees"] = axis.get("degrees")
            axis["degrees"] = None
            axis["magnitude_publishable"] = False
            axis["note_phase4b1"] = "Cross-source agreement supports direction/class only, not angular magnitude."
            warnings.append(f"head_{axis_name}_magnitude_withheld_direction_only")
        else:
            axis["magnitude_publishable"] = bool(axis.get("publishable") and axis.get("degrees") is not None)

    horizontal = head.get("horizontal") if isinstance(head.get("horizontal"), dict) else {}
    if horizontal.get("authority") == "corroborated_direction" and horizontal.get("publishable"):
        if head.get("yaw_strength") is not None:
            head["yaw_strength_candidate"] = head.get("yaw_strength")
        head["yaw_strength"] = None
        head["yaw_strength_publishable"] = False
    else:
        head["yaw_strength_publishable"] = bool(head.get("yaw_strength") is not None and horizontal.get("publishable"))
    return warnings


def _apply_torso_qwen_handoff(sheet: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    conflicts: list[dict[str, Any]] = []
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    torso = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []

    for item in configuration:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text or not TORSO_CAMERA_ORIENTATION_RE.search(text):
            continue

        item["domain_handoff"] = "torso_camera_orientation"
        item["specialist_owner"] = "sam3d_plus_dwpose_observation_gate"

        if torso.get("available") and torso.get("composer_eligible"):
            item["composer_text"] = None
            item["promotion_status"] = "superseded_by_torso_geometry"
            item["note"] = (
                "Qwen camera-relative torso orientation is held because strongly observed torso geometry grants "
                "the specialist domain authority."
            )
            warnings.append("configuration_torso_orientation_resolved_by_specialist")
            continue

        item["composer_text"] = None
        item["promotion_status"] = "held_for_torso_orientation_review"
        item["note"] = (
            "Qwen camera-relative torso orientation and SAM3D geometry are preserved diagnostically, but neither "
            "is allowed to override the other without strongly observed torso support."
        )
        conflict = {
            "domain": "torso_camera_orientation",
            "qwen_text": text,
            "sam3d_available": bool(torso.get("available")),
            "sam3d_orientation": torso.get("torso_camera_orientation"),
            "sam3d_yaw_magnitude_deg": torso.get("torso_yaw_magnitude_deg"),
            "observed_torso_visibility": torso.get("observed_torso_visibility"),
            "route_mode": torso.get("route_mode"),
            "resolution": "manual_review_or_omit",
        }
        conflicts.append(conflict)
        warnings.append("configuration_torso_orientation_review_conflict")
    return warnings, conflicts


def _apply_phase4b1(sheet: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(sheet)
    warnings: list[str] = []

    warnings.extend(_apply_torso_semantic_scope(out))
    warnings.extend(_apply_direction_only_head_authority(out))
    handoff_warnings, conflicts = _apply_torso_qwen_handoff(out)
    warnings.extend(handoff_warnings)

    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    torso = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}

    reserved = out.get("reserved_domains") if isinstance(out.get("reserved_domains"), dict) else {}
    torso_reserved = reserved.get("torso_geometry") if isinstance(reserved.get("torso_geometry"), dict) else None
    if torso_reserved is not None:
        if not torso.get("available"):
            torso_reserved["status"] = "resolved_null"
        elif torso.get("composer_eligible"):
            torso_reserved["status"] = "resolved"
        else:
            torso_reserved["status"] = "resolved_diagnostic_only"

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    existing_warnings = [str(x) for x in (audit.get("warnings") or []) if x]
    audit["warnings"] = sorted(set(existing_warnings + warnings))
    audit["review_conflicts"] = conflicts
    audit["phase"] = "4B.1"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        crop_governor_caps_torso_composer_bandwidth=True,
        corroborated_direction_does_not_publish_angular_magnitude=True,
        qwen_torso_camera_orientation_requires_specialist_handoff=True,
        partial_torso_conflict_is_held_not_auto_resolved=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


_BASE_BUILD_FACT_SHEET = base._build_fact_sheet


def _build_fact_sheet(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _apply_phase4b1(_BASE_BUILD_FACT_SHEET(*args, **kwargs))


def main() -> int:
    # Keep the stable v0.1 CLI/file discovery logic, but replace only the fact-sheet
    # builder and emitted schema for this small 4B.1 contract refinement.
    base._build_fact_sheet = _build_fact_sheet
    base.SCHEMA_VERSION = SCHEMA_VERSION
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
