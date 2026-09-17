from __future__ import annotations

from typing import Any, Mapping

from . import caption_perception_policy as base

SCHEMA_VERSION = "caption-perception-policy-0.2"


def _tier_state(count: int, *, strong: int = 2) -> str:
    return "strong" if count >= strong else ("partial" if count else "absent")


def _visibility(points: dict[str, tuple[float, float] | None], width: int, height: int) -> dict[str, Any]:
    """Return observation-only visibility with coherent anatomical tiers.

    Unlike the legacy policy, one ankle/knee cannot promote the crop to a
    full-length/long framing class or independently authorize broad posture.
    Broad pose requires bilateral hips and bilateral knees.  Head strength is
    based on observed face landmarks rather than neck alone.
    """
    visible = {name for name, p in points.items() if p is not None}

    face_names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
    face_count = sum(name in visible for name in face_names)
    head_partial = bool(face_count or "neck" in visible)
    shoulders = sum(name in visible for name in ("left_shoulder", "right_shoulder"))
    hips = sum(name in visible for name in ("left_hip", "right_hip"))
    knees = sum(name in visible for name in ("left_knee", "right_knee"))
    ankles = sum(name in visible for name in ("left_ankle", "right_ankle"))

    head_state = "strong" if face_count >= 2 else ("partial" if head_partial else "absent")
    shoulder_state = _tier_state(shoulders)
    hip_state = _tier_state(hips)
    knee_state = _tier_state(knees)
    ankle_state = _tier_state(ankles)

    # Reconstruction is never allowed to fill missing crop evidence.  Broad
    # posture requires a coherent observed pelvis->knee structure on both sides.
    broad = hips == 2 and knees == 2

    # Keep the legacy extent field for old consumers, but make it conservative:
    # only coherent bilateral tiers can select lower-body labels.  Newer stages
    # should prefer anatomical_tiers / broad_pose_supported over this hint.
    if ankles == 2 and knees == 2 and hips == 2:
        extent = "full_length"
    elif knees == 2 and hips == 2:
        extent = "three_quarter_or_long"
    elif hips == 2:
        extent = "waist_or_upper_body"
    elif shoulders == 2:
        extent = "close_or_medium_close"
    else:
        extent = "face_or_partial_body"

    return {
        "head": head_state,
        "shoulders": shoulder_state,
        "torso": "strong" if shoulders == hips == 2 else ("partial" if shoulders and (hips or shoulders == 2) else "absent"),
        "hips": hip_state,
        "knees": knee_state,
        "feet": ankle_state,
        "observed_landmarks": [name for name in base.BODY18 if name in visible],
        "observed_bbox": base._bbox(points, width, height),
        "extent_hint": extent,
        "broad_pose_supported": broad,
        "anatomical_tiers": {
            "head": {"state": head_state, "observed_face_landmark_count": face_count},
            "shoulders": {"state": shoulder_state, "observed_count": shoulders},
            "hips": {"state": hip_state, "observed_count": hips},
            "knees": {"state": knee_state, "observed_count": knees},
            "ankles": {"state": ankle_state, "observed_count": ankles},
        },
        "broad_pose_gate": {
            "supported": broad,
            "requires_bilateral_hips": True,
            "requires_bilateral_knees": True,
            "reason": (
                "bilateral_hips_and_bilateral_knees_observed"
                if broad
                else "withheld_without_bilateral_hip_and_knee_visibility"
            ),
        },
    }


def _local_configuration_gate(geometry: Mapping[str, Any]) -> dict[str, Any]:
    cues = [str(v) for v in (geometry.get("configuration_cues") or [])]
    # A directly observed arm relationship is sufficient to justify the narrow
    # configuration observer.  The observer is still forbidden from naming a
    # broad posture and may abstain with NO RELIABLE POSE FACTS.
    direct = [cue for cue in cues if cue == "visible_arm_relationship"]
    supported = bool(direct) or int(geometry.get("configuration_score") or 0) >= 2
    return {
        "supported": supported,
        "configuration_score": int(geometry.get("configuration_score") or 0),
        "configuration_cues": cues,
        "direct_qualifying_cues": direct,
        "broad_pose_authority_created": False,
        "reason": (
            "directly_observed_arm_relationship_justifies_local_configuration_observer"
            if direct
            else (
                "multiple_local_configuration_cues_justify_observer"
                if supported
                else "insufficient_local_configuration_evidence"
            )
        ),
        "note": (
            "Local configuration eligibility only permits directly visible body relationships. "
            "It does not authorize standing, seated, crouching, squatting, kneeling, reclining, or lying."
        ),
    }


def route_policy(
    points: dict[str, tuple[float, float] | None],
    *,
    width: int,
    height: int,
    sam3d: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Observation-gated broad pose plus independent local configuration routing."""
    visibility = _visibility(points, width, height)
    geometry = base._geometry(points, width, height)
    local_gate = _local_configuration_gate(geometry)

    sam = dict(sam3d or {})
    sam.setdefault("available", False)
    sam["reconstruction_is_observation"] = False
    sam["can_promote_observability"] = False

    broad = bool(visibility["broad_pose_supported"])
    guidance_usable = bool(
        sam["available"] and int(sam.get("projected_selected_joint_count") or 0) >= 8
    )
    reasons: list[str] = []

    if broad:
        reasons.append(
            "Observed DWPose crop evidence includes bilateral hips and bilateral knees, so broad-pose wording is supportable."
        )
        if int(geometry.get("pose_complexity_score") or 0) >= 2 and guidance_usable:
            mode, relevance = "pose_guided", "high"
            reasons.append("Observed 2D geometry is non-routine enough to benefit from SAM3D relational guidance.")
        else:
            mode, relevance = "pose_allowed", "medium"
            reasons.append("Broad pose is visible, but geometry-heavy SAM3D guidance is not required or not sufficiently supported.")
    else:
        reasons.append("Observed crop evidence does not expose bilateral hip-and-knee structure required for a broad posture claim.")
        if local_gate["supported"]:
            mode, relevance = "configuration", "low"
            reasons.append("Visible local body relationships justify a narrow configuration observation without naming broad posture.")
        else:
            mode, relevance = "framing_only", "negligible"
            reasons.append("No sufficiently useful local relationship is observed; framing/head semantics should dominate.")
        if sam["available"]:
            reasons.append("SAM3D reconstruction stayed diagnostic and did not promote hidden/out-of-crop anatomy into observed evidence.")

    return {
        "visibility": visibility,
        "geometry": geometry,
        "local_configuration_gate": local_gate,
        "sam3d": sam,
        "pose_relevance": relevance,
        "policy": {"mode": mode},
        "reasons": reasons,
    }


def main() -> int:
    # Reuse the validated v0.1 artifact discovery/writer while replacing only
    # visibility/routing semantics and output identity.
    old_visibility = base._visibility
    old_route_policy = base.route_policy
    old_schema = base.SCHEMA_VERSION
    try:
        base._visibility = _visibility
        base.route_policy = route_policy
        base.SCHEMA_VERSION = SCHEMA_VERSION
        return base.main()
    finally:
        base._visibility = old_visibility
        base.route_policy = old_route_policy
        base.SCHEMA_VERSION = old_schema


if __name__ == "__main__":
    raise SystemExit(main())
