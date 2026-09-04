from __future__ import annotations

from copy import deepcopy
from typing import Any


AUTHORITY_VERSION = "semantic-v3-gestalt-source-authority-0.1"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_blob(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return " ".join(_text_blob(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_text_blob(v) for v in value)
    return ""


def _has_explicit_camera_perspective_evidence(evidence: dict[str, Any]) -> bool:
    """Require a dedicated composition observation carrying actual perspective geometry.

    Centered faces, gaze, crop scale, foreground objects, and absence of floor/ceiling are
    intentionally excluded: those are exactly the priors that caused the v0.2 failures.
    """
    tokens = (
        "camera_angle",
        "camera_elevation",
        "camera_pitch",
        "perspective",
        "vanishing",
        "horizon",
        "foreshorten",
        "top_plane",
        "bottom_plane",
        "looking_up",
        "looking_down",
        "low_angle",
        "high_angle",
    )
    for item in _list(evidence.get("composition_observations")):
        text = _text_blob(item)
        if any(token in text for token in tokens):
            return True
    return False


def _has_explicit_capture_evidence(evidence: dict[str, Any]) -> bool:
    """Positive capture evidence only; a generic device (for example a laptop) is not enough."""
    tokens = ("mirror", "selfie", "phone", "camera", "tripod", "remote_shutter")
    candidates = []
    candidates.extend(_list(evidence.get("entities")))
    candidates.extend(_list(evidence.get("relations")))
    candidates.extend(_list(evidence.get("composition_observations")))
    candidates.extend(_list(_dict(evidence.get("subject_evidence")).get("interactions")))
    return any(any(token in _text_blob(item) for token in tokens) for item in candidates)


def _has_explicit_near_lens_evidence(evidence: dict[str, Any]) -> bool:
    tokens = ("near_lens", "near lens", "camera_between", "camera between", "at_lens", "toward_lens")
    for item in _list(evidence.get("composition_observations")):
        text = _text_blob(item)
        if any(token in text for token in tokens):
            return True
    return False


def apply_source_authority(
    gestalt: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Withhold Gestalt claims when the required Extract evidence channel is absent.

    This is an assertion-authority layer, not semantic repair. It never chooses a replacement
    camera angle, orientation, capture mode, or proximity relation. Unsupported claims become
    unknown/omitted while the original model output remains preserved by the enclosing artifact.
    """
    governed = deepcopy(gestalt)
    actions: list[dict[str, Any]] = []

    subject_evidence = _dict(evidence.get("subject_evidence"))
    orientation_cues = _list(subject_evidence.get("orientation_cues"))
    orientation_authorized = bool(orientation_cues)

    orientation = _dict(governed.get("subject_orientation"))
    if not orientation_authorized and orientation:
        original = deepcopy(orientation)
        orientation["body_orientation"] = "unknown"
        orientation["body_faces_frame"] = "unknown"
        orientation["body_confidence"] = min(float(orientation.get("body_confidence") or 0.0), 0.25)
        orientation["torso_evidence_quality"] = "weak" if orientation.get("body_evidence") else "unknown"
        orientation["head_relative_body"] = "unknown"
        orientation["head_confidence"] = min(float(orientation.get("head_confidence") or 0.0), 0.25)
        orientation["body_counterevidence"] = list(orientation.get("body_counterevidence") or []) + [
            "withheld by source authority: Extract orientation_cues is empty"
        ]
        orientation["head_evidence"] = []
        actions.append(
            {
                "rule": "orientation_requires_extract_orientation_cues",
                "original": original,
                "governed": deepcopy(orientation),
            }
        )

    camera_authorized = _has_explicit_camera_perspective_evidence(evidence)
    camera = _dict(governed.get("camera"))
    if not camera_authorized and camera and (
        camera.get("elevation") != "unknown" or camera.get("pitch") != "unknown"
    ):
        original = deepcopy(camera)
        camera["elevation"] = "unknown"
        camera["pitch"] = "unknown"
        camera["confidence"] = min(float(camera.get("confidence") or 0.0), 0.25)
        camera["evidence"] = []
        camera["counterevidence"] = list(camera.get("counterevidence") or []) + [
            "withheld by source authority: no explicit perspective/camera observation"
        ]
        actions.append(
            {
                "rule": "camera_requires_explicit_perspective_evidence",
                "original": original,
                "governed": deepcopy(camera),
            }
        )

    capture_authorized = _has_explicit_capture_evidence(evidence)
    capture = _dict(governed.get("capture"))
    if not capture_authorized and capture and capture.get("mode") != "unknown":
        original = deepcopy(capture)
        capture["mode"] = "unknown"
        capture["confidence"] = min(float(capture.get("confidence") or 0.0), 0.25)
        capture["evidence"] = []
        actions.append(
            {
                "rule": "capture_requires_positive_capture_evidence",
                "original": original,
                "governed": deepcopy(capture),
            }
        )

    near_lens_authorized = _has_explicit_near_lens_evidence(evidence)
    if not near_lens_authorized:
        original_relations = _list(governed.get("foreground_relations"))
        kept: list[Any] = []
        removed: list[Any] = []
        gated_types = {"object_near_lens", "limb_near_lens", "camera_between_legs"}
        for item in original_relations:
            if isinstance(item, dict) and item.get("type") in gated_types:
                removed.append(deepcopy(item))
            else:
                kept.append(item)
        if removed:
            governed["foreground_relations"] = kept
            actions.append(
                {
                    "rule": "near_lens_requires_explicit_composition_evidence",
                    "removed": removed,
                }
            )

    # A free-text summary must not smuggle a withheld orientation/camera/capture claim back
    # into Fusion. Preserve it in raw model output, but withhold the canonical summary when
    # it contains a claim whose structured field was just denied authority.
    summary = governed.get("composition_summary")
    if isinstance(summary, str):
        lower = summary.lower()
        tainted = False
        reasons: list[str] = []
        if not orientation_authorized and any(
            term in lower
            for term in ("frontal", "three-quarter", "three quarter", "side-on", "side on", "rear view", "facing left", "facing right", "facing slightly")
        ):
            tainted = True
            reasons.append("unsupported orientation language")
        if not camera_authorized and any(term in lower for term in ("eye-level", "eye level", "low-angle", "low angle", "high-angle", "high angle")):
            tainted = True
            reasons.append("unsupported camera language")
        if not capture_authorized and any(term in lower for term in ("selfie", "external camera", "mirror")):
            tainted = True
            reasons.append("unsupported capture language")
        if tainted:
            governed["composition_summary"] = None
            actions.append(
                {
                    "rule": "summary_cannot_reintroduce_withheld_claims",
                    "original": summary,
                    "governed": None,
                    "reasons": reasons,
                }
            )

    audit = {
        "schema_version": AUTHORITY_VERSION,
        "policy": {
            "semantic_repair": "none",
            "unsupported_claim_action": "withhold to unknown/omit",
            "raw_model_output_mutated": False,
        },
        "authority_surface": {
            "orientation": orientation_authorized,
            "camera_perspective": camera_authorized,
            "capture": capture_authorized,
            "near_lens": near_lens_authorized,
        },
        "action_count": len(actions),
        "actions": actions,
    }
    return governed, audit
