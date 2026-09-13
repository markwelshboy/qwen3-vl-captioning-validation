from __future__ import annotations

from typing import Any

HORIZONTAL_NEAR_CENTER_MAX_DEG = 15.0
HORIZONTAL_CLEAR_LATERAL_MIN_DEG = 25.0


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _horizontal_semantics(gaze: dict[str, Any]) -> dict[str, Any]:
    raw_value = _clean(gaze.get("horizontal"))
    yaw = gaze.get("yaw_deg")
    try:
        yaw_deg = float(yaw)
    except (TypeError, ValueError):
        yaw_deg = None

    out: dict[str, Any] = {
        "raw_value": raw_value,
        "raw_yaw_deg": yaw_deg,
        "composer_value": None,
        "publishable": False,
    }
    if yaw_deg is None:
        out.update(
            semantic_class="unavailable",
            reason="missing_gaze_yaw",
        )
        return out

    magnitude = abs(yaw_deg)
    out["yaw_magnitude_deg"] = magnitude

    if magnitude <= HORIZONTAL_NEAR_CENTER_MAX_DEG:
        out.update(
            semantic_class="near_center",
            reason="lateral_offset_below_caption_salience_threshold",
        )
        return out

    if magnitude < HORIZONTAL_CLEAR_LATERAL_MIN_DEG:
        out.update(
            semantic_class="mild_lateral_uncorroborated",
            reason="single_model_lateral_offset_not_large_enough_for_caption_direction",
        )
        return out

    if raw_value in {"frame_left", "frame_right"}:
        composer_value = raw_value
    else:
        # head-gaze-evidence-v03 defines +yaw=frame_left/-yaw=frame_right.
        composer_value = "frame_left" if yaw_deg > 0 else "frame_right"
    out.update(
        semantic_class="clear_lateral",
        composer_value=composer_value,
        publishable=True,
        reason="lateral_offset_is_caption_salient",
    )
    return out


def _vertical_semantics(gaze: dict[str, Any]) -> dict[str, Any]:
    raw_value = _clean(gaze.get("vertical"))
    out: dict[str, Any] = {
        "raw_value": raw_value,
        "raw_pitch_deg": gaze.get("pitch_deg"),
        "composer_value": None,
        "publishable": False,
    }
    if raw_value in {"up", "down"}:
        out.update(
            semantic_class="directional",
            composer_value=raw_value,
            publishable=True,
            reason="upstream_vertical_direction_is_caption_salient",
        )
    elif raw_value == "center":
        out.update(
            semantic_class="near_center",
            reason="centered_vertical_gaze_is_not_caption_salient",
        )
    else:
        out.update(
            semantic_class="unavailable",
            reason="no_publishable_vertical_direction",
        )
    return out


def _camera_relationship_semantics(gaze: dict[str, Any]) -> dict[str, Any]:
    raw_value = _clean(gaze.get("camera_relationship"))
    publishable = raw_value in {"toward_camera", "off_camera"}
    return {
        "raw_value": raw_value,
        "composer_value": raw_value if publishable else None,
        "publishable": publishable,
        "reason": "camera_relationship_resolved" if publishable else "camera_relationship_uncertain_or_unavailable",
    }


def build_gaze_caption_semantics(gaze: dict[str, Any]) -> dict[str, Any]:
    """Project precise gaze evidence into caption-worthy semantics.

    Observability/authority answers whether the gaze estimator is allowed to run.
    This layer separately asks whether the measured displacement is semantically
    strong enough to deserve words in a training caption.
    """
    horizontal = _horizontal_semantics(gaze)
    vertical = _vertical_semantics(gaze)
    camera_relationship = _camera_relationship_semantics(gaze)
    publishable = any(
        bool(axis.get("publishable"))
        for axis in (horizontal, vertical, camera_relationship)
    )
    return {
        "available": bool(gaze.get("available")),
        "publishable": publishable,
        "horizontal": horizontal,
        "vertical": vertical,
        "camera_relationship": camera_relationship,
        "policy": {
            "horizontal_near_center_max_deg": HORIZONTAL_NEAR_CENTER_MAX_DEG,
            "horizontal_clear_lateral_min_deg": HORIZONTAL_CLEAR_LATERAL_MIN_DEG,
            "mild_single_model_lateral_offsets_are_caption_suppressed": True,
            "centered_axes_are_not_caption_salient": True,
            "raw_measurement_is_preserved": True,
        },
        "note": (
            "Gaze observability and estimator authority are retained separately from caption salience. "
            "A clean observable eye pair does not by itself make a modest directional point estimate caption-worthy."
        ),
    }
