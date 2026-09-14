from __future__ import annotations

from typing import Any, Callable


def compact_orientation(value: Any, clean: Callable[[Any], str | None]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    band = clean(value.get("orientation_band"))
    yaw = value.get("yaw_magnitude_deg")
    approx = value.get("approx_yaw_deg")
    direction = clean(value.get("turn_direction")) if value.get("turn_direction_publishable", True) else None
    out: dict[str, Any] = {}
    if band:
        out["camera_orientation"] = band
    if isinstance(yaw, (int, float)):
        out["yaw_magnitude_deg"] = round(float(yaw), 1)
    if isinstance(approx, (int, float)):
        out["approx_yaw_deg"] = int(approx)
    if direction:
        out["turn_direction"] = direction
    return out or None


def torso_fact(body: dict[str, Any], clean: Callable[[Any], str | None]) -> dict[str, Any] | None:
    torso = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}
    if not torso.get("available") or not torso.get("composer_eligible"):
        return None

    summary = torso.get("caption_orientation") if isinstance(torso.get("caption_orientation"), dict) else {}
    body_orientation = compact_orientation(torso.get("body_root_orientation"), clean)
    upper_orientation = compact_orientation(torso.get("upper_torso_orientation"), clean)
    mode = clean(summary.get("mode"))

    if upper_orientation:
        if mode == "articulated" and body_orientation:
            out: dict[str, Any] = {
                "mode": "articulated",
                "body_orientation": body_orientation,
                "upper_torso_orientation": upper_orientation,
                "preferred": dict(upper_orientation),
            }
            twist = summary.get("relative_twist_magnitude_deg")
            if isinstance(twist, (int, float)):
                out["relative_twist_magnitude_deg"] = round(float(twist), 1)
            return out
        return dict(upper_orientation)

    orientation = clean(torso.get("torso_camera_orientation"))
    if not orientation:
        return None
    out: dict[str, Any] = {"camera_orientation": orientation}
    yaw = torso.get("torso_yaw_magnitude_deg")
    if isinstance(yaw, (int, float)):
        out["yaw_magnitude_deg"] = round(abs(float(yaw)), 1)
        out["approx_yaw_deg"] = int(5 * round(abs(float(yaw)) / 5.0))
    direction = clean(summary.get("preferred_turn_direction")) if summary.get("turn_direction_publishable") else None
    if direction:
        out["turn_direction"] = direction
    return out