from __future__ import annotations

from typing import Any


RELIABLE_AXIS_AUTHORITIES = {"high", "corroborated", "corroborated_direction"}
PUBLISHABLE_GAZE_AUTHORITIES = {"high", "corroborated", "reduced"}


def _axis_authority(head: dict[str, Any], axis: str) -> str | None:
    axes = head.get("axis_authority") if isinstance(head.get("axis_authority"), dict) else {}
    value = axes.get(axis)
    if isinstance(value, dict):
        authority = value.get("authority")
        return str(authority) if authority else None
    return None


def _axis_reliable(head: dict[str, Any], axis: str) -> bool:
    authority = _axis_authority(head, axis)
    if authority is not None:
        return authority in RELIABLE_AXIS_AUTHORITIES
    # A single-source UniFace estimate can be marked high when py-feat is absent.
    return head.get("authority") == "high"


def _image_direction(value: str | None) -> str:
    return {
        "frame_left": "image-left",
        "frame_right": "image-right",
        "center": "approximately centered",
    }.get(value or "", "unknown")


def _head_turn_phrase(head: dict[str, Any]) -> str:
    direction = _image_direction(head.get("frame_horizontal"))
    strength = head.get("yaw_strength")
    if direction == "approximately centered":
        return "approximately frontal/centered"
    if strength == "profile":
        return f"near-profile toward {direction}"
    if strength == "strong_turn":
        return f"strongly turned toward {direction}"
    if strength == "turned":
        return f"turned toward {direction}"
    return f"oriented toward {direction}"


def _head_pitch_phrase(value: str | None) -> str:
    return {
        "up": "tilted upward",
        "down": "tilted downward",
        "center": "approximately level",
    }.get(value or "", "unknown")


def _gaze_components(gaze: dict[str, Any]) -> str:
    parts: list[str] = []
    horizontal = gaze.get("horizontal")
    vertical = gaze.get("vertical")
    if horizontal == "frame_left":
        parts.append("toward image-left")
    elif horizontal == "frame_right":
        parts.append("toward image-right")
    elif horizontal == "center":
        parts.append("horizontally near center")

    if vertical == "up":
        parts.append("upward")
    elif vertical == "down":
        parts.append("downward")
    elif vertical == "center":
        parts.append("vertically near center")
    return " and ".join(parts) if parts else "direction unavailable"


def format_head_gaze_facts(record: dict[str, Any] | None) -> str:
    """Render only caption-safe specialist facts; never expose numeric angles."""
    if not record or record.get("status") != "ok":
        return (
            "- Head specialist evidence unavailable; do not correct head direction from specialist evidence.\n"
            "- Gaze specialist evidence unavailable/non-publishable; do not add, remove, or correct gaze."
        )

    head = record.get("head") if isinstance(record.get("head"), dict) else {}
    gaze = record.get("gaze") if isinstance(record.get("gaze"), dict) else {}
    lines: list[str] = []

    if head.get("available") and _axis_reliable(head, "yaw"):
        lines.append(f"- Reliable head yaw: {_head_turn_phrase(head)}.")
    else:
        lines.append("- Head yaw is not reliable enough for correction; do not use it to change head left/right orientation.")

    if head.get("available") and _axis_reliable(head, "pitch"):
        lines.append(f"- Reliable head pitch: {_head_pitch_phrase(head.get('vertical'))}.")
    else:
        lines.append("- Head pitch is not reliable enough for correction; do not use it to change head up/down orientation.")

    if not gaze.get("available") or not gaze.get("publishable") or gaze.get("authority") not in PUBLISHABLE_GAZE_AUTHORITIES:
        lines.append(
            "- Gaze is unavailable/non-publishable; do not add, remove, or correct gaze or eye-contact language from specialist evidence."
        )
        return "\n".join(lines)

    authority = str(gaze.get("authority") or "high")
    prefix = "Publishable gaze" if authority != "reduced" else "Publishable gaze, reduced authority"
    lines.append(f"- {prefix}: {_gaze_components(gaze)}.")

    camera = gaze.get("camera_relationship")
    if camera == "toward_camera":
        lines.append("- Reliable camera relationship: gaze is broadly toward the camera.")
    elif camera == "off_camera":
        lines.append("- Reliable camera relationship: gaze is clearly off-camera.")
    else:
        lines.append(
            "- Camera relationship is uncertain; do not claim toward-camera or off-camera from specialist evidence, although the publishable directional components above may still be used."
        )

    return "\n".join(lines)


def render_specialist_prompt(template: str, caption: str, laterality_text: str, specialist_text: str) -> str:
    return (
        template
        .replace("{{CURRENT_CAPTION}}", caption.strip())
        .replace("{{LATERALITY_FACTS}}", laterality_text.strip())
        .replace("{{HEAD_GAZE_FACTS}}", specialist_text.strip())
    )
