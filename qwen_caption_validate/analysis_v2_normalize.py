from __future__ import annotations

from copy import deepcopy
from typing import Any


def _record(actions: list[dict[str, Any]], path: str, before: Any, after: Any) -> None:
    if before == after:
        return
    actions.append({"path": path, "from": before, "to": after})


def _map_value(
    obj: dict[str, Any],
    key: str,
    mapping: dict[str, str],
    path: str,
    actions: list[dict[str, Any]],
) -> None:
    value = obj.get(key)
    if not isinstance(value, str):
        return
    mapped = mapping.get(value.strip().lower())
    if mapped is None or mapped == value:
        return
    _record(actions, path, value, mapped)
    obj[key] = mapped


def normalize_analysis_v2(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Canonicalize only unambiguous Analyze-v2 vocabulary aliases.

    Raw model text remains preserved by the runner in ``raw_response``. This
    function deliberately does not change geometry, confidence, ownership, or
    any other semantic judgement. It only maps obvious lexical aliases onto the
    schema vocabulary so ``camera`` vs ``camera_lens`` does not turn otherwise
    usable structured evidence into a schema failure.
    """
    if not isinstance(data, dict) or data.get("schema_version") != "2.0":
        return data, []

    out = deepcopy(data)
    actions: list[dict[str, Any]] = []

    subject = out.get("target_subject") or {}
    gaze = subject.get("gaze") or {}
    if isinstance(gaze, dict):
        _map_value(
            gaze,
            "target",
            {
                "camera": "camera_lens",
                "camera lens": "camera_lens",
                "lens": "camera_lens",
                "phone": "object",
                "smartphone": "object",
                "phone screen": "object",
                "smartphone screen": "object",
            },
            "target_subject.gaze.target",
            actions,
        )
        _map_value(
            gaze,
            "image_direction",
            {
                "left": "image_left",
                "center": "image_center",
                "centre": "image_center",
                "right": "image_right",
            },
            "target_subject.gaze.image_direction",
            actions,
        )

    orientation = subject.get("orientation") or {}
    if isinstance(orientation, dict):
        image_axis = orientation.get("image_plane_body_axis") or {}
        if isinstance(image_axis, dict):
            _map_value(
                image_axis,
                "direction",
                {
                    "near-horizontal": "near_horizontal",
                    "near horizontal": "near_horizontal",
                },
                "target_subject.orientation.image_plane_body_axis.direction",
                actions,
            )

    parts = subject.get("visible_body_parts") or []
    if isinstance(parts, list):
        for index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            # The schema is intentionally singular-side only. When the model emits
            # a collective "both" for a plural part (e.g. shoulders), retain the
            # fact in raw_response but canonicalize the structured side to unknown
            # rather than inventing one anatomical side.
            _map_value(
                part,
                "anatomical_side",
                {"both": "unknown"},
                f"target_subject.visible_body_parts.{index}.anatomical_side",
                actions,
            )

    return out, actions
