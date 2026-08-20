from __future__ import annotations

import json
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


def _compact_text(value: Any) -> str | None:
    """Losslessly collapse schema-text shape drift into compact readable text.

    Analyze-v2 body-part ``geometry``, ``contact`` and ``support`` are intentionally
    opaque short-text fields. Qwen can occasionally emit a small object or array
    instead. Converting that shape to text is a representation normalization, not
    a semantic rewrite; the untouched raw model response remains available for
    audit in the runner artifact.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(item, str) for item in value):
            return "; ".join(str(item) for item in value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if isinstance(value, dict):
        if not value:
            return None
        simple = all(
            item is None or isinstance(item, (str, int, float, bool))
            for item in value.values()
        )
        if simple:
            return "; ".join(
                f"{key}={value[key]}"
                for key in sorted(value)
                if value[key] is not None
            ) or None
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return str(value)


def _normalize_text_field(
    obj: dict[str, Any],
    key: str,
    path: str,
    actions: list[dict[str, Any]],
) -> None:
    if key not in obj:
        return
    before = obj.get(key)
    if before is None or isinstance(before, str):
        return
    after = _compact_text(before)
    _record(actions, path, before, after)
    obj[key] = after


def normalize_analysis_v2(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Canonicalize unambiguous Analyze-v2/v2.1 schema drift.

    Raw model text remains preserved by the runner in ``raw_response``. This
    function does not change geometry, visibility, confidence, ownership, or any
    other semantic judgement. It maps obvious lexical aliases and converts only
    schema-declared opaque text fields from accidental object/array shapes into
    compact text so otherwise valid evidence is not discarded.
    """
    if not isinstance(data, dict) or data.get("schema_version") not in {"2.0", "2.1"}:
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
            _map_value(
                part,
                "anatomical_side",
                {"both": "unknown"},
                f"target_subject.visible_body_parts.{index}.anatomical_side",
                actions,
            )
            for field in ("geometry", "contact", "support"):
                _normalize_text_field(
                    part,
                    field,
                    f"target_subject.visible_body_parts.{index}.{field}",
                    actions,
                )

    return out, actions
