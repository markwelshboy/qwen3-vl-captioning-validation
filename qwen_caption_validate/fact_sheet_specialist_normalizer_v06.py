from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v05 as phase4b4

SCHEMA_VERSION = "caption-fact-sheet-0.2.5"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.5"

KNEE_RAISED_RE = re.compile(
    r"(?:\b(?:(?:left|right|one|other)\s+)?knee\b.{0,24}\b(?:raised|lifted)\b|"
    r"\b(?:raised|lifted)\b.{0,24}\b(?:(?:left|right|one|other)\s+)?knee\b)",
    re.I,
)
FOOT_PLANTED_RE = re.compile(
    r"\b(?:(?:left|right|one|other)\s+)?foot\b.{0,28}\b(?:planted|flat\s+on\s+(?:the\s+)?(?:floor|ground))\b",
    re.I,
)
FOOT_LIFTED_RE = re.compile(
    r"\b(?:(?:left|right|one|other)\s+)?foot\b.{0,30}\b(?:lifted|raised|off\s+(?:the\s+)?(?:floor|ground))\b",
    re.I,
)
BOTH_LEGS_RE = re.compile(r"\b(?:both|two)\s+(?:knees?|legs?|feet)\b", re.I)

KNEE_RELATIVE_HEIGHT_MIN_MARGIN = 0.20
FOOT_RELATIVE_HEIGHT_MIN_MARGIN = 0.15


def _source_side(text: str, noun: str) -> str | None:
    match = re.search(rf"\b(left|right)\s+{re.escape(noun)}\b", text, re.I)
    return match.group(1).lower() if match else None


def _leg_scale(points: dict[str, tuple[float, float] | None]) -> float | None:
    return phase4b4._body_scale(points)


def _relative_drop(
    points: dict[str, tuple[float, float] | None],
    side: str,
    distal_joint: str,
) -> float | None:
    hip = points.get(f"{side}_hip")
    distal = points.get(f"{side}_{distal_joint}")
    if hip is None or distal is None:
        return None
    return float(distal[1] - hip[1])


def _raised_knee_binding(points: dict[str, tuple[float, float] | None]) -> dict[str, Any] | None:
    scale = _leg_scale(points)
    if scale is None:
        return None
    left = _relative_drop(points, "left", "knee")
    right = _relative_drop(points, "right", "knee")
    if left is None or right is None:
        return None

    # Image y increases downward.  Relative hip->knee drop is smaller for the
    # knee that is raised toward the torso.  Comparing each knee to its own hip
    # is more robust than comparing absolute frame y under torso tilt.
    margin = abs(left - right) / scale
    if margin < KNEE_RELATIVE_HEIGHT_MIN_MARGIN:
        return None
    side = "left" if left < right else "right"
    return {
        "anatomical_side": side,
        "authority": "dwpose_bilateral_hip_knee_relative_height",
        "left_hip_knee_drop_norm": round(left / scale, 3),
        "right_hip_knee_drop_norm": round(right / scale, 3),
        "score_margin": round(margin, 3),
    }


def _foot_height_binding(points: dict[str, tuple[float, float] | None]) -> dict[str, Any] | None:
    scale = _leg_scale(points)
    if scale is None:
        return None
    left = _relative_drop(points, "left", "ankle")
    right = _relative_drop(points, "right", "ankle")
    if left is None or right is None:
        return None

    # The planted foot normally extends farther downward from its own hip than
    # a visibly lifted foot.  Only publish when the bilateral difference is
    # large enough to be useful; otherwise keep Qwen's relation unsigned.
    margin = abs(left - right) / scale
    if margin < FOOT_RELATIVE_HEIGHT_MIN_MARGIN:
        return None
    planted = "left" if left > right else "right"
    return {
        "planted_side": planted,
        "lifted_side": phase4b4._opposite(planted),
        "authority": "dwpose_bilateral_hip_ankle_relative_height",
        "left_hip_ankle_drop_norm": round(left / scale, 3),
        "right_hip_ankle_drop_norm": round(right / scale, 3),
        "score_margin": round(margin, 3),
    }


def _side_leg_observed(points: dict[str, tuple[float, float] | None], side: str) -> bool:
    return sum(points.get(f"{side}_{joint}") is not None for joint in ("hip", "knee", "ankle")) >= 2


def _bind_item(
    item: dict[str, Any],
    *,
    side: str,
    composer_text: str,
    semantic_relation: str,
    authority: str,
    diagnostics: dict[str, Any],
    source_noun: str,
) -> dict[str, Any]:
    source_text = str(item.get("text") or "")
    item["composer_text"] = composer_text
    item["normalized_text"] = composer_text
    item["promotion_status"] = "accepted_specialist_lateralized_candidate"
    item["specialist_owner"] = "dwpose_anatomical_relation_binding"
    item["laterality_binding"] = {
        "anatomical_side": side,
        "authority": authority,
        "semantic_relation": semantic_relation,
        "source_text": source_text,
        "source_side_label": _source_side(source_text, source_noun),
        "source_side_label_trusted": False,
        **diagnostics,
    }
    return copy.deepcopy(item["laterality_binding"])


def _single_relation_items(
    configuration: list[dict[str, Any]],
    pattern: re.Pattern[str],
) -> list[dict[str, Any]]:
    return [
        item for item in configuration
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and pattern.search(str(item.get("text")))
        and not BOTH_LEGS_RE.search(str(item.get("text")))
    ]


def _bind_leg_laterality(
    configuration: list[dict[str, Any]],
    points: dict[str, tuple[float, float] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    out = copy.deepcopy(configuration)
    bindings: list[dict[str, Any]] = []
    warnings: list[str] = []

    knee_items = _single_relation_items(out, KNEE_RAISED_RE)
    planted_items = _single_relation_items(out, FOOT_PLANTED_RE)
    lifted_items = _single_relation_items(out, FOOT_LIFTED_RE)

    knee_binding = None
    if len(knee_items) == 1:
        knee_binding = _raised_knee_binding(points)
        if knee_binding is None:
            warnings.append("raised_knee_anatomical_laterality_unresolved")
        else:
            side = str(knee_binding["anatomical_side"])
            item = knee_items[0]
            diagnostics = {k: v for k, v in knee_binding.items() if k not in {"anatomical_side", "authority"}}
            bindings.append(_bind_item(
                item,
                side=side,
                composer_text=f"{side} knee raised",
                semantic_relation="knee_raised",
                authority=str(knee_binding["authority"]),
                diagnostics=diagnostics,
                source_noun="knee",
            ))
    elif len(knee_items) > 1:
        warnings.append("raised_knee_laterality_unresolved_multiple_relations")

    foot_binding = None
    if len(planted_items) == 1 and len(lifted_items) == 1:
        foot_binding = _foot_height_binding(points)
        if foot_binding is None:
            warnings.append("asymmetric_feet_anatomical_laterality_unresolved")
        else:
            planted_side = str(foot_binding["planted_side"])
            lifted_side = str(foot_binding["lifted_side"])
            diagnostics = {k: v for k, v in foot_binding.items() if k not in {"planted_side", "lifted_side", "authority"}}
            bindings.append(_bind_item(
                planted_items[0],
                side=planted_side,
                composer_text=f"{planted_side} foot planted",
                semantic_relation="foot_planted",
                authority=str(foot_binding["authority"]),
                diagnostics=diagnostics,
                source_noun="foot",
            ))
            bindings.append(_bind_item(
                lifted_items[0],
                side=lifted_side,
                composer_text=f"{lifted_side} foot slightly lifted",
                semantic_relation="foot_lifted",
                authority=str(foot_binding["authority"]),
                diagnostics=diagnostics,
                source_noun="foot",
            ))
    elif len(planted_items) > 1 or len(lifted_items) > 1:
        warnings.append("asymmetric_feet_laterality_unresolved_multiple_relations")

    # Common pose-candidate pairing: "one knee raised" + "one foot planted".
    # If the knee has been independently bound, the planted support foot can be
    # attached to the opposite observed leg without trusting Qwen's side word.
    if len(planted_items) == 1 and not lifted_items and knee_binding is not None:
        planted_item = planted_items[0]
        if not isinstance(planted_item.get("laterality_binding"), dict):
            raised_side = str(knee_binding["anatomical_side"])
            planted_side = phase4b4._opposite(raised_side)
            if _side_leg_observed(points, planted_side):
                bindings.append(_bind_item(
                    planted_item,
                    side=planted_side,
                    composer_text=f"{planted_side} foot planted",
                    semantic_relation="foot_planted",
                    authority="opposite_of_dwpose_bound_raised_knee_with_observed_leg_chain",
                    diagnostics={},
                    source_noun="foot",
                ))
            else:
                warnings.append("planted_foot_laterality_unresolved_missing_opposite_dwpose_chain")

    return out, bindings, warnings


_BASE_BIND_CONFIGURATION_LATERALITY = phase4b4._bind_configuration_laterality


def _bind_configuration_laterality(
    configuration: list[dict[str, Any]],
    points: dict[str, tuple[float, float] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    normalized, bindings, warnings = _BASE_BIND_CONFIGURATION_LATERALITY(configuration, points)
    normalized, leg_bindings, leg_warnings = _bind_leg_laterality(normalized, points)
    return normalized, bindings + leg_bindings, warnings + leg_warnings


def main() -> int:
    phase4b4._bind_configuration_laterality = _bind_configuration_laterality
    phase4b4.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b4.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b4.main()


if __name__ == "__main__":
    raise SystemExit(main())
