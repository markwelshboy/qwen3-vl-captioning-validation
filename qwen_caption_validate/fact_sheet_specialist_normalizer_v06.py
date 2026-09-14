from __future__ import annotations

import copy
import math
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
KNEE_THIGH_ANGLE_MIN_MARGIN_DEG = 15.0
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


def _thigh_angle_from_down_vertical(
    points: dict[str, tuple[float, float] | None],
    side: str,
) -> float | None:
    """Return hip->knee angle away from straight-down image vertical.

    A normally supporting thigh points mostly down (near 0 degrees). A knee
    raised toward the torso makes the thigh more horizontal and therefore has
    a larger angle. This is useful when two hip->knee vertical drops differ
    only modestly because of perspective, but the full 2D joint geometry still
    clearly identifies which thigh is lifted.
    """
    hip = points.get(f"{side}_hip")
    knee = points.get(f"{side}_knee")
    if hip is None or knee is None:
        return None
    dx = float(knee[0] - hip[0])
    dy = float(knee[1] - hip[1])
    length = math.hypot(dx, dy)
    if length <= 1e-8:
        return None
    cosine = max(-1.0, min(1.0, dy / length))
    return math.degrees(math.acos(cosine))


def _raised_knee_binding(points: dict[str, tuple[float, float] | None]) -> dict[str, Any] | None:
    scale = _leg_scale(points)
    if scale is None:
        return None
    left = _relative_drop(points, "left", "knee")
    right = _relative_drop(points, "right", "knee")
    if left is None or right is None:
        return None

    left_angle = _thigh_angle_from_down_vertical(points, "left")
    right_angle = _thigh_angle_from_down_vertical(points, "right")

    # Signal 1: image y increases downward. Relative hip->knee drop is smaller
    # for a knee raised toward the torso. Comparing each knee to its own hip
    # avoids much of the error from a tilted pelvis.
    drop_margin = abs(left - right) / scale
    drop_side = None
    if drop_margin >= KNEE_RELATIVE_HEIGHT_MIN_MARGIN:
        drop_side = "left" if left < right else "right"

    # Signal 2: in a full-body DWPose skeleton, a raised thigh usually rotates
    # substantially away from straight-down vertical. This can remain clear
    # even when the raw vertical-drop margin is modest. Qwen has already
    # supplied the semantic relation "one knee raised"; this geometry is used
    # only to bind that relation to anatomical left/right.
    angle_margin = None
    angle_side = None
    if left_angle is not None and right_angle is not None:
        angle_margin = abs(left_angle - right_angle)
        if angle_margin >= KNEE_THIGH_ANGLE_MIN_MARGIN_DEG:
            angle_side = "left" if left_angle > right_angle else "right"

    # If both independent 2D cues are strong but disagree, do not publish a
    # side. Otherwise use the strong drop signal first, then the thigh-angle
    # fallback.
    if drop_side and angle_side and drop_side != angle_side:
        return None
    side = drop_side or angle_side
    if side is None:
        return None

    authority = (
        "dwpose_bilateral_hip_knee_relative_height"
        if drop_side
        else "dwpose_bilateral_thigh_angle_from_vertical"
    )
    return {
        "anatomical_side": side,
        "authority": authority,
        "left_hip_knee_drop_norm": round(left / scale, 3),
        "right_hip_knee_drop_norm": round(right / scale, 3),
        "score_margin": round(drop_margin, 3),
        "left_thigh_angle_from_down_deg": round(left_angle, 1) if left_angle is not None else None,
        "right_thigh_angle_from_down_deg": round(right_angle, 1) if right_angle is not None else None,
        "thigh_angle_margin_deg": round(angle_margin, 1) if angle_margin is not None else None,
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
    # a visibly lifted foot. Only publish when the bilateral difference is
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
        if knee_binding is not None:
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

    # When both planted and lifted feet are explicitly described, bind both
    # directly from the bilateral ankle geometry.
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

    # Common semantic pairing: "one knee raised" + "one foot planted". There
    # are two valid ways to resolve this without trusting Qwen's side word:
    #
    # 1) If the raised knee is directly bound from knee/thigh geometry, assign
    #    the planted support foot to the opposite observed leg.
    # 2) If knee geometry is ambiguous but the full-body ankle geometry clearly
    #    identifies the planted support foot, bind that foot directly and assign
    #    the raised-knee relation to the opposite observed leg.
    #
    # The second path is important for poses where both thighs are nearly
    # horizontal (so the knees have similar frame height) but one ankle clearly
    # reaches the floor while the other leg is folded upward.
    if len(planted_items) == 1 and not lifted_items and len(knee_items) == 1:
        planted_item = planted_items[0]
        knee_item = knee_items[0]

        if knee_binding is not None:
            raised_side = str(knee_binding["anatomical_side"])
            planted_side = phase4b4._opposite(raised_side)
            if not isinstance(planted_item.get("laterality_binding"), dict):
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
        else:
            support_binding = _foot_height_binding(points)
            if support_binding is not None:
                planted_side = str(support_binding["planted_side"])
                raised_side = phase4b4._opposite(planted_side)
                diagnostics = {
                    k: v for k, v in support_binding.items()
                    if k not in {"planted_side", "lifted_side", "authority"}
                }
                if _side_leg_observed(points, planted_side) and _side_leg_observed(points, raised_side):
                    if not isinstance(planted_item.get("laterality_binding"), dict):
                        bindings.append(_bind_item(
                            planted_item,
                            side=planted_side,
                            composer_text=f"{planted_side} foot planted",
                            semantic_relation="foot_planted",
                            authority=str(support_binding["authority"]),
                            diagnostics=diagnostics,
                            source_noun="foot",
                        ))
                    if not isinstance(knee_item.get("laterality_binding"), dict):
                        knee_diagnostics = {
                            **diagnostics,
                            "support_planted_side": planted_side,
                        }
                        bindings.append(_bind_item(
                            knee_item,
                            side=raised_side,
                            composer_text=f"{raised_side} knee raised",
                            semantic_relation="knee_raised",
                            authority="opposite_of_dwpose_bound_planted_foot_with_observed_leg_chain",
                            diagnostics=knee_diagnostics,
                            source_noun="knee",
                        ))
                else:
                    warnings.append("raised_knee_support_pair_unresolved_missing_leg_chain")

    if len(knee_items) == 1 and not isinstance(knee_items[0].get("laterality_binding"), dict):
        warnings.append("raised_knee_anatomical_laterality_unresolved")

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
