from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v06 as phase4b5

phase4b4 = phase4b5.phase4b4

SCHEMA_VERSION = "caption-fact-sheet-0.2.6"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.6"

SUPPORT_AXIS_MOSTLY_UPRIGHT_MAX_DEG = 15.0
ELEVATED_LEG_HIGH_ANKLE_GAP_NORM = 0.75
FORWARD_TORSO_RE = re.compile(
    r"\b(?:torso|upper\s+body|body)\b.{0,36}\b(?:bent|bend(?:s|ing)?|lean(?:s|ing|ed)?|hing(?:e|es|ed|ing))\b.{0,20}\bforward\b",
    re.I,
)

_BASE_APPLY_RELATION_LATERALITY = phase4b4._apply_relation_laterality


def _segment_angle_from_down_vertical(
    points: dict[str, tuple[float, float] | None],
    side: str,
    distal_joint: str,
) -> float | None:
    origin = points.get(f"{side}_hip")
    distal = points.get(f"{side}_{distal_joint}")
    if origin is None or distal is None:
        return None
    dx = float(distal[0] - origin[0])
    dy = float(distal[1] - origin[1])
    length = math.hypot(dx, dy)
    if length <= 1e-8:
        return None
    cosine = max(-1.0, min(1.0, dy / length))
    return math.degrees(math.acos(cosine))


def _relation_items(configuration: list[dict[str, Any]], relation: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in configuration:
        if not isinstance(item, dict):
            continue
        binding = item.get("laterality_binding") if isinstance(item.get("laterality_binding"), dict) else {}
        if binding.get("semantic_relation") == relation and binding.get("anatomical_side") in {"left", "right"}:
            out.append(item)
    return out


def _support_geometry(
    configuration: list[dict[str, Any]],
    points: dict[str, tuple[float, float] | None],
) -> dict[str, Any] | None:
    """Resolve whole-body support shape for an already-semantic asymmetric leg pose.

    Qwen still supplies the semantic relation (planted/lifted foot or raised knee).
    DWPose is used only after those relations have been anatomically bound.  The
    goal is to preserve the global silhouette: a near-vertical weight-bearing
    leg can coexist with a locally forward-bent upper body.
    """
    planted = _relation_items(configuration, "foot_planted")
    raised_knee = _relation_items(configuration, "knee_raised")
    lifted_foot = _relation_items(configuration, "foot_lifted")
    if len(planted) != 1:
        return None

    elevated_candidates = raised_knee if len(raised_knee) == 1 else lifted_foot
    if len(elevated_candidates) != 1:
        return None

    planted_binding = planted[0]["laterality_binding"]
    elevated_binding = elevated_candidates[0]["laterality_binding"]
    support_side = str(planted_binding.get("anatomical_side"))
    elevated_side = str(elevated_binding.get("anatomical_side"))
    if support_side not in {"left", "right"} or elevated_side not in {"left", "right"}:
        return None
    if support_side == elevated_side:
        return None

    scale = phase4b5._leg_scale(points)
    support_ankle = points.get(f"{support_side}_ankle")
    elevated_ankle = points.get(f"{elevated_side}_ankle")
    if scale is None or support_ankle is None or elevated_ankle is None:
        return None

    support_axis_angle = _segment_angle_from_down_vertical(points, support_side, "ankle")
    if support_axis_angle is None:
        return None

    ankle_height_gap_norm = float(support_ankle[1] - elevated_ankle[1]) / scale
    mostly_upright = support_axis_angle <= SUPPORT_AXIS_MOSTLY_UPRIGHT_MAX_DEG
    elevated_leg_height = (
        "high" if ankle_height_gap_norm >= ELEVATED_LEG_HIGH_ANKLE_GAP_NORM else "unspecified"
    )

    return {
        "available": True,
        "composer_eligible": bool(mostly_upright),
        "overall_shape": "mostly_upright_over_support_leg" if mostly_upright else "unresolved",
        "support_side": support_side,
        "elevated_side": elevated_side,
        "elevated_relation": str(elevated_binding.get("semantic_relation")),
        "elevated_leg_height": elevated_leg_height,
        "support_axis_angle_from_vertical_deg": round(support_axis_angle, 1),
        "elevated_ankle_height_gap_norm": round(ankle_height_gap_norm, 3),
        "authority": "dwpose_bound_leg_relations_plus_hip_to_support_ankle_axis",
        "note": (
            "Global support shape is distinct from local torso bend: a near-vertical planted-leg axis "
            "means the overall stance remains mostly upright even when the upper body reaches forward."
        ),
    }


def _apply_support_shape_to_configuration(
    configuration: list[dict[str, Any]],
    support: dict[str, Any],
) -> list[dict[str, Any]]:
    out = copy.deepcopy(configuration)
    if not support.get("composer_eligible"):
        return out

    support_side = str(support.get("support_side") or "")
    elevated_side = str(support.get("elevated_side") or "")
    elevated_height = str(support.get("elevated_leg_height") or "")

    # Scope an existing Qwen forward-bend relation to the upper body.  This is
    # the key distinction between local hinge and global body silhouette.
    for item in out:
        if not isinstance(item, dict):
            continue
        source_text = str(item.get("text") or "")
        composer_text = str(item.get("composer_text") or "")
        if FORWARD_TORSO_RE.search(source_text) or FORWARD_TORSO_RE.search(composer_text):
            item["composer_text"] = "upper body bent forward from the hips while overall stance remains mostly upright"
            item["normalized_text"] = item["composer_text"]
            item["promotion_status"] = "accepted_specialist_shape_refined_candidate"
            item["specialist_owner"] = "dwpose_global_support_shape_plus_route_scoped_body_relation"
            item["support_shape_refinement"] = {
                "source_text": source_text,
                "support_side": support_side,
                "overall_shape": support.get("overall_shape"),
                "support_axis_angle_from_vertical_deg": support.get("support_axis_angle_from_vertical_deg"),
            }

    # If the elevated limb is far above the support ankle, do not retain weak
    # wording such as "slightly lifted".  Preserve the already-bound anatomical
    # side while strengthening only the magnitude supplied by visible geometry.
    if elevated_height == "high":
        for item in out:
            if not isinstance(item, dict):
                continue
            binding = item.get("laterality_binding") if isinstance(item.get("laterality_binding"), dict) else {}
            if binding.get("anatomical_side") != elevated_side:
                continue
            relation = binding.get("semantic_relation")
            if relation == "knee_raised":
                item["composer_text"] = f"{elevated_side} knee raised high"
                item["normalized_text"] = item["composer_text"]
                item["promotion_status"] = "accepted_specialist_shape_refined_candidate"
            elif relation == "foot_lifted":
                item["composer_text"] = f"{elevated_side} leg lifted high with foot well off the floor"
                item["normalized_text"] = item["composer_text"]
                item["promotion_status"] = "accepted_specialist_shape_refined_candidate"
            else:
                continue
            item["specialist_owner"] = "dwpose_global_support_shape_plus_route_scoped_body_relation"
            item["support_shape_refinement"] = {
                "elevated_side": elevated_side,
                "elevated_leg_height": "high",
                "elevated_ankle_height_gap_norm": support.get("elevated_ankle_height_gap_norm"),
            }

    return out


def _apply_relation_laterality(sheet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out, warnings = _BASE_APPLY_RELATION_LATERALITY(sheet)
    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    if not configuration:
        return out, warnings

    points, error = phase4b4._load_dwpose_points_for_sheet(out)
    if points is None:
        return out, warnings + ([error] if error else [])

    support = _support_geometry(configuration, points)
    if support is None:
        return out, warnings

    body["support_geometry"] = support
    body["configuration"] = _apply_support_shape_to_configuration(configuration, support)
    return out, warnings


def main() -> int:
    # v06 still owns relation laterality.  Wrap only the relation-laterality
    # application so support-shape semantics are added after side binding.
    phase4b4._apply_relation_laterality = _apply_relation_laterality
    phase4b5.SCHEMA_VERSION = SCHEMA_VERSION
    phase4b5.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return phase4b5.main()


if __name__ == "__main__":
    raise SystemExit(main())
