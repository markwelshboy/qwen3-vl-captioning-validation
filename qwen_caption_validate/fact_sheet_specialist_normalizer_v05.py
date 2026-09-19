from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v04 as base
from .caption_perception_policy import _dwpose_points

SCHEMA_VERSION = "caption-fact-sheet-0.2.4"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.4"

TURN_DIRECTION_MIN_ABS_YAW_DEG = 15.0

HAND_HIP_RE = re.compile(
    r"(?:\b(?:one\s+)?hand\b.{0,28}\b(?:hip|waist)\b|\b(?:hip|waist)\b.{0,28}\b(?:one\s+)?hand\b)",
    re.I,
)
HAND_HIP_PROXIMITY_RE = re.compile(
    r"\b(?:near|beside|adjacent\s+to|close\s+to|positioned\s+near|placed\s+near)\b",
    re.I,
)
HAND_HIP_CONTACT_RE = re.compile(
    r"\b(?:on|against|touch(?:es|ing|ed)?|press(?:ed|ing)?\s+against)\b",
    re.I,
)
RELAXED_ARM_RE = re.compile(
    r"\b(?:(?:left|right|other|one)\s+)?arm\b.{0,32}\b(?:relaxed|hang(?:s|ing)?|at\s+(?:the\s+)?side)\b",
    re.I,
)
BOTH_ARMS_RE = re.compile(r"\b(?:both|two)\s+arms?\b", re.I)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _joint_angle(a: tuple[float, float] | None, b: tuple[float, float] | None, c: tuple[float, float] | None) -> float | None:
    if a is None or b is None or c is None:
        return None
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    denom = math.hypot(ux, uy) * math.hypot(vx, vy)
    if denom <= 1e-8:
        return None
    cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / denom))
    return math.degrees(math.acos(cosine))


def _turn_direction(yaw_deg: Any) -> str | None:
    if not isinstance(yaw_deg, (int, float)) or not math.isfinite(float(yaw_deg)):
        return None
    yaw = float(yaw_deg)
    if abs(yaw) <= TURN_DIRECTION_MIN_ABS_YAW_DEG:
        return None
    # Camera coordinates use +X toward image/frame right.  With the signed-yaw
    # convention in sam3d_caption_orientation_v01, positive yaw means the body
    # forward vector has rotated toward frame left relative to the subject->camera ray.
    return "frame_left" if yaw > 0 else "frame_right"


def _apply_signed_torso_direction(sheet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(sheet)
    warnings: list[str] = []
    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    torso = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}
    if not torso or not torso.get("available"):
        return out, warnings

    for key in ("body_root_orientation", "upper_torso_orientation"):
        orientation = torso.get(key) if isinstance(torso.get(key), dict) else {}
        if not orientation:
            continue
        direction = _turn_direction(orientation.get("yaw_deg"))
        orientation["turn_direction"] = direction
        orientation["turn_direction_publishable"] = bool(direction and torso.get("composer_eligible"))
        orientation["turn_direction_convention"] = "positive_signed_yaw=frame_left; negative_signed_yaw=frame_right"

    summary = torso.get("caption_orientation") if isinstance(torso.get("caption_orientation"), dict) else {}
    upper = torso.get("upper_torso_orientation") if isinstance(torso.get("upper_torso_orientation"), dict) else {}
    root = torso.get("body_root_orientation") if isinstance(torso.get("body_root_orientation"), dict) else {}
    preferred = upper if upper.get("orientation_band") is not None else root
    summary["preferred_turn_direction"] = preferred.get("turn_direction")
    summary["turn_direction_publishable"] = bool(
        torso.get("composer_eligible") and preferred.get("turn_direction_publishable")
    )
    summary["turn_direction_reference"] = "direction the resolved torso faces in image-frame coordinates"
    torso["caption_orientation"] = summary
    torso["note_phase4b4_signed_yaw"] = (
        "Signed SAM3D camera-relative yaw is retained as frame-left/frame-right turn direction. "
        "Direction is omitted inside the near-frontal deadband; unsigned magnitude remains separately available."
    )
    return out, warnings


def _body_scale(points: dict[str, tuple[float, float] | None]) -> float | None:
    shoulder = _distance(points.get("left_shoulder"), points.get("right_shoulder"))
    hip = _distance(points.get("left_hip"), points.get("right_hip"))
    values = [v for v in (shoulder, hip) if isinstance(v, (int, float)) and v > 1e-6]
    return max(values) if values else None


def _hand_on_hip_binding(points: dict[str, tuple[float, float] | None]) -> dict[str, Any] | None:
    scale = _body_scale(points)
    if scale is None:
        return None

    candidates: list[dict[str, Any]] = []
    for side in ("left", "right"):
        wrist = points.get(f"{side}_wrist")
        hip = points.get(f"{side}_hip")
        if wrist is None or hip is None:
            continue
        distance = _distance(wrist, hip)
        if distance is None:
            continue
        distance_norm = distance / scale
        elbow_angle = _joint_angle(
            points.get(f"{side}_shoulder"),
            points.get(f"{side}_elbow"),
            wrist,
        )
        # A hand-on-hip arm is usually both spatially close to its hip and bent.
        # Penalize nearly straight arms so a hanging wrist beside the pelvis is
        # not mistaken for the hip-contact hand.
        if elbow_angle is None:
            straight_penalty = 0.15
        elif elbow_angle >= 150.0:
            straight_penalty = 0.55
        elif elbow_angle >= 130.0:
            straight_penalty = 0.30
        elif elbow_angle >= 115.0:
            straight_penalty = 0.15
        else:
            straight_penalty = 0.0
        candidates.append({
            "side": side,
            "wrist_hip_distance_norm": round(distance_norm, 3),
            "elbow_angle_deg": round(elbow_angle, 1) if elbow_angle is not None else None,
            "score": distance_norm + straight_penalty,
        })

    if not candidates:
        return None
    candidates.sort(key=lambda x: float(x["score"]))
    best = candidates[0]
    if float(best["wrist_hip_distance_norm"]) > 0.95:
        return None

    if len(candidates) >= 2:
        runner_up = candidates[1]
        margin = float(runner_up["score"]) - float(best["score"])
        if margin < 0.20:
            return None
    else:
        # With only one visible candidate, require a much stronger geometric fit.
        if float(best["wrist_hip_distance_norm"]) > 0.45:
            return None
        margin = None

    return {
        "anatomical_side": best["side"],
        "authority": "dwpose_named_joints_plus_relation_geometry",
        "wrist_hip_distance_norm": best["wrist_hip_distance_norm"],
        "elbow_angle_deg": best["elbow_angle_deg"],
        "score_margin": round(margin, 3) if margin is not None else None,
    }


def _opposite(side: str) -> str:
    return "right" if side == "left" else "left"


def _side_arm_observed(points: dict[str, tuple[float, float] | None], side: str) -> bool:
    return sum(points.get(f"{side}_{joint}") is not None for joint in ("shoulder", "elbow", "wrist")) >= 2


def _source_arm_side(text: str) -> str | None:
    match = re.search(r"\b(left|right)\s+arm\b", text, re.I)
    return match.group(1).lower() if match else None


def _hand_hip_relation_strength(text: str) -> str | None:
    """Classify Qwen's semantic strength without allowing geometry to strengthen it."""
    if not HAND_HIP_RE.search(text):
        return None
    if HAND_HIP_PROXIMITY_RE.search(text):
        return "proximity"
    if HAND_HIP_CONTACT_RE.search(text):
        return "contact"
    return "unspecified"


def _bind_configuration_laterality(
    configuration: list[dict[str, Any]],
    points: dict[str, tuple[float, float] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    out = copy.deepcopy(configuration)
    bindings: list[dict[str, Any]] = []
    warnings: list[str] = []

    proximity_items = [
        item for item in out
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and _hand_hip_relation_strength(str(item.get("text"))) == "proximity"
        and not re.search(r"\b(?:both|two)\s+hands\b", str(item.get("text")), re.I)
    ]
    if proximity_items:
        warnings.append("hand_near_hip_relation_not_promoted_to_contact")

    hip_items = [
        item for item in out
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and _hand_hip_relation_strength(str(item.get("text"))) == "contact"
        and not re.search(r"\b(?:both|two)\s+hands\b", str(item.get("text")), re.I)
    ]
    if len(hip_items) != 1:
        return out, bindings, warnings

    binding = _hand_on_hip_binding(points)
    if binding is None:
        warnings.append("hand_on_hip_anatomical_laterality_unresolved")
        return out, bindings, warnings

    side = str(binding["anatomical_side"])
    hip_item = hip_items[0]
    source_text = str(hip_item.get("text") or "")
    target = "waist" if re.search(r"\bwaist\b", source_text, re.I) else "hip"
    composer_text = f"{side} hand resting on {target}"
    hip_item["composer_text"] = composer_text
    hip_item["normalized_text"] = composer_text
    hip_item["promotion_status"] = "accepted_specialist_lateralized_candidate"
    hip_item["specialist_owner"] = "dwpose_anatomical_relation_binding"
    hip_item["laterality_binding"] = {
        **binding,
        "semantic_relation": "hand_on_hip" if target == "hip" else "hand_at_waist",
        "source_text": source_text,
        "source_side_label_trusted": False,
    }
    bindings.append(copy.deepcopy(hip_item["laterality_binding"]))

    # A paired asymmetric description often arrives as e.g. "right hand on hip"
    # plus "left arm relaxed at side". Qwen's left/right token is not authority:
    # once geometry has bound the hip hand, the distinct relaxed-arm relation can
    # be assigned to the opposite anatomical chain when that chain is observed.
    relaxed_items = [
        item for item in out
        if isinstance(item, dict)
        and item is not hip_item
        and isinstance(item.get("text"), str)
        and RELAXED_ARM_RE.search(str(item.get("text")))
        and not BOTH_ARMS_RE.search(str(item.get("text")))
    ]
    if len(relaxed_items) > 1:
        warnings.append("relaxed_arm_laterality_unresolved_multiple_relations")
        return out, bindings, warnings
    if not relaxed_items:
        return out, bindings, warnings

    other_side = _opposite(side)
    item = relaxed_items[0]
    text = str(item.get("text") or "")
    if not _side_arm_observed(points, other_side):
        warnings.append("relaxed_arm_laterality_unresolved_missing_opposite_dwpose_chain")
        return out, bindings, warnings

    arm_text = f"{other_side} arm relaxed at side"
    item["composer_text"] = arm_text
    item["normalized_text"] = arm_text
    item["promotion_status"] = "accepted_specialist_lateralized_candidate"
    item["specialist_owner"] = "dwpose_anatomical_relation_binding"
    item["laterality_binding"] = {
        "anatomical_side": other_side,
        "authority": "opposite_of_dwpose_bound_hand_on_hip_with_observed_arm_chain",
        "semantic_relation": "arm_relaxed_at_side",
        "source_text": text,
        "source_side_label": _source_arm_side(text),
        "source_side_label_trusted": False,
    }
    bindings.append(copy.deepcopy(item["laterality_binding"]))

    return out, bindings, warnings


def _load_dwpose_points_for_sheet(sheet: dict[str, Any]) -> tuple[dict[str, tuple[float, float] | None] | None, str | None]:
    sources = sheet.get("sources") if isinstance(sheet.get("sources"), dict) else {}
    policy_text = sources.get("perception_policy")
    if not policy_text:
        return None, "perception_policy_source_missing"
    policy_path = Path(str(policy_text)).expanduser()
    if not policy_path.is_file():
        return None, "perception_policy_source_not_found"
    try:
        policy = _read_json(policy_path)
        size = policy.get("image_size")
        dwpose_text = (policy.get("sources") or {}).get("dwpose") if isinstance(policy.get("sources"), dict) else None
        if not isinstance(size, list) or len(size) < 2 or not dwpose_text:
            return None, "dwpose_binding_inputs_missing"
        dwpose_path = Path(str(dwpose_text)).expanduser()
        if not dwpose_path.is_file():
            return None, "dwpose_source_not_found"
        points = _dwpose_points(_read_json(dwpose_path), int(size[0]), int(size[1]))
        return points, None
    except Exception as exc:
        return None, f"dwpose_relation_binding_failed:{type(exc).__name__}"


def _apply_relation_laterality(sheet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(sheet)
    warnings: list[str] = []
    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    if not configuration:
        return out, warnings

    points, error = _load_dwpose_points_for_sheet(out)
    if points is None:
        if error:
            warnings.append(error)
        return out, warnings

    normalized, bindings, bind_warnings = _bind_configuration_laterality(configuration, points)
    warnings.extend(bind_warnings)
    body["configuration"] = normalized
    laterality = body.get("anatomical_laterality") if isinstance(body.get("anatomical_laterality"), dict) else {}
    laterality["relation_bindings"] = bindings
    laterality["relation_binding_policy"] = (
        "Qwen supplies the semantic relation but never the side label. DWPose named joints and observed 2D geometry bind "
        "the relation to anatomical left/right; unresolved relations remain unlateralized."
    )
    body["anatomical_laterality"] = laterality
    return out, warnings


def _apply_phase4b4(sheet: dict[str, Any]) -> dict[str, Any]:
    out, torso_warnings = _apply_signed_torso_direction(sheet)
    out, relation_warnings = _apply_relation_laterality(out)
    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    existing = [str(x) for x in (audit.get("warnings") or []) if x]
    audit["warnings"] = sorted(set(existing + torso_warnings + relation_warnings))
    audit["phase"] = "4B.4"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        signed_sam3d_yaw_can_publish_frame_turn_direction=True,
        near_frontal_torso_does_not_publish_turn_direction=True,
        qwen_never_owns_anatomical_side_assignment=True,
        qwen_source_side_labels_are_ignored_during_relation_rebinding=True,
        dwpose_named_joints_can_bind_supported_semantic_relations_to_anatomical_side=True,
        unresolved_relation_laterality_remains_unlateralized=True,
        hand_hip_proximity_is_never_strengthened_to_contact=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


_BASE_BUILD_FACT_SHEET = base._build_fact_sheet


def _build_fact_sheet(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _apply_phase4b4(_BASE_BUILD_FACT_SHEET(*args, **kwargs))


def main() -> int:
    base._build_fact_sheet = _build_fact_sheet
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
