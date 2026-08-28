from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .dwpose_compat import _candidate_array
from .runner import model_slug, resolve_model_id


BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
IDX = {name: i for i, name in enumerate(BODY18)}

POSTURE_RE = {
    "standing": re.compile(r"\b(?:stands?|standing|stood)\b", re.I),
    "seated": re.compile(r"\b(?:sit(?:s|ting)?|sat|seated)\b", re.I),
    "kneeling": re.compile(r"\b(?:kneel(?:s|ed|ing)?)\b", re.I),
    "lying": re.compile(r"\b(?:lies|lying|lay|reclin(?:es|ed|ing)|reclined)\b", re.I),
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _analysis_root(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("analysis")
    return nested if isinstance(nested, dict) else value


def _fusion_root(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("fusion")
    return nested if isinstance(nested, dict) else value


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _visible(point: np.ndarray | None) -> bool:
    return bool(
        point is not None
        and len(point) >= 2
        and np.isfinite(point[0])
        and np.isfinite(point[1])
        and point[0] >= 0.0
        and point[1] >= 0.0
    )


def _target_person(dwpose: dict[str, Any]) -> np.ndarray | None:
    candidates = _candidate_array(dwpose.get("raw_pose") or {})
    target_index = (dwpose.get("derived") or {}).get("target_person_index")
    if isinstance(target_index, int) and 0 <= target_index < len(candidates):
        return candidates[target_index]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _point(person: np.ndarray | None, name: str) -> np.ndarray | None:
    if person is None or name not in IDX or IDX[name] >= len(person):
        return None
    point = np.asarray(person[IDX[name]], dtype=np.float64)[:2]
    return point if _visible(point) else None


def calculate_angle(p1: Any, p2: Any, p3: Any) -> float | None:
    """Return the 2-D angle p1 -> p2 -> p3 in degrees, or None if unavailable."""
    if p1 is None or p2 is None or p3 is None:
        return None
    a = np.asarray(p1, dtype=np.float64)[:2]
    b = np.asarray(p2, dtype=np.float64)[:2]
    c = np.asarray(p3, dtype=np.float64)[:2]
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return None
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-9:
        return None
    cosine = float(np.dot(v1, v2) / denom)
    return round(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), 2)


def _distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _angle_from_vertical(top: np.ndarray | None, bottom: np.ndarray | None) -> float | None:
    if top is None or bottom is None:
        return None
    dx = float(bottom[0] - top[0])
    dy = float(bottom[1] - top[1])
    if math.hypot(dx, dy) < 1e-9:
        return None
    return round(float(math.degrees(math.atan2(dx, dy))), 2)


def _angle_from_horizontal(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    if math.hypot(dx, dy) < 1e-9:
        return None
    raw = abs(float(math.degrees(math.atan2(dy, dx)))) % 180.0
    if raw > 90.0:
        raw = 180.0 - raw
    return round(raw, 2)


def _band(score: float) -> str:
    if score >= 0.80:
        return "strong"
    if score >= 0.60:
        return "moderate"
    if score >= 0.40:
        return "weak"
    return "withheld"


def _primitive(
    primitive_id: str,
    label: str,
    score: float,
    *,
    support: list[str] | None = None,
    limitations: list[str] | None = None,
    subsumes: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = max(0.0, min(1.0, float(score)))
    return {
        "id": primitive_id,
        "label": label,
        "support_score": round(score, 3),
        "confidence_band": _band(score),
        "caption_preferred": score >= 0.60,
        "support": support or [],
        "limitations": limitations or [],
        "subsumes": subsumes or [],
        "details": details or {},
    }


def _geometry_features(person: np.ndarray | None, dwpose: dict[str, Any]) -> dict[str, Any]:
    if person is None:
        return {
            "target_skeleton_available": False,
            "visible_joints": [],
            "angles_deg": {},
            "normalized_distances": {},
            "directions_deg": {},
            "connectivity": {},
        }

    def p(name: str) -> np.ndarray | None:
        return _point(person, name)

    left_shoulder, right_shoulder = p("left_shoulder"), p("right_shoulder")
    left_hip, right_hip = p("left_hip"), p("right_hip")
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    neck_point = p("neck")
    neck = neck_point if neck_point is not None else shoulder_mid

    torso_len = _distance(neck, hip_mid)
    shoulder_width = _distance(left_shoulder, right_shoulder)
    hip_width = _distance(left_hip, right_hip)
    scales = [value for value in (torso_len, shoulder_width, hip_width) if value is not None and value > 1e-6]
    scale = max(scales) if scales else None

    angles: dict[str, float | None] = {}
    distances: dict[str, float | None] = {}
    directions: dict[str, float | None] = {}
    for side in ("left", "right"):
        angles[f"{side}_elbow"] = calculate_angle(p(f"{side}_shoulder"), p(f"{side}_elbow"), p(f"{side}_wrist"))
        angles[f"{side}_knee"] = calculate_angle(p(f"{side}_hip"), p(f"{side}_knee"), p(f"{side}_ankle"))
        angles[f"{side}_hip"] = calculate_angle(shoulder_mid, p(f"{side}_hip"), p(f"{side}_knee"))
        directions[f"{side}_arm_from_vertical"] = _angle_from_vertical(p(f"{side}_shoulder"), p(f"{side}_wrist"))
        directions[f"{side}_thigh_from_horizontal"] = _angle_from_horizontal(p(f"{side}_hip"), p(f"{side}_knee"))
        wrist_hip = _distance(p(f"{side}_wrist"), p(f"{side}_hip"))
        wrist_face = _distance(p(f"{side}_wrist"), p("nose"))
        distances[f"{side}_wrist_to_same_hip"] = round(wrist_hip / scale, 4) if wrist_hip is not None and scale else None
        distances[f"{side}_wrist_to_face"] = round(wrist_face / scale, 4) if wrist_face is not None and scale else None

    directions["torso_axis_from_vertical"] = _angle_from_vertical(neck, hip_mid)
    directions["shoulder_line_from_horizontal"] = _angle_from_horizontal(right_shoulder, left_shoulder)
    directions["hip_line_from_horizontal"] = _angle_from_horizontal(right_hip, left_hip)

    target = ((dwpose.get("derived") or {}).get("target") or {})
    visible_joints = [name for name in BODY18 if p(name) is not None]
    return {
        "target_skeleton_available": True,
        "visible_joints": visible_joints,
        "visible_joint_count": len(visible_joints),
        "pose_extent_hint": target.get("pose_extent_hint"),
        "connectivity": target.get("connectivity") or {},
        "scale": {
            "normalizer": "max(torso_length, shoulder_width, hip_width)",
            "value_normalized_image_coordinates": round(scale, 5) if scale else None,
        },
        "angles_deg": angles,
        "normalized_distances": distances,
        "directions_deg": directions,
    }


def _body_part_records(fusion: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in (fusion.get("qualified_body_parts") or []) if isinstance(item, dict)]


def _qualified_side(item: dict[str, Any]) -> str | None:
    state = item.get("fusion_v2") or {}
    side = state.get("qualified_anatomical_side") or item.get("anatomical_side")
    if side in {"left", "right"} and state.get("laterality_selection_usable", True):
        return str(side)
    return None


def _part_text(item: dict[str, Any]) -> str:
    return " ".join(str(item.get(field) or "") for field in ("geometry", "contact", "support")).strip()


def _leg_semantic_sides(fusion: dict[str, Any], pattern: re.Pattern[str]) -> set[str]:
    sides: set[str] = set()
    for item in _body_part_records(fusion):
        state = item.get("fusion_v2") or {}
        if state.get("selection_usable") is False:
            continue
        label = str(item.get("part") or "").lower().replace("_", " ")
        if not re.search(r"\b(?:leg|thigh|knee)\b", label):
            continue
        side = _qualified_side(item)
        if side and pattern.search(_part_text(item)):
            sides.add(side)
    return sides


def _posture_hypotheses(features: dict[str, Any], fusion: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    angles = features.get("angles_deg") or {}
    directions = features.get("directions_deg") or {}
    connectivity = features.get("connectivity") or {}
    summary = str(analysis.get("image_summary") or "")

    complete_legs = [side for side in ("left", "right") if bool((connectivity.get(f"{side}_leg") or {}).get("complete"))]
    hip_knee_sides = [side for side in ("left", "right") if int((connectivity.get(f"{side}_leg") or {}).get("visible_count") or 0) >= 2]
    knee_angles = [float(angles[f"{side}_knee"]) for side in complete_legs if angles.get(f"{side}_knee") is not None]
    thigh_angles = [float(directions[f"{side}_thigh_from_horizontal"]) for side in hip_knee_sides if directions.get(f"{side}_thigh_from_horizontal") is not None]
    torso_axis = _safe_float(directions.get("torso_axis_from_vertical"))

    hypotheses: list[dict[str, Any]] = []

    standing_score = 0.0
    standing_support: list[str] = []
    standing_limits: list[str] = []
    if len(knee_angles) == 2 and min(knee_angles) >= 155.0:
        standing_score += 0.55
        standing_support.append(f"both complete DWPose knee angles are near straight ({knee_angles[0]:.1f}°, {knee_angles[1]:.1f}°)")
    elif len(knee_angles) == 1 and knee_angles[0] >= 155.0:
        standing_score += 0.35
        standing_support.append(f"one complete DWPose knee angle is near straight ({knee_angles[0]:.1f}°)")
    if len(thigh_angles) >= 2 and min(thigh_angles) >= 55.0:
        standing_score += 0.15
        standing_support.append("both observed thighs project mostly downward")
    if torso_axis is not None and abs(torso_axis) <= 20.0:
        standing_score += 0.10
        standing_support.append(f"2-D torso axis is near vertical ({torso_axis:.1f}°)")
    if POSTURE_RE["standing"].search(summary):
        standing_score += 0.30
        standing_support.append("Analyze summary explicitly reports standing")
    standing_semantic_sides = _leg_semantic_sides(fusion, POSTURE_RE["standing"])
    if standing_semantic_sides == {"left", "right"} and len(hip_knee_sides) == 2:
        standing_score += 0.25
        standing_support.append("both governed leg records report standing and DWPose observes both hip-to-knee chains")
    if len(complete_legs) < 2:
        standing_limits.append("incomplete distal leg geometry; knee straightness is not invented")
    hypotheses.append(_primitive(
        "posture_standing", "standing", min(1.0, standing_score),
        support=standing_support, limitations=standing_limits,
        subsumes=["per-leg standing clauses", "knee/foot support prose"],
    ))

    seated_score = 0.0
    seated_support: list[str] = []
    seated_limits: list[str] = []
    seated_knees = [value for value in knee_angles if 60.0 <= value <= 130.0]
    horizontal_thighs = [value for value in thigh_angles if value <= 35.0]
    if len(seated_knees) == 2:
        seated_score += 0.45
        seated_support.append(f"both complete knee angles are seated-like ({seated_knees[0]:.1f}°, {seated_knees[1]:.1f}°)")
    elif len(seated_knees) == 1:
        seated_score += 0.28
        seated_support.append(f"one complete knee angle is seated-like ({seated_knees[0]:.1f}°)")
    if len(horizontal_thighs) >= 2:
        seated_score += 0.30
        seated_support.append("both observed thighs project substantially horizontally")
    elif len(horizontal_thighs) == 1:
        seated_score += 0.16
        seated_support.append("one observed thigh projects substantially horizontally")
    if POSTURE_RE["seated"].search(summary):
        seated_score += 0.30
        seated_support.append("Analyze summary explicitly reports sitting/seated")
    if not complete_legs:
        seated_limits.append("no complete hip-knee-ankle chain; seated geometry remains underdetermined")
    hypotheses.append(_primitive(
        "posture_seated", "seated", min(1.0, seated_score),
        support=seated_support, limitations=seated_limits,
        subsumes=["individual knee bends", "thigh direction"],
    ))

    kneeling_score = 0.0
    kneeling_support: list[str] = []
    deeply_bent = [value for value in knee_angles if value <= 90.0]
    if len(deeply_bent) == 2:
        kneeling_score += 0.45
        kneeling_support.append("both complete legs have deeply flexed knees")
    elif len(deeply_bent) == 1:
        kneeling_score += 0.22
        kneeling_support.append("one complete leg has a deeply flexed knee")
    if POSTURE_RE["kneeling"].search(summary):
        kneeling_score += 0.40
        kneeling_support.append("Analyze summary explicitly reports kneeling")
    hypotheses.append(_primitive("posture_kneeling", "kneeling", min(1.0, kneeling_score), support=kneeling_support))

    lying_score = 0.0
    lying_support: list[str] = []
    if torso_axis is not None and abs(torso_axis) >= 55.0:
        lying_score += 0.40
        lying_support.append(f"2-D torso axis is strongly horizontal/diagonal ({torso_axis:.1f}° from vertical)")
    if POSTURE_RE["lying"].search(summary):
        lying_score += 0.45
        lying_support.append("Analyze summary explicitly reports lying/reclining")
    hypotheses.append(_primitive(
        "posture_lying_or_reclining", "lying or reclining", min(1.0, lying_score),
        support=lying_support,
        limitations=["2-D torso orientation alone cannot distinguish rotated-camera or extreme-lean cases"],
    ))
    return hypotheses


def _select_posture(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(hypotheses, key=lambda item: float(item.get("support_score") or 0.0), reverse=True)
    if not ranked or float(ranked[0].get("support_score") or 0.0) < 0.60:
        return {"status": "withheld", "label": None, "reason": "no posture hypothesis reached MODERATE support", "hypotheses": ranked}
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if second and float(second.get("support_score") or 0.0) >= 0.60:
        margin = float(best.get("support_score") or 0.0) - float(second.get("support_score") or 0.0)
        if margin < 0.15:
            return {"status": "withheld", "label": None, "reason": "multiple posture hypotheses have similar support", "hypotheses": ranked}
    return {
        "status": "qualified",
        "label": best.get("label"),
        "primitive_id": best.get("id"),
        "support_score": best.get("support_score"),
        "confidence_band": best.get("confidence_band"),
        "support": best.get("support"),
        "limitations": best.get("limitations"),
        "subsumes": best.get("subsumes"),
        "hypotheses": ranked,
    }


def _torso_orientation(fusion: dict[str, Any]) -> dict[str, Any]:
    sam = fusion.get("sam3d_geometry_audit") or {}
    provenance = sam.get("target_provenance") or {}
    if provenance.get("context_risk") == "requires_review":
        return {"status": "withheld", "reason": "SAM3D target provenance requires review"}

    shoulder = sam.get("shoulder_depth_rotation") or {}
    hip = sam.get("hip_depth_rotation") or {}
    shoulder_deg = _safe_float(shoulder.get("magnitude_deg"))
    hip_deg = _safe_float(hip.get("magnitude_deg"))
    shoulder_ok = shoulder.get("authority") == "qualified_component_geometry" and shoulder_deg is not None
    hip_ok = hip.get("authority") == "qualified_component_geometry" and hip_deg is not None
    values = [value for ok, value in ((shoulder_ok, shoulder_deg), (hip_ok, hip_deg)) if ok and value is not None]

    if not values:
        upper = fusion.get("qualified_upper_torso_depth_relation") or {}
        upper_deg = _safe_float(upper.get("source_magnitude_deg"))
        if upper.get("authority") == "qualified_visible_shoulder_depth_rotation" and upper_deg is not None:
            values = [upper_deg]
            shoulder_ok = True
        else:
            return {"status": "withheld", "reason": "no visibility-qualified torso depth component"}

    magnitude = sum(values) / len(values)
    if magnitude >= 65.0:
        label = "near side-on to the camera"
    elif magnitude >= 35.0:
        label = "turned at a three-quarter angle to the camera"
    elif magnitude >= 15.0:
        label = "slightly turned in depth from the camera"
    else:
        label = "approximately square-on to the camera"

    signed = fusion.get("signed_depth_authority_audit") or {}
    torso_direction = signed.get("torso_direction") or {}
    nearer = torso_direction.get("nearer_anatomical_side") if torso_direction.get("action") == "qualified" else None
    if nearer not in {"left", "right"}:
        shoulder_signed = ((signed.get("components") or {}).get("shoulder") or {})
        nearer = shoulder_signed.get("nearer_anatomical_side") if shoulder_signed.get("action") == "qualified" else None

    score = 0.72 + (0.12 if shoulder_ok else 0.0) + (0.12 if hip_ok else 0.0) + (0.04 if nearer in {"left", "right"} else 0.0)
    support = [f"visibility-qualified SAM3D depth proxy ≈ {magnitude:.1f}°"]
    if shoulder_ok:
        support.append("shoulder depth component is visibility-qualified")
    if hip_ok:
        support.append("hip depth component is visibility-qualified")
    if nearer in {"left", "right"}:
        support.append(f"signed-depth audit qualifies the {nearer} side as nearer")
    return {
        "status": "qualified",
        "label": label,
        "depth_rotation_deg": round(magnitude, 2),
        "nearer_anatomical_side": nearer if nearer in {"left", "right"} else None,
        "support_score": round(min(1.0, score), 3),
        "confidence_band": _band(min(1.0, score)),
        "support": support,
        "subsumes": ["shoulder depth staggering", "hip depth staggering", "component torso-yaw prose"],
        "note": "depth_rotation_deg is a camera-relative SAM3D proxy, not calibrated world yaw",
    }


def _head_semantics(fusion: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    relation = fusion.get("qualified_head_torso_relation") or {}
    if relation.get("camera_relation") == "toward_camera":
        out.append(_primitive(
            "head_turn_toward_camera_relative_torso",
            "head turned toward the camera relative to the torso",
            0.95,
            support=[str(relation.get("authority") or "qualified Fusion head/torso relation")],
            subsumes=["absolute frontal head-yaw wording"],
        ))

    orientation = fusion.get("orientation_semantics") or {}
    pitch = orientation.get("head_pitch") or {}
    direction = str(pitch.get("direction") or "").lower()
    magnitude = str(pitch.get("magnitude") or "").lower()
    confidence = _safe_float(pitch.get("confidence")) or 0.0
    if direction not in {"", "neutral", "unknown"} and magnitude not in {"", "none", "unknown"} and confidence >= 0.70:
        out.append(_primitive(
            "head_pitch", f"head angled {direction.replace('_', ' ')}", min(0.90, confidence),
            support=[f"governed head pitch is {direction}/{magnitude} at confidence {confidence:.2f}"],
        ))

    gaze = (analysis.get("target_subject") or {}).get("gaze") or {}
    target = str(gaze.get("target") or "")
    confidence = _safe_float(gaze.get("confidence")) or 0.0
    if target and target not in {"unknown", "unclear"} and confidence >= 0.75:
        labels = {
            "camera_lens": "looking toward the camera",
            "near_camera": "looking near the camera",
            "off_camera": "looking off-camera",
            "down": "looking downward",
        }
        out.append(_primitive(
            "gaze_direction", labels.get(target, f"gaze directed toward {target.replace('_', ' ')}"), min(0.90, confidence),
            support=[f"Analyze gaze target is {target} at confidence {confidence:.2f}"],
        ))
    return out


def _interaction_gestures(features: dict[str, Any], fusion: dict[str, Any]) -> list[dict[str, Any]]:
    angles = features.get("angles_deg") or {}
    distances = features.get("normalized_distances") or {}
    gestures: list[dict[str, Any]] = []

    for index, item in enumerate(fusion.get("qualified_interactions") or []):
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        if state.get("selection_usable") is False:
            continue
        confidence = _safe_float(item.get("confidence")) or 0.0
        kind = str(item.get("type") or "").lower()
        actor = str(item.get("actor_part") or "").lower().replace("_", " ")
        target = str(item.get("target") or "").lower().replace("_", " ").strip()
        actor_side = state.get("qualified_actor_anatomical_side") or item.get("actor_anatomical_side")
        if actor_side not in {"left", "right"}:
            match = re.search(r"\b(left|right)\b", actor)
            actor_side = match.group(1) if match else None

        if "hand" in actor and re.search(r"\bhip\b", target) and kind in {"contact", "support", "rest", "touch"}:
            score = 0.60 + min(0.25, confidence * 0.25)
            support = [f"qualified interaction: {actor or 'hand'} -> {target} ({kind}, {confidence:.2f})"]
            if actor_side in {"left", "right"}:
                distance = _safe_float(distances.get(f"{actor_side}_wrist_to_same_hip"))
                elbow = _safe_float(angles.get(f"{actor_side}_elbow"))
                if distance is not None and distance <= 0.55:
                    score += 0.10
                    support.append(f"DWPose wrist is near same-side hip ({distance:.2f} normalized units)")
                if elbow is not None:
                    score += 0.05
                    support.append(f"complete arm gives elbow angle {elbow:.1f}°")
            side_text = f"{actor_side} " if actor_side in {"left", "right"} else ""
            gestures.append(_primitive(
                f"gesture_hand_on_hip_{index}", f"{side_text}hand resting on the hip", min(1.0, score),
                support=support,
                subsumes=[f"{side_text}elbow bend", f"{side_text}wrist position", "hand/hip contact clause"],
                details={"class": "hand_on_hip", "actor_side": actor_side, "target": "hip"},
            ))
            continue

        if "hand" in actor and re.search(r"\b(?:chin|head|face)\b", target) and kind in {"contact", "support", "rest", "touch"}:
            score = 0.65 + min(0.25, confidence * 0.25)
            side_text = f"{actor_side} " if actor_side in {"left", "right"} else ""
            gestures.append(_primitive(
                f"gesture_head_support_{index}", f"chin/head resting on the {side_text}hand", min(1.0, score),
                support=[f"qualified hand-to-{target} {kind} interaction at confidence {confidence:.2f}"],
                subsumes=["finger curl detail", "forearm support chain", "hand/head contact clause"],
                details={"class": "head_support", "actor_side": actor_side, "target": target},
            ))
            continue

        if kind in {"hold", "holding", "grip", "grasp", "carry", "carrying"} and target:
            score = 0.58 + min(0.30, confidence * 0.30)
            side_text = f"{actor_side} " if actor_side in {"left", "right"} else ""
            verb = "holding" if kind in {"hold", "holding", "grip", "grasp"} else "carrying"
            gestures.append(_primitive(
                f"gesture_object_{index}", f"{side_text}hand {verb} {target}", min(1.0, score),
                support=[f"qualified {kind} interaction at confidence {confidence:.2f}"],
                subsumes=["component arm/hand geometry used only to establish the object interaction"],
                details={"class": "object_interaction", "actor_side": actor_side, "target": target, "verb": verb},
            ))
    return gestures


def _arm_geometry_gestures(features: dict[str, Any], fusion: dict[str, Any], interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    directions = features.get("directions_deg") or {}
    connectivity = features.get("connectivity") or {}
    occupied_sides = {
        str((item.get("details") or {}).get("actor_side"))
        for item in interactions
        if (item.get("details") or {}).get("actor_side") in {"left", "right"}
    }
    out: list[dict[str, Any]] = []
    for side in ("left", "right"):
        if side in occupied_sides:
            continue
        chain = connectivity.get(f"{side}_arm") or {}
        if not chain.get("complete"):
            continue
        angle = _safe_float(directions.get(f"{side}_arm_from_vertical"))
        if angle is None:
            continue
        semantic_text = " ".join(
            _part_text(item)
            for item in _body_part_records(fusion)
            if "arm" in str(item.get("part") or "").lower() and _qualified_side(item) == side
        ).lower()
        if abs(angle) <= 30.0:
            score = 0.68 + (0.20 if re.search(r"\b(?:hang|down|side)\b", semantic_text) else 0.0)
            support = [f"complete DWPose {side} arm projects down ({angle:.1f}° from vertical)"]
            if re.search(r"\b(?:hang|down|side)\b", semantic_text):
                support.append("governed arm semantics agree that the arm hangs/is down")
            out.append(_primitive(
                f"gesture_{side}_arm_down", f"{side} arm hanging at the side", min(1.0, score),
                support=support, subsumes=[f"{side} shoulder/elbow/wrist component geometry"],
                details={"class": "arm_down", "actor_side": side},
            ))
        elif abs(angle) >= 120.0:
            out.append(_primitive(
                f"gesture_{side}_arm_raised", f"{side} arm raised", 0.68,
                support=[f"complete DWPose {side} arm projects up ({angle:.1f}° from vertical)"],
                limitations=["2-D direction does not establish the purpose of the raised arm"],
                details={"class": "arm_raised", "actor_side": side},
            ))
    return out


def _merge_gestures(gestures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compress duplicate/bilateral primitives before they reach human-facing prose."""
    by_label: dict[str, dict[str, Any]] = {}
    for item in gestures:
        label = str(item.get("label") or "")
        previous = by_label.get(label)
        if previous is None or float(item.get("support_score") or 0.0) > float(previous.get("support_score") or 0.0):
            by_label[label] = item
    values = list(by_label.values())

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in values:
        details = item.get("details") or {}
        if details.get("class") != "object_interaction":
            continue
        key = (str(details.get("target") or ""), str(details.get("verb") or "holding"))
        groups.setdefault(key, []).append(item)

    consumed: set[str] = set()
    merged: list[dict[str, Any]] = []
    for (target, verb), items in groups.items():
        sides = {(item.get("details") or {}).get("actor_side") for item in items}
        if {"left", "right"}.issubset(sides):
            chosen = [item for item in items if (item.get("details") or {}).get("actor_side") in {"left", "right"}]
            score = min(float(item.get("support_score") or 0.0) for item in chosen)
            support = [entry for item in chosen for entry in (item.get("support") or [])]
            merged.append(_primitive(
                f"gesture_both_hands_{verb}_{re.sub(r'[^a-z0-9]+', '_', target).strip('_')}",
                f"both hands {verb} {target}", score,
                support=support,
                subsumes=["separate left/right hand object interactions", "component hand/arm geometry"],
                details={"class": "bilateral_object_interaction", "target": target, "verb": verb},
            ))
            consumed.update(str(item.get("id")) for item in chosen)

    merged.extend(item for item in values if str(item.get("id")) not in consumed)
    return sorted(merged, key=lambda item: float(item.get("support_score") or 0.0), reverse=True)


def _framing_summary(analysis: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    framing = analysis.get("framing") or {}
    source_scale = str(framing.get("shot_scale") or "").lower()
    extent = str(framing.get("subject_extent") or "").strip()
    text = extent.lower()
    pose_extent = str(features.get("pose_extent_hint") or "")
    if any(token in text for token in ("mid-calf", "mid calf", "feet cropped", "feet partially cropped")):
        label = "near-full-length"
    elif any(token in text for token in ("upper thigh", "mid-thigh", "mid thigh", "knees")) or pose_extent == "three_quarter_or_long":
        label = "three-quarter / medium-full"
    elif "waist" in text or "hip" in text:
        label = "medium / waist-up"
    elif "upper chest" in text or "head and shoulders" in text:
        label = "medium close-up"
    elif source_scale == "close_up":
        label = "close-up"
    elif pose_extent == "full_length" and "crop" not in text:
        label = "full-length"
    else:
        label = source_scale.replace("_", " ") if source_scale else "unspecified"
    return {
        "label": label,
        "source_shot_scale": source_scale or None,
        "subject_extent": extent or None,
        "dwpose_extent_hint": pose_extent or None,
    }


def _human_summary(posture: dict[str, Any], torso: dict[str, Any], gestures: list[dict[str, Any]], head: list[dict[str, Any]], framing: dict[str, Any]) -> str:
    clauses: list[str] = []
    if posture.get("status") == "qualified" and posture.get("label"):
        clauses.append(str(posture["label"]).capitalize())
    if torso.get("status") == "qualified":
        clause = f"body {torso.get('label')}"
        nearer = torso.get("nearer_anatomical_side")
        if nearer in {"left", "right"}:
            clause += f", {nearer} side/shoulder nearer the camera"
        clauses.append(clause)
    clauses.extend(str(item.get("label")) for item in gestures if item.get("caption_preferred"))
    clauses.extend(str(item.get("label")) for item in head if item.get("caption_preferred"))
    if framing.get("label") and framing.get("label") != "unspecified":
        clauses.append(f"{framing['label']} framing")
    if not clauses:
        return "Pose withheld: insufficient deterministic support for a simple human-level summary."
    return "; ".join(clauses) + "."


def build_pose_semantics(dwpose: dict[str, Any], fused_payload: dict[str, Any], analysis_payload: dict[str, Any]) -> dict[str, Any]:
    """Build report-only pose primitives; support scores are not probabilities."""
    analysis = _analysis_root(analysis_payload)
    fusion = _fusion_root(fused_payload)
    features = _geometry_features(_target_person(dwpose), dwpose)
    posture = _select_posture(_posture_hypotheses(features, fusion, analysis))
    torso = _torso_orientation(fusion)
    interactions = _interaction_gestures(features, fusion)
    gestures = _merge_gestures(interactions + _arm_geometry_gestures(features, fusion, interactions))
    head = _head_semantics(fusion, analysis)
    framing = _framing_summary(analysis, features)

    preferred = {
        "posture": posture.get("label") if posture.get("status") == "qualified" else None,
        "torso_orientation": torso.get("label") if torso.get("status") == "qualified" else None,
        "nearer_anatomical_side": torso.get("nearer_anatomical_side") if torso.get("status") == "qualified" else None,
        "gestures": [item.get("label") for item in gestures if item.get("caption_preferred")],
        "head_and_gaze": [item.get("label") for item in head if item.get("caption_preferred")],
        "framing": framing.get("label"),
    }
    return {
        "schema_version": "pose-semantics-0.1",
        "status": "experimental_report_only",
        "score_semantics": "deterministic support score, not calibrated probability",
        "authority_policy": {
            "dwpose": "observed 2-D joints, complete-chain angles, normalized projected geometry",
            "sam3d": "qualified camera-relative depth/orientation only; reconstruction never creates visibility",
            "fusion": "qualified interactions/laterality and already-audited cross-source relations",
            "unknown_is_first_class": True,
            "preferred_threshold": 0.60,
        },
        "preferred_pose": preferred,
        "human_summary": _human_summary(posture, torso, gestures, head, framing),
        "posture": posture,
        "torso_orientation": torso,
        "gestures": gestures,
        "head_and_gaze": head,
        "framing": framing,
        "geometry_features": features,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics",
        description="Experimental deterministic pose semanticizer over cached DWPose + Fusion/SAM3D evidence.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", help="Only process result keys containing one of these strings.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.5" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    analysis_dir = run_dir / slug
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.1" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion"), (dwpose_dir, "DWPose"), (analysis_dir, "Analyze")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_paths = sorted(fusion_dir.glob("*.fused_v2_3.json"))
    if args.only:
        needles = tuple(args.only)
        fusion_paths = [path for path in fusion_paths if any(needle in path.name for needle in needles)]

    records: list[dict[str, Any]] = []
    for fusion_path in fusion_paths:
        key = fusion_path.name.removesuffix(".fused_v2_3.json")
        out_json = output_dir / f"{key}.pose_semantics.json"
        out_txt = output_dir / f"{key}.pose_semantics.txt"
        if out_json.exists() and out_txt.exists() and not args.overwrite:
            result = _read(out_json)
            records.append({"image_key": key, "status": "reused", "human_summary": result.get("human_summary")})
            continue

        dw_path = dwpose_dir / f"{key}.dwpose.json"
        analysis_path = analysis_dir / f"{key}.analysis.json"
        if not dw_path.is_file() or not analysis_path.is_file():
            records.append({"image_key": key, "status": "missing_source"})
            continue

        result = build_pose_semantics(_read(dw_path), _read(fusion_path), _read(analysis_path))
        result.update({
            "image_key": key,
            "source_paths": {"fusion": str(fusion_path), "dwpose": str(dw_path), "analysis": str(analysis_path)},
        })
        _write(out_json, result)
        out_txt.write_text(str(result.get("human_summary") or "") + "\n", encoding="utf-8")
        records.append({
            "image_key": key,
            "status": "written",
            "posture": (result.get("preferred_pose") or {}).get("posture"),
            "human_summary": result.get("human_summary"),
        })

    index = {
        "schema_version": "pose-semantics-0.1-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "records": records,
    }
    _write(output_dir / "pose_semantics.index.json", index)
    print(f"Pose semantics: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
