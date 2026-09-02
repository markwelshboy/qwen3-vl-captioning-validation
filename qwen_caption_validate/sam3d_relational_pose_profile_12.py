from __future__ import annotations

"""v0.12 support-area and directional physical-governance refinement.

Changes relative to v0.11:
* foot support is a 2D X/Z contact hull (ankle/heel/toe points), not a line
  between foot centroids;
* low-stance hard vetoes are authority-aware: unobserved support geometry may
  lower confidence but cannot prove a crouch/squat impossible;
* strong forward compensation suppresses/rejects recline and supports a
  flexed crouch/squat family instead of being interpreted as recline;
* head_supported_by_hand requires wrist/palm-root support topology, not merely
  finger/head proximity.

All support/contact reasoning remains reconstructed geometry. Crop support is
kept separate and still governs whether a pose may be serialized as observed.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import pose_atlas_v3_03 as atlas03
from . import sam3d_hand_geometry as hand
from . import sam3d_relational_pose_profile_11 as v11


v10 = v11.v10
v09 = v11.v09
v08 = v09.v08
MHR = v11.MHR

LOW_SUPPORT_AUTHORITY = 0.20
FORWARD_COMPENSATION_START = 0.35
FORWARD_COMPENSATION_STRONG = 0.60
FORWARD_RECLINE_HARD_MAX_LOWER = 0.70
SUPPORT_HULL_PADDING_SW = 0.08

HEAD_SUPPORT_MAX_PALM_DISTANCE_SW = 0.80
HEAD_SUPPORT_MAX_WRIST_DISTANCE_SW = 1.00
HEAD_SUPPORT_MIN_PALM_VERTICAL_SW = -0.25


def _round(value: float | None, digits: int = 3) -> float | None:
    return v11._round(value, digits)


def _ramp(value: float, low: float, high: float) -> float:
    return v11._ramp(value, low, high)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _point(keypoints: np.ndarray, name: str) -> np.ndarray | None:
    idx = MHR.get(name)
    if idx is None or idx >= len(keypoints):
        return None
    p = np.asarray(keypoints[idx, :3], dtype=np.float64)
    if p.size < 3 or not np.all(np.isfinite(p)):
        return None
    return p


def _convex_hull(points: list[np.ndarray]) -> np.ndarray:
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside_convex(point: np.ndarray, hull: np.ndarray) -> bool:
    if len(hull) < 3:
        return False
    p = np.asarray(point, dtype=np.float64)
    signs: list[float] = []
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        edge = b - a
        rel = p - a
        signs.append(float(edge[0] * rel[1] - edge[1] * rel[0]))
    return all(v >= -1e-9 for v in signs) or all(v <= 1e-9 for v in signs)


def _distance_to_hull(point: np.ndarray, hull: np.ndarray) -> float:
    if len(hull) == 0:
        return float("inf")
    if len(hull) == 1:
        return float(np.linalg.norm(np.asarray(point) - hull[0]))
    if len(hull) >= 3 and _inside_convex(point, hull):
        return 0.0
    return min(
        _segment_distance(point, hull[i], hull[(i + 1) % len(hull)])
        for i in range(len(hull))
    )


def _support_area_geometry(keypoints: np.ndarray) -> dict[str, Any]:
    required = {
        name: _point(keypoints, name)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    }
    if any(v is None for v in required.values()):
        return {"available": False}

    ls, rs = required["left_shoulder"], required["right_shoulder"]
    lh, rh = required["left_hip"], required["right_hip"]
    assert ls is not None and rs is not None and lh is not None and rh is not None
    shoulder_width = float(np.linalg.norm(ls - rs))
    if shoulder_width <= 1e-9:
        return {"available": False}

    contacts: list[np.ndarray] = []
    contacts_by_side: dict[str, list[list[float]]] = {"left": [], "right": []}
    for side in ("left", "right"):
        for suffix in ("ankle", "heel", "big_toe"):
            p = _point(keypoints, f"{side}_{suffix}")
            if p is None:
                continue
            xz = np.asarray(p[[0, 2]], dtype=np.float64)
            contacts.append(xz)
            contacts_by_side[side].append([float(xz[0]), float(xz[1])])

    if len(contacts) < 2:
        return {"available": False}

    hull = _convex_hull(contacts)
    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    torso_proxy = 0.42 * shoulder_mid + 0.58 * hip_mid

    pad = SUPPORT_HULL_PADDING_SW * shoulder_width

    def norm_distance(p3: np.ndarray) -> float:
        raw = _distance_to_hull(np.asarray(p3[[0, 2]], dtype=np.float64), hull)
        return max(0.0, raw - pad) / shoulder_width

    pelvis_dist = norm_distance(hip_mid)
    shoulder_dist = norm_distance(shoulder_mid)
    torso_dist = norm_distance(torso_proxy)

    foot_center = np.mean(np.stack(contacts, axis=0), axis=0)
    hip_to_feet = foot_center - hip_mid[[0, 2]]
    shoulder_from_hip = shoulder_mid[[0, 2]] - hip_mid[[0, 2]]
    foot_distance = float(np.linalg.norm(hip_to_feet) / shoulder_width)
    if float(np.linalg.norm(hip_to_feet)) > 1e-9:
        unit = hip_to_feet / float(np.linalg.norm(hip_to_feet))
        shoulder_shift = float(np.dot(shoulder_from_hip, unit) / shoulder_width)
        compensation_fraction = shoulder_shift / max(0.05, foot_distance)
    else:
        shoulder_shift = 0.0
        compensation_fraction = 1.0

    pelvis_support = 1.0 - _ramp(pelvis_dist, 0.08, 0.70)
    torso_support = 1.0 - _ramp(torso_dist, 0.08, 0.78)
    distance_feasibility = _clamp(0.55 * pelvis_support + 0.45 * torso_support)

    compensation_need = _ramp(pelvis_dist, 0.06, 0.40)
    if compensation_need <= 0.05:
        compensation_score = 1.0
    else:
        compensation_score = _ramp(compensation_fraction, 0.02, 0.65)

    compensation_rescue = _clamp(0.62 * compensation_score * compensation_need)
    feasibility = _clamp(max(distance_feasibility, compensation_rescue))

    hull_norm = (hull / shoulder_width).tolist()
    return {
        "available": True,
        "model": "convex_foot_contact_hull_xz_with_padding",
        "shoulder_width_3d": _round(shoulder_width),
        "support_hull_padding_shoulder_widths": SUPPORT_HULL_PADDING_SW,
        "contact_points_xz": contacts_by_side,
        "support_hull_xz_shoulder_widths": [[_round(x, 4), _round(z, 4)] for x, z in hull_norm],
        "pelvis_to_support_area_shoulder_widths": _round(pelvis_dist),
        "shoulder_to_support_area_shoulder_widths": _round(shoulder_dist),
        "torso_to_support_area_shoulder_widths": _round(torso_dist),
        "hip_to_foot_centroid_horizontal_distance_shoulder_widths": _round(foot_distance),
        "shoulder_shift_toward_feet_shoulder_widths": _round(shoulder_shift),
        "shoulder_compensation_fraction": _round(compensation_fraction, 4),
        "pelvis_support_score": _round(pelvis_support, 4),
        "torso_support_score": _round(torso_support, 4),
        "distance_feasibility_score": _round(distance_feasibility, 4),
        "compensation_need_score": _round(compensation_need, 4),
        "compensation_score": _round(compensation_score, 4),
        "compensation_rescue_score": _round(compensation_rescue, 4),
        "support_feasibility_score": _round(feasibility, 4),
        "support_feasibility_score_percent": int(round(100.0 * feasibility)),
    }


def _replace_support_diagnostic(profile: dict[str, Any], keypoints: np.ndarray) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    old = projected.get("independent_support_diagnostic") or {}
    leg = projected.get("leg_state_diagnostic") or {}
    geometry = _support_area_geometry(keypoints)
    if not geometry.get("available") or not leg.get("available"):
        return

    flex = float(leg.get("bilateral_flexion_score") or 0.0)
    straight = float(leg.get("bilateral_straight_score") or 0.0)
    asym = float(leg.get("asymmetry_score") or 0.0)
    feasibility = float(geometry.get("support_feasibility_score") or 0.0)
    foot_supported = _clamp(flex * feasibility)
    external = _clamp(flex * (1.0 - feasibility))
    asymmetric_rescue = _clamp(asym * (1.0 - flex))
    standing_conflict = _clamp(flex * (1.0 - asymmetric_rescue))

    candidates = {
        "bilateral_straight_stance": straight,
        "foot_supported_flexed_stance": foot_supported,
        "externally_supported_flexed_posture": external,
        "asymmetric_leg_state": asym,
    }
    best_name, best_value = max(candidates.items(), key=lambda item: item[1])
    candidate = best_name if best_value >= 0.35 else "indeterminate"

    crop_support = float(old.get("crop_support") or 0.0)
    support_class = old.get("support_class") or v08.v07.v06.v05._projected_support_class(crop_support)

    projected["independent_support_diagnostic_v08_line"] = old
    projected["independent_support_diagnostic"] = {
        "report_only": False,
        "available": True,
        "independent_of_existing_posture_scores": True,
        "support_model_version": "v0.12_foot_contact_area",
        "candidate": candidate,
        "candidate_scores": {k: _round(v, 4) for k, v in candidates.items()},
        "candidate_score_percent": {k: int(round(100.0 * v)) for k, v in candidates.items()},
        "foot_supported_flexed_stance_feasibility": _round(foot_supported, 4),
        "foot_supported_flexed_stance_feasibility_percent": int(round(100.0 * foot_supported)),
        "external_support_requirement": _round(external, 4),
        "external_support_requirement_percent": int(round(100.0 * external)),
        "standing_joint_conflict": _round(standing_conflict, 4),
        "standing_joint_conflict_percent": int(round(100.0 * standing_conflict)),
        "support_feasibility_score": geometry.get("support_feasibility_score"),
        "support_feasibility_score_percent": geometry.get("support_feasibility_score_percent"),
        "geometry": geometry,
        "crop_support": _round(crop_support, 4),
        "crop_support_percent": int(round(100.0 * crop_support)),
        "support_class": support_class,
        "hard_veto_authority_available": bool(crop_support >= LOW_SUPPORT_AUTHORITY),
        "interpretation": (
            "Foot support is modeled as a finite X/Z contact hull from reconstructed "
            "ankle/heel/toe points. Hard low-stance exclusions must also have enough "
            "observed support authority; unobserved feet cannot prove a squat impossible."
        ),
    }
    profile["sam3d_projected_pose"] = projected


def _restore_unobserved_support_rejections(profile: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    rows = governance.get("per_pose") or {}
    support = projected.get("independent_support_diagnostic") or {}
    leg = projected.get("leg_state_diagnostic") or {}
    support_crop = float(support.get("crop_support") or 0.0)
    if support_crop >= LOW_SUPPORT_AUTHORITY:
        return

    flex = float(leg.get("bilateral_flexion_score") or 0.0)
    flex_gate = _ramp(flex, 0.18, 0.58)
    support_reasons = {
        "bilateral_flexion_with_near_zero_foot_support_feasibility",
        "strong_external_support_requirement",
        "pelvis_displaced_and_shoulders_retreat_from_foot_support",
    }
    restored: list[str] = []
    for name in ("crouching", "squatting"):
        row = rows.get(name) or {}
        reasons = list(row.get("hard_rejection_reasons") or [])
        if not row.get("hard_rejected") or not reasons or not set(reasons).issubset(support_reasons):
            continue
        raw = float(row.get("raw_score") or 0.0)
        factor = 0.80 + 0.20 * flex_gate
        row["hard_rejected"] = False
        row["hard_rejection_reasons"] = []
        row["deferred_rejection_reasons_due_low_support_authority"] = reasons
        row["soft_feasibility_factor"] = _round(factor, 4)
        row["governed_score"] = _round(raw * factor, 4)
        row["governed_score_percent"] = int(round(100.0 * raw * factor))
        row["support_veto_deferred"] = True
        rows[name] = row
        restored.append(name)

    if restored:
        governance["per_pose"] = rows
        governance["support_veto_authority"] = {
            "minimum_support_crop": LOW_SUPPORT_AUTHORITY,
            "actual_support_crop": _round(support_crop, 4),
            "restored_low_stance_families": restored,
        }
        projected["physical_governance"] = governance
        profile["sam3d_projected_pose"] = projected


def _strengthen_directional_recline(directional: dict[str, Any]) -> None:
    shift = directional.get("shoulder_shift_toward_feet_shoulder_widths")
    shift = float(shift) if shift is not None else None
    advance = float(directional.get("advance_toward_support_score") or 0.0)
    lower = float(directional.get("lower_body_recline_score") or 0.0)
    hard = bool(
        shift is not None
        and shift >= 0.25
        and advance >= FORWARD_COMPENSATION_STRONG
        and lower <= FORWARD_RECLINE_HARD_MAX_LOWER
    )
    if hard:
        directional["hard_forward_bend_recline_rejection"] = True
        directional["v12_forward_recline_veto"] = True
        directional["v12_forward_recline_veto_reason"] = (
            "strong_upper_body_advance_toward_support_not_recline"
        )


def _forward_compensation_refine(profile: dict[str, Any], directional: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    rows = governance.get("per_pose") or {}
    if not rows:
        return

    leg = projected.get("leg_state_diagnostic") or {}
    support = projected.get("independent_support_diagnostic") or {}
    flex = float(leg.get("bilateral_flexion_score") or 0.0)
    advance = float(directional.get("advance_toward_support_score") or 0.0)
    feasibility = float(support.get("support_feasibility_score") or 0.0)
    support_crop = float(support.get("crop_support") or 0.0)
    external = float(support.get("external_support_requirement") or 0.0)

    if flex >= 0.45 and advance >= FORWARD_COMPENSATION_START:
        crouch = rows.get("crouching") or {}
        if not crouch.get("hard_rejected"):
            raw = float(crouch.get("raw_score") or 0.0)
            authority_relief = 1.0 - _ramp(support_crop, 0.10, 0.50)
            effective_feasibility = max(feasibility, 0.75 * authority_relief)
            target = _clamp(
                raw
                * (0.75 + 0.25 * advance)
                * (0.65 + 0.35 * effective_feasibility)
            )
            current = float(crouch.get("governed_score") or 0.0)
            if target > current:
                crouch["forward_compensated_flexed_stance_candidate"] = _round(target, 4)
                crouch["governed_score"] = _round(target, 4)
                crouch["governed_score_percent"] = int(round(100.0 * target))
            rows["crouching"] = crouch

        sitting = rows.get("sitting") or {}
        if not sitting.get("hard_rejected"):
            observed_external = _clamp(external * min(1.0, support_crop / 0.50))
            factor = 1.0 - 0.55 * advance * flex * (1.0 - observed_external)
            current = float(sitting.get("governed_score") or 0.0)
            sitting["forward_compensation_sitting_factor"] = _round(factor, 4)
            sitting["governed_score"] = _round(current * factor, 4)
            sitting["governed_score_percent"] = int(round(100.0 * current * factor))
            rows["sitting"] = sitting

    reclined = rows.get("reclined") or {}
    if not reclined.get("hard_rejected") and advance > 0.15:
        lower = float(directional.get("lower_body_recline_score") or 0.0)
        suppression = _clamp(1.0 - 0.90 * advance * (1.0 - 0.55 * lower))
        current = float(reclined.get("governed_score") or 0.0)
        reclined["forward_direction_recline_factor"] = _round(suppression, 4)
        reclined["governed_score"] = _round(current * suppression, 4)
        reclined["governed_score_percent"] = int(round(100.0 * current * suppression))
        rows["reclined"] = reclined

    governance["per_pose"] = rows
    governance["forward_compensation_refine"] = {
        "bilateral_flexion": _round(flex, 4),
        "advance_toward_support": _round(advance, 4),
        "support_feasibility": _round(feasibility, 4),
        "support_crop_authority": _round(support_crop, 4),
    }
    projected["physical_governance"] = governance
    profile["sam3d_projected_pose"] = projected


def _support_class_for_relation(relation: dict[str, Any]) -> str:
    if not relation.get("geometry_match"):
        return "not_matched"
    support = float(relation.get("crop_support") or 0.0)
    if support >= 0.50:
        return "crop_supported"
    if support > 0.0:
        return "weakly_crop_supported"
    return "reconstruction_only"


def _head_center_2d(sam2d: np.ndarray) -> np.ndarray | None:
    points: list[np.ndarray] = []
    for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
        idx = MHR.get(name)
        if idx is not None and idx < len(sam2d):
            p = np.asarray(sam2d[idx, :2], dtype=np.float64)
            if np.all(np.isfinite(p)):
                points.append(p)
    return np.mean(np.stack(points, axis=0), axis=0) if points else None


def _hand_support_topology(sam2d: np.ndarray, side: str) -> dict[str, Any]:
    head = _head_center_2d(sam2d)
    ls_i, rs_i = MHR.get("left_shoulder"), MHR.get("right_shoulder")
    if head is None or ls_i is None or rs_i is None or max(ls_i, rs_i) >= len(sam2d):
        return {"available": False}
    ls, rs = sam2d[ls_i, :2], sam2d[rs_i, :2]
    if not np.all(np.isfinite(ls)) or not np.all(np.isfinite(rs)):
        return {"available": False}
    shoulder_span = float(np.linalg.norm(ls - rs))
    order = hand.mhr_hand_order(side)
    if shoulder_span <= 1e-9 or len(order) < 18 or order[0] >= len(sam2d):
        return {"available": False}

    wrist = np.asarray(sam2d[order[0], :2], dtype=np.float64)
    palm_indices = [order[i] for i in (0, 1, 5, 9, 13, 17) if i < len(order)]
    palm_points = [
        np.asarray(sam2d[idx, :2], dtype=np.float64)
        for idx in palm_indices
        if idx < len(sam2d) and np.all(np.isfinite(sam2d[idx, :2]))
    ]
    if not np.all(np.isfinite(wrist)) or len(palm_points) < 3:
        return {"available": False}
    palm = np.mean(np.stack(palm_points, axis=0), axis=0)

    palm_dist = float(np.linalg.norm(palm - head) / shoulder_span)
    wrist_dist = float(np.linalg.norm(wrist - head) / shoulder_span)
    palm_vertical = float((palm[1] - head[1]) / shoulder_span)
    match = bool(
        palm_dist <= HEAD_SUPPORT_MAX_PALM_DISTANCE_SW
        and wrist_dist <= HEAD_SUPPORT_MAX_WRIST_DISTANCE_SW
        and palm_vertical >= HEAD_SUPPORT_MIN_PALM_VERTICAL_SW
    )
    return {
        "available": True,
        "geometry_match": match,
        "palm_root_to_head_shoulder_widths": _round(palm_dist),
        "wrist_to_head_shoulder_widths": _round(wrist_dist),
        "palm_root_vertical_from_head_shoulder_widths": _round(palm_vertical),
        "thresholds": {
            "max_palm_root_distance_shoulder_widths": HEAD_SUPPORT_MAX_PALM_DISTANCE_SW,
            "max_wrist_distance_shoulder_widths": HEAD_SUPPORT_MAX_WRIST_DISTANCE_SW,
            "min_palm_vertical_from_head_shoulder_widths": HEAD_SUPPORT_MIN_PALM_VERTICAL_SW,
        },
    }


def _guard_head_supported_by_hand(profile: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    relations = profile.get("relations") or {}
    broad = relations.get("head_supported_by_hand") or {}
    if not broad.get("geometry_match"):
        return

    sam2d = atlas03._sam2d_points(arrays)
    sides = [str(broad.get("side"))] if str(broad.get("side")) in ("left", "right") else ["left", "right"]
    candidates = [(side, _hand_support_topology(sam2d, side)) for side in sides]
    candidates = [(side, row) for side, row in candidates if row.get("available")]
    if not candidates:
        broad["support_topology_guard"] = {"available": False}
        relations["head_supported_by_hand"] = broad
        profile["relations"] = relations
        return

    side, topology = min(
        candidates,
        key=lambda item: float(item[1].get("palm_root_to_head_shoulder_widths") or 999.0),
    )
    broad["support_topology_guard"] = topology
    broad["support_topology_side"] = side
    if not topology.get("geometry_match"):
        broad["geometry_match_before_support_topology_guard"] = True
        broad["geometry_match"] = False
        broad["crop_support_before_support_topology_guard"] = broad.get("crop_support")
        broad["crop_support"] = 0.0
        broad["crop_support_percent"] = 0
        broad["support_class"] = "not_matched"
        broad["rejection_reason"] = "finger_proximity_without_wrist_palm_support_topology"

        fist = relations.get("head_supported_by_fist") or {}
        if fist.get("geometry_match"):
            fist["geometry_match_before_support_topology_guard"] = True
            fist["geometry_match"] = False
            fist["crop_support"] = 0.0
            fist["crop_support_percent"] = 0
            fist["support_class"] = "not_matched"
            fist["rejection_reason"] = "parent_head_hand_support_topology_rejected"
            relations["head_supported_by_fist"] = fist
    else:
        broad["support_class"] = _support_class_for_relation(broad)

    relations["head_supported_by_hand"] = broad
    profile["relations"] = relations


def _recompute_public(profile: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    rows = governance.get("per_pose") or {}
    scores = {name: float((row or {}).get("governed_score") or 0.0) for name, row in rows.items()}
    if not scores:
        return
    candidate_pose, best_candidate, best_score, margin = v08.v07.v06.v05._choose_posture(scores)
    region_support = {str(k): float(v or 0.0) for k, v in (projected.get("region_support") or {}).items()}
    coverage, authority, coverage_regions, support_regions = v08.v07.v06.v05._crop_support(
        region_support, candidate_pose, best_candidate
    )

    directional = projected.get("directional_recline_diagnostic") or {}
    upper = projected.get("upper_body_recline_diagnostic") or {}
    upper_used = False
    if (
        best_candidate == "reclined"
        and float(directional.get("directional_upper_recline_score") or 0.0) >= v11.UPPER_RECLINE_AUTHORITY_MIN_SCORE
        and float(directional.get("retreat_from_support_score") or 0.0) >= v11.RECLINE_RETREAT_AUTHORITY_MIN
    ):
        upper_authority = float(upper.get("path_authority") or 0.0)
        if upper_authority > authority:
            authority = upper_authority
            upper_used = True

    support_class = v08.v07.v06.v05._projected_support_class(authority)
    reconstruction_dominant = authority < v09.MIN_POSE_AUTHORITY
    usable = bool(candidate_pose != "uncertain" and not reconstruction_dominant)
    public_pose = candidate_pose if usable else "uncertain"

    governance["architecture"] = (
        "raw_similarity_then_support_area_physical_exclusion_then_directional_governance_"
        "then_forward_compensation_then_path_authority"
    )
    governance["governed_pose_before_authority"] = candidate_pose
    governance["governed_best_candidate_pose"] = best_candidate
    governance["governed_best_score"] = _round(best_score, 4)
    governance["governed_best_score_percent"] = int(round(100.0 * best_score))
    governance["governed_winner_margin"] = _round(margin, 4)
    governance["governed_winner_margin_percent"] = int(round(100.0 * margin))
    governance["authority"] = {
        "minimum_pose_authority": v09.MIN_POSE_AUTHORITY,
        "crop_support": _round(authority, 4),
        "crop_support_percent": int(round(100.0 * authority)),
        "support_class": support_class,
        "reconstruction_dominant": reconstruction_dominant,
        "usable_as_projected_pose": usable,
        "authority_path": "directional_upper_body_recline" if upper_used else "posture_region_weights",
        "upper_body_path_available_percent": upper.get("path_authority_percent"),
        "withheld_reason": (
            "insufficient_observed_support" if reconstruction_dominant
            else ("insufficient_governed_score_or_margin" if candidate_pose == "uncertain" else None)
        ),
    }
    projected["physical_governance"] = governance
    projected["pose"] = public_pose
    projected["best_candidate_pose"] = best_candidate
    projected["posture_scores"] = {name: _round(value, 4) for name, value in scores.items()}
    projected["posture_score_percent"] = {name: int(round(100.0 * value)) for name, value in scores.items()}
    projected["winner_margin"] = _round(margin, 4)
    projected["winner_margin_percent"] = int(round(100.0 * margin))
    projected["reconstruction_match"] = _round(best_score, 4)
    projected["reconstruction_match_percent"] = int(round(100.0 * best_score))
    projected["crop_coverage"] = _round(coverage, 4)
    projected["crop_coverage_percent"] = int(round(100.0 * coverage))
    projected["crop_support"] = _round(authority, 4)
    projected["crop_support_percent"] = int(round(100.0 * authority))
    projected["pose_support"] = _round(authority, 4)
    projected["pose_support_percent"] = int(round(100.0 * authority))
    projected["crop_supported_regions"] = coverage_regions
    projected["pose_support_regions"] = support_regions
    projected["support_class"] = support_class
    profile["sam3d_projected_pose"] = projected


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v08.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.12"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]

    _replace_support_diagnostic(profile, keypoints)
    v09._govern_postures(profile)
    _restore_unobserved_support_rejections(profile)

    upper = v10._upper_body_recline(profile, keypoints)
    directional = v11._directional_upper_recline(profile, upper)
    _strengthen_directional_recline(directional)
    v11._refine_directional_sitting_recline(profile, upper, directional)
    _forward_compensation_refine(profile, directional)
    _recompute_public(profile)
    _guard_head_supported_by_hand(profile, arrays)

    projected = profile.get("sam3d_projected_pose") or {}
    projected["support_area_diagnostic"] = projected.get("independent_support_diagnostic") or {}
    profile["sam3d_projected_pose"] = projected

    policy = profile.get("policy") or {}
    policy.update({
        "v12_foot_support_is_contact_area_not_centroid_line": True,
        "v12_low_stance_hard_veto_requires_support_crop_authority": True,
        "v12_forward_compensation_can_raise_flexed_crouch_candidate": True,
        "v12_forward_compensation_suppresses_inherited_unsigned_recline": True,
        "v12_head_supported_by_hand_requires_wrist_palm_support_topology": True,
    })
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-12",
        description=(
            "Build v0.12 governed pose profiles with finite foot-support area, "
            "authority-aware low-stance vetoes, directional recline suppression, "
            "and wrist/palm head-support topology."
        ),
    )
    parser.add_argument("sam3d_dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")
    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else sam3d_dir.parent / "dwpose"
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else sam3d_dir.parent / "images"
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.12")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    helpers = v09.v08.v07.v06.v05.v04.v03
    rows: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        dwpose = helpers._read_json(dwpose_path)
        width = int(dwpose.get("image_width") or 0)
        height = int(dwpose.get("image_height") or 0)
        if width <= 0 or height <= 0:
            image_matches = [p for p in images_dir.rglob(f"{key}.*") if p.is_file()] if images_dir.is_dir() else []
            if not image_matches:
                raise SystemExit(f"Cannot determine image size for {key}")
            with Image.open(image_matches[0]) as im:
                width, height = im.size

        profile = build_profile(arrays, dwpose or None, width, height)
        record = {
            "image_key": key,
            "sam3d_arrays": str(path),
            "dwpose": str(dwpose_path) if dwpose_path.is_file() else None,
            "image_width": width,
            "image_height": height,
            "profile": profile,
        }
        out_path = output / f"{key}.sam3d_relational_pose.json"
        helpers._write_json(out_path, record)
        rows.append(record)

        projected = profile["sam3d_projected_pose"]
        scores = projected.get("posture_score_percent") or {}
        support = projected.get("independent_support_diagnostic") or {}
        direction = projected.get("directional_recline_diagnostic") or {}
        gov = projected.get("physical_governance") or {}
        relations = profile.get("relations") or {}
        head = relations.get("head_supported_by_hand") or {}
        print(
            f"{key}: pose={projected.get('pose')} best={projected.get('best_candidate_pose')} "
            f"scores=stand:{scores.get('standing',0)} crouch:{scores.get('crouching',0)} "
            f"squat:{scores.get('squatting',0)} sit:{scores.get('sitting',0)} recl:{scores.get('reclined',0)} "
            f"foot_area:{support.get('support_feasibility_score_percent',0)} "
            f"support_crop:{support.get('crop_support_percent',0)} "
            f"dir:{direction.get('direction','-')} advance:{direction.get('advance_toward_support_score_percent',0)} "
            f"authority:{(gov.get('authority') or {}).get('crop_support_percent',0)}% "
            f"head_support:{'yes' if head.get('geometry_match') else 'no'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.12",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    helpers._write_json(output / "sam3d_relational_pose.index.json", index)
    print(f"Index: {output / 'sam3d_relational_pose.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
