from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dataset_evidence import (
    COVERAGE_SIGNAL,
    POSE_SIGNAL,
    SHOT_IDENTITY_MODIFIER,
    REGION_COVERAGE,
    RELEVANCE_DISCOUNT,
    RELEVANCE_RANK,
    _confound_burden,
    _read_json,
)
from .dataset_evidence_v2 import _effective_framing, _facial_pose
from .pose_evidence import build_pose_evidence
from .runner import model_slug, resolve_model_id


ENTROPY_COMPLEXITY = {"low": 0.15, "medium": 0.80, "high": 1.60}
HIGH_VALUE_ACTION_TAGS = {
    "chin_or_head_support",
    "hand_face_contact",
    "hands_in_pockets",
    "holding_object",
    "reaching",
    "arms_extended",
    "arms_crossed",
    "hand_at_hip",
    "arm_or_hand_on_table",
    "head_torso_counter_rotation",
    "strong_shoulder_cant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "Source-aware dataset profiler combining cached Qwen analysis, DWPose geometry, "
            "action/contact evidence, and a lightweight deterministic background-texture proxy."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence", help="Output basename inside run_dir.")
    return parser.parse_args()


def _model_size_b(model_id: str) -> float | None:
    values = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)B(?![A-Za-z])", model_id, flags=re.IGNORECASE)
    if not values:
        return None
    try:
        return max(float(v) for v in values)
    except ValueError:
        return None


def _capability_profile(model_id: str) -> dict[str, Any]:
    """Policy defaults, not benchmark scores.

    These tiers encode what this harness is willing to claim from cached vision
    output. They are intentionally conservative because our own validation has
    already shown spatial failure modes even at 32B.
    """
    size_b = _model_size_b(model_id)
    if size_b is not None and size_b <= 4.5:
        tier = "low"
        breadth = "broad_screening"
        strength = "screening_only"
        semantics = "low_to_medium"
        spatial = "low"
        exact_pose_protection = "deterministic_only"
    elif size_b is not None and size_b <= 10:
        tier = "medium"
        breadth = "moderately_qualified"
        strength = "qualified"
        semantics = "medium"
        spatial = "low_to_medium"
        exact_pose_protection = "unique_only"
    elif size_b is not None and size_b >= 24:
        tier = "high"
        breadth = "detailed_but_qualified"
        strength = "qualified_to_strong"
        semantics = "high"
        spatial = "medium"
        exact_pose_protection = "rare_or_unique"
    else:
        tier = "medium"
        breadth = "moderately_qualified"
        strength = "qualified"
        semantics = "medium"
        spatial = "low_to_medium"
        exact_pose_protection = "unique_only"

    return {
        "model_id": model_id,
        "parameter_size_b": size_b,
        "policy_tier": tier,
        "judgement_breadth": breadth,
        "recommendation_strength": strength,
        "vision_semantics_authority": semantics,
        "vision_spatial_authority": spatial,
        "exact_vlm_pose_protection": exact_pose_protection,
        "dwpose_2d_geometry_authority": "high_secondary_evidence",
        "deterministic_pixel_evidence_authority": "high_measurement_evidence",
        "laterality_policy": "never treat VLM or DWPose anatomical laterality as infallible",
        "note": (
            "This is a harness policy for qualification/breadth, informed by observed model-size behavior in this validation work; "
            "it is not a universal benchmark score for the model."
        ),
    }


def _coarse_yaw(value: str) -> str:
    if value in {"frontal", "slight_left", "slight_right"}:
        return "frontalish"
    if value == "three_quarter_left":
        return "three_quarter_left"
    if value == "three_quarter_right":
        return "three_quarter_right"
    if value == "strong_left":
        return "strong_left"
    if value == "strong_right":
        return "strong_right"
    return "unknown"


def _coarse_pitch(value: str) -> str:
    if value in {"neutral", "slight_up", "slight_down"}:
        return "neutralish"
    if value == "down":
        return "down"
    if value == "strong_down":
        return "strong_down"
    if value == "up":
        return "up"
    if value == "strong_up":
        return "strong_up"
    return "unknown"


def _coarse_shot(value: str) -> str:
    if value in {"extreme_close_up", "close_up", "medium_close_up"}:
        return "identity_close"
    if value in {"medium", "three_quarter"}:
        return "body_context"
    if value == "full_length":
        return "full_body"
    return "unknown"


def _trusted_composition_class(framing: dict[str, Any], capability: dict[str, Any]) -> str:
    if capability["policy_tier"] != "low":
        return _coarse_shot(str(framing.get("effective_shot_scale") or "unknown"))
    extent = str(framing.get("dwpose_extent_hint") or "unknown")
    return {
        "full_length": "full_body",
        "three_quarter_or_long": "long_body",
        "waist_or_upper_body": "upper_body",
        "close_or_medium_close": "close_or_medium_close",
    }.get(extent, "unknown")


def _nuisance_profile(analysis: dict[str, Any]) -> dict[str, Any]:
    weighted: list[dict[str, Any]] = []
    occupancy_total = 0.0
    entropy_total = 0.0
    declared_entropy_focus = 0

    for region in analysis.get("nuisance_regions") or []:
        if not isinstance(region, dict):
            continue
        coverage = str(region.get("frame_coverage") or "small")
        complexity = str(region.get("visual_complexity") or "low")
        identity_rel = str(region.get("identity_relevance") or "none")
        pose_rel = str(region.get("pose_relevance") or "none")
        relevance = identity_rel
        if RELEVANCE_RANK.get(pose_rel, 0) > RELEVANCE_RANK.get(identity_rel, 0):
            relevance = pose_rel

        occupancy = REGION_COVERAGE.get(coverage, 1.0) * RELEVANCE_DISCOUNT.get(relevance, 1.0)
        entropy = occupancy * ENTROPY_COMPLEXITY.get(complexity, 0.8)
        occupancy_total += occupancy
        entropy_total += entropy
        declared = bool(region.get("entropy_focus_candidate"))
        declared_entropy_focus += int(declared)
        weighted.append(
            {
                "description": region.get("description"),
                "frame_coverage": coverage,
                "visual_complexity": complexity,
                "identity_relevance": identity_rel,
                "pose_relevance": pose_rel,
                "entropy_focus_candidate": declared,
                "occupancy_burden_points": round(occupancy, 3),
                "semantic_entropy_burden_points": round(entropy, 3),
            }
        )

    weighted.sort(key=lambda r: float(r["semantic_entropy_burden_points"]), reverse=True)
    return {
        "background_occupancy_burden": round(occupancy_total, 3),
        "semantic_entropy_burden": round(entropy_total, 3),
        "region_count": len(weighted),
        "vlm_entropy_focus_candidate_count": declared_entropy_focus,
        "regions": weighted,
    }


def _background_texture_proxy(image_path: Path | None, pose: dict[str, Any]) -> dict[str, Any]:
    unavailable = {
        "available": False,
        "texture_label": "unknown",
        "sample_fraction": None,
        "tonal_entropy_norm": None,
        "mean_gradient": None,
        "high_gradient_fraction": None,
        "texture_score": None,
        "subject_exclusion": "unavailable",
        "note": "No readable report image was available for deterministic texture measurement.",
    }
    if image_path is None or not image_path.exists():
        return unavailable

    try:
        with Image.open(image_path) as im:
            gray_im = im.convert("L")
            w, h = gray_im.size
            max_side = max(w, h)
            if max_side > 640:
                scale = 640.0 / max_side
                gray_im = gray_im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
            gray = np.asarray(gray_im, dtype=np.float32) / 255.0
    except Exception as exc:
        out = dict(unavailable)
        out["note"] = f"Texture measurement failed: {exc}"
        return out

    if gray.ndim != 2 or min(gray.shape) < 4:
        return unavailable

    mask = np.ones(gray.shape, dtype=bool)
    bbox = ((pose.get("target_2d_geometry") or {}).get("clipped_in_frame_keypoint_bbox") or {})
    exclusion = "full_frame_fallback"
    if all(bbox.get(k) is not None for k in ("x0", "y0", "x1", "y1")):
        pad = 0.035
        x0 = max(0.0, float(bbox["x0"]) - pad)
        y0 = max(0.0, float(bbox["y0"]) - pad)
        x1 = min(1.0, float(bbox["x1"]) + pad)
        y1 = min(1.0, float(bbox["y1"]) + pad)
        hh, ww = gray.shape
        ix0, iy0 = int(x0 * ww), int(y0 * hh)
        ix1, iy1 = max(ix0 + 1, int(math.ceil(x1 * ww))), max(iy0 + 1, int(math.ceil(y1 * hh)))
        mask[iy0:iy1, ix0:ix1] = False
        if float(mask.mean()) >= 0.08:
            exclusion = "outside_padded_dwpose_keypoint_bbox"
        else:
            mask[:] = True

    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = 0.5 * (gx + gy)
    pixels = gray[mask]
    grads = grad[mask]
    if pixels.size < 32:
        return unavailable

    hist, _ = np.histogram(pixels, bins=64, range=(0.0, 1.0))
    probs = hist.astype(np.float64)
    probs = probs[probs > 0]
    probs /= probs.sum()
    entropy = float(-(probs * np.log2(probs)).sum()) / math.log2(64)
    mean_grad = float(grads.mean())
    edge_fraction = float((grads > 0.08).mean())

    # Tuned only as a broad proxy: edge structure dominates; tonal diversity is
    # intentionally weak so a smooth sky/gradient is not treated like a rug.
    grad_component = min(mean_grad / 0.12, 1.0)
    edge_component = min(edge_fraction / 0.35, 1.0)
    score = 0.15 * entropy + 0.55 * grad_component + 0.30 * edge_component
    label = "low" if score < 0.30 else "medium" if score < 0.55 else "high"
    return {
        "available": True,
        "texture_label": label,
        "sample_fraction": round(float(mask.mean()), 4),
        "tonal_entropy_norm": round(entropy, 4),
        "mean_gradient": round(mean_grad, 5),
        "high_gradient_fraction": round(edge_fraction, 4),
        "texture_score": round(score, 4),
        "subject_exclusion": exclusion,
        "note": (
            "Provisional deterministic proxy measured outside a padded DWPose keypoint rectangle when possible. "
            "It is not a subject matte; a real mask will make this substantially more accurate."
        ),
    }


def _action_profile(analysis: dict[str, Any], pose: dict[str, Any], face: dict[str, Any]) -> dict[str, Any]:
    tags: set[str] = set()
    evidence: list[dict[str, str]] = []
    limbs = ((analysis.get("target_subject") or {}).get("visible_limbs") or [])

    for limb in limbs:
        if not isinstance(limb, dict):
            continue
        fields = {
            "part": limb.get("part"),
            "geometry": limb.get("geometry"),
            "contact": limb.get("contact"),
            "support": limb.get("support"),
            "foreshortening": limb.get("foreshortening"),
        }
        text = " ".join(str(v) for v in fields.values() if v not in (None, "")).lower()
        contact_text = " ".join(str(limb.get(k) or "") for k in ("contact", "support")).lower()

        def add(tag: str) -> None:
            tags.add(tag)
            evidence.append({"tag": tag, "source": "qwen_visible_limbs", "text": text[:280]})

        if "pocket" in text:
            add("hands_in_pockets")
        if any(w in text for w in ("hold", "holding", "grip", "grasp")):
            add("holding_object")
        if "reach" in text:
            add("reaching")
        if "extend" in text:
            add("arms_extended")
        if "cross" in text and any(w in text for w in ("arm", "forearm", "wrist", "hand")):
            add("arms_crossed")
        if "hip" in contact_text and "pocket" not in text:
            add("hand_at_hip")
        if "table" in contact_text:
            add("arm_or_hand_on_table")
        if any(w in contact_text for w in ("chin", "cheek", "face", "head")):
            add("hand_face_contact")
            if any(w in contact_text for w in ("support", "rest", "knuckle", "fist")) or limb.get("support"):
                add("chin_or_head_support")

    summary = str(analysis.get("image_summary") or "").lower()
    posture: list[str] = []
    if re.search(r"\b(seated|sitting|sits)\b", summary):
        posture.append("seated")
    if re.search(r"\b(standing|stands)\b", summary):
        posture.append("standing")

    torso = str(face.get("torso_yaw") or "unknown")
    head = str(face.get("head_yaw") or "unknown")
    if torso not in {"frontal", "unknown"} and head not in {"unknown", torso}:
        tags.add("head_torso_counter_rotation")
        evidence.append({"tag": "head_torso_counter_rotation", "source": "qwen_orientation_fusion", "text": f"torso={torso}; head={head}"})

    shoulder = ((pose.get("target_2d_geometry") or {}).get("shoulder_line_angle_from_horizontal_deg"))
    if shoulder is not None:
        try:
            angle = abs(float(shoulder))
            if angle >= 15.0:
                tags.add("strong_shoulder_cant")
                evidence.append({"tag": "strong_shoulder_cant", "source": "dwpose_geometry", "text": f"abs shoulder-line angle={angle:.2f} deg"})
            elif angle >= 8.0:
                tags.add("moderate_shoulder_cant")
                evidence.append({"tag": "moderate_shoulder_cant", "source": "dwpose_geometry", "text": f"abs shoulder-line angle={angle:.2f} deg"})
        except (TypeError, ValueError):
            pass

    # Deduplicate repeated evidence created by multiple visible-limb records.
    dedup: dict[tuple[str, str], dict[str, str]] = {}
    for item in evidence:
        dedup[(item["tag"], item["source"])] = item
    return {
        "tags": sorted(tags),
        "posture_tags": posture,
        "evidence": list(dedup.values()),
    }


def _signal_profile(analysis: dict[str, Any], pose: dict[str, Any], framing: dict[str, Any], capability: dict[str, Any]) -> dict[str, Any]:
    coverage = framing["subject_frame_coverage"]
    shot = framing["effective_shot_scale"]
    identity_score = COVERAGE_SIGNAL.get(coverage, 2.2) * SHOT_IDENTITY_MODIFIER.get(shot, 0.9)
    target_pose = pose.get("target_2d_geometry") or {}
    connectivity = target_pose.get("connectivity") or {}
    complete_chains = sum(
        bool((connectivity.get(name) or {}).get("complete"))
        for name in ("left_arm", "right_arm", "left_leg", "right_leg")
    )
    pose_score = POSE_SIGNAL.get(shot, 1.5) + 0.15 * complete_chains
    if target_pose.get("pose_extent_hint") == "full_length":
        pose_score += 0.35
    clipped_bbox = target_pose.get("clipped_in_frame_keypoint_bbox") or {}
    return {
        "subject_frame_coverage": coverage,
        "qwen_shot_scale": framing["qwen_shot_scale"],
        "effective_shot_scale": shot,
        "trusted_composition_class": _trusted_composition_class(framing, capability),
        "dwpose_extent_hint": framing["dwpose_extent_hint"],
        "framing_conflict": framing["framing_conflict"],
        "identity_signal_score": round(identity_score, 3),
        "pose_signal_score": round(pose_score, 3),
        "clipped_keypoint_bbox_height_fraction": clipped_bbox.get("height_fraction"),
        "clipped_keypoint_bbox_area_fraction": clipped_bbox.get("area_fraction"),
        "complete_limb_chains": complete_chains,
    }


def _active_snr(identity: float, nuisance: dict[str, Any], confound: float, texture: dict[str, Any], coverage: str) -> dict[str, Any]:
    occupancy = float(nuisance["background_occupancy_burden"])
    semantic_entropy = float(nuisance["semantic_entropy_burden"])
    texture_score = texture.get("texture_score")
    modifier = 1.0 if texture_score is None else 0.75 + 0.50 * float(texture_score)
    effective_entropy = semantic_entropy * modifier
    score = identity / (1.0 + 0.18 * occupancy + 0.55 * effective_entropy + 0.60 * confound)
    if coverage == "small" and occupancy >= 3.0:
        label = "low"
    elif score < 0.75:
        label = "low"
    elif score < 1.45:
        label = "medium"
    else:
        label = "high"
    return {
        "label": label,
        "heuristic_score": round(score, 3),
        "background_occupancy_component": round(occupancy, 3),
        "effective_entropy_component": round(effective_entropy, 3),
        "definition": (
            "provisional training-signal-density heuristic separating background occupancy from visual entropy; "
            "not photometric SNR and not yet subject-mask based"
        ),
    }


def _axis_quality_flags(counts: dict[str, Counter[str]], total: int) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for axis in ("torso", "head_roll", "head_yaw", "head_pitch"):
        counter = counts[axis]
        if not counter or total <= 0:
            continue
        value, n = counter.most_common(1)[0]
        share = n / total
        if share >= 0.90:
            flags.append(
                {
                    "axis": axis,
                    "dominant_value": value,
                    "dominant_count": n,
                    "dominant_share": round(share, 3),
                    "interpretation": (
                        "distribution is highly degenerate; this may reflect a real dataset bias, analyzer under-detection, "
                        "or both. Do not treat the axis as high-confidence dataset truth without spot-checking."
                    ),
                }
            )
    return flags


def _protection_limit(capability: dict[str, Any]) -> int:
    if capability["policy_tier"] == "high":
        return 2
    if capability["policy_tier"] == "medium":
        return 1
    return 0


def _protected_dimensions(item: dict[str, Any], counts: dict[str, Counter[str]], capability: dict[str, Any]) -> list[dict[str, Any]]:
    protected: list[dict[str, Any]] = []
    limit = _protection_limit(capability)
    face = item["facial_pose"]
    coarse = item["coverage_signature"]
    shot = coarse["composition"]

    def add(dimension: str, value: str, count: int, source: str, authority: str) -> None:
        protected.append(
            {
                "dimension": dimension,
                "value": value,
                "count": count,
                "remaining_if_removed": max(0, count - 1),
                "source": source,
                "authority": authority,
            }
        )

    if shot == "full_body" and counts["composition"].get(shot, 0) <= 2:
        add("composition", shot, counts["composition"][shot], "dwpose+qwen_framing_fusion", "high_for_visible_extent")

    torso = face["torso_yaw"]
    if torso not in {"frontal", "unknown"} and limit and counts["torso"].get(torso, 0) <= limit:
        add("torso_yaw", torso, counts["torso"][torso], "qwen_analysis", capability["vision_spatial_authority"])

    cyaw = coarse["head_yaw"]
    if cyaw not in {"frontalish", "unknown"} and limit and counts["coarse_yaw"].get(cyaw, 0) <= limit:
        add("head_yaw", cyaw, counts["coarse_yaw"][cyaw], "qwen_analysis", capability["vision_spatial_authority"])

    cpitch = coarse["head_pitch"]
    if cpitch not in {"neutralish", "unknown"} and limit and counts["coarse_pitch"].get(cpitch, 0) <= limit:
        add("head_pitch", cpitch, counts["coarse_pitch"][cpitch], "qwen_analysis", capability["vision_spatial_authority"])

    face_key = coarse["face_pose"]
    if face_key != "yaw:frontalish|pitch:neutralish" and limit and counts["coarse_face_pose"].get(face_key, 0) <= limit:
        add("face_pose", face_key, counts["coarse_face_pose"][face_key], "qwen_analysis", capability["vision_spatial_authority"])

    for tag in item["action_contact"]["tags"]:
        if tag not in HIGH_VALUE_ACTION_TAGS:
            continue
        n = counts["action"].get(tag, 0)
        source = "dwpose_geometry" if tag == "strong_shoulder_cant" else "qwen_visible_limbs"
        if source == "dwpose_geometry" and n <= 2:
            add("action_or_geometry", tag, n, source, "high_secondary_evidence")
        elif limit and n <= limit:
            add("action_or_contact", tag, n, source, capability["vision_semantics_authority"])

    for posture in item["action_contact"]["posture_tags"]:
        n = counts["posture"].get(posture, 0)
        if posture == "seated" and limit and n <= limit:
            add("posture", posture, n, "qwen_image_summary", capability["vision_semantics_authority"])

    return protected


def _marginal_values(item: dict[str, Any]) -> dict[str, str]:
    protected = item["protected_dimensions"]
    identity_dims = {"head_yaw", "head_pitch", "face_pose"}
    body_dims = {"composition", "torso_yaw", "action_or_geometry", "action_or_contact", "posture"}
    identity = "high" if any(p["dimension"] in identity_dims for p in protected) else (
        "medium" if item["coverage_signature"]["head_yaw"] != "frontalish" or item["coverage_signature"]["head_pitch"] != "neutralish" else "low"
    )
    body = "high" if any(p["dimension"] in body_dims for p in protected) else (
        "medium" if float(item["signal"]["pose_signal_score"]) >= 2.5 or item["action_contact"]["tags"] else "low"
    )
    order = {"low": 0, "medium": 1, "high": 2}
    overall = identity if order[identity] >= order[body] else body
    return {"identity_view_value": identity, "body_action_composition_value": body, "overall_marginal_value": overall}


def _base_recommendation(item: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    snr = item["active_snr"]["label"]
    protected = item["protected_dimensions"]
    nuisance = item["nuisance"]
    texture = item["background_texture_proxy"]
    reasons: list[str] = []
    replacement: list[str] = []

    entropy = float(item["active_snr"]["effective_entropy_component"])
    texture_label = str(texture.get("texture_label") or "unknown")
    entropy_focus_worthy = entropy >= 2.5 and texture_label in {"medium", "high", "unknown"}

    if snr == "low":
        reasons.append("low active subject-to-irrelevant-detail signal density")
        if protected:
            summary = ", ".join(f"{p['dimension']}={p['value']} (n={p['count']}→{p['remaining_if_removed']})" for p in protected)
            reasons.append(f"removing it would collapse thin/rare coverage: {summary}")
            replacement.append("preserve the protected coverage listed above")
            replacement.append("prefer higher subject signal density and lower irrelevant texture/confound burden")
            return "keep_until_cleaner_equivalent", reasons, replacement
        reasons.append("no currently trusted rare identity/body/action coverage depends on this image")
        replacement.append("replace with a cleaner image that fills a trusted dataset coverage gap")
        return "replace_candidate", reasons, replacement

    if entropy_focus_worthy:
        reasons.append("useful subject signal is present and irrelevant background texture/complexity is substantial")
        reasons.append("a subject/entropy-focus mask is worth testing before replacement")
        if protected:
            reasons.append("the image also carries thin/rare coverage, so masking is lower-risk than deletion")
        return "consider_entropy_focus", reasons, replacement

    if float(nuisance["background_occupancy_burden"]) >= 4.0 and texture_label == "low":
        reasons.append("background occupancy is substantial but measured texture is low; masking is not automatically prioritized")
    else:
        reasons.append("useful signal is not heavily diluted by currently identified high-entropy nuisance burden")
    return "keep", reasons, replacement


def _present_recommendation(base: str, capability: dict[str, Any]) -> dict[str, str]:
    tier = capability["policy_tier"]
    if tier == "high":
        return {"label": base, "strength": "qualified", "scope": "detailed"}
    if tier == "medium":
        mapping = {
            "replace_candidate": "review_for_replacement",
            "keep_until_cleaner_equivalent": "preserve_pending_review",
            "consider_entropy_focus": "mask_test_candidate",
            "keep": "keep_provisionally",
        }
        return {"label": mapping.get(base, base), "strength": "qualified", "scope": "moderate"}
    mapping = {
        "replace_candidate": "manual_review_candidate",
        "keep_until_cleaner_equivalent": "do_not_remove_without_review",
        "consider_entropy_focus": "possible_mask_test",
        "keep": "no_issue_flagged",
    }
    return {"label": mapping.get(base, base), "strength": "screening_only", "scope": "broad"}


def _highest_value_additions(counts: dict[str, Counter[str]], capability: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tier = capability["policy_tier"]
    full = counts["composition"].get("full_body", 0)
    if full <= 2:
        out.append({"priority": 1, "target": "clean full-length image with a different facial/action pose", "reason": f"trusted full-body coverage is thin (n={full})"})

    if tier == "low":
        out.extend(
            [
                {"priority": 1, "target": "more clearly non-frontal face views", "reason": "small-model vision is not trusted enough here to distinguish precise three-quarter/profile coverage automatically"},
                {"priority": 2, "target": "at least one clearly raised/upward head pose", "reason": "broad pitch diversity should be checked manually or with stronger vision"},
            ]
        )
    else:
        for cls, label in (
            ("strong_left", "clean strong-left-yaw / near-profile-or-profile face"),
            ("strong_right", "clean strong-right-yaw / near-profile-or-profile face"),
            ("three_quarter_left", "clean left three-quarter face"),
            ("three_quarter_right", "clean right three-quarter face"),
        ):
            n = counts["coarse_yaw"].get(cls, 0)
            cutoff = 1 if tier == "medium" else 2
            if n <= cutoff:
                out.append({"priority": 1 if n == 0 else 2, "target": label, "reason": f"qualified head-yaw coverage is thin (n={n})"})
        up = counts["coarse_pitch"].get("up", 0) + counts["coarse_pitch"].get("strong_up", 0)
        if up <= 1:
            out.append({"priority": 1, "target": "face with clearly raised/upward head pitch", "reason": f"qualified upward-pitch coverage is thin (n={up})"})

    nonfrontal_torso = sum(v for k, v in counts["torso"].items() if k not in {"frontal", "unknown"})
    if tier == "high" and nonfrontal_torso <= 2:
        out.append({"priority": 2, "target": "clean non-frontal torso view with the face usefully visible", "reason": f"VLM-reported non-frontal torso coverage is thin (n={nonfrontal_torso}); axis quality should still be spot-checked"})

    out.sort(key=lambda x: (int(x["priority"]), str(x["target"])))
    return out[:8]


def _make_markdown(payload: dict[str, Any]) -> str:
    s = payload["dataset_summary"]
    cap = payload["analysis_capability_policy"]
    lines = [
        "# Dataset evidence report — source-aware v3",
        "",
        f"> Analysis source: **{cap['model_id']}** · policy tier **{cap['policy_tier']}** · judgement breadth **{cap['judgement_breadth']}**.",
        "> This qualification is a harness policy, not a universal model benchmark. DWPose and deterministic pixel measurements retain their own evidence authority.",
        "",
        "## Dataset summary",
        "",
        f"- Images profiled: **{s['image_count']}**",
        f"- Effective shot scale: `{json.dumps(s['effective_shot_scale_counts'], sort_keys=True)}`",
        f"- Trusted composition classes: `{json.dumps(s['trusted_composition_counts'], sort_keys=True)}`",
        f"- Active SNR: `{json.dumps(s['active_snr_counts'], sort_keys=True)}`",
        f"- Presented recommendations: `{json.dumps(s['presented_recommendation_counts'], sort_keys=True)}`",
        "",
        "## Identity-view coverage",
        "",
        f"- Coarse head yaw: `{json.dumps(s['coarse_head_yaw_counts'], sort_keys=True)}`",
        f"- Coarse head pitch: `{json.dumps(s['coarse_head_pitch_counts'], sort_keys=True)}`",
        f"- Expressions: `{json.dumps(s['expression_counts'], sort_keys=True)}`",
        "",
        "## Body / action / composition coverage",
        "",
        f"- Torso yaw: `{json.dumps(s['torso_yaw_counts'], sort_keys=True)}`",
        f"- Action/contact tags: `{json.dumps(s['action_tag_counts'], sort_keys=True)}`",
        f"- Posture tags: `{json.dumps(s['posture_tag_counts'], sort_keys=True)}`",
        "",
        "## Analyzer-axis caution flags",
        "",
    ]
    if s["axis_quality_flags"]:
        for flag in s["axis_quality_flags"]:
            lines.append(f"- **{flag['axis']}**: {flag['dominant_value']} = {flag['dominant_count']}/{s['image_count']} ({flag['dominant_share']:.1%}). {flag['interpretation']}")
    else:
        lines.append("- None triggered by the current degeneracy heuristic.")

    lines.extend(["", "## Highest-value additions", ""])
    for i, item in enumerate(s["highest_value_additions"], 1):
        lines.append(f"{i}. **{item['target']}** — {item['reason']}")

    lines.extend(
        [
            "",
            "## Per-image evidence",
            "",
            "| image | composition | face pose | actions | SNR | bg texture | protected evidence | judgement |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    severity = {"replace_candidate": 0, "keep_until_cleaner_equivalent": 1, "consider_entropy_focus": 2, "keep": 3}
    records = sorted(payload["records"], key=lambda r: (severity.get(r["base_recommendation"], 9), float(r["active_snr"]["heuristic_score"])))
    for r in records:
        prot = ", ".join(f"{p['dimension']}:{p['value']}" for p in r["protected_dimensions"]) or "—"
        actions = ", ".join(r["action_contact"]["tags"]) or "—"
        face = f"{r['coverage_signature']['head_yaw']}/{r['coverage_signature']['head_pitch']}"
        lines.append(
            f"| {r['relative_path']} | {r['coverage_signature']['composition']} | {face} | {actions} | "
            f"{r['active_snr']['label']} ({r['active_snr']['heuristic_score']:.3f}) | "
            f"{r['background_texture_proxy']['texture_label']} | {prot} | **{r['presented_recommendation']['label']}** |"
        )

    lines.extend(["", "## Automated suggestions", ""])
    for r in records:
        if r["base_recommendation"] == "keep":
            continue
        lines.append(f"### {r['relative_path']} — {r['presented_recommendation']['label']}")
        lines.append("")
        lines.append(f"- Base action: `{r['base_recommendation']}`; presentation strength: `{r['presented_recommendation']['strength']}`.")
        for reason in r["recommendation_reasons"]:
            lines.append(f"- {reason}")
        if r["replacement_target"]:
            lines.append("- Replacement target:")
            for target in r["replacement_target"]:
                lines.append(f"  - {target}")
        top = (r["nuisance"].get("regions") or [])[:3]
        if top:
            lines.append("- Largest semantic entropy-burden regions:")
            for region in top:
                lines.append(
                    f"  - {region.get('description') or 'unnamed region'} "
                    f"({region.get('frame_coverage')}, {region.get('visual_complexity')}; "
                    f"occupancy {region.get('occupancy_burden_points')}, entropy {region.get('semantic_entropy_burden_points')})"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = _read_json(run_dir / "run.json")
    if manifest is None:
        print(f"Missing/invalid run manifest: {run_dir / 'run.json'}", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    model_dir = run_dir / slug
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    if not model_dir.is_dir():
        print(f"Analysis model directory not found: {model_dir}", file=sys.stderr)
        return 2
    if not dwpose_dir.is_dir():
        print(f"DWPose directory not found: {dwpose_dir}", file=sys.stderr)
        return 2

    capability = _capability_profile(model_id)
    records: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {
        "shot": Counter(),
        "composition": Counter(),
        "torso": Counter(),
        "head_yaw": Counter(),
        "head_pitch": Counter(),
        "head_roll": Counter(),
        "coarse_yaw": Counter(),
        "coarse_pitch": Counter(),
        "coarse_face_pose": Counter(),
        "expression": Counter(),
        "action": Counter(),
        "posture": Counter(),
    }
    framing_conflicts = 0

    for record in manifest.get("images") or []:
        key = record.get("result_key")
        if not key:
            continue
        analysis_record = _read_json(model_dir / f"{key}.analysis.json")
        dwpose_record = _read_json(dwpose_dir / f"{key}.dwpose.json")
        analysis = (analysis_record or {}).get("analysis")
        if not isinstance(analysis, dict) or dwpose_record is None:
            continue

        pose = build_pose_evidence(dwpose_record)
        face = _facial_pose(analysis)
        framing = _effective_framing(analysis, pose)
        signal = _signal_profile(analysis, pose, framing, capability)
        nuisance = _nuisance_profile(analysis)
        confound = _confound_burden(analysis, pose)
        action = _action_profile(analysis, pose, face)
        report_image = record.get("report_image")
        image_path = run_dir / str(report_image) if report_image else None
        texture = _background_texture_proxy(image_path, pose)
        active_snr = _active_snr(float(signal["identity_signal_score"]), nuisance, float(confound["score"]), texture, signal["subject_frame_coverage"])

        cyaw = _coarse_yaw(face["head_yaw"])
        cpitch = _coarse_pitch(face["head_pitch"])
        composition = signal["trusted_composition_class"]
        face_key = f"yaw:{cyaw}|pitch:{cpitch}"
        signature = {"composition": composition, "head_yaw": cyaw, "head_pitch": cpitch, "face_pose": face_key}

        counts["shot"][signal["effective_shot_scale"]] += 1
        counts["composition"][composition] += 1
        counts["torso"][face["torso_yaw"]] += 1
        counts["head_yaw"][face["head_yaw"]] += 1
        counts["head_pitch"][face["head_pitch"]] += 1
        counts["head_roll"][face["head_roll"]] += 1
        counts["coarse_yaw"][cyaw] += 1
        counts["coarse_pitch"][cpitch] += 1
        counts["coarse_face_pose"][face_key] += 1
        counts["expression"][face["expression"]["primary"]] += 1
        for tag in action["tags"]:
            counts["action"][tag] += 1
        for posture in action["posture_tags"]:
            counts["posture"][posture] += 1
        framing_conflicts += int(bool(framing["framing_conflict"]))

        records.append(
            {
                "relative_path": record.get("relative_path"),
                "result_key": key,
                "image_summary": analysis.get("image_summary"),
                "signal": signal,
                "framing_fusion": framing,
                "facial_pose": face,
                "coverage_signature": signature,
                "action_contact": action,
                "nuisance": nuisance,
                "confound": confound,
                "background_texture_proxy": texture,
                "active_snr": active_snr,
                "pose_evidence": pose,
            }
        )

    for item in records:
        item["protected_dimensions"] = _protected_dimensions(item, counts, capability)
        item["marginal_value"] = _marginal_values(item)
        base, reasons, replacement = _base_recommendation(item)
        item["base_recommendation"] = base
        item["presented_recommendation"] = _present_recommendation(base, capability)
        item["recommendation_reasons"] = reasons
        item["replacement_target"] = replacement

    total = len(records)
    base_counts = Counter(r["base_recommendation"] for r in records)
    presented_counts = Counter(r["presented_recommendation"]["label"] for r in records)
    snr_counts = Counter(r["active_snr"]["label"] for r in records)
    summary = {
        "image_count": total,
        "effective_shot_scale_counts": dict(sorted(counts["shot"].items())),
        "trusted_composition_counts": dict(sorted(counts["composition"].items())),
        "torso_yaw_counts": dict(sorted(counts["torso"].items())),
        "head_yaw_counts": dict(sorted(counts["head_yaw"].items())),
        "head_pitch_counts": dict(sorted(counts["head_pitch"].items())),
        "head_roll_counts": dict(sorted(counts["head_roll"].items())),
        "coarse_head_yaw_counts": dict(sorted(counts["coarse_yaw"].items())),
        "coarse_head_pitch_counts": dict(sorted(counts["coarse_pitch"].items())),
        "expression_counts": dict(sorted(counts["expression"].items())),
        "action_tag_counts": dict(sorted(counts["action"].items())),
        "posture_tag_counts": dict(sorted(counts["posture"].items())),
        "framing_conflict_count": framing_conflicts,
        "active_snr_counts": dict(sorted(snr_counts.items())),
        "base_recommendation_counts": dict(sorted(base_counts.items())),
        "presented_recommendation_counts": dict(sorted(presented_counts.items())),
        "axis_quality_flags": _axis_quality_flags(counts, total),
        "highest_value_additions": _highest_value_additions(counts, capability),
    }

    payload = {
        "schema_version": "dataset-evidence-3.0",
        "analysis_model": model_id,
        "analysis_source": str(model_dir),
        "dwpose_source": str(dwpose_dir),
        "analysis_capability_policy": capability,
        "method_notes": [
            "Judgement breadth and recommendation wording are qualified by the source vision model tier.",
            "DWPose retains independent authority for measurable 2D geometry/visible extent and can outweigh weaker VLM framing evidence.",
            "Exact VLM-derived pose/action rarity protects images less aggressively for smaller vision models.",
            "Facial-pose rarity uses coarse bins so a one-off slight yaw does not become artificially precious.",
            "Action/contact coverage is extracted conservatively from cached visible_limbs and kept provenance-tagged.",
            "Background occupancy and background visual entropy are separated; low-texture large regions no longer automatically imply Entropy Focus.",
            "The deterministic texture proxy excludes a padded DWPose keypoint rectangle when possible, but is not a real subject mask.",
            "Per-image training loss remains separate empirical evidence and should later join these profile features rather than be hidden inside active_snr.",
        ],
        "dataset_summary": summary,
        "records": records,
    }

    out_json = run_dir / f"{args.output_prefix}_{slug}.json"
    out_md = run_dir / f"{args.output_prefix}_{slug}.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_make_markdown(payload), encoding="utf-8")

    print(f"Done. JSON:   {out_json}")
    print(f"      Report: {out_md}")
    print(f"Analysis policy: {capability['policy_tier']} / {capability['judgement_breadth']}")
    print(f"Coarse head yaw: {dict(sorted(counts['coarse_yaw'].items()))}")
    print(f"Coarse head pitch: {dict(sorted(counts['coarse_pitch'].items()))}")
    print(f"Action/contact tags: {dict(sorted(counts['action'].items()))}")
    print(f"Active SNR: {dict(sorted(snr_counts.items()))}")
    print(f"Recommendations: {dict(sorted(presented_counts.items()))}")
    if summary["axis_quality_flags"]:
        print("Analyzer-axis caution flags:", [f["axis"] for f in summary["axis_quality_flags"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
