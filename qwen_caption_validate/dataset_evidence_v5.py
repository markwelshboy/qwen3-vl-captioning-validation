from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .dataset_evidence import _confound_burden, _read_json
from .dataset_evidence_v2 import _effective_framing, _facial_pose
from .dataset_evidence_v3 import (
    _axis_quality_flags,
    _background_texture_proxy,
    _capability_profile,
    _coarse_pitch,
    _coarse_yaw,
    _nuisance_profile,
    _present_recommendation,
    _signal_profile,
)
from .pose_evidence import build_pose_evidence
from .runner import model_slug, resolve_model_id


SEMANTIC_DEFAULT_CONFIDENCE = {"low_to_medium": 0.45, "medium": 0.62, "high": 0.78}
SPATIAL_DEFAULT_CONFIDENCE = {"low": 0.35, "low_to_medium": 0.50, "medium": 0.65}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "Source-calibrated dataset profiler with dynamic VLM-axis quarantine, "
            "model-independent measured signal-density evidence, and source-dependent semantic granularity."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence_v5", help="Output basename inside run_dir.")
    return parser.parse_args()


def _semantic_default(capability: dict[str, Any]) -> float:
    return SEMANTIC_DEFAULT_CONFIDENCE.get(str(capability.get("vision_semantics_authority") or "medium"), 0.62)


def _spatial_default(capability: dict[str, Any]) -> float:
    return SPATIAL_DEFAULT_CONFIDENCE.get(str(capability.get("vision_spatial_authority") or "low_to_medium"), 0.50)


def _contains_word(text: str, *words: str) -> bool:
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", text) for word in words)


def _action_profile(
    analysis: dict[str, Any],
    pose: dict[str, Any],
    face: dict[str, Any],
    capability: dict[str, Any],
) -> dict[str, Any]:
    """Extract coarse action/contact evidence with conservative semantic ownership.

    Exact strings remain inspectable evidence, but coverage classes stay broad.
    Word-boundary matching deliberately prevents `headrest` from becoming `head`.
    """
    raw_tags: set[str] = set()
    evidence: list[dict[str, Any]] = []
    class_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    warnings: list[str] = []
    semantic_default = _semantic_default(capability)
    limbs = ((analysis.get("target_subject") or {}).get("visible_limbs") or [])

    unsupported_hands = [
        h for h in (pose.get("hand_candidates") or [])
        if not h.get("supported_by_nearby_visible_target_wrist") and int(h.get("visible_keypoints") or 0) >= 5
    ]

    def add_raw(tag: str, source: str, text: str, confidence: float, coarse_class: str | None = None) -> None:
        raw_tags.add(tag)
        item = {
            "tag": tag,
            "source": source,
            "text": text[:320],
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
        }
        evidence.append(item)
        if coarse_class:
            class_evidence[coarse_class].append(item)

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
        part_text = str(limb.get("part") or "").lower()
        conf = _safe_confidence(limb.get("confidence"), semantic_default)

        if _contains_word(text, "unknown"):
            conf = min(conf, 0.60)
        if unsupported_hands and _contains_word(part_text, "hand", "wrist", "arm", "forearm"):
            conf = min(conf, 0.75)

        if _contains_word(text, "pocket", "pockets"):
            add_raw("hands_in_pockets", "qwen_visible_limbs", text, conf, "hands_near_hips")
        if _contains_word(contact_text, "hip", "hips") and not _contains_word(text, "pocket", "pockets"):
            add_raw("hand_at_hip", "qwen_visible_limbs", text, min(conf, 0.62), "hands_near_hips")
        if _contains_word(text, "hold", "holds", "holding", "grip", "grips", "gripping", "grasp", "grasps", "grasping"):
            add_raw("holding_object", "qwen_visible_limbs", text, conf, "object_interaction")
        if _contains_word(text, "reach", "reaches", "reaching"):
            add_raw("reaching", "qwen_visible_limbs", text, conf, "object_interaction")
        if _contains_word(text, "cross", "crossed", "crossing") and _contains_word(text, "arm", "arms", "forearm", "forearms", "wrist", "wrists", "hand", "hands"):
            add_raw("arms_crossed", "qwen_visible_limbs", text, conf, "arms_crossed")
        if _contains_word(contact_text, "table", "desk", "counter"):
            add_raw("surface_contact", "qwen_visible_limbs", text, conf, "surface_contact")

        # Exact anatomical tokens only: "headrest" must not count as "head".
        if _contains_word(contact_text, "chin", "cheek", "face", "head"):
            add_raw("hand_face_contact", "qwen_visible_limbs", text, conf, "head_support")
            if (
                _contains_word(contact_text, "support", "supports", "supported", "rest", "rests", "resting", "knuckle", "knuckles", "fist")
                or bool(limb.get("support"))
            ):
                add_raw("chin_or_head_support", "qwen_visible_limbs", text, conf, "head_support")

        # "extends downward and out of frame" is ordinary crop geometry, not a
        # meaningful forward-extension action. Require explicit forward/reach/lens evidence.
        forward = _contains_word(text, "forward", "forwards") or "toward the camera" in text or "towards the camera" in text or "toward the lens" in text
        reaching = _contains_word(text, "reach", "reaches", "reaching")
        extension = _contains_word(text, "extend", "extends", "extended", "extending")
        if reaching or (forward and extension):
            add_raw("forward_arm_extension", "qwen_visible_limbs", text, conf, "forward_arm_extension")

    summary = str(analysis.get("image_summary") or "").lower()
    posture: list[str] = []
    if _contains_word(summary, "seated", "sitting", "sits"):
        posture.append("seated")
    if _contains_word(summary, "standing", "stands"):
        posture.append("standing")

    torso = str(face.get("torso_yaw") or "unknown")
    head = str(face.get("head_yaw") or "unknown")
    if torso not in {"frontal", "unknown"} and head not in {"unknown", torso}:
        add_raw(
            "head_torso_counter_rotation",
            "qwen_orientation_fusion",
            f"torso={torso}; head={head}",
            _spatial_default(capability),
            "head_torso_counter_rotation",
        )

    shoulder = ((pose.get("target_2d_geometry") or {}).get("shoulder_line_angle_from_horizontal_deg"))
    if shoulder is not None:
        try:
            angle = abs(float(shoulder))
            if angle >= 15.0:
                add_raw("strong_shoulder_cant", "dwpose_geometry", f"abs shoulder-line angle={angle:.2f} deg", 0.92, "strong_shoulder_cant")
            elif angle >= 8.0:
                add_raw("moderate_shoulder_cant", "dwpose_geometry", f"abs shoulder-line angle={angle:.2f} deg", 0.86, None)
        except (TypeError, ValueError):
            pass

    if unsupported_hands:
        warnings.append(
            "one or more detected hands lack a nearby visible target wrist; hand ownership/contact semantics are confidence-capped"
        )

    coarse_classes: list[dict[str, Any]] = []
    for name, items in sorted(class_evidence.items()):
        coarse_classes.append(
            {
                "class": name,
                "confidence": round(max(float(i["confidence"]) for i in items), 3),
                "sources": sorted({str(i["source"]) for i in items}),
                "evidence_tags": sorted({str(i["tag"]) for i in items}),
            }
        )

    components = [c["class"] for c in coarse_classes]
    if "object_interaction" in components and "forward_arm_extension" in components:
        components.remove("forward_arm_extension")
    components = sorted(set(components))

    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in evidence:
        dedup[(str(item["tag"]), str(item["source"]), str(item["text"]))] = item

    return {
        "raw_tags": sorted(raw_tags),
        "coarse_classes": coarse_classes,
        "action_signature": "+".join(components) if components else "none",
        "signature_components": components,
        "posture_tags": posture,
        "evidence": list(dedup.values()),
        "semantic_warnings": warnings,
    }


def _safe_confidence(value: Any, fallback: float) -> float:
    try:
        v = float(value)
        if 0.0 <= v <= 1.0:
            return v
    except (TypeError, ValueError):
        pass
    return fallback


def _measurement_composition(pose: dict[str, Any]) -> str:
    extent = str((pose.get("target_2d_geometry") or {}).get("pose_extent_hint") or "unknown")
    return {
        "close_or_medium_close": "identity_close_proxy",
        "waist_or_upper_body": "upper_body_proxy",
        "three_quarter_or_long": "long_body_proxy",
        "full_length": "full_body_proxy",
    }.get(extent, "unknown")


def _measured_signal_density(pose: dict[str, Any], texture: dict[str, Any]) -> dict[str, Any]:
    """Model-independent provisional signal-density core.

    This intentionally uses only DWPose geometry and deterministic pixels. It is
    not a true subject/face-pixel measurement yet; a matte/face detector should
    supersede the rectangle proxies later.
    """
    geom = pose.get("target_2d_geometry") or {}
    bbox = geom.get("clipped_in_frame_keypoint_bbox") or {}
    height = _as_float(bbox.get("height_fraction"), 0.0)
    area = _as_float(bbox.get("area_fraction"), 0.0)
    connectivity = geom.get("connectivity") or {}
    complete = sum(bool((connectivity.get(name) or {}).get("complete")) for name in ("left_arm", "right_arm", "left_leg", "right_leg"))
    extent = str(geom.get("pose_extent_hint") or "unknown")

    base = {
        "close_or_medium_close": 1.00,
        "waist_or_upper_body": 0.84,
        "three_quarter_or_long": 0.72,
        "full_length": 0.64,
    }.get(extent, 0.70)
    height_factor = 0.70 + 0.30 * min(1.0, height / 0.70) if height > 0 else 0.70
    area_factor = 0.82 + 0.18 * min(1.0, area / 0.45) if area > 0 else 0.82
    connectivity_factor = 1.0 + 0.025 * complete
    signal_proxy = base * height_factor * area_factor * connectivity_factor

    sample_fraction = texture.get("sample_fraction")
    outside_fraction = _as_float(sample_fraction, max(0.0, 1.0 - area))
    texture_score = _as_float(texture.get("texture_score"), 0.35)
    people = pose.get("person_evidence") or {}
    significant_secondary = int(people.get("significant_secondary_people") or 0)
    small_secondary = int(people.get("small_secondary_people") or 0)
    measured_burden = 0.55 * outside_fraction + 0.75 * texture_score + 0.18 * significant_secondary + 0.06 * small_secondary
    score = signal_proxy / (0.35 + measured_burden)
    label = "low" if score < 0.75 else "medium" if score < 1.05 else "high"

    return {
        "label": label,
        "score": round(score, 3),
        "signal_proxy": round(signal_proxy, 4),
        "measured_background_burden": round(measured_burden, 4),
        "dwpose_extent_class": _measurement_composition(pose),
        "keypoint_bbox_height_fraction": round(height, 5) if height else None,
        "keypoint_bbox_area_fraction": round(area, 5) if area else None,
        "outside_padded_bbox_sample_fraction": round(outside_fraction, 4),
        "deterministic_texture_score": round(texture_score, 4),
        "complete_limb_chains": complete,
        "secondary_people_penalty": {
            "significant": significant_secondary,
            "small": small_secondary,
        },
        "authority": "high_measurement_evidence_but_proxy_geometry",
        "definition": (
            "provisional model-independent training-signal-density core using DWPose extent/rectangle geometry and deterministic background texture; "
            "it is not photometric SNR and is not yet a true subject/face-pixel measurement"
        ),
    }


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _semantic_burden_profile(
    nuisance: dict[str, Any],
    confound: dict[str, Any],
    capability: dict[str, Any],
    measured: dict[str, Any],
) -> dict[str, Any]:
    occupancy = _as_float(nuisance.get("background_occupancy_burden"), 0.0)
    entropy = _as_float(nuisance.get("semantic_entropy_burden"), 0.0)
    confound_score = _as_float(confound.get("score"), 0.0)
    raw = 0.08 * occupancy + 0.18 * entropy + 0.25 * confound_score
    tier_weight = {"high": 0.85, "medium": 0.45, "low": 0.20}.get(str(capability.get("policy_tier")), 0.45)
    adjusted_score = float(measured["score"]) / (1.0 + 0.12 * tier_weight * raw)
    adjusted_label = "low" if adjusted_score < 0.75 else "medium" if adjusted_score < 1.05 else "high"
    reported = int(nuisance.get("region_count") or 0) > 0 or confound_score > 0
    return {
        "observed_burden_score": round(raw, 3),
        "authority": capability.get("vision_semantics_authority"),
        "source_weight": tier_weight,
        "observation_status": "reported_evidence" if reported else "not_reported_not_assumed_zero",
        "qualified_adjusted_score": round(adjusted_score, 3),
        "qualified_adjusted_label": adjusted_label,
        "used_for_dataset_action": False,
        "note": (
            "VLM nuisance/confound evidence is shown separately and does not decide the v5 keep/replace action. "
            "Absence of a reported nuisance is not interpreted as evidence that the scene is clean."
        ),
    }


def _dynamic_axis_quarantine(counts: dict[str, Counter[str]], total: int) -> dict[str, dict[str, Any]]:
    quarantined: dict[str, dict[str, Any]] = {}
    if total < 12:
        return quarantined
    for flag in _axis_quality_flags(counts, total):
        axis = str(flag["axis"])
        unique_values = len(counts.get(axis, {}))
        if float(flag.get("dominant_share") or 0.0) >= 0.90 and unique_values <= 2:
            quarantined[axis] = {
                **flag,
                "status": "quarantined",
                "effect": "excluded from coverage buckets, representative protection, and dataset-addition advice",
                "reason": "dataset-level output is too degenerate to distinguish real bias from analyzer collapse safely",
            }
    return quarantined


def _coverage_availability(quarantine: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for axis in ("head_yaw", "head_pitch", "head_roll", "torso"):
        if axis in quarantine:
            out[axis] = {
                "status": "unavailable_from_this_analysis_source",
                "reason": quarantine[axis]["reason"],
            }
        else:
            out[axis] = {"status": "available_qualified", "reason": "axis passed the current dataset-level degeneracy check"}
    out["face_pose"] = {
        "status": "unavailable_from_this_analysis_source" if "head_yaw" in quarantine or "head_pitch" in quarantine else "available_qualified",
        "reason": "depends on both head_yaw and head_pitch",
    }
    return out


def _action_class_confidence(action: dict[str, Any], name: str) -> float:
    for item in action.get("coarse_classes") or []:
        if item.get("class") == name:
            return float(item.get("confidence") or 0.0)
    return 0.0


def _coverage_memberships(item: dict[str, Any], capability: dict[str, Any], quarantine: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tier = str(capability["policy_tier"])
    sig = item["coverage_signature"]
    face = item["facial_pose"]

    def add(dimension: str, value: str, quota: int, source: str, authority: str, confidence: float) -> None:
        if quota <= 0 or value in {"unknown", "none", ""}:
            return
        out.append(
            {
                "dimension": dimension,
                "value": value,
                "quota": quota,
                "source": source,
                "authority": authority,
                "confidence": round(confidence, 3),
            }
        )

    # Composition remains protectable independent of VLM spatial quality because
    # full-length status is anchored by DWPose visible-joint extent.
    if sig["composition"] == "full_body":
        add("composition", "full_body", 2, "dwpose+qwen_framing_fusion", "high_for_visible_extent", 0.90)

    if tier == "high":
        pose_quota = 2
        combo_quota = 1
    elif tier == "medium":
        pose_quota = 1
        combo_quota = 1
    else:
        pose_quota = 0
        combo_quota = 0

    if "head_yaw" not in quarantine and sig["head_yaw"] not in {"frontalish", "unknown"}:
        add("head_yaw", sig["head_yaw"], pose_quota, "qwen_analysis", capability["vision_spatial_authority"], _spatial_default(capability))
    if "head_pitch" not in quarantine and sig["head_pitch"] not in {"neutralish", "unknown"}:
        add("head_pitch", sig["head_pitch"], pose_quota, "qwen_analysis", capability["vision_spatial_authority"], _spatial_default(capability))
    if "head_yaw" not in quarantine and "head_pitch" not in quarantine and sig["face_pose"] != "yaw:frontalish|pitch:neutralish":
        add("face_pose", sig["face_pose"], combo_quota, "qwen_analysis", capability["vision_spatial_authority"], _spatial_default(capability))

    torso = str(face.get("torso_yaw") or "unknown")
    if tier == "high" and "torso" not in quarantine and torso not in {"frontal", "unknown"}:
        add("torso_yaw", torso, 1, "qwen_analysis", capability["vision_spatial_authority"], _spatial_default(capability))

    components = list(item["action_contact"].get("signature_components") or [])
    if "strong_shoulder_cant" in components:
        add("geometry_class", "strong_shoulder_cant", 1, "dwpose_geometry", "high_secondary_evidence", 0.92)

    # Fine semantic action coverage is only allowed to protect images for the
    # high-tier analyzer. Medium/low analyzers still report broad action hints,
    # but those hints cannot make a weak image irreplaceable.
    if tier == "high":
        semantic_components = [c for c in components if c != "strong_shoulder_cant"]
        eligible = [c for c in semantic_components if _action_class_confidence(item["action_contact"], c) >= 0.75]
        if eligible:
            value = "+".join(sorted(eligible))
            dimension = "compound_action_signature" if len(eligible) >= 2 else "action_class"
            confidence = min(_action_class_confidence(item["action_contact"], c) for c in eligible)
            add(dimension, value, 1, "qwen_visible_limbs", capability["vision_semantics_authority"], confidence)

    return out


def _representative_score(item: dict[str, Any]) -> float:
    measured = item["measured_signal_density"]
    score = float(measured["score"])
    score += 0.04 * int(measured.get("complete_limb_chains") or 0)
    # Confidence is applied separately as a tie-breaker inside each bucket.
    return score


def _apply_coverage_quotas(records: list[dict[str, Any]], capability: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, item in enumerate(records):
        memberships = _coverage_memberships(item, capability, quarantine)
        item["coverage_memberships"] = memberships
        item["representative_score"] = round(_representative_score(item), 4)
        for membership in memberships:
            buckets[(membership["dimension"], membership["value"])].append((idx, membership))

    summaries: list[dict[str, Any]] = []
    for (dimension, value), members in sorted(buckets.items()):
        quota = max(int(m["quota"]) for _, m in members)
        ranked = sorted(
            members,
            key=lambda pair: (float(records[pair[0]]["representative_score"]), float(pair[1].get("confidence") or 0.0)),
            reverse=True,
        )
        selected = ranked[:quota]
        selected_indices = {idx for idx, _ in selected}
        for rank, (idx, membership) in enumerate(ranked, 1):
            membership["bucket_count"] = len(ranked)
            membership["representative_rank"] = rank
            membership["selected_as_representative"] = idx in selected_indices
            membership["remaining_bucket_members_if_removed"] = max(0, len(ranked) - 1)
        summaries.append(
            {
                "dimension": dimension,
                "value": value,
                "quota": quota,
                "candidate_count": len(ranked),
                "selected_representatives": [records[idx]["relative_path"] for idx, _ in selected],
                "ranked_candidates": [
                    {
                        "image": records[idx]["relative_path"],
                        "representative_score": records[idx]["representative_score"],
                        "confidence": membership.get("confidence"),
                    }
                    for idx, membership in ranked
                ],
            }
        )

    for item in records:
        item["protected_dimensions"] = [m for m in item["coverage_memberships"] if m.get("selected_as_representative")]
    return {"buckets": summaries}


def _marginal_values(item: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, str]:
    protected = item["protected_dimensions"]
    identity_dims = {"head_yaw", "head_pitch", "face_pose"}
    body_dims = {"composition", "torso_yaw", "geometry_class", "action_class", "compound_action_signature"}

    if "head_yaw" in quarantine and "head_pitch" in quarantine:
        identity = "unassessed_from_source"
    elif any(p["dimension"] in identity_dims for p in protected):
        identity = "high"
    elif item["coverage_signature"]["head_yaw"] != "frontalish" or item["coverage_signature"]["head_pitch"] != "neutralish":
        identity = "medium"
    else:
        identity = "low"

    if any(p["dimension"] in body_dims for p in protected):
        body = "high"
    elif int(item["measured_signal_density"].get("complete_limb_chains") or 0) >= 2 or item["action_contact"]["signature_components"]:
        body = "medium"
    else:
        body = "low"

    rank = {"low": 0, "unassessed_from_source": 0, "medium": 1, "high": 2}
    overall = identity if rank.get(identity, 0) >= rank.get(body, 0) else body
    return {"identity_view_value": identity, "body_action_composition_value": body, "overall_marginal_value": overall}


def _dataset_action(item: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    density = item["measured_signal_density"]
    protected = item["protected_dimensions"]
    reasons: list[str] = []
    replacement: list[str] = []

    if density["label"] == "low":
        reasons.append("low model-independent measured signal-density proxy")
        if protected:
            summary = ", ".join(
                f"{p['dimension']}={p['value']} (rank {p['representative_rank']}/{p['bucket_count']}, quota {p['quota']})"
                for p in protected
            )
            reasons.append(f"selected as a current best representative for trusted coverage: {summary}")
            replacement.extend(
                [
                    "preserve or improve the protected coverage listed above",
                    "prefer higher measured subject signal density and lower deterministic background burden",
                ]
            )
            return "keep_until_cleaner_equivalent", reasons, replacement
        reasons.append("not selected among the current best representatives for any trusted coverage bucket")
        replacement.append("replace with a cleaner image that fills a trusted coverage gap or outranks a current representative")
        return "replace_candidate", reasons, replacement

    reasons.append("model-independent measured signal density is not currently low enough to justify replacement")
    if protected:
        reasons.append("the image is also selected as a current representative for one or more trusted coverage buckets")
    return "keep", reasons, replacement


def _interventions(item: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    measured = item["measured_signal_density"]
    texture = item["background_texture_proxy"]
    composition = str(item["coverage_signature"].get("composition") or "unknown")
    bbox_area = measured.get("keypoint_bbox_area_fraction")

    if str(texture.get("texture_label") or "unknown") in {"medium", "high"} and _as_float(texture.get("texture_score"), 0.0) >= 0.30:
        out.append(
            {
                "type": "entropy_focus_test",
                "reason": "deterministic background texture is substantial enough to test masking independently of the keep/replace decision",
            }
        )
    if measured["label"] == "low":
        out.append(
            {
                "type": "subject_mask_measurement",
                "reason": "measured signal density is low and current geometry still uses a DWPose rectangle rather than a true subject matte/face-pixel measurement",
            }
        )
    if composition != "full_body" and bbox_area is not None and float(bbox_area) < 0.36 and measured["label"] == "low":
        out.append(
            {
                "type": "crop_review",
                "reason": "target keypoint extent is relatively small; a tighter crop may improve effective subject resolution if useful pose evidence is preserved",
            }
        )
    if composition == "full_body" and measured["label"] == "low":
        out.append(
            {
                "type": "seek_cleaner_full_body_equivalent",
                "reason": "cropping would destroy full-body coverage, so a cleaner equivalent is preferable if available",
            }
        )
    if any(c["class"] == "hands_near_hips" for c in item["action_contact"].get("coarse_classes") or []):
        out.append(
            {
                "type": "action_semantics_review",
                "reason": "hand placement around hips/pockets is intentionally represented coarsely because exact contact may be visually ambiguous",
            }
        )
    if "strong_shoulder_cant" in item["action_contact"].get("signature_components", []):
        out.append(
            {
                "type": "geometry_caption_audit",
                "reason": "DWPose measures a strong shoulder-line cant that should not be flattened by later caption composition",
            }
        )
    if item["action_contact"].get("semantic_warnings"):
        out.append(
            {
                "type": "ownership_semantics_review",
                "reason": "hand/contact semantics were confidence-capped because pose evidence contains an unsupported hand candidate",
            }
        )
    return out


def _highest_value_additions(counts: dict[str, Counter[str]], quarantine: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(priority: int, target: str, reason: str) -> None:
        out.append({"priority": priority, "target": target, "reason": reason})

    full = counts["composition"].get("full_body", 0)
    if full < 3:
        add(1, "clean full-length image with a different facial/action pose where possible", f"trusted full-body coverage is thin (n={full})")

    if "head_yaw" not in quarantine:
        for direction in ("strong_left", "strong_right"):
            n = counts["coarse_yaw"].get(direction, 0)
            if n == 0:
                add(1, f"clean {direction.replace('_', '-')} / near-profile-or-profile face", f"qualified head-yaw coverage is thin (n={n})")
        left34 = counts["coarse_yaw"].get("three_quarter_left", 0)
        right34 = counts["coarse_yaw"].get("three_quarter_right", 0)
        if left34 <= 1:
            add(2, "clean left three-quarter face", f"qualified head-yaw coverage is thin (n={left34})")
        if right34 <= 1:
            add(2, "clean right three-quarter face", f"qualified head-yaw coverage is thin (n={right34})")

    if "head_pitch" not in quarantine:
        up = counts["coarse_pitch"].get("up", 0) + counts["coarse_pitch"].get("strong_up", 0)
        if up == 0:
            add(1, "face with clearly raised/upward head pitch", "qualified upward-pitch coverage is thin (n=0)")

    if "torso" not in quarantine:
        nonfrontal = sum(v for k, v in counts["torso"].items() if k not in {"frontal", "unknown"})
        if nonfrontal <= 1:
            add(2, "clean non-frontal torso view with the face usefully visible", f"qualified non-frontal torso coverage is thin (n={nonfrontal})")

    return sorted(out, key=lambda x: (int(x["priority"]), x["target"]))


def _assessment_gaps(quarantine: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    labels = {
        "head_yaw": "facial left/right view coverage",
        "head_pitch": "facial up/down view coverage",
        "head_roll": "head-roll coverage",
        "torso": "torso-orientation coverage",
    }
    for axis, item in quarantine.items():
        gaps.append(
            {
                "axis": axis,
                "coverage": labels.get(axis, axis),
                "status": "unavailable_from_this_analysis_source",
                "suggestion": "re-analyze with a stronger/calibrated analyzer or verify manually before making dataset-addition/removal decisions on this axis",
                "reason": str(item.get("reason")),
            }
        )
    return gaps


def _make_markdown(payload: dict[str, Any]) -> str:
    s = payload["dataset_summary"]
    cap = payload["analysis_capability_policy"]
    quarantine = payload["dynamic_axis_quarantine"]
    lines = [
        "# Dataset evidence report — source-calibrated v5",
        "",
        f"> Analysis source: **{cap['model_id']}** · policy tier **{cap['policy_tier']}** · judgement breadth **{cap['judgement_breadth']}**.",
        "> Keep/replace decisions use the model-independent measured signal-density proxy. VLM semantic burden is reported separately and absence is never interpreted as zero burden.",
        "> VLM pose axes that fail the dataset-level sanity check are quarantined from coverage protection and addition advice.",
        "",
        "## Dataset summary",
        "",
        f"- Images profiled: **{s['image_count']}**",
        f"- Trusted composition classes: `{json.dumps(s['trusted_composition_counts'], sort_keys=True)}`",
        f"- Measured signal density: `{json.dumps(s['measured_signal_density_counts'], sort_keys=True)}`",
        f"- Dataset actions: `{json.dumps(s['presented_recommendation_counts'], sort_keys=True)}`",
        f"- Interventions: `{json.dumps(s['intervention_counts'], sort_keys=True)}`",
        f"- Quarantined VLM axes: `{json.dumps(sorted(quarantine))}`",
        "",
        "## Evidence authority",
        "",
        f"- Vision semantics: **{cap['vision_semantics_authority']}**",
        f"- Vision spatial reasoning prior: **{cap['vision_spatial_authority']}**",
        "- DWPose 2-D geometry: **high secondary evidence**",
        "- Deterministic pixel evidence: **high measurement evidence (still rectangle-proxy based)**",
        f"- Semantic action preservation policy: **{cap['semantic_action_protection']}**",
        "",
        "## Identity-view coverage",
        "",
    ]

    availability = payload["coverage_availability"]
    for axis, count_key in (("head_yaw", "coarse_head_yaw_counts"), ("head_pitch", "coarse_head_pitch_counts"), ("head_roll", "head_roll_counts")):
        if availability[axis]["status"].startswith("unavailable"):
            lines.append(f"- **{axis}**: unavailable from this analysis source — quarantined after a degenerate dataset-level distribution.")
        else:
            lines.append(f"- **{axis}**: `{json.dumps(s[count_key], sort_keys=True)}`")

    lines.extend(["", "## Body / action / composition coverage", ""])
    if availability["torso"]["status"].startswith("unavailable"):
        lines.append("- **torso yaw**: unavailable from this analysis source — quarantined after a degenerate dataset-level distribution.")
    else:
        lines.append(f"- Torso yaw: `{json.dumps(s['torso_yaw_counts'], sort_keys=True)}`")
    lines.extend(
        [
            f"- Coarse action classes (reported evidence): `{json.dumps(s['action_class_counts'], sort_keys=True)}`",
            f"- Action signatures (reported evidence): `{json.dumps(s['action_signature_counts'], sort_keys=True)}`",
            f"- Posture tags: `{json.dumps(s['posture_tag_counts'], sort_keys=True)}`",
            "",
            "## Assessment gaps",
            "",
        ]
    )
    if payload["assessment_gaps"]:
        for gap in payload["assessment_gaps"]:
            lines.append(f"- **{gap['coverage']}** — {gap['suggestion']}")
    else:
        lines.append("- None triggered by the current axis sanity check.")

    lines.extend(["", "## Highest-value additions", ""])
    if s["highest_value_additions"]:
        for i, item in enumerate(s["highest_value_additions"], 1):
            lines.append(f"{i}. **{item['target']}** — {item['reason']}")
    else:
        lines.append("- No addition is asserted from currently trusted coverage dimensions.")

    lines.extend(["", "## Coverage representative buckets", ""])
    for bucket in payload["coverage_selection"]["buckets"]:
        selected = ", ".join(bucket["selected_representatives"]) or "—"
        lines.append(f"- **{bucket['dimension']}={bucket['value']}** · quota {bucket['quota']} · candidates {bucket['candidate_count']} · selected: {selected}")

    lines.extend(
        [
            "",
            "## Per-image evidence",
            "",
            "| image | composition | face pose | action signature | measured density | semantic burden | protected representative | dataset action | interventions |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    severity = {"replace_candidate": 0, "keep_until_cleaner_equivalent": 1, "keep": 2}
    records = sorted(payload["records"], key=lambda r: (severity.get(r["base_recommendation"], 9), float(r["measured_signal_density"]["score"])))
    for r in records:
        prot = ", ".join(f"{p['dimension']}:{p['value']}" for p in r["protected_dimensions"]) or "—"
        face = f"{r['coverage_signature']['head_yaw']}/{r['coverage_signature']['head_pitch']}"
        interventions = ", ".join(i["type"] for i in r["interventions"]) or "—"
        semantic = r["semantic_burden"]
        lines.append(
            f"| {r['relative_path']} | {r['coverage_signature']['composition']} | {face} | {r['action_contact']['action_signature']} | "
            f"{r['measured_signal_density']['label']} ({r['measured_signal_density']['score']:.3f}) | "
            f"{semantic['observation_status']} ({semantic['observed_burden_score']:.3f}) | {prot} | "
            f"**{r['presented_recommendation']['label']}** | {interventions} |"
        )

    lines.extend(["", "## Automated suggestions", ""])
    for r in records:
        if r["base_recommendation"] == "keep" and not r["interventions"]:
            continue
        lines.append(f"### {r['relative_path']}")
        lines.append("")
        lines.append(f"- Dataset action: **{r['presented_recommendation']['label']}** (base `{r['base_recommendation']}`).")
        for reason in r["recommendation_reasons"]:
            lines.append(f"- {reason}")
        if r["replacement_target"]:
            lines.append("- Replacement target:")
            for target in r["replacement_target"]:
                lines.append(f"  - {target}")
        if r["interventions"]:
            lines.append("- Optional interventions:")
            for intervention in r["interventions"]:
                lines.append(f"  - `{intervention['type']}` — {intervention['reason']}")
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
    capability["semantic_action_protection"] = (
        "coarse_and_compound_protection" if capability["policy_tier"] == "high" else "report_only_no_dataset_protection"
    )
    capability["dynamic_axis_validation"] = "enabled"

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
        "expression": Counter(),
        "action_class": Counter(),
        "action_signature": Counter(),
        "posture": Counter(),
    }

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
        action = _action_profile(analysis, pose, face, capability)
        report_image = record.get("report_image")
        image_path = run_dir / str(report_image) if report_image else None
        texture = _background_texture_proxy(image_path, pose)
        measured = _measured_signal_density(pose, texture)
        semantic_burden = _semantic_burden_profile(nuisance, confound, capability, measured)

        cyaw = _coarse_yaw(face["head_yaw"])
        cpitch = _coarse_pitch(face["head_pitch"])
        composition = signal["trusted_composition_class"]
        signature = {
            "composition": composition,
            "head_yaw": cyaw,
            "head_pitch": cpitch,
            "face_pose": f"yaw:{cyaw}|pitch:{cpitch}",
        }

        counts["shot"][signal["effective_shot_scale"]] += 1
        counts["composition"][composition] += 1
        counts["torso"][face["torso_yaw"]] += 1
        counts["head_yaw"][face["head_yaw"]] += 1
        counts["head_pitch"][face["head_pitch"]] += 1
        counts["head_roll"][face["head_roll"]] += 1
        counts["coarse_yaw"][cyaw] += 1
        counts["coarse_pitch"][cpitch] += 1
        counts["expression"][face["expression"]["primary"]] += 1
        for cls in action["coarse_classes"]:
            counts["action_class"][cls["class"]] += 1
        if action["action_signature"] != "none":
            counts["action_signature"][action["action_signature"]] += 1
        for posture in action["posture_tags"]:
            counts["posture"][posture] += 1

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
                "measured_signal_density": measured,
                "semantic_burden": semantic_burden,
                "pose_evidence": pose,
            }
        )

    total = len(records)
    quarantine = _dynamic_axis_quarantine(counts, total)
    capability["dynamic_axis_quarantine"] = sorted(quarantine)
    availability = _coverage_availability(quarantine)
    coverage_selection = _apply_coverage_quotas(records, capability, quarantine)

    for item in records:
        item["marginal_value"] = _marginal_values(item, quarantine)
        base, reasons, replacement = _dataset_action(item)
        item["base_recommendation"] = base
        item["presented_recommendation"] = _present_recommendation(base, capability)
        item["recommendation_reasons"] = reasons
        item["replacement_target"] = replacement
        item["interventions"] = _interventions(item)

    base_counts = Counter(r["base_recommendation"] for r in records)
    presented_counts = Counter(r["presented_recommendation"]["label"] for r in records)
    measured_counts = Counter(r["measured_signal_density"]["label"] for r in records)
    semantic_adjusted_counts = Counter(r["semantic_burden"]["qualified_adjusted_label"] for r in records)
    intervention_counts = Counter(i["type"] for r in records for i in r["interventions"])
    action_warnings = sum(bool(r["action_contact"].get("semantic_warnings")) for r in records)

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
        "action_class_counts": dict(sorted(counts["action_class"].items())),
        "action_signature_counts": dict(sorted(counts["action_signature"].items())),
        "posture_tag_counts": dict(sorted(counts["posture"].items())),
        "measured_signal_density_counts": dict(sorted(measured_counts.items())),
        "qualified_semantic_adjusted_density_counts": dict(sorted(semantic_adjusted_counts.items())),
        "base_recommendation_counts": dict(sorted(base_counts.items())),
        "presented_recommendation_counts": dict(sorted(presented_counts.items())),
        "intervention_counts": dict(sorted(intervention_counts.items())),
        "action_semantic_warning_images": action_warnings,
        "axis_quality_flags": _axis_quality_flags(counts, total),
        "highest_value_additions": _highest_value_additions(counts, quarantine),
    }

    payload = {
        "schema_version": "dataset-evidence-5.0",
        "analysis_model": model_id,
        "analysis_source": str(model_dir),
        "dwpose_source": str(dwpose_dir),
        "analysis_capability_policy": capability,
        "dynamic_axis_quarantine": quarantine,
        "coverage_availability": availability,
        "assessment_gaps": _assessment_gaps(quarantine),
        "method_notes": [
            "V5 dynamically quarantines degenerate VLM spatial axes before coverage buckets or recommendations are built.",
            "Keep/replace action is driven by a model-independent provisional signal-density core built from DWPose geometry and deterministic pixel texture, not VLM nuisance completeness.",
            "VLM semantic nuisance/confound evidence is reported as a qualified secondary burden estimate; unreported nuisance is treated as unknown, never as proof of a clean scene.",
            "High-tier analyzers may protect coarse/compound semantic action coverage; medium/low analyzers report action hints but cannot preserve images solely because of those semantics.",
            "Action parsing uses word-boundary matching so terms such as headrest do not become anatomical head contact, and unsupported-hand evidence confidence-caps hand semantics.",
            "Coverage representative ranking now uses the measured signal-density core, making same-bucket quality ranking substantially less dependent on analyzer size.",
            "The measured core still uses a padded DWPose rectangle rather than a true matte/face detector; subject masks and effective face pixels remain the next measurement upgrade.",
            "Per-image training loss remains separate empirical evidence and should later be joined with these profile features.",
        ],
        "dataset_summary": summary,
        "coverage_selection": coverage_selection,
        "records": records,
    }

    out_json = run_dir / f"{args.output_prefix}_{slug}.json"
    out_md = run_dir / f"{args.output_prefix}_{slug}.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_make_markdown(payload), encoding="utf-8")

    print(f"Done. JSON:   {out_json}")
    print(f"      Report: {out_md}")
    print(f"Analysis policy: {capability['policy_tier']} / {capability['judgement_breadth']}")
    print(f"Quarantined VLM axes: {sorted(quarantine)}")
    print(f"Coverage availability: { {k: v['status'] for k, v in availability.items()} }")
    print(f"Measured signal density: {dict(sorted(measured_counts.items()))}")
    print(f"Qualified semantic-adjusted density: {dict(sorted(semantic_adjusted_counts.items()))}")
    print(f"Action classes: {dict(sorted(counts['action_class'].items()))}")
    print(f"Action signatures: {dict(sorted(counts['action_signature'].items()))}")
    print(f"Dataset actions: {dict(sorted(presented_counts.items()))}")
    print(f"Interventions: {dict(sorted(intervention_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
