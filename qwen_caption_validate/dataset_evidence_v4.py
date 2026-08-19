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
    _active_snr,
    _axis_quality_flags,
    _background_texture_proxy,
    _capability_profile,
    _coarse_pitch,
    _coarse_yaw,
    _highest_value_additions,
    _nuisance_profile,
    _present_recommendation,
    _signal_profile,
)
from .pose_evidence import build_pose_evidence
from .runner import model_slug, resolve_model_id


SEMANTIC_DEFAULT_CONFIDENCE = {"low": 0.45, "medium": 0.62, "high": 0.78}
SPATIAL_DEFAULT_CONFIDENCE = {"low": 0.35, "low_to_medium": 0.50, "medium": 0.65}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "Source-aware dataset profiler with quota-ranked coverage representatives, "
            "coarse action/contact signatures, and independent intervention suggestions."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence", help="Output basename inside run_dir.")
    return parser.parse_args()


def _safe_confidence(value: Any, fallback: float) -> float:
    try:
        v = float(value)
        if 0.0 <= v <= 1.0:
            return v
    except (TypeError, ValueError):
        pass
    return fallback


def _semantic_default(capability: dict[str, Any]) -> float:
    authority = str(capability.get("vision_semantics_authority") or "medium")
    return SEMANTIC_DEFAULT_CONFIDENCE.get(authority, 0.62)


def _spatial_default(capability: dict[str, Any]) -> float:
    authority = str(capability.get("vision_spatial_authority") or "low_to_medium")
    return SPATIAL_DEFAULT_CONFIDENCE.get(authority, 0.50)


def _action_profile(
    analysis: dict[str, Any],
    pose: dict[str, Any],
    face: dict[str, Any],
    capability: dict[str, Any],
) -> dict[str, Any]:
    """Extract coarse action/contact evidence without pretending to know more than v1 says.

    Specific semantic labels remain in raw_tags, but coverage uses deliberately broader
    classes such as hands_near_hips and head_support. This avoids turning an uncertain
    rear-pocket/on-hip distinction into false dataset uniqueness.
    """
    raw_tags: set[str] = set()
    evidence: list[dict[str, Any]] = []
    class_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    semantic_default = _semantic_default(capability)
    limbs = ((analysis.get("target_subject") or {}).get("visible_limbs") or [])

    def add_raw(tag: str, source: str, text: str, confidence: float, coarse_class: str | None = None) -> None:
        raw_tags.add(tag)
        item = {
            "tag": tag,
            "source": source,
            "text": text[:320],
            "confidence": round(confidence, 3),
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
        conf = _safe_confidence(limb.get("confidence"), semantic_default)

        if "pocket" in text:
            add_raw("hands_in_pockets", "qwen_visible_limbs", text, conf, "hands_near_hips")
        if "hip" in contact_text and "pocket" not in text:
            # Keep exact semantic wording as evidence, but deliberately lower the
            # coverage confidence because hand-on-hip vs rear-pocket placement can
            # be visually ambiguous even to a human.
            add_raw("hand_at_hip", "qwen_visible_limbs", text, min(conf, 0.62), "hands_near_hips")
        if any(w in text for w in ("hold", "holding", "grip", "grasp")):
            add_raw("holding_object", "qwen_visible_limbs", text, conf, "object_interaction")
        if "reach" in text:
            add_raw("reaching", "qwen_visible_limbs", text, conf, "object_interaction")
        if "cross" in text and any(w in text for w in ("arm", "forearm", "wrist", "hand")):
            add_raw("arms_crossed", "qwen_visible_limbs", text, conf, "arms_crossed")
        if "table" in contact_text:
            add_raw("arm_or_hand_on_table", "qwen_visible_limbs", text, conf, "surface_contact")
        if any(w in contact_text for w in ("chin", "cheek", "face", "head")):
            add_raw("hand_face_contact", "qwen_visible_limbs", text, conf, "head_support")
            if any(w in contact_text for w in ("support", "rest", "knuckle", "fist")) or limb.get("support"):
                add_raw("chin_or_head_support", "qwen_visible_limbs", text, conf, "head_support")

        # "extends downward" is ordinary limb geometry, not a distinct action.
        # Require forward/reaching/off-frame language before assigning a coarse
        # forward-extension class.
        if (
            ("forward" in text and any(w in text for w in ("extend", "reach")))
            or ("out of frame" in text and any(w in text for w in ("extend", "reach")))
        ):
            add_raw("forward_arm_extension", "qwen_visible_limbs", text, conf, "forward_arm_extension")

    summary = str(analysis.get("image_summary") or "").lower()
    posture: list[str] = []
    if re.search(r"\b(seated|sitting|sits)\b", summary):
        posture.append("seated")
    if re.search(r"\b(standing|stands)\b", summary):
        posture.append("standing")

    torso = str(face.get("torso_yaw") or "unknown")
    head = str(face.get("head_yaw") or "unknown")
    if torso not in {"frontal", "unknown"} and head not in {"unknown", torso}:
        conf = _spatial_default(capability)
        add_raw(
            "head_torso_counter_rotation",
            "qwen_orientation_fusion",
            f"torso={torso}; head={head}",
            conf,
            "head_torso_counter_rotation",
        )

    shoulder = ((pose.get("target_2d_geometry") or {}).get("shoulder_line_angle_from_horizontal_deg"))
    if shoulder is not None:
        try:
            angle = abs(float(shoulder))
            if angle >= 15.0:
                add_raw(
                    "strong_shoulder_cant",
                    "dwpose_geometry",
                    f"abs shoulder-line angle={angle:.2f} deg",
                    0.92,
                    "strong_shoulder_cant",
                )
            elif angle >= 8.0:
                add_raw(
                    "moderate_shoulder_cant",
                    "dwpose_geometry",
                    f"abs shoulder-line angle={angle:.2f} deg",
                    0.86,
                    None,
                )
        except (TypeError, ValueError):
            pass

    coarse_classes: list[dict[str, Any]] = []
    for name, items in sorted(class_evidence.items()):
        confidence = max(float(i["confidence"]) for i in items)
        sources = sorted({str(i["source"]) for i in items})
        coarse_classes.append(
            {
                "class": name,
                "confidence": round(confidence, 3),
                "sources": sources,
                "evidence_tags": sorted({str(i["tag"]) for i in items}),
            }
        )

    # Remove generic forward-extension if a more meaningful object interaction is
    # already present. Likewise head_support already subsumes hand-face contact.
    signature_components = [c["class"] for c in coarse_classes]
    if "object_interaction" in signature_components and "forward_arm_extension" in signature_components:
        signature_components.remove("forward_arm_extension")
    signature_components = sorted(set(signature_components))

    # Dedupe repeated raw evidence created by multiple limb records.
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in evidence:
        dedup[(str(item["tag"]), str(item["source"]), str(item["text"]))] = item

    return {
        "raw_tags": sorted(raw_tags),
        "coarse_classes": coarse_classes,
        "action_signature": "+".join(signature_components) if signature_components else "none",
        "signature_components": signature_components,
        "posture_tags": posture,
        "evidence": list(dedup.values()),
    }


def _action_class_confidence(action: dict[str, Any], name: str) -> float:
    for item in action.get("coarse_classes") or []:
        if item.get("class") == name:
            return float(item.get("confidence") or 0.0)
    return 0.0


def _coverage_memberships(item: dict[str, Any], capability: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate coverage buckets and their desired representative quotas."""
    out: list[dict[str, Any]] = []
    tier = capability["policy_tier"]
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

    if sig["composition"] == "full_body":
        add("composition", "full_body", 2, "dwpose+qwen_framing_fusion", "high_for_visible_extent", 0.90)

    if tier == "high":
        qwen_pose_quota = 2
        combo_quota = 1
    elif tier == "medium":
        qwen_pose_quota = 1
        combo_quota = 1
    else:
        qwen_pose_quota = 0
        combo_quota = 0

    if sig["head_yaw"] not in {"frontalish", "unknown"}:
        add("head_yaw", sig["head_yaw"], qwen_pose_quota, "qwen_analysis", capability["vision_spatial_authority"], _spatial_default(capability))
    if sig["head_pitch"] not in {"neutralish", "unknown"}:
        add("head_pitch", sig["head_pitch"], qwen_pose_quota, "qwen_analysis", capability["vision_spatial_authority"], _spatial_default(capability))
    if sig["face_pose"] != "yaw:frontalish|pitch:neutralish":
        add("face_pose", sig["face_pose"], combo_quota, "qwen_analysis", capability["vision_spatial_authority"], _spatial_default(capability))

    torso = str(face.get("torso_yaw") or "unknown")
    if tier == "high" and torso not in {"frontal", "unknown"}:
        add("torso_yaw", torso, 1, "qwen_analysis", capability["vision_spatial_authority"], _spatial_default(capability))

    components = list(item["action_contact"].get("signature_components") or [])
    semantic_threshold = 0.70 if tier == "high" else 0.78 if tier == "medium" else 1.1

    # Deterministic strong shoulder cant can protect one representative regardless
    # of VLM size because the evidence comes from DWPose geometry.
    if "strong_shoulder_cant" in components:
        add("geometry_class", "strong_shoulder_cant", 1, "dwpose_geometry", "high_secondary_evidence", 0.92)

    semantic_components = [c for c in components if c != "strong_shoulder_cant"]
    eligible = [c for c in semantic_components if _action_class_confidence(item["action_contact"], c) >= semantic_threshold]
    if eligible:
        # Compound signatures preserve combinations such as head-support + object
        # interaction without preserving every lower-quality example carrying one
        # of the component tags.
        value = "+".join(sorted(eligible))
        dim = "compound_action_signature" if len(eligible) >= 2 else "action_class"
        confidence = min(_action_class_confidence(item["action_contact"], c) for c in eligible)
        add(dim, value, 1, "qwen_visible_limbs", capability["vision_semantics_authority"], confidence)

    return out


def _representative_score(item: dict[str, Any]) -> float:
    """Rank competing representatives inside the same coverage bucket."""
    snr = float(item["active_snr"]["heuristic_score"])
    identity = float(item["signal"]["identity_signal_score"])
    pose = float(item["signal"]["pose_signal_score"])
    texture = item["background_texture_proxy"].get("texture_score")
    texture_penalty = 0.0 if texture is None else 0.20 * float(texture)
    return snr + 0.04 * identity + 0.05 * pose - texture_penalty


def _apply_coverage_quotas(records: list[dict[str, Any]], capability: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, item in enumerate(records):
        memberships = _coverage_memberships(item, capability)
        item["coverage_memberships"] = memberships
        item["representative_score"] = round(_representative_score(item), 4)
        for membership in memberships:
            buckets[(membership["dimension"], membership["value"])].append((idx, membership))

    bucket_summary: list[dict[str, Any]] = []
    for (dimension, value), members in sorted(buckets.items()):
        quota = max(int(m["quota"]) for _, m in members)
        ranked = sorted(
            members,
            key=lambda pair: (
                float(records[pair[0]]["representative_score"]),
                float(pair[1].get("confidence") or 0.0),
            ),
            reverse=True,
        )
        selected = ranked[:quota]
        selected_indices = {idx for idx, _ in selected}
        for rank, (idx, membership) in enumerate(ranked, 1):
            membership["bucket_count"] = len(ranked)
            membership["representative_rank"] = rank
            membership["selected_as_representative"] = idx in selected_indices
            membership["remaining_bucket_members_if_removed"] = max(0, len(ranked) - 1)

        bucket_summary.append(
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

    return {"buckets": bucket_summary}


def _marginal_values(item: dict[str, Any]) -> dict[str, str]:
    protected = item["protected_dimensions"]
    identity_dims = {"head_yaw", "head_pitch", "face_pose"}
    body_dims = {"composition", "torso_yaw", "geometry_class", "action_class", "compound_action_signature"}
    identity = "high" if any(p["dimension"] in identity_dims for p in protected) else (
        "medium" if item["coverage_signature"]["head_yaw"] != "frontalish" or item["coverage_signature"]["head_pitch"] != "neutralish" else "low"
    )
    body = "high" if any(p["dimension"] in body_dims for p in protected) else (
        "medium" if float(item["signal"]["pose_signal_score"]) >= 2.5 or item["action_contact"]["signature_components"] else "low"
    )
    order = {"low": 0, "medium": 1, "high": 2}
    overall = identity if order[identity] >= order[body] else body
    return {"identity_view_value": identity, "body_action_composition_value": body, "overall_marginal_value": overall}


def _dataset_action(item: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    snr = item["active_snr"]["label"]
    protected = item["protected_dimensions"]
    reasons: list[str] = []
    replacement: list[str] = []

    if snr == "low":
        reasons.append("low active subject-to-irrelevant-detail signal density")
        if protected:
            summary = ", ".join(
                f"{p['dimension']}={p['value']} (rank {p['representative_rank']}/{p['bucket_count']}, quota {p['quota']})"
                for p in protected
            )
            reasons.append(f"selected as a current best representative for protected coverage: {summary}")
            replacement.extend(
                [
                    "preserve or improve the protected coverage listed above",
                    "prefer higher subject signal density and lower irrelevant texture/confound burden",
                ]
            )
            return "keep_until_cleaner_equivalent", reasons, replacement
        reasons.append("not selected among the current best representatives for any trusted coverage bucket")
        replacement.append("replace with a cleaner image that fills a trusted dataset coverage gap or outranks a current representative")
        return "replace_candidate", reasons, replacement

    reasons.append("useful signal is not currently low enough to justify replacement on SNR grounds")
    if protected:
        reasons.append("the image is also selected as a current representative for one or more coverage buckets")
    return "keep", reasons, replacement


def _interventions(item: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    entropy = float(item["active_snr"]["effective_entropy_component"])
    texture_label = str(item["background_texture_proxy"].get("texture_label") or "unknown")
    composition = str(item["coverage_signature"].get("composition") or "unknown")
    bbox_area = item["signal"].get("clipped_keypoint_bbox_area_fraction")

    if entropy >= 2.5 and texture_label in {"medium", "high"}:
        out.append(
            {
                "type": "entropy_focus_test",
                "reason": "background texture/complexity is substantial enough to test masking independently of the keep/replace decision",
            }
        )

    if item["active_snr"]["label"] == "low":
        out.append(
            {
                "type": "subject_mask_measurement",
                "reason": "active SNR is low and the current pixel proxy still excludes only a DWPose rectangle rather than a true subject matte",
            }
        )

    if composition != "full_body" and bbox_area is not None and float(bbox_area) < 0.36 and item["active_snr"]["label"] == "low":
        out.append(
            {
                "type": "crop_review",
                "reason": "the target skeleton occupies a relatively small in-frame rectangle; a tighter crop may improve effective subject resolution if useful pose evidence is preserved",
            }
        )

    if composition == "full_body" and item["active_snr"]["label"] == "low":
        out.append(
            {
                "type": "seek_cleaner_full_body_equivalent",
                "reason": "cropping would destroy full-body coverage, so replacement is preferable to aggressive crop if a cleaner equivalent exists",
            }
        )

    action = item["action_contact"]
    if any(c["class"] == "hands_near_hips" for c in action.get("coarse_classes") or []):
        out.append(
            {
                "type": "action_semantics_review",
                "reason": "hand placement around the hips/pockets is represented coarsely because exact contact may be visually ambiguous",
            }
        )

    if "strong_shoulder_cant" in action.get("signature_components", []):
        out.append(
            {
                "type": "geometry_caption_audit",
                "reason": "DWPose measures a strong shoulder-line cant that should not be flattened by later caption composition",
            }
        )

    return out


def _make_markdown(payload: dict[str, Any]) -> str:
    s = payload["dataset_summary"]
    cap = payload["analysis_capability_policy"]
    lines = [
        "# Dataset evidence report — source-aware v4",
        "",
        f"> Analysis source: **{cap['model_id']}** · policy tier **{cap['policy_tier']}** · judgement breadth **{cap['judgement_breadth']}**.",
        "> Dataset action and optional interventions are intentionally separate. Coverage protection uses quotas and best-representative ranking, not raw rarity alone.",
        "",
        "## Dataset summary",
        "",
        f"- Images profiled: **{s['image_count']}**",
        f"- Trusted composition classes: `{json.dumps(s['trusted_composition_counts'], sort_keys=True)}`",
        f"- Active SNR: `{json.dumps(s['active_snr_counts'], sort_keys=True)}`",
        f"- Dataset actions: `{json.dumps(s['presented_recommendation_counts'], sort_keys=True)}`",
        f"- Interventions: `{json.dumps(s['intervention_counts'], sort_keys=True)}`",
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
        f"- Coarse action classes: `{json.dumps(s['action_class_counts'], sort_keys=True)}`",
        f"- Action signatures: `{json.dumps(s['action_signature_counts'], sort_keys=True)}`",
        f"- Posture tags: `{json.dumps(s['posture_tag_counts'], sort_keys=True)}`",
        "",
        "## Analyzer-axis caution flags",
        "",
    ]
    if s["axis_quality_flags"]:
        for flag in s["axis_quality_flags"]:
            lines.append(
                f"- **{flag['axis']}**: {flag['dominant_value']} = {flag['dominant_count']}/{s['image_count']} "
                f"({flag['dominant_share']:.1%}). {flag['interpretation']}"
            )
    else:
        lines.append("- None triggered by the current degeneracy heuristic.")

    lines.extend(["", "## Highest-value additions", ""])
    for i, item in enumerate(s["highest_value_additions"], 1):
        lines.append(f"{i}. **{item['target']}** — {item['reason']}")

    lines.extend(["", "## Coverage representative buckets", ""])
    for bucket in payload["coverage_selection"]["buckets"]:
        selected = ", ".join(bucket["selected_representatives"]) or "—"
        lines.append(
            f"- **{bucket['dimension']}={bucket['value']}** · quota {bucket['quota']} · candidates {bucket['candidate_count']} · selected: {selected}"
        )

    lines.extend(
        [
            "",
            "## Per-image evidence",
            "",
            "| image | composition | face pose | action signature | SNR | protected representative | dataset action | interventions |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    severity = {"replace_candidate": 0, "keep_until_cleaner_equivalent": 1, "keep": 2}
    records = sorted(payload["records"], key=lambda r: (severity.get(r["base_recommendation"], 9), float(r["active_snr"]["heuristic_score"])))
    for r in records:
        prot = ", ".join(f"{p['dimension']}:{p['value']}" for p in r["protected_dimensions"]) or "—"
        face = f"{r['coverage_signature']['head_yaw']}/{r['coverage_signature']['head_pitch']}"
        interventions = ", ".join(i["type"] for i in r["interventions"]) or "—"
        lines.append(
            f"| {r['relative_path']} | {r['coverage_signature']['composition']} | {face} | {r['action_contact']['action_signature']} | "
            f"{r['active_snr']['label']} ({r['active_snr']['heuristic_score']:.3f}) | {prot} | "
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
        active_snr = _active_snr(
            float(signal["identity_signal_score"]),
            nuisance,
            float(confound["score"]),
            texture,
            signal["subject_frame_coverage"],
        )

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
                "active_snr": active_snr,
                "pose_evidence": pose,
            }
        )

    coverage_selection = _apply_coverage_quotas(records, capability)

    for item in records:
        item["marginal_value"] = _marginal_values(item)
        base, reasons, replacement = _dataset_action(item)
        item["base_recommendation"] = base
        item["presented_recommendation"] = _present_recommendation(base, capability)
        item["recommendation_reasons"] = reasons
        item["replacement_target"] = replacement
        item["interventions"] = _interventions(item)

    total = len(records)
    base_counts = Counter(r["base_recommendation"] for r in records)
    presented_counts = Counter(r["presented_recommendation"]["label"] for r in records)
    snr_counts = Counter(r["active_snr"]["label"] for r in records)
    intervention_counts = Counter(i["type"] for r in records for i in r["interventions"])
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
        "active_snr_counts": dict(sorted(snr_counts.items())),
        "base_recommendation_counts": dict(sorted(base_counts.items())),
        "presented_recommendation_counts": dict(sorted(presented_counts.items())),
        "intervention_counts": dict(sorted(intervention_counts.items())),
        "axis_quality_flags": _axis_quality_flags(counts, total),
        "highest_value_additions": _highest_value_additions(counts, capability),
    }

    payload = {
        "schema_version": "dataset-evidence-4.0",
        "analysis_model": model_id,
        "analysis_source": str(model_dir),
        "dwpose_source": str(dwpose_dir),
        "analysis_capability_policy": capability,
        "method_notes": [
            "Coverage protection now uses explicit quotas and ranks competing representatives by useful-signal quality instead of protecting every rare-labelled image.",
            "Action/contact coverage is coarse-first: ambiguous hand-at-hip versus pocket placement maps to hands_near_hips rather than pretending exact contact is certain.",
            "Compound action signatures preserve meaningful combinations such as head_support + object_interaction without preserving every duplicate component pose.",
            "Dataset action (keep/replace) is independent from interventions such as Entropy Focus, crop review, or geometry-caption audit.",
            "DWPose geometry and deterministic pixel evidence retain authority independent of VLM size; VLM-derived coverage is qualified by model tier.",
            "The deterministic texture proxy still excludes a padded DWPose rectangle rather than a real subject mask; future matte-based measurements should supersede it.",
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
    print(f"Coarse head yaw: {dict(sorted(counts['coarse_yaw'].items()))}")
    print(f"Coarse head pitch: {dict(sorted(counts['coarse_pitch'].items()))}")
    print(f"Action classes: {dict(sorted(counts['action_class'].items()))}")
    print(f"Action signatures: {dict(sorted(counts['action_signature'].items()))}")
    print(f"Active SNR: {dict(sorted(snr_counts.items()))}")
    print(f"Dataset actions: {dict(sorted(presented_counts.items()))}")
    print(f"Interventions: {dict(sorted(intervention_counts.items()))}")
    if summary["axis_quality_flags"]:
        print("Analyzer-axis caution flags:", [f["axis"] for f in summary["axis_quality_flags"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
