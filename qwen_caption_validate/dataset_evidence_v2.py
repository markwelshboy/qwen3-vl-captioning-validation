from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset_evidence import (
    COVERAGE_SIGNAL,
    POSE_SIGNAL,
    SHOT_IDENTITY_MODIFIER,
    _confound_burden,
    _nuisance_burden,
    _rarity,
    _read_json,
    _snr_label,
)
from .pose_evidence import build_pose_evidence
from .runner import model_slug, resolve_model_id


YAW_PRIORITY = [
    "strong_left",
    "strong_right",
    "three_quarter_left",
    "three_quarter_right",
    "slight_left",
    "slight_right",
    "frontal",
]
PITCH_PRIORITY = ["strong_up", "up", "slight_up", "strong_down", "down", "slight_down", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "Join cached Qwen analysis with DWPose and profile identity-view, body-pose, "
            "composition, nuisance burden, active-SNR, and dataset marginal value."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence", help="Output basename inside run_dir.")
    return parser.parse_args()


def _axis(analysis: dict[str, Any], name: str) -> dict[str, Any]:
    orientation = ((analysis.get("target_subject") or {}).get("orientation") or {})
    value = orientation.get(name) or {}
    return value if isinstance(value, dict) else {}


def _side_class(direction: str, magnitude: str, *, allow_back: bool = False) -> str:
    direction = direction or "unknown"
    magnitude = magnitude or "unknown"
    if allow_back and direction == "back_to_camera":
        return "back_to_camera"
    if direction == "frontal" or magnitude == "none":
        return "frontal"
    if direction not in {"anatomical_left", "anatomical_right"}:
        return "unknown"
    side = "left" if direction == "anatomical_left" else "right"
    if magnitude == "slight":
        return f"slight_{side}"
    if magnitude == "moderate":
        return f"three_quarter_{side}"
    if magnitude == "strong":
        # Strong yaw is a useful profile-coverage proxy, but the current v1
        # analysis schema does not establish a true silhouette profile.
        return f"strong_{side}"
    return f"{side}_unknown"


def _pitch_class(direction: str, magnitude: str) -> str:
    direction = direction or "unknown"
    magnitude = magnitude or "unknown"
    if direction == "neutral" or magnitude == "none":
        return "neutral"
    if direction not in {"up", "down"}:
        return "unknown"
    if magnitude == "slight":
        return f"slight_{direction}"
    if magnitude == "moderate":
        return direction
    if magnitude == "strong":
        return f"strong_{direction}"
    return f"{direction}_unknown"


def _roll_class(direction: str, magnitude: str) -> str:
    direction = direction or "unknown"
    magnitude = magnitude or "unknown"
    if direction == "neutral" or magnitude == "none":
        return "neutral"
    if direction not in {"anatomical_left", "anatomical_right"}:
        return "unknown"
    side = "left" if direction == "anatomical_left" else "right"
    if magnitude in {"slight", "moderate", "strong"}:
        return f"{magnitude}_{side}"
    return f"{side}_unknown"


def _expression_profile(analysis: dict[str, Any]) -> dict[str, Any]:
    states = ((analysis.get("target_subject") or {}).get("expression_state") or [])
    text = " ".join(str(v) for v in states).lower()
    modifiers: list[str] = []
    if "squint" in text or "narrow" in text:
        modifiers.append("squint")
    if "eyes closed" in text or "closed eyes" in text:
        modifiers.append("eyes_closed")
    if "brow" in text and ("raise" in text or "elevat" in text):
        modifiers.append("raised_brows")
    if "mouth open" in text or "open mouth" in text or "open-mouthed" in text:
        modifiers.append("mouth_open")

    if "smile" in text:
        if "open" in text and "mouth" in text:
            primary = "open_mouth_smile"
        elif any(word in text for word in ("broad", "wide", "toothy")):
            primary = "broad_smile"
        else:
            primary = "smile"
    elif any(word in text for word in ("neutral", "relaxed")):
        primary = "neutral_or_relaxed"
    elif any(word in text for word in ("serious", "stern", "frown")):
        primary = "non_smiling_serious"
    elif text.strip():
        primary = "other"
    else:
        primary = "unknown"
    return {"primary": primary, "modifiers": modifiers, "raw": states}


def _facial_pose(analysis: dict[str, Any]) -> dict[str, Any]:
    yaw = _axis(analysis, "head_yaw")
    pitch = _axis(analysis, "head_pitch")
    roll = _axis(analysis, "head_roll")
    torso = _axis(analysis, "torso_yaw")
    yaw_class = _side_class(str(yaw.get("direction") or "unknown"), str(yaw.get("magnitude") or "unknown"))
    pitch_class = _pitch_class(str(pitch.get("direction") or "unknown"), str(pitch.get("magnitude") or "unknown"))
    roll_class = _roll_class(str(roll.get("direction") or "unknown"), str(roll.get("magnitude") or "unknown"))
    torso_class = _side_class(
        str(torso.get("direction") or "unknown"),
        str(torso.get("magnitude") or "unknown"),
        allow_back=True,
    )
    return {
        "head_yaw": yaw_class,
        "head_pitch": pitch_class,
        "head_roll": roll_class,
        "torso_yaw": torso_class,
        "face_pose_key": f"yaw:{yaw_class}|pitch:{pitch_class}",
        "body_face_key": f"torso:{torso_class}|yaw:{yaw_class}|pitch:{pitch_class}",
        "expression": _expression_profile(analysis),
        "profile_note": (
            "strong_left/right is a strong-yaw proxy from the cached v1 VLM analysis; "
            "it is not yet a verified true facial profile class"
        ),
    }


def _effective_framing(analysis: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    framing = analysis.get("framing") or {}
    qwen_shot = str(framing.get("shot_scale") or "unknown")
    extent = str(((pose.get("target_2d_geometry") or {}).get("pose_extent_hint") or "unknown"))
    effective = qwen_shot
    reason = "Qwen framing retained"
    conflict = False

    # An ankle-bearing DWPose skeleton is strong independent evidence that the
    # visible subject extent is genuinely full-length. This corrects the known
    # failure where Qwen called a full-body smartphone image merely 'medium'.
    if extent == "full_length" and qwen_shot != "full_length":
        effective = "full_length"
        conflict = True
        reason = "DWPose ankle/full-skeleton evidence overrides narrower Qwen shot label"
    elif qwen_shot == "unknown":
        if extent == "three_quarter_or_long":
            effective = "three_quarter"
            reason = "Qwen unknown; DWPose knee extent used as fallback"
        elif extent == "waist_or_upper_body":
            effective = "medium"
            reason = "Qwen unknown; DWPose hip extent used as fallback"
        elif extent == "close_or_medium_close":
            effective = "medium_close_up"
            reason = "Qwen unknown; DWPose shoulder extent used as fallback"

    return {
        "qwen_shot_scale": qwen_shot,
        "dwpose_extent_hint": extent,
        "effective_shot_scale": effective,
        "framing_conflict": conflict,
        "resolution_reason": reason,
        "subject_frame_coverage": str(framing.get("subject_frame_coverage") or "unknown"),
    }


def _signal_profile(analysis: dict[str, Any], pose: dict[str, Any], framing: dict[str, Any]) -> dict[str, Any]:
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
        "dwpose_extent_hint": framing["dwpose_extent_hint"],
        "framing_conflict": framing["framing_conflict"],
        "identity_signal_score": round(identity_score, 3),
        "pose_signal_score": round(pose_score, 3),
        "clipped_keypoint_bbox_height_fraction": clipped_bbox.get("height_fraction"),
        "clipped_keypoint_bbox_area_fraction": clipped_bbox.get("area_fraction"),
        "complete_limb_chains": complete_chains,
    }


def _coverage_status(count: int) -> str:
    if count == 0:
        return "missing"
    if count == 1:
        return "unique"
    if count == 2:
        return "thin"
    return "represented"


def _nonneutral_face(face: dict[str, Any]) -> bool:
    return face["head_yaw"] not in {"frontal", "unknown"} or face["head_pitch"] not in {"neutral", "unknown"}


def _protected_dimensions(item: dict[str, Any]) -> list[dict[str, Any]]:
    cv = item["coverage_value"]
    face = item["facial_pose"]
    protected: list[dict[str, Any]] = []

    if cv["shot_scale_count"] <= 2:
        protected.append({"dimension": "shot_scale", "value": item["signal"]["effective_shot_scale"], "count": cv["shot_scale_count"]})
    if face["torso_yaw"] not in {"frontal", "unknown"} and cv["torso_yaw_count"] <= 2:
        protected.append({"dimension": "torso_yaw", "value": face["torso_yaw"], "count": cv["torso_yaw_count"]})
    if face["head_yaw"] not in {"frontal", "unknown"} and cv["head_yaw_count"] <= 2:
        protected.append({"dimension": "head_yaw", "value": face["head_yaw"], "count": cv["head_yaw_count"]})
    if face["head_pitch"] not in {"neutral", "unknown"} and cv["head_pitch_count"] <= 2:
        protected.append({"dimension": "head_pitch", "value": face["head_pitch"], "count": cv["head_pitch_count"]})
    if _nonneutral_face(face) and cv["face_pose_count"] <= 2:
        protected.append({"dimension": "face_pose", "value": face["face_pose_key"], "count": cv["face_pose_count"]})
    return protected


def _marginal_values(item: dict[str, Any]) -> dict[str, str]:
    cv = item["coverage_value"]
    face = item["facial_pose"]
    identity = "low"
    if _nonneutral_face(face):
        identity = "high" if cv["face_pose_count"] <= 2 else "medium"
    elif face["expression"]["primary"] in {"neutral_or_relaxed", "non_smiling_serious"} and cv["expression_count"] <= 2:
        identity = "medium"

    body = "low"
    if cv["shot_scale_count"] <= 2 or (face["torso_yaw"] not in {"frontal", "unknown"} and cv["torso_yaw_count"] <= 2):
        body = "high"
    elif float(item["signal"]["pose_signal_score"]) >= 2.5:
        body = "medium"

    order = {"low": 0, "medium": 1, "high": 2}
    overall = identity if order[identity] >= order[body] else body
    return {
        "identity_view_value": identity,
        "body_composition_value": body,
        "overall_marginal_value": overall,
    }


def _recommendation(item: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    snr = item["active_snr"]["label"]
    burden = float(item["nuisance"]["irrelevant_visual_burden"])
    entropy_regions = int(item["nuisance"]["entropy_focus_region_count"])
    protected = item["protected_dimensions"]
    values = item["marginal_value"]
    reasons: list[str] = []
    replacement: list[str] = []

    if snr == "low":
        reasons.append("low active subject-to-irrelevant-detail signal density")
        if protected:
            protected_text = ", ".join(f"{p['dimension']}={p['value']} (n={p['count']})" for p in protected)
            reasons.append(f"deleting it now would remove thin/rare coverage: {protected_text}")
            if values["identity_view_value"] == "low" and values["body_composition_value"] == "high":
                reasons.append("its marginal value is mainly body/composition evidence; facial-view diversity is not a major contribution")
                replacement.append("preserve the useful body/framing coverage while using a less-represented facial yaw/pitch if possible")
            else:
                replacement.append("preserve the rare identity/body coverage listed above")
            replacement.append("prefer a cleaner background and higher subject signal density")
            return "keep_until_cleaner_equivalent", reasons, replacement
        reasons.append("no currently protected identity-view/body-composition dimension depends on this image")
        replacement.append("replace with a cleaner image that fills one of the dataset's thin identity-view or body-coverage gaps")
        return "replace_candidate", reasons, replacement

    if burden >= 3.0 or entropy_regions:
        reasons.append("useful subject signal is present but irrelevant visual burden is substantial")
        reasons.append("a subject/entropy-focus mask is worth testing before replacement")
        if protected:
            reasons.append("the image also carries thin/rare coverage, so masking is lower-risk than deletion")
        return "consider_entropy_focus", reasons, replacement

    reasons.append("useful signal is not heavily diluted by currently identified nuisance burden")
    return "keep", reasons, replacement


def _coverage_gaps(counts: dict[str, Counter[str]], total: int) -> list[dict[str, Any]]:
    yaw = counts["head_yaw"]
    pitch = counts["head_pitch"]
    shot = counts["shot"]
    torso = counts["torso"]
    expr = counts["expression"]
    suggestions: list[dict[str, Any]] = []

    for cls, label in (
        ("strong_left", "clean strong-left-yaw / near-profile-or-profile face"),
        ("strong_right", "clean strong-right-yaw / near-profile-or-profile face"),
        ("three_quarter_left", "clean left three-quarter face"),
        ("three_quarter_right", "clean right three-quarter face"),
    ):
        n = yaw.get(cls, 0)
        if n <= 1:
            suggestions.append({"priority": 1 if n == 0 else 2, "target": label, "reason": f"head-yaw coverage is {_coverage_status(n)} (n={n})"})

    up_total = sum(pitch.get(k, 0) for k in ("slight_up", "up", "strong_up"))
    down_total = sum(pitch.get(k, 0) for k in ("slight_down", "down", "strong_down"))
    if up_total <= 1:
        suggestions.append({"priority": 1 if up_total == 0 else 2, "target": "face with clearly raised/upward head pitch", "reason": f"upward-pitch coverage is {_coverage_status(up_total)} (n={up_total})"})
    if down_total <= 1:
        suggestions.append({"priority": 2, "target": "face with clearly downward head pitch", "reason": f"downward-pitch coverage is {_coverage_status(down_total)} (n={down_total})"})

    full = shot.get("full_length", 0)
    if full <= 2:
        suggestions.append({"priority": 1, "target": "clean full-length image with a less-represented facial yaw/pitch", "reason": f"effective full-length coverage is thin (n={full}); avoid adding another redundant frontal/identical head pose"})

    nonfrontal_torso = sum(v for k, v in torso.items() if k not in {"frontal", "unknown"})
    if nonfrontal_torso <= 2:
        suggestions.append({"priority": 2, "target": "clean non-frontal torso view with the face still usefully visible", "reason": f"non-frontal torso coverage is thin (n={nonfrontal_torso})"})

    smile_total = sum(v for k, v in expr.items() if "smile" in k)
    neutral_total = expr.get("neutral_or_relaxed", 0) + expr.get("non_smiling_serious", 0)
    if total and smile_total / total >= 0.70 and neutral_total <= 2:
        suggestions.append({"priority": 3, "target": "neutral or non-smiling portrait", "reason": f"expression coverage is smile-heavy ({smile_total}/{total}) with only {neutral_total} neutral/non-smiling examples"})

    suggestions.sort(key=lambda s: (int(s["priority"]), str(s["target"])))
    return suggestions[:8]


def _make_markdown(payload: dict[str, Any]) -> str:
    summary = payload["dataset_summary"]
    lines = [
        "# Dataset evidence report — identity-view aware",
        "",
        "> `active_snr` remains a transparent training-signal-density heuristic, not photometric SNR. This v2 report separates identity-view coverage (head yaw/pitch/roll/expression), body/composition coverage, nuisance burden, confounds, and marginal dataset value. Per-image training loss is still intentionally separate.",
        "",
        "## Dataset summary",
        "",
        f"- Images profiled: **{summary['image_count']}**",
        f"- Effective shot scale counts: `{json.dumps(summary['effective_shot_scale_counts'], sort_keys=True)}`",
        f"- Qwen↔DWPose framing conflicts corrected: **{summary['framing_conflict_count']}**",
        f"- Active SNR counts: `{json.dumps(summary['active_snr_counts'], sort_keys=True)}`",
        f"- Recommendations: `{json.dumps(summary['recommendation_counts'], sort_keys=True)}`",
        "",
        "## Identity-view coverage",
        "",
        f"- Head yaw: `{json.dumps(summary['head_yaw_counts'], sort_keys=True)}`",
        f"- Head pitch: `{json.dumps(summary['head_pitch_counts'], sort_keys=True)}`",
        f"- Head roll: `{json.dumps(summary['head_roll_counts'], sort_keys=True)}`",
        f"- Expression: `{json.dumps(summary['expression_counts'], sort_keys=True)}`",
        "",
        "> `strong_left` / `strong_right` are strong-yaw proxies from the cached v1 analysis. They should not yet be read as verified true profile photographs; a future analyzer schema can make profile evidence explicit.",
        "",
        "## Body / composition coverage",
        "",
        f"- Torso yaw: `{json.dumps(summary['torso_yaw_counts'], sort_keys=True)}`",
        f"- Effective shot scale: `{json.dumps(summary['effective_shot_scale_counts'], sort_keys=True)}`",
        "",
        "## Highest-value additions",
        "",
    ]
    for i, item in enumerate(summary["highest_value_additions"], 1):
        lines.append(f"{i}. **{item['target']}** — {item['reason']}")

    lines.extend([
        "",
        "## Per-image evidence",
        "",
        "| image | shot | head yaw | head pitch | identity-view value | body value | active SNR | nuisance | protected coverage | recommendation |",
        "|---|---|---|---|---|---|---|---:|---|---|",
    ])
    rank = {"replace_candidate": 0, "keep_until_cleaner_equivalent": 1, "consider_entropy_focus": 2, "keep": 3}
    records = sorted(payload["records"], key=lambda r: (rank.get(r["recommendation"], 9), -float(r["nuisance"]["irrelevant_visual_burden"])))
    for r in records:
        protected = ", ".join(p["dimension"] for p in r["protected_dimensions"]) or "—"
        lines.append(
            "| {image} | {shot} | {yaw} | {pitch} | {iv} | {bv} | {snr} | {burden:.2f} | {protected} | **{rec}** |".format(
                image=r["relative_path"],
                shot=r["signal"]["effective_shot_scale"],
                yaw=r["facial_pose"]["head_yaw"],
                pitch=r["facial_pose"]["head_pitch"],
                iv=r["marginal_value"]["identity_view_value"],
                bv=r["marginal_value"]["body_composition_value"],
                snr=r["active_snr"]["label"],
                burden=float(r["nuisance"]["irrelevant_visual_burden"]),
                protected=protected,
                rec=r["recommendation"],
            )
        )

    lines.extend(["", "## Automated suggestions", ""])
    for r in records:
        if r["recommendation"] == "keep":
            continue
        lines.append(f"### {r['relative_path']} — {r['recommendation']}")
        lines.append("")
        for reason in r["recommendation_reasons"]:
            lines.append(f"- {reason}")
        if r["replacement_target"]:
            lines.append("- Replacement target:")
            for target in r["replacement_target"]:
                lines.append(f"  - {target}")
        top = (r["nuisance"].get("regions") or [])[:3]
        if top:
            lines.append("- Largest currently identified irrelevant-burden regions:")
            for region in top:
                desc = region.get("description") or "unnamed region"
                lines.append(f"  - {desc} ({region.get('frame_coverage')}, {region.get('visual_complexity')} complexity; burden {region.get('irrelevant_burden_points')})")
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

    provisional: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = {
        "shot": Counter(),
        "torso": Counter(),
        "head_yaw": Counter(),
        "head_pitch": Counter(),
        "head_roll": Counter(),
        "face_pose": Counter(),
        "shot_face": Counter(),
        "body_face": Counter(),
        "expression": Counter(),
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
        signal = _signal_profile(analysis, pose, framing)
        nuisance = _nuisance_burden(analysis)
        confound = _confound_burden(analysis, pose)
        shot = signal["effective_shot_scale"]
        shot_face_key = f"shot:{shot}|{face['face_pose_key']}"

        counters["shot"][shot] += 1
        counters["torso"][face["torso_yaw"]] += 1
        counters["head_yaw"][face["head_yaw"]] += 1
        counters["head_pitch"][face["head_pitch"]] += 1
        counters["head_roll"][face["head_roll"]] += 1
        counters["face_pose"][face["face_pose_key"]] += 1
        counters["shot_face"][shot_face_key] += 1
        counters["body_face"][face["body_face_key"]] += 1
        counters["expression"][face["expression"]["primary"]] += 1
        framing_conflicts += int(bool(framing["framing_conflict"]))

        snr_label, snr_score = _snr_label(
            float(signal["identity_signal_score"]),
            float(nuisance["irrelevant_visual_burden"]),
            float(confound["score"]),
            signal["subject_frame_coverage"],
        )
        provisional.append({
            "relative_path": record.get("relative_path"),
            "result_key": key,
            "image_summary": analysis.get("image_summary"),
            "signal": signal,
            "framing_fusion": framing,
            "facial_pose": face,
            "nuisance": nuisance,
            "confound": confound,
            "pose_evidence": pose,
            "shot_face_key": shot_face_key,
            "active_snr": {
                "label": snr_label,
                "heuristic_score": round(snr_score, 3),
                "definition": "estimated useful identity signal relative to currently identified irrelevant visual/confound burden; not photometric SNR",
            },
        })

    for item in provisional:
        face = item["facial_pose"]
        shot = item["signal"]["effective_shot_scale"]
        expression = face["expression"]["primary"]
        cv = {
            "shot_scale_count": counters["shot"][shot],
            "shot_scale_rarity": _rarity(counters["shot"][shot]),
            "torso_yaw_count": counters["torso"][face["torso_yaw"]],
            "torso_yaw_rarity": _rarity(counters["torso"][face["torso_yaw"]]),
            "head_yaw_count": counters["head_yaw"][face["head_yaw"]],
            "head_yaw_rarity": _rarity(counters["head_yaw"][face["head_yaw"]]),
            "head_pitch_count": counters["head_pitch"][face["head_pitch"]],
            "head_pitch_rarity": _rarity(counters["head_pitch"][face["head_pitch"]]),
            "face_pose_count": counters["face_pose"][face["face_pose_key"]],
            "face_pose_rarity": _rarity(counters["face_pose"][face["face_pose_key"]]),
            "shot_face_count": counters["shot_face"][item["shot_face_key"]],
            "shot_face_rarity": _rarity(counters["shot_face"][item["shot_face_key"]]),
            "body_face_count": counters["body_face"][face["body_face_key"]],
            "body_face_rarity": _rarity(counters["body_face"][face["body_face_key"]]),
            "expression_count": counters["expression"][expression],
            "expression_rarity": _rarity(counters["expression"][expression]),
        }
        item["coverage_value"] = cv
        item["protected_dimensions"] = _protected_dimensions(item)
        item["marginal_value"] = _marginal_values(item)
        recommendation, reasons, replacement = _recommendation(item)
        item["recommendation"] = recommendation
        item["recommendation_reasons"] = reasons
        item["replacement_target"] = replacement

    recommendation_counts = Counter(item["recommendation"] for item in provisional)
    snr_counts = Counter(item["active_snr"]["label"] for item in provisional)
    highest_value = _coverage_gaps(counters, len(provisional))

    payload = {
        "schema_version": "dataset-evidence-2.0",
        "analysis_model": model_id,
        "analysis_source": str(model_dir),
        "dwpose_source": str(dwpose_dir),
        "method_notes": [
            "Identity-view coverage is now independent of shot scale/body coverage.",
            "Head yaw, pitch, roll, expression, torso yaw, effective framing, and cross-combinations are counted separately.",
            "DWPose full-length ankle/skeleton evidence can override an erroneously narrow Qwen shot-scale label.",
            "Strong head yaw is only a profile proxy in cached analysis_v1; true profile classification should become an explicit future Analyze field.",
            "Rare identity-view or body/composition evidence vetoes immediate replacement of a low-SNR image; the recommendation becomes keep-until-cleaner-equivalent.",
            "Per-image training loss remains separate empirical evidence and is not hidden inside active_snr.",
        ],
        "dataset_summary": {
            "image_count": len(provisional),
            "effective_shot_scale_counts": dict(sorted(counters["shot"].items())),
            "torso_yaw_counts": dict(sorted(counters["torso"].items())),
            "head_yaw_counts": dict(sorted(counters["head_yaw"].items())),
            "head_pitch_counts": dict(sorted(counters["head_pitch"].items())),
            "head_roll_counts": dict(sorted(counters["head_roll"].items())),
            "face_pose_counts": dict(sorted(counters["face_pose"].items())),
            "expression_counts": dict(sorted(counters["expression"].items())),
            "framing_conflict_count": framing_conflicts,
            "active_snr_counts": dict(sorted(snr_counts.items())),
            "recommendation_counts": dict(sorted(recommendation_counts.items())),
            "highest_value_additions": highest_value,
        },
        "records": provisional,
    }

    out_json = run_dir / f"{args.output_prefix}_{slug}.json"
    out_md = run_dir / f"{args.output_prefix}_{slug}.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_make_markdown(payload), encoding="utf-8")

    print(f"Done. JSON:   {out_json}")
    print(f"      Report: {out_md}")
    print(f"Head yaw: {payload['dataset_summary']['head_yaw_counts']}")
    print(f"Head pitch: {payload['dataset_summary']['head_pitch_counts']}")
    print(f"Effective shot scale: {payload['dataset_summary']['effective_shot_scale_counts']}")
    print(f"Framing conflicts corrected: {framing_conflicts}")
    print(f"Recommendations: {payload['dataset_summary']['recommendation_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
