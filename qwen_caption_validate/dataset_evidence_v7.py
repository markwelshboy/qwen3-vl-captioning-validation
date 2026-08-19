from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .runner import model_slug, resolve_model_id


DEFAULT_GUIDANCE = Path(__file__).resolve().parents[1] / "guidance_profiles" / "identity_lora_balanced_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "V7 guidance-aware acquisition/rebalancing planner. Builds on v6 evidence, "
            "adds conservative scene/lighting diversity observations, then proposes "
            "multi-debt acquisition targets and low-cost donor swaps."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence_v7", help="Output basename inside run_dir.")
    parser.add_argument(
        "--guidance-profile",
        type=Path,
        default=DEFAULT_GUIDANCE,
        help=f"Guidance profile JSON (default: {DEFAULT_GUIDANCE}).",
    )
    parser.add_argument(
        "--base-v6-json",
        type=Path,
        help="Optional existing v6 JSON to post-process instead of regenerating v6 from cached analysis.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_guidance(path: Path) -> dict[str, Any]:
    payload = _read_json(path.expanduser().resolve())
    if payload.get("schema_version") != "dataset-guidance-profile-1.0":
        raise ValueError(f"Unsupported guidance schema: {payload.get('schema_version')!r}")
    return payload


def _run_v6_base(args: argparse.Namespace, run_dir: Path, slug: str) -> tuple[dict[str, Any], list[Path]]:
    if args.base_v6_json:
        path = args.base_v6_json.expanduser().resolve()
        return _read_json(path), []

    internal_prefix = f".__v7_base_{args.output_prefix}"
    cmd = [
        sys.executable,
        "-m",
        "qwen_caption_validate.dataset_evidence_v6",
        str(run_dir),
        "--model",
        args.model,
        "--output-prefix",
        internal_prefix,
        "--guidance-profile",
        str(args.guidance_profile.expanduser().resolve()),
    ]
    if args.dwpose_dir:
        cmd.extend(["--dwpose-dir", str(args.dwpose_dir.expanduser().resolve())])

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"v6 base profiler failed with exit code {proc.returncode}")

    base_json = run_dir / f"{internal_prefix}_{slug}.json"
    base_md = run_dir / f"{internal_prefix}_{slug}.md"
    if not base_json.exists():
        raise RuntimeError(f"v6 base profiler did not create expected file: {base_json}")
    return _read_json(base_json), [base_json, base_md]


def _norm_text(record: dict[str, Any]) -> str:
    chunks: list[str] = [str(record.get("image_summary") or "")]
    nuisance = record.get("nuisance") or {}
    for region in nuisance.get("regions") or []:
        chunks.append(str(region.get("description") or ""))
    camera = record.get("camera") or {}
    chunks.append(str(camera.get("perspective_notes") or ""))
    return " ".join(chunks).lower()


def _matched_cues(text: str, cues: dict[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    matches: list[str] = []
    for cue, weight in cues.items():
        pattern = r"(?<!\w)" + re.escape(cue).replace(r"\ ", r"\s+") + r"(?!\w)"
        if re.search(pattern, text):
            score += weight
            matches.append(cue)
    return score, matches


OUTDOOR_CUES = {
    "outdoor": 1.7,
    "outdoors": 1.7,
    "park": 1.3,
    "park-like": 1.3,
    "beach": 1.4,
    "sky": 1.2,
    "clear blue sky": 1.6,
    "landscape": 1.2,
    "foliage": 1.0,
    "trees": 1.0,
    "shrubs": 0.8,
    "grassy": 0.9,
    "grass": 0.8,
    "garden": 1.0,
    "deck": 1.0,
    "patio": 1.0,
    "terrain": 1.0,
    "pavement": 0.6,
    "street": 0.8,
    "hills": 0.8,
    "water": 0.7,
}

INDOOR_CUES = {
    "indoor": 1.7,
    "indoors": 1.7,
    "airplane cabin": 1.7,
    "cabin": 1.0,
    "vehicle interior": 1.5,
    "seatback": 0.9,
    "headrest": 0.7,
    "overhead panel": 1.0,
    "ceiling": 0.8,
    "cabinet": 1.0,
    "framed portrait": 0.8,
    "framed photograph": 0.8,
    "wall behind": 0.7,
    "studio": 0.9,
    "room": 1.0,
}

DAYLIGHT_CUES = {
    "daylight": 1.8,
    "sunlight": 1.8,
    "sunlit": 1.8,
    "sunny": 1.6,
    "clear blue sky": 1.5,
    "dappled shadows": 1.4,
    "bright outdoor": 1.2,
    "bright, hazy outdoor": 1.2,
}

ARTIFICIAL_CUES = {
    "overhead lighting": 1.8,
    "overhead lights": 1.8,
    "ceiling light": 1.7,
    "ceiling lights": 1.7,
    "fluorescent": 1.8,
    "lamp": 1.4,
    "artificial light": 1.8,
    "studio lighting": 1.8,
}

LOW_LIGHT_CUES = {
    "low light": 1.8,
    "low-light": 1.8,
    "dim": 1.5,
    "night": 1.5,
    "nighttime": 1.7,
    "dark interior": 1.3,
}

WINDOW_LIGHT_CUES = {
    "bright window": 1.7,
    "window light": 1.8,
    "window-lit": 1.8,
    "window lit": 1.8,
    "backlit window": 1.8,
}


def _confidence_from_score(score: float, explicit: bool = False) -> float:
    if explicit or score >= 2.6:
        return 0.92
    if score >= 1.6:
        return 0.84
    if score >= 1.0:
        return 0.74
    if score >= 0.6:
        return 0.62
    return 0.0


def _classify_capture_context(record: dict[str, Any]) -> dict[str, Any]:
    text = _norm_text(record)
    outdoor_score, outdoor_matches = _matched_cues(text, OUTDOOR_CUES)
    indoor_score, indoor_matches = _matched_cues(text, INDOOR_CUES)

    explicit_outdoor = bool(re.search(r"(?<!\w)outdoors?(?!\w)", text))
    explicit_indoor = bool(re.search(r"(?<!\w)indoors?(?!\w)", text))

    if outdoor_score >= 1.0 and indoor_score >= 1.0:
        environment = "mixed_or_transitional"
        env_conf = min(0.9, 0.70 + 0.05 * min(outdoor_score, indoor_score))
        env_basis = "both indoor/enclosed and outdoor cues are visible"
    elif outdoor_score >= 0.6:
        environment = "outdoor"
        env_conf = _confidence_from_score(outdoor_score, explicit_outdoor)
        env_basis = "outdoor scene cues"
    elif indoor_score >= 0.6:
        environment = "indoor_enclosed"
        env_conf = _confidence_from_score(indoor_score, explicit_indoor)
        env_basis = "indoor/enclosed scene cues"
    else:
        environment = "unknown"
        env_conf = 0.0
        env_basis = "no sufficiently explicit scene cue in cached v1 analysis"

    daylight_score, daylight_matches = _matched_cues(text, DAYLIGHT_CUES)
    artificial_score, artificial_matches = _matched_cues(text, ARTIFICIAL_CUES)
    low_score, low_matches = _matched_cues(text, LOW_LIGHT_CUES)
    window_score, window_matches = _matched_cues(text, WINDOW_LIGHT_CUES)

    if low_score >= 1.3:
        illumination = "low_light"
        light_score = low_score
        light_matches = low_matches
        light_basis = "explicit low-light/night cue"
    elif window_score >= 1.3 or (environment == "mixed_or_transitional" and "window" in text and artificial_score > 0):
        illumination = "mixed_window_light"
        light_score = max(window_score, artificial_score + 0.4)
        light_matches = sorted(set(window_matches + artificial_matches + (["window"] if "window" in text else [])))
        light_basis = "explicit window/mixed-light cue"
    elif daylight_score >= 1.0:
        illumination = "outdoor_daylight"
        light_score = daylight_score
        light_matches = daylight_matches
        light_basis = "explicit daylight/sun cue"
    elif artificial_score >= 1.0:
        illumination = "indoor_artificial"
        light_score = artificial_score
        light_matches = artificial_matches
        light_basis = "explicit artificial/overhead-light cue"
    elif environment == "outdoor":
        illumination = "outdoor_ambient_unspecified"
        light_score = 0.6
        light_matches = outdoor_matches[:3]
        light_basis = "inferred only from outdoor context; exact lighting was not requested in Analyze v1"
    elif environment == "indoor_enclosed":
        illumination = "indoor_ambient_unspecified"
        light_score = 0.5
        light_matches = indoor_matches[:3]
        light_basis = "inferred only from indoor/enclosed context; exact lighting was not requested in Analyze v1"
    else:
        illumination = "unknown"
        light_score = 0.0
        light_matches = []
        light_basis = "cached v1 analysis does not support a useful lighting judgement"

    light_conf = _confidence_from_score(light_score)
    if illumination.endswith("_unspecified"):
        light_conf = min(light_conf, 0.62)

    return {
        "environment": {
            "value": environment,
            "confidence": round(env_conf, 3),
            "basis": env_basis,
            "outdoor_cues": outdoor_matches,
            "indoor_cues": indoor_matches,
            "source": "cached_v1_text_cues",
            "authority": "qualified_semantic_context",
        },
        "illumination": {
            "value": illumination,
            "confidence": round(light_conf, 3),
            "basis": light_basis,
            "cues": light_matches,
            "source": "cached_v1_text_cues",
            "authority": "provisional_semantic_context",
        },
        "note": (
            "V7 derives broad capture context conservatively from explicit cached Analyze-v1 text. "
            "Lighting was not a first-class Analyze-v1 field, so illumination remains provisional "
            "and should become structured evidence in Analyze v2."
        ),
    }


def _context_guidance_summary(records: list[dict[str, Any]], guidance: dict[str, Any]) -> dict[str, Any]:
    context_policy = guidance.get("context") or {}
    env_policy = context_policy.get("environment") or {}
    light_policy = context_policy.get("illumination") or {}
    env_floor_conf = float(env_policy.get("confidence_floor", 0.65))
    light_floor_conf = float(light_policy.get("confidence_floor", 0.75))

    env_counts: Counter[str] = Counter()
    light_counts: Counter[str] = Counter()
    env_unknown = 0
    light_unknown = 0

    for record in records:
        context = record["capture_context"]
        env = context["environment"]
        light = context["illumination"]
        if env["value"] != "unknown" and float(env["confidence"]) >= env_floor_conf:
            env_counts[env["value"]] += 1
        else:
            env_unknown += 1
        if light["value"] != "unknown" and float(light["confidence"]) >= light_floor_conf:
            light_counts[light["value"]] += 1
        else:
            light_unknown += 1

    n = len(records)
    env_required = env_policy.get("required_categories") or {}
    env_debt: list[dict[str, Any]] = []
    env_floors: dict[str, int] = {}
    env_category_status: dict[str, dict[str, Any]] = {}
    for category, rule in env_required.items():
        min_share = float(rule.get("minimum_share", 0.0))
        min_count = int(rule.get("minimum_count", 0))
        floor = max(min_count, int(math.ceil(min_share * n)))
        env_floors[category] = floor
        current = int(env_counts.get(category, 0))
        possible_max = current + env_unknown
        if current >= floor:
            status = "satisfied"
        elif possible_max >= floor:
            status = "partially_unassessed"
        else:
            status = "deficit"
            env_debt.append(
                {
                    "category": category,
                    "current_count": current,
                    "possible_count_including_unassessed": possible_max,
                    "guidance_floor_count": floor,
                    "debt_count": floor - possible_max,
                }
            )
        env_category_status[category] = {
            "status": status,
            "current_count": current,
            "possible_count_including_unassessed": possible_max,
            "guidance_floor_count": floor,
            "known_shortfall_count": max(0, floor - current),
        }

    env_min_distinct = int(env_policy.get("minimum_distinct", 2))
    env_distinct = len([k for k, v in env_counts.items() if v > 0])
    if env_distinct >= env_min_distinct:
        env_diversity_status = "satisfied"
        env_diversity_debt = 0
    elif env_unknown:
        env_diversity_status = "partially_unassessed"
        env_diversity_debt = None
    else:
        env_diversity_status = "deficit"
        env_diversity_debt = env_min_distinct - env_distinct

    light_min_distinct = int(light_policy.get("minimum_distinct", 2))
    light_distinct = len([k for k, v in light_counts.items() if v > 0])
    if light_distinct >= light_min_distinct:
        light_diversity_status = "satisfied"
        light_diversity_debt = 0
    elif light_unknown:
        light_diversity_status = "partially_unassessed"
        light_diversity_debt = None
    else:
        light_diversity_status = "deficit"
        light_diversity_debt = light_min_distinct - light_distinct
    desired_lights = list(light_policy.get("desired_categories") or [])
    missing_lights = [x for x in desired_lights if light_counts.get(x, 0) == 0]

    return {
        "environment": {
            "confidence_floor": env_floor_conf,
            "counts": dict(sorted(env_counts.items())),
            "unassessed_count": env_unknown,
            "minimum_distinct": env_min_distinct,
            "distinct_count": env_distinct,
            "diversity_status": env_diversity_status,
            "diversity_debt": env_diversity_debt,
            "required_category_floors": env_floors,
            "required_category_status": env_category_status,
            "category_debts": env_debt,
        },
        "illumination": {
            "confidence_floor": light_floor_conf,
            "counts": dict(sorted(light_counts.items())),
            "unassessed_count": light_unknown,
            "minimum_distinct": light_min_distinct,
            "distinct_count": light_distinct,
            "diversity_status": light_diversity_status,
            "diversity_debt": light_diversity_debt,
            "desired_categories": desired_lights,
            "missing_desired_categories": missing_lights,
            "analysis_gap": light_unknown > max(2, int(0.35 * n)),
        },
    }


def _global_identity_gaps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    availability = payload.get("coverage_availability") or {}
    summary = payload.get("dataset_summary") or {}
    gaps: list[dict[str, Any]] = []

    yaw_available = str((availability.get("head_yaw") or {}).get("status") or "").startswith("available")
    pitch_available = str((availability.get("head_pitch") or {}).get("status") or "").startswith("available")

    if yaw_available:
        yaw = summary.get("head_yaw_counts") or {}
        if int(yaw.get("strong_left", 0)) == 0:
            gaps.append({"axis": "head_yaw", "target": "strong_left_or_profile", "weight_key": "head_yaw_debt"})
        if int(yaw.get("strong_right", 0)) == 0:
            gaps.append({"axis": "head_yaw", "target": "strong_right_or_profile", "weight_key": "head_yaw_debt"})
        if int(yaw.get("three_quarter_left", 0)) <= 1:
            gaps.append({"axis": "head_yaw", "target": "left_three_quarter", "weight_key": "head_yaw_debt", "secondary": True})

    if pitch_available:
        pitch = summary.get("head_pitch_counts") or {}
        upward = int(pitch.get("up", 0)) + int(pitch.get("strong_up", 0))
        if upward == 0:
            gaps.append({"axis": "head_pitch", "target": "up_or_strong_up", "weight_key": "head_pitch_debt"})

    return gaps


def _within_composition_diversity_gaps(payload: dict[str, Any]) -> dict[str, list[str]]:
    policy = payload.get("guidance_policy") or {}
    out: dict[str, list[str]] = {}
    for cls, item in (policy.get("composition") or {}).items():
        missing: list[str] = []
        for feature, div in (item.get("diversity") or {}).items():
            if div.get("status") == "debt":
                missing.append(feature)
        out[cls] = missing
    return out


def _donor_context_penalty(
    record: dict[str, Any],
    context_summary: dict[str, Any],
    guidance: dict[str, Any],
) -> tuple[float, list[str]]:
    planner = guidance.get("acquisition_planner") or {}
    weights = planner.get("weights") or {}
    penalty = 0.0
    reasons: list[str] = []

    env = record["capture_context"]["environment"]
    env_summary = context_summary["environment"]
    if float(env["confidence"]) >= float(env_summary["confidence_floor"]):
        value = env["value"]
        floor = int((env_summary.get("required_category_floors") or {}).get(value, 0))
        count = int((env_summary.get("counts") or {}).get(value, 0))
        if floor and count <= floor:
            penalty += float(weights.get("context_floor_protection", 1.0))
            reasons.append(f"removal would pressure environment floor for {value}")
        elif count <= 2:
            penalty += 0.5 * float(weights.get("context_floor_protection", 1.0))
            reasons.append(f"{value} environment is relatively uncommon")

    light = record["capture_context"]["illumination"]
    light_summary = context_summary["illumination"]
    if float(light["confidence"]) >= float(light_summary["confidence_floor"]):
        value = light["value"]
        count = int((light_summary.get("counts") or {}).get(value, 0))
        if count <= 1:
            penalty += float(weights.get("lighting_rarity_protection", 0.5))
            reasons.append(f"sole confidently observed illumination context {value}")
    return penalty, reasons


def _rank_donors(
    records: list[dict[str, Any]],
    context_summary: dict[str, Any],
    guidance: dict[str, Any],
) -> list[dict[str, Any]]:
    donors: list[dict[str, Any]] = []
    for record in records:
        if not record.get("guidance_swap_candidate"):
            continue
        measured = float((record.get("measured_signal_density") or {}).get("score") or 0.0)
        penalty, reasons = _donor_context_penalty(record, context_summary, guidance)
        cost = measured + penalty
        donors.append(
            {
                "image": record.get("relative_path"),
                "from_class": (record.get("guidance_composition") or {}).get("class"),
                "measured_signal_density": round(measured, 4),
                "context_penalty": round(penalty, 4),
                "removal_cost": round(cost, 4),
                "context_reasons": reasons,
                "environment": record["capture_context"]["environment"]["value"],
                "illumination": record["capture_context"]["illumination"]["value"],
            }
        )
    donors.sort(key=lambda x: (x["removal_cost"], x["measured_signal_density"]))
    return donors


def _build_acquisition_plan(
    payload: dict[str, Any],
    guidance: dict[str, Any],
    context_summary: dict[str, Any],
) -> dict[str, Any]:
    planner = guidance.get("acquisition_planner") or {}
    weights = planner.get("weights") or {}
    max_targets = int(planner.get("max_targets", 8))
    composition = (payload.get("guidance_policy") or {}).get("composition") or {}
    identity_gaps = _global_identity_gaps(payload)
    within_gaps = _within_composition_diversity_gaps(payload)

    env_debt_queue: list[str] = []
    for item in context_summary["environment"]["category_debts"]:
        env_debt_queue.extend([item["category"]] * int(item["debt_count"]))

    light_missing = list(context_summary["illumination"]["missing_desired_categories"])
    if context_summary["illumination"]["analysis_gap"]:
        light_missing = []

    raw_slots: list[str] = []
    priority_classes = ["full_body", "upper_body", "identity_close"]
    for cls in priority_classes:
        debt = int((composition.get(cls) or {}).get("coverage_debt_count") or 0)
        raw_slots.extend([cls] * debt)

    targets: list[dict[str, Any]] = []
    yaw_gaps = [g for g in identity_gaps if g["axis"] == "head_yaw" and not g.get("secondary")]
    pitch_gaps = [g for g in identity_gaps if g["axis"] == "head_pitch"]
    secondary_gaps = [g for g in identity_gaps if g.get("secondary")]

    for idx, cls in enumerate(raw_slots[:max_targets], 1):
        goals: list[dict[str, Any]] = []
        score = float(weights.get("composition_debt", 3.0))

        if yaw_gaps:
            gap = yaw_gaps.pop(0)
            goals.append({"type": "identity_view", "axis": gap["axis"], "target": gap["target"]})
            score += float(weights.get(gap["weight_key"], 2.0))
        elif secondary_gaps:
            gap = secondary_gaps.pop(0)
            goals.append({"type": "identity_view", "axis": gap["axis"], "target": gap["target"]})
            score += 0.7 * float(weights.get(gap["weight_key"], 2.0))

        if pitch_gaps and (cls == "full_body" or idx <= 3):
            gap = pitch_gaps.pop(0)
            goals.append({"type": "identity_view", "axis": gap["axis"], "target": gap["target"]})
            score += float(weights.get(gap["weight_key"], 1.5))

        if env_debt_queue:
            env = env_debt_queue.pop(0)
            goals.append({"type": "environment", "target": env})
            score += float(weights.get("environment_debt", 1.2))

        if light_missing:
            light = light_missing.pop(0)
            goals.append({"type": "illumination", "target": light})
            score += float(weights.get("illumination_diversity", 0.7))

        for feature in within_gaps.get(cls, []):
            if feature == "action_signature":
                goals.append({"type": "within_class_diversity", "axis": feature, "target": "different_from_current_representatives"})
                score += float(weights.get("within_class_diversity", 0.8))
                break

        targets.append(
            {
                "priority": idx,
                "primary_composition": cls,
                "goals": goals,
                "debt_payoff_count": 1 + len(goals),
                "priority_score": round(score, 3),
            }
        )

    if not targets and identity_gaps:
        for idx, gap in enumerate(identity_gaps[:max_targets], 1):
            targets.append(
                {
                    "priority": idx,
                    "primary_composition": "best_fit_without_creating_composition_debt",
                    "goals": [{"type": "identity_view", "axis": gap["axis"], "target": gap["target"]}],
                    "debt_payoff_count": 1,
                    "priority_score": round(float(weights.get(gap["weight_key"], 2.0)), 3),
                }
            )

    targets.sort(key=lambda x: (x["priority_score"], x["debt_payoff_count"]), reverse=True)
    for idx, target in enumerate(targets, 1):
        target["priority"] = idx

    donors = _rank_donors(payload.get("records") or [], context_summary, guidance)
    swaps: list[dict[str, Any]] = []
    for target, donor in zip(targets, donors):
        swaps.append(
            {
                "acquisition_priority": target["priority"],
                "add_target": {
                    "primary_composition": target["primary_composition"],
                    "goals": target["goals"],
                },
                "preferred_donor": donor,
                "reason": (
                    "candidate donor comes from a guidance-surplus composition class and has low estimated removal cost "
                    "after context-rarity penalties"
                ),
            }
        )

    return {
        "identity_view_gaps": identity_gaps,
        "within_composition_diversity_gaps": within_gaps,
        "targets": targets,
        "donor_ranking": donors,
        "proposed_fixed_size_swaps": swaps,
        "notes": [
            "Targets are multi-debt acquisition briefs: one new image can pay composition, identity-view, context, and quality debt at the same time.",
            "Context diversity is secondary policy evidence in v7. It can alter acquisition preference and donor cost, but it does not override trusted v5/v6 evidence protection.",
            "Lighting classification is intentionally conservative because Analyze v1 did not request structured illumination fields; unknown is preferable to invented precision.",
        ],
    }


def _make_markdown(payload: dict[str, Any]) -> str:
    cap = payload.get("analysis_capability_policy") or {}
    guidance = payload.get("guidance_policy") or {}
    context = payload["context_diversity"]
    plan = payload["acquisition_plan"]
    summary = payload.get("dataset_summary") or {}

    lines = [
        "# Dataset evidence report — acquisition planner v7",
        "",
        f"> Analysis source: **{cap.get('model_id')}** · policy tier **{cap.get('policy_tier')}**.",
        f"> Guidance profile: **{guidance.get('profile_id')}** · authority **{guidance.get('authority')}**.",
        "> Scene and lighting diversity are policy/context observations, not intrinsic image-quality facts.",
        "",
        "## Dataset summary",
        "",
        f"- Images profiled: **{len(payload.get('records') or [])}**",
        f"- Guidance classes: `{json.dumps(summary.get('guidance_class_counts', {}), sort_keys=True)}`",
        f"- Dataset actions: `{json.dumps(summary.get('presented_recommendation_counts_v6', summary.get('presented_recommendation_counts', {})), sort_keys=True)}`",
        f"- Quarantined VLM axes: `{json.dumps(sorted((payload.get('dynamic_axis_quarantine') or {}).keys()))}`",
        "",
        "## Capture-context diversity",
        "",
        f"- Environment counts above confidence floor: `{json.dumps(context['environment']['counts'], sort_keys=True)}`",
        f"- Environment unassessed: **{context['environment']['unassessed_count']}**",
        f"- Environment category debt: `{json.dumps(context['environment']['category_debts'])}`",
        f"- Illumination counts above confidence floor: `{json.dumps(context['illumination']['counts'], sort_keys=True)}`",
        f"- Illumination unassessed: **{context['illumination']['unassessed_count']}**",
    ]
    if context["illumination"]["analysis_gap"]:
        lines.append(
            "- **Lighting-analysis gap:** cached Analyze v1 did not explicitly characterize illumination for enough images; treat light-diversity conclusions as provisional and add structured lighting fields in Analyze v2."
        )

    lines.extend(["", "## Acquisition targets", ""])
    if plan["targets"]:
        for target in plan["targets"]:
            goals = "; ".join(
                f"{g.get('type')}:{g.get('axis') + '=' if g.get('axis') else ''}{g.get('target')}"
                for g in target["goals"]
            ) or "clean/high-signal example"
            lines.append(
                f"{target['priority']}. **{target['primary_composition']}** — {goals} "
                f"(debt payoff {target['debt_payoff_count']}, priority score {target['priority_score']:.2f})"
            )
    else:
        lines.append("- No composition/identity acquisition debt triggered by the currently assessable policy axes.")

    lines.extend(["", "## Preferred donor ranking", ""])
    if plan["donor_ranking"]:
        lines.extend(
            [
                "| rank | image | class | measured density | context penalty | removal cost | environment | illumination |",
                "|---:|---|---|---:|---:|---:|---|---|",
            ]
        )
        for i, donor in enumerate(plan["donor_ranking"], 1):
            lines.append(
                f"| {i} | {donor['image']} | {donor['from_class']} | {donor['measured_signal_density']:.3f} | "
                f"{donor['context_penalty']:.3f} | {donor['removal_cost']:.3f} | {donor['environment']} | {donor['illumination']} |"
            )
    else:
        lines.append("- No guidance-surplus donor candidates are available.")

    lines.extend(["", "## Proposed fixed-size swaps", ""])
    if plan["proposed_fixed_size_swaps"]:
        for swap in plan["proposed_fixed_size_swaps"]:
            target = swap["add_target"]
            donor = swap["preferred_donor"]
            goal_text = ", ".join(g["target"] for g in target["goals"]) or "cleaner evidence"
            lines.append(
                f"- Add **{target['primary_composition']}** ({goal_text}); consider retiring **{donor['image']}** "
                f"from `{donor['from_class']}` if the challenger is demonstrably stronger."
            )
    else:
        lines.append("- No fixed-size swap pairing asserted.")

    lines.extend(
        [
            "",
            "## Per-image context",
            "",
            "| image | guidance class | environment | env conf | illumination | light conf | guidance action |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )
    for record in payload.get("records") or []:
        ctx = record["capture_context"]
        lines.append(
            f"| {record.get('relative_path')} | {(record.get('guidance_composition') or {}).get('class')} | "
            f"{ctx['environment']['value']} | {float(ctx['environment']['confidence']):.2f} | "
            f"{ctx['illumination']['value']} | {float(ctx['illumination']['confidence']):.2f} | "
            f"{(record.get('guidance_action') or {}).get('label', 'neutral')} |"
        )

    lines.extend(
        [
            "",
            "## V7 interpretation notes",
            "",
            "- Environment diversity is broad by design: indoor/enclosed, outdoor, mixed/transitional, or unknown.",
            "- Illumination is more conservative than environment because Analyze v1 did not request structured lighting observations.",
            "- Context diversity can make an otherwise redundant image strategically useful, but v7 does not allow weak light/context inference to overrule trusted composition/geometry protection.",
            "- The next Analyze schema should make environment and illumination first-class fields so lighting diversity can move from text-derived guidance evidence to calibrated visual evidence.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    try:
        guidance = _load_guidance(args.guidance_profile)
        model_id = resolve_model_id(args.model)
        slug = model_slug(model_id)
        payload, cleanup = _run_v6_base(args, run_dir, slug)

        for record in payload.get("records") or []:
            record["capture_context"] = _classify_capture_context(record)

        context_summary = _context_guidance_summary(payload.get("records") or [], guidance)
        acquisition_plan = _build_acquisition_plan(payload, guidance, context_summary)

        payload["schema_version"] = "dataset-evidence-7.0"
        payload["context_diversity"] = context_summary
        payload["acquisition_plan"] = acquisition_plan
        payload.setdefault("method_notes", []).extend(
            [
                "V7 adds scene/environment and illumination diversity as a separate guidance/context layer rather than folding them into image-quality scores.",
                "Environment is conservatively derived from explicit cached Analyze-v1 scene cues; illumination is lower-authority because lighting was not a first-class Analyze-v1 field.",
                "Acquisition targets combine multiple debts so a single new sample can repay composition, identity-view, within-class diversity, and context debt simultaneously.",
                "Donor ranking starts from guidance-surplus images and adds removal penalties when an image carries rare or floor-critical context evidence.",
                "Context evidence never overrides trusted v5/v6 geometry/composition protection in v7.",
            ]
        )
        summary = payload.setdefault("dataset_summary", {})
        summary["environment_counts_v7"] = context_summary["environment"]["counts"]
        summary["illumination_counts_v7"] = context_summary["illumination"]["counts"]
        summary["acquisition_target_count_v7"] = len(acquisition_plan["targets"])
        summary["donor_candidate_count_v7"] = len(acquisition_plan["donor_ranking"])

        out_json = run_dir / f"{args.output_prefix}_{slug}.json"
        out_md = run_dir / f"{args.output_prefix}_{slug}.md"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        out_md.write_text(_make_markdown(payload), encoding="utf-8")

        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        print(f"Done. JSON:   {out_json}")
        print(f"      Report: {out_md}")
        print(
            f"Analysis policy: {(payload.get('analysis_capability_policy') or {}).get('policy_tier')} / "
            f"{(payload.get('analysis_capability_policy') or {}).get('judgement_breadth')}"
        )
        print(f"Guidance profile: {(payload.get('guidance_policy') or {}).get('profile_id')} / {(payload.get('guidance_policy') or {}).get('authority')}")
        print(f"Environment: {context_summary['environment']['counts']} (unassessed={context_summary['environment']['unassessed_count']})")
        print(f"Environment debt: {context_summary['environment']['category_debts']}")
        print(f"Illumination: {context_summary['illumination']['counts']} (unassessed={context_summary['illumination']['unassessed_count']})")
        print(f"Lighting analysis gap: {context_summary['illumination']['analysis_gap']}")
        print(f"Acquisition targets: {len(acquisition_plan['targets'])}")
        print(f"Donor candidates: {len(acquisition_plan['donor_ranking'])}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
