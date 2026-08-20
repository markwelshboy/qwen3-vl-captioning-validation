from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import dataset_evidence_v8 as v8
from .runner import model_slug, resolve_model_id


DEFAULT_GUIDANCE = v8.DEFAULT_GUIDANCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "V8.1 dataset-selection workspace. Treats minimum counts as hard safety floors, "
            "preferred composition shares as soft objectives, qualifies semantic evidence before "
            "selection, and evaluates candidates as incremental positive-gain changes to the active set."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence_v8_1", help="Output basename inside run_dir.")
    parser.add_argument("--guidance-profile", type=Path, default=DEFAULT_GUIDANCE)
    parser.add_argument("--base-v7-json", type=Path, help="Optional existing v7 JSON instead of regenerating v7.")
    parser.add_argument("--selection-manifest", type=Path, help="Optional dataset-selection-workspace-1.0 manifest.")
    parser.add_argument(
        "--candidate-glob",
        action="append",
        default=[],
        help="Glob matching records to treat as candidates when no manifest entry overrides them. Repeatable.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["preserve_size", "target_size", "flexible"],
        default="preserve_size",
        help="preserve_size evaluates positive-gain swaps; target_size grows/shrinks to --target-size then swaps; flexible never forces removals.",
    )
    parser.add_argument("--target-size", type=int, help="Final portfolio size for target_size mode.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_guidance(path: Path) -> dict[str, Any]:
    return v8._load_guidance(path)


def _run_v7_base(args: argparse.Namespace, run_dir: Path, slug: str) -> tuple[dict[str, Any], list[Path]]:
    if args.base_v7_json:
        return _read_json(args.base_v7_json.expanduser().resolve()), []
    internal_prefix = f".__v8_1_base_{args.output_prefix}"
    cmd = [
        sys.executable,
        "-m",
        "qwen_caption_validate.dataset_evidence_v7",
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
        raise RuntimeError(f"v7 base profiler failed with exit code {proc.returncode}")
    base_json = run_dir / f"{internal_prefix}_{slug}.json"
    base_md = run_dir / f"{internal_prefix}_{slug}.md"
    if not base_json.exists():
        raise RuntimeError(f"v7 base profiler did not create expected file: {base_json}")
    return _read_json(base_json), [base_json, base_md]


def _preferred_low(rule: dict[str, Any], n: int) -> int:
    low = float((rule.get("preferred_share") or [0.0, 1.0])[0])
    return max(int(rule.get("minimum_count", 0)), int(math.ceil(low * n)))


def _preferred_high(rule: dict[str, Any], n: int) -> int:
    high = float((rule.get("preferred_share") or [0.0, 1.0])[1])
    by_share = int(math.ceil(high * n))
    soft_cap = int(rule.get("soft_cap_count", by_share))
    return max(_preferred_low(rule, n), min(by_share, soft_cap))


def _fine_action(record: dict[str, Any], entry: dict[str, Any], tier: str) -> dict[str, Any]:
    result = dict(v8._fine_action(record, entry))
    if result.get("source") == "manual_override":
        result["usable_for_selection"] = True
        result["authority_reason"] = "manual workspace override"
        return result
    if tier != "high":
        result["usable_for_selection"] = False
        result["authority_reason"] = "cached fine-action semantics are report-only below the high analyzer tier"
    elif float(result.get("confidence") or 0.0) < 0.85:
        result["usable_for_selection"] = False
        result["authority_reason"] = "fine-action confidence below selection threshold"
    else:
        result["usable_for_selection"] = True
        result["authority_reason"] = "high-tier cached semantic evidence above confidence threshold"
    return result


def _qualified_action_signature(
    record: dict[str, Any],
    tier: str,
    axis_health: dict[str, Any],
    shoulder: dict[str, Any],
) -> dict[str, Any]:
    action = record.get("action_contact") or {}
    raw_components = [str(x) for x in (action.get("signature_components") or []) if x]
    coarse = {
        str(item.get("class")): float(item.get("confidence") or 0.0)
        for item in (action.get("coarse_classes") or [])
        if item.get("class")
    }
    accepted: list[str] = []
    dropped: list[dict[str, Any]] = []

    if tier != "high":
        return {
            "raw": "+".join(raw_components) if raw_components else None,
            "trusted": None,
            "trusted_components": [],
            "dropped": [{"component": x, "reason": "semantic action protection disabled below high tier"} for x in raw_components],
            "authority": "report_only_for_source_tier",
        }

    for component in raw_components:
        conf = float(coarse.get(component, 0.0))
        if component == "strong_shoulder_cant":
            if shoulder.get("status") != "usable":
                dropped.append({"component": component, "reason": "shoulder geometry is sanity-review/unusable"})
                continue
            if float(shoulder.get("abs_cant_deg") or 0.0) < 15.0:
                dropped.append({"component": component, "reason": "normalized shoulder cant below strong-cant threshold"})
                continue
        if component == "head_torso_counter_rotation":
            if axis_health.get("torso", {}).get("status") == "quarantined":
                dropped.append({"component": component, "reason": "torso yaw axis is quarantined"})
                continue
            if axis_health.get("head_yaw", {}).get("status") == "quarantined":
                dropped.append({"component": component, "reason": "head yaw axis is quarantined"})
                continue
        if conf < 0.75:
            dropped.append({"component": component, "reason": f"component confidence {conf:.2f} below 0.75 selection threshold"})
            continue
        accepted.append(component)

    return {
        "raw": "+".join(raw_components) if raw_components else None,
        "trusted": "+".join(accepted) if accepted else None,
        "trusted_components": accepted,
        "dropped": dropped,
        "authority": "high_tier_confidence_and_geometry_qualified",
    }


def _record_features(
    record: dict[str, Any],
    entry: dict[str, Any],
    axis_health: dict[str, Any],
    guidance: dict[str, Any],
    tier: str,
) -> dict[str, Any]:
    comp = str(((record.get("guidance_composition") or {}).get("class")) or "unknown")
    quality = float(((record.get("measured_signal_density") or {}).get("score")) or 0.0)
    coverage = record.get("coverage_signature") or {}
    yaw = coverage.get("head_yaw") if axis_health["head_yaw"]["status"] != "quarantined" else None
    pitch = coverage.get("head_pitch") if axis_health["head_pitch"]["status"] != "quarantined" else None

    geom = ((record.get("pose_evidence") or {}).get("target_2d_geometry") or {})
    shoulder = v8._normalize_line_angle(geom.get("shoulder_line_angle_from_horizontal_deg"))
    geometry_class = None
    if shoulder["status"] == "usable" and shoulder["abs_cant_deg"] is not None and shoulder["abs_cant_deg"] >= 15.0:
        geometry_class = "strong_shoulder_cant"

    qualified_action = _qualified_action_signature(record, tier, axis_health, shoulder)
    fine_action = _fine_action(record, entry, tier)

    context = record.get("capture_context") or {}
    env = context.get("environment") or {}
    light = context.get("illumination") or {}
    env_floor = float((((guidance.get("context") or {}).get("environment") or {}).get("confidence_floor", 0.65)))
    light_floor = float((((guidance.get("context") or {}).get("illumination") or {}).get("confidence_floor", 0.75)))
    environment = env.get("value") if float(env.get("confidence") or 0.0) >= env_floor and env.get("value") != "unknown" else None
    illumination = light.get("value") if float(light.get("confidence") or 0.0) >= light_floor and light.get("value") != "unknown" else None

    return {
        "class": comp,
        "quality": round(quality, 6),
        "head_yaw": yaw,
        "head_pitch": pitch,
        "action_signature": qualified_action.get("trusted"),
        "action_qualification": qualified_action,
        "fine_action": fine_action,
        "geometry_class": geometry_class,
        "shoulder_geometry": shoulder,
        "environment": environment,
        "illumination": illumination,
    }


def _profile(records: list[dict[str, Any]], selected_paths: set[str], guidance: dict[str, Any], target_n: int) -> dict[str, Any]:
    selected = [r for r in records if str(r.get("relative_path")) in selected_paths]
    counts = Counter(str(((r.get("guidance_composition") or {}).get("class")) or "unknown") for r in selected)
    composition: dict[str, Any] = {}
    for cls, rule in (guidance.get("composition") or {}).items():
        hard_min = int(rule.get("minimum_count", 0))
        preferred_low = _preferred_low(rule, target_n)
        preferred_high = _preferred_high(rule, target_n)
        current = int(counts.get(cls, 0))
        if current < hard_min:
            status = "hard_deficit"
        elif current < preferred_low:
            status = "below_preferred"
        elif current > preferred_high:
            status = "above_preferred"
        else:
            status = "within_preferred"
        composition[cls] = {
            "count": current,
            "hard_minimum": hard_min,
            "preferred_low": preferred_low,
            "preferred_high": preferred_high,
            "hard_debt": max(0, hard_min - current),
            "preference_debt": max(0, preferred_low - current),
            "surplus_above_preferred": max(0, current - preferred_high),
            "status": status,
        }
    return {"image_count": len(selected), "composition": composition}


def _optimizer_weights(guidance: dict[str, Any]) -> dict[str, float]:
    defaults = {
        "quality": 1.0,
        "hard_minimum_penalty": 3.0,
        "preferred_shortfall_penalty": 0.14,
        "preferred_surplus_penalty": 0.06,
        "head_yaw_diversity": 0.10,
        "head_pitch_diversity": 0.06,
        "action_diversity": 0.08,
        "fine_action_diversity": 0.05,
        "environment_diversity": 0.04,
        "illumination_diversity": 0.025,
        "candidate_churn_penalty": 0.04,
        "min_swap_gain": 0.05,
    }
    configured = ((guidance.get("selection_optimizer") or {}).get("weights") or {})
    for key in defaults:
        if key in configured:
            defaults[key] = float(configured[key])
    if "min_swap_gain" in (guidance.get("selection_optimizer") or {}):
        defaults["min_swap_gain"] = float(guidance["selection_optimizer"]["min_swap_gain"])
    return defaults


def _dataset_objective(
    selected: set[str],
    features: dict[str, dict[str, Any]],
    states: dict[str, str],
    guidance: dict[str, Any],
    target_n: int,
) -> tuple[float, dict[str, float]]:
    w = _optimizer_weights(guidance)
    quality = sum(float(features[p]["quality"]) for p in selected) * w["quality"]
    counts = Counter(str(features[p]["class"]) for p in selected)

    hard_term = 0.0
    preferred_term = 0.0
    for cls, rule in (guidance.get("composition") or {}).items():
        count = int(counts.get(cls, 0))
        hard_min = int(rule.get("minimum_count", 0))
        low = _preferred_low(rule, target_n)
        high = _preferred_high(rule, target_n)
        hard_term -= max(0, hard_min - count) * w["hard_minimum_penalty"]
        preferred_term -= max(0, low - count) * w["preferred_shortfall_penalty"]
        preferred_term -= max(0, count - high) * w["preferred_surplus_penalty"]

    diversity_term = 0.0
    diversity_weights = {
        "head_yaw": w["head_yaw_diversity"],
        "head_pitch": w["head_pitch_diversity"],
        "action_signature": w["action_diversity"],
    }
    for cls, policy in (guidance.get("diversity") or {}).items():
        members = [features[p] for p in selected if features[p]["class"] == cls]
        for field, desired in (policy.get("minimum_distinct") or {}).items():
            values = {str(f[field]) for f in members if f.get(field)}
            diversity_term += min(len(values), int(desired)) * diversity_weights.get(field, 0.0)

    fine_values = {
        str((features[p].get("fine_action") or {}).get("value"))
        for p in selected
        if (features[p].get("fine_action") or {}).get("usable_for_selection")
        and (features[p].get("fine_action") or {}).get("value")
    }
    diversity_term += min(len(fine_values), 3) * w["fine_action_diversity"]

    env_values = {str(features[p]["environment"]) for p in selected if features[p].get("environment")}
    light_values = {str(features[p]["illumination"]) for p in selected if features[p].get("illumination")}
    context_term = min(len(env_values), 2) * w["environment_diversity"]
    context_term += min(len(light_values), 2) * w["illumination_diversity"]

    churn_term = -sum(1 for p in selected if states[p] == "candidate") * w["candidate_churn_penalty"]
    parts = {
        "quality": quality,
        "hard_minimum": hard_term,
        "preferred_composition": preferred_term,
        "qualified_diversity": diversity_term,
        "qualified_context": context_term,
        "candidate_churn": churn_term,
    }
    total = sum(parts.values())
    return total, {k: round(v, 6) for k, v in parts.items()}


def _hard_debt(selected: set[str], features: dict[str, dict[str, Any]], guidance: dict[str, Any]) -> dict[str, int]:
    counts = Counter(str(features[p]["class"]) for p in selected)
    return {
        cls: max(0, int(rule.get("minimum_count", 0)) - int(counts.get(cls, 0)))
        for cls, rule in (guidance.get("composition") or {}).items()
    }


def _does_not_worsen_hard_debt(
    before: set[str],
    after: set[str],
    features: dict[str, dict[str, Any]],
    guidance: dict[str, Any],
) -> bool:
    a = _hard_debt(before, features, guidance)
    b = _hard_debt(after, features, guidance)
    return all(b[k] <= a[k] for k in a)


def _delta_parts(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: round(after.get(k, 0.0) - before.get(k, 0.0), 4) for k in before}


def _optimize_preserve_size(
    included_paths: set[str],
    candidate_paths: set[str],
    features: dict[str, dict[str, Any]],
    states: dict[str, str],
    guidance: dict[str, Any],
    target_n: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    selected = set(included_paths)
    remaining = set(candidate_paths)
    trace: list[dict[str, Any]] = []
    min_gain = _optimizer_weights(guidance)["min_swap_gain"]

    while remaining:
        before_score, before_parts = _dataset_objective(selected, features, states, guidance, target_n)
        best: tuple[float, str, str, float, dict[str, float]] | None = None
        for cand in remaining:
            for donor in selected:
                trial = (selected - {donor}) | {cand}
                if not _does_not_worsen_hard_debt(selected, trial, features, guidance):
                    continue
                after_score, after_parts = _dataset_objective(trial, features, states, guidance, target_n)
                gain = after_score - before_score
                if best is None or gain > best[0]:
                    best = (gain, cand, donor, after_score, after_parts)
        if best is None or best[0] < min_gain:
            break
        gain, cand, donor, after_score, after_parts = best
        selected.remove(donor)
        selected.add(cand)
        remaining.remove(cand)
        trace.append({
            "step": len(trace) + 1,
            "add": cand,
            "remove": donor,
            "gain": round(gain, 4),
            "quality_delta": round(float(features[cand]["quality"]) - float(features[donor]["quality"]), 4),
            "add_class": features[cand]["class"],
            "remove_class": features[donor]["class"],
            "objective_before": round(before_score, 4),
            "objective_after": round(after_score, 4),
            "component_delta": _delta_parts(before_parts, after_parts),
        })
    return sorted(selected), trace


def _grow_or_shrink_to_target(
    included_paths: set[str],
    candidate_paths: set[str],
    features: dict[str, dict[str, Any]],
    states: dict[str, str],
    guidance: dict[str, Any],
    target_n: int,
) -> tuple[set[str], list[dict[str, Any]]]:
    selected = set(included_paths)
    remaining = set(candidate_paths)
    trace: list[dict[str, Any]] = []
    while len(selected) < target_n and remaining:
        best: tuple[float, str, float, dict[str, float]] | None = None
        before_score, before_parts = _dataset_objective(selected, features, states, guidance, target_n)
        for cand in remaining:
            trial = selected | {cand}
            score, parts = _dataset_objective(trial, features, states, guidance, target_n)
            gain = score - before_score
            if best is None or gain > best[0]:
                best = (gain, cand, score, parts)
        if best is None:
            break
        gain, cand, score, parts = best
        selected.add(cand)
        remaining.remove(cand)
        trace.append({"step": len(trace) + 1, "action": "grow", "add": cand, "gain": round(gain, 4), "component_delta": _delta_parts(before_parts, parts)})

    while len(selected) > target_n:
        before_score, before_parts = _dataset_objective(selected, features, states, guidance, target_n)
        best_remove: tuple[float, str, float, dict[str, float]] | None = None
        for donor in selected:
            trial = selected - {donor}
            if not _does_not_worsen_hard_debt(selected, trial, features, guidance):
                continue
            score, parts = _dataset_objective(trial, features, states, guidance, target_n)
            loss = before_score - score
            if best_remove is None or loss < best_remove[0]:
                best_remove = (loss, donor, score, parts)
        if best_remove is None:
            break
        loss, donor, score, parts = best_remove
        selected.remove(donor)
        trace.append({"step": len(trace) + 1, "action": "shrink", "remove": donor, "loss": round(loss, 4), "component_delta": _delta_parts(before_parts, parts)})
    return selected, trace


def _best_final_swap(
    cand: str,
    selected: set[str],
    features: dict[str, dict[str, Any]],
    states: dict[str, str],
    guidance: dict[str, Any],
    target_n: int,
) -> dict[str, Any] | None:
    if cand in selected:
        return None
    before_score, before_parts = _dataset_objective(selected, features, states, guidance, target_n)
    best: tuple[float, str, dict[str, float]] | None = None
    for donor in selected:
        trial = (selected - {donor}) | {cand}
        if not _does_not_worsen_hard_debt(selected, trial, features, guidance):
            continue
        score, parts = _dataset_objective(trial, features, states, guidance, target_n)
        gain = score - before_score
        if best is None or gain > best[0]:
            best = (gain, donor, parts)
    if best is None:
        return None
    gain, donor, parts = best
    return {
        "donor": donor,
        "gain": round(gain, 4),
        "quality_delta": round(float(features[cand]["quality"]) - float(features[donor]["quality"]), 4),
        "component_delta": _delta_parts(before_parts, parts),
    }


def _render_report(payload: dict[str, Any]) -> str:
    ws = payload["selection_workspace"]
    baseline = payload["baseline_profile"]
    proposed = payload["proposed_profile"]
    lines = [
        "# Dataset evidence report — selection workspace v8.1",
        "",
        f"> Analysis source: **{payload['analysis_model']}** · mode **{ws['mode']}**.",
        f"> Guidance profile: **{payload['guidance_profile']['profile_id']}** · authority **{payload['guidance_profile']['authority']}**.",
        "> `minimum_count` is a hard safety floor. Preferred percentage bands are soft portfolio objectives, not quotas.",
        "",
        "## Workspace",
        "",
        f"- Included: **{ws['state_counts'].get('included', 0)}**",
        f"- Candidates: **{ws['state_counts'].get('candidate', 0)}**",
        f"- Target portfolio size: **{ws['target_size']}**",
        f"- Minimum accepted swap gain: **{payload['optimizer']['min_swap_gain']:.3f}**",
        "",
        "## Composition — active vs proposed",
        "",
        "| class | active | hard min | preferred | proposed | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for cls in payload["guidance_profile"]["composition_classes"]:
        b = baseline["composition"].get(cls, {})
        p = proposed["composition"].get(cls, {})
        pref = f"{p.get('preferred_low', 0)}–{p.get('preferred_high', 0)}"
        lines.append(f"| {cls} | {b.get('count', 0)} | {p.get('hard_minimum', 0)} | {pref} | {p.get('count', 0)} | {p.get('status', 'n/a')} |")

    lines += ["", "## Accepted incremental changes", ""]
    if payload["proposal"]["swap_trace"]:
        lines += ["| step | add | remove | gain | quality Δ | composition Δ | diversity Δ |", "|---:|---|---|---:|---:|---:|---:|"]
        for item in payload["proposal"]["swap_trace"]:
            delta = item.get("component_delta") or {}
            lines.append(
                f"| {item['step']} | {item.get('add','')} | {item.get('remove','')} | {item.get('gain',0):.3f} | "
                f"{item.get('quality_delta',0):+.3f} | {delta.get('preferred_composition',0):+.3f} | {delta.get('qualified_diversity',0):+.3f} |"
            )
    else:
        lines.append("No candidate produced enough positive portfolio gain to justify a change.")

    lines += ["", "## Candidate decisions", "", "| image | class | quality | decision | best remaining gain | donor | fine action |", "|---|---|---:|---|---:|---|---|"]
    for row in payload["candidate_decisions"]:
        opportunity = row.get("best_final_swap") or {}
        gain = opportunity.get("gain")
        gain_text = "—" if gain is None else f"{gain:.3f}"
        donor = opportunity.get("donor") or "—"
        fine = row.get("fine_action") or {}
        fine_text = fine.get("value") or "unassessed"
        if fine_text != "unassessed" and not fine.get("usable_for_selection"):
            fine_text += " (report-only)"
        lines.append(f"| {row['image']} | {row['class']} | {row['quality']:.3f} | {row['decision']} | {gain_text} | {donor} | {fine_text} |")

    lines += ["", "## Analyzer authority anchored to active dataset", ""]
    for axis, info in payload["axis_health"].items():
        if axis == "face_pose":
            lines.append(f"- {axis}: **{info['status']}**")
        else:
            lines.append(f"- {axis}: **{info['status']}** (dominant={info.get('dominant_value')}, share={info.get('dominant_share')})")

    if payload["geometry_sanity_reviews"]:
        lines += ["", "## Geometry sanity reviews", ""]
        for item in payload["geometry_sanity_reviews"]:
            lines.append(f"- **{item['image']}**: raw shoulder angle {item['raw_deg']}° → undirected normalized {item['normalized_deg']}°; excluded from trusted action/geometry novelty until reviewed.")

    lines += [
        "",
        "## V8.1 interpretation notes",
        "",
        "- Candidate audition starts from the current included dataset instead of reconstructing a new portfolio from scratch.",
        "- Preferred composition bands can improve a swap score, but cannot by themselves force a low-quality candidate into the dataset.",
        "- Hard minimum counts are protected separately from soft preferred ranges.",
        "- Action signatures are rebuilt from evidence-qualified components before they can contribute novelty.",
        "- Quarantined torso/head axes and geometry sanity failures cannot leak back into selection through compound action labels.",
        "- Cached fine-action semantics only affect selection at the high analyzer tier and confidence >= 0.85; manual overrides remain authoritative.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    guidance = _load_guidance(args.guidance_profile)
    manifest = v8._load_selection_manifest(args.selection_manifest)
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    base, temp_paths = _run_v7_base(args, run_dir, slug)
    records = list(base.get("records") or [])
    if not records:
        print("V7 base contains no records", file=sys.stderr)
        return 2

    entries = v8._manifest_entry_map(manifest)
    states: dict[str, str] = {}
    record_entries: dict[str, dict[str, Any]] = {}
    for record in records:
        path = str(record.get("relative_path") or "")
        state, entry = v8._assign_selection_state(record, manifest, entries, args.candidate_glob)
        states[path] = state
        record_entries[path] = entry

    state_counts = Counter(states.values())
    included_paths = {p for p, s in states.items() if s == "included"}
    candidate_paths = {p for p, s in states.items() if s == "candidate"}
    if not included_paths:
        print("Selection workspace has no included images", file=sys.stderr)
        return 2

    if args.selection_mode == "preserve_size":
        target_size = len(included_paths)
    elif args.selection_mode == "target_size":
        if not args.target_size or args.target_size <= 0:
            print("--target-size must be > 0 in target_size mode", file=sys.stderr)
            return 2
        target_size = args.target_size
    else:
        target_size = len(included_paths)

    tier = str(((base.get("analysis_capability_policy") or {}).get("policy_tier")) or "medium")
    active_records = [r for r in records if str(r.get("relative_path")) in included_paths]
    axis_health = v8._axis_health(active_records, tier)
    features: dict[str, dict[str, Any]] = {}
    geometry_reviews: list[dict[str, Any]] = []
    for record in records:
        path = str(record.get("relative_path") or "")
        f = _record_features(record, record_entries[path], axis_health, guidance, tier)
        features[path] = f
        if f["shoulder_geometry"]["status"] == "sanity_review":
            geometry_reviews.append({"image": path, **f["shoulder_geometry"]})

    extra_trace: list[dict[str, Any]] = []
    if args.selection_mode == "preserve_size":
        selected, swap_trace = _optimize_preserve_size(included_paths, candidate_paths, features, states, guidance, target_size)
        proposal_target_size = target_size
    elif args.selection_mode == "target_size":
        staged, extra_trace = _grow_or_shrink_to_target(included_paths, candidate_paths, features, states, guidance, target_size)
        remaining_candidates = candidate_paths - staged
        selected, swap_trace = _optimize_preserve_size(staged, remaining_candidates, features, states, guidance, target_size)
        proposal_target_size = target_size
    else:
        baseline = _profile(records, included_paths, guidance, len(included_paths))
        selected_set = set(included_paths)
        seen_candidates = sorted(candidate_paths, key=lambda p: float(features[p]["quality"]), reverse=True)
        flexible_trace: list[dict[str, Any]] = []
        for cand in seen_candidates:
            cls = str(features[cand]["class"])
            class_info = baseline["composition"].get(cls) or {}
            preference_debt = int(class_info.get("preference_debt", 0))
            quality = float(features[cand]["quality"])
            if quality >= 1.0 or (preference_debt > 0 and quality >= 0.8):
                selected_set.add(cand)
                flexible_trace.append({"step": len(flexible_trace) + 1, "action": "add", "add": cand, "quality": round(quality, 4)})
        selected = sorted(selected_set)
        swap_trace = flexible_trace
        proposal_target_size = len(selected)

    selected_set = set(selected)
    add_candidates = sorted(selected_set & candidate_paths)
    remove_included = sorted(included_paths - selected_set)
    rejected_candidates = sorted(candidate_paths - selected_set)

    baseline_profile = _profile(records, included_paths, guidance, target_size)
    proposed_profile = _profile(records, selected_set, guidance, proposal_target_size)
    weights = _optimizer_weights(guidance)
    baseline_objective, baseline_parts = _dataset_objective(included_paths, features, states, guidance, target_size)
    proposed_objective, proposed_parts = _dataset_objective(selected_set, features, states, guidance, proposal_target_size)

    accepted_by_candidate = {str(item.get("add")): item for item in swap_trace if item.get("add")}
    candidate_decisions: list[dict[str, Any]] = []
    for path in sorted(candidate_paths):
        f = features[path]
        candidate_decisions.append({
            "image": path,
            "class": f["class"],
            "quality": round(float(f["quality"]), 4),
            "decision": "proposed_add" if path in selected_set else "not_selected",
            "accepted_change": accepted_by_candidate.get(path),
            "best_final_swap": None if path in selected_set or args.selection_mode == "flexible" else _best_final_swap(path, selected_set, features, states, guidance, proposal_target_size),
            "fine_action": f["fine_action"],
            "action_qualification": f["action_qualification"],
            "shoulder_geometry": f["shoulder_geometry"],
        })

    for record in records:
        path = str(record.get("relative_path") or "")
        record["selection_workspace_v8_1"] = {
            "state": states[path],
            "proposed_selected": path in selected_set,
            "proposed_action": (
                "add" if states[path] == "candidate" and path in selected_set
                else "remove_or_exclude" if states[path] == "included" and path not in selected_set
                else "retain" if states[path] == "included" and path in selected_set
                else "reject_candidate" if states[path] == "candidate" and path not in selected_set
                else "inactive"
            ),
            "features": features[path],
        }

    payload = {
        "schema_version": "dataset-evidence-8.1",
        "analysis_model": base.get("analysis_model") or model_id,
        "analysis_source": base.get("analysis_source"),
        "dwpose_source": base.get("dwpose_source"),
        "guidance_profile": {
            "profile_id": guidance.get("profile_id"),
            "version": guidance.get("version"),
            "authority": guidance.get("authority"),
            "composition_classes": list((guidance.get("composition") or {}).keys()),
        },
        "selection_workspace": {
            "schema_version": "dataset-selection-workspace-1.0",
            "mode": args.selection_mode,
            "state_counts": dict(sorted(state_counts.items())),
            "initial_active_size": len(included_paths),
            "candidate_pool_size": len(candidate_paths),
            "target_size": proposal_target_size,
            "manifest": str(args.selection_manifest.expanduser().resolve()) if args.selection_manifest else None,
            "candidate_globs": args.candidate_glob,
        },
        "optimizer": {
            "strategy": "incremental_positive_gain",
            "weights": weights,
            "min_swap_gain": weights["min_swap_gain"],
            "baseline_objective": round(baseline_objective, 4),
            "proposed_objective": round(proposed_objective, 4),
            "objective_gain": round(proposed_objective - baseline_objective, 4),
            "baseline_components": baseline_parts,
            "proposed_components": proposed_parts,
        },
        "axis_health": axis_health,
        "baseline_profile": baseline_profile,
        "proposed_profile": proposed_profile,
        "proposal": {
            "selected_paths": selected,
            "add_candidates": add_candidates,
            "remove_included": remove_included,
            "reject_candidates": rejected_candidates,
            "swap_trace": swap_trace,
            "target_size_adjustment_trace": extra_trace,
        },
        "candidate_decisions": candidate_decisions,
        "geometry_sanity_reviews": geometry_reviews,
        "records": records,
        "method_notes": [
            "V8.1 preserves the V8 selection-workspace state model and changes only portfolio reasoning.",
            "minimum_count is a hard safety floor; preferred_share is a soft bounded objective rather than a forced quota.",
            "Preserve-size optimization starts from the active included dataset and applies only positive-gain candidate swaps above a configured churn threshold.",
            "Semantic action signatures are rebuilt from qualified component evidence before selection; quarantined axes and geometry sanity failures cannot leak back through compound labels.",
            "Cached fine-action evidence is selection-authoritative only for high-tier analyzers above confidence threshold; manual workspace overrides remain authoritative at all tiers.",
            "Candidate pool size never changes the guidance denominator until a candidate is actually proposed into the final portfolio.",
        ],
    }

    json_path = run_dir / f"{args.output_prefix}_{slug}.json"
    md_path = run_dir / f"{args.output_prefix}_{slug}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_report(payload), encoding="utf-8")

    for path in temp_paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"Done. JSON:   {json_path}")
    print(f"      Report: {md_path}")
    print(f"Workspace states: {dict(sorted(state_counts.items()))}")
    print(f"Selection mode: {args.selection_mode}; target size: {proposal_target_size}")
    print(f"Objective gain: {proposed_objective - baseline_objective:.4f}")
    print(f"Proposed adds: {add_candidates}")
    print(f"Proposed removals: {remove_included}")
    print(f"Rejected candidates: {rejected_candidates}")
    print("Axis health: " + str({k: v.get("status") for k, v in axis_health.items()}))
    if geometry_reviews:
        print("Geometry sanity reviews: " + str([x["image"] for x in geometry_reviews]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
