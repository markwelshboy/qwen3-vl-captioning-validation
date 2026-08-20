from __future__ import annotations

import argparse
import fnmatch
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .runner import model_slug, resolve_model_id


DEFAULT_GUIDANCE = Path(__file__).resolve().parents[1] / "guidance_profiles" / "identity_lora_balanced_v1.json"
VALID_STATES = {"included", "candidate", "excluded", "superseded"}
AXES = ("head_yaw", "head_pitch", "head_roll", "torso")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "V8 dataset-selection workspace. Keeps included/candidate/excluded state separate "
            "from image evidence, evaluates candidates against the active dataset denominator, "
            "and proposes a portfolio without committing selection changes."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence_v8", help="Output basename inside run_dir.")
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
        help="preserve_size keeps the initial included count; target_size uses --target-size; flexible recommends additive candidates without forced removals.",
    )
    parser.add_argument("--target-size", type=int, help="Final portfolio size for target_size mode.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_guidance(path: Path) -> dict[str, Any]:
    payload = _read_json(path.expanduser().resolve())
    if payload.get("schema_version") != "dataset-guidance-profile-1.0":
        raise ValueError(f"Unsupported guidance schema: {payload.get('schema_version')!r}")
    return payload


def _run_v7_base(args: argparse.Namespace, run_dir: Path, slug: str) -> tuple[dict[str, Any], list[Path]]:
    if args.base_v7_json:
        return _read_json(args.base_v7_json.expanduser().resolve()), []

    internal_prefix = f".__v8_base_{args.output_prefix}"
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


def _load_selection_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _read_json(path.expanduser().resolve())
    if payload.get("schema_version") != "dataset-selection-workspace-1.0":
        raise ValueError(f"Unsupported selection manifest schema: {payload.get('schema_version')!r}")
    return payload


def _manifest_entry_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("entries") or []:
        path = str(entry.get("path") or "")
        if path:
            out[path] = entry
    return out


def _assign_selection_state(
    record: dict[str, Any],
    manifest: dict[str, Any],
    entries: dict[str, dict[str, Any]],
    candidate_globs: list[str],
) -> tuple[str, dict[str, Any]]:
    rel = str(record.get("relative_path") or "")
    entry = entries.get(rel, {})
    if entry:
        state = str(entry.get("state") or manifest.get("default_state") or "included")
    elif any(fnmatch.fnmatch(rel, pat) for pat in candidate_globs):
        state = "candidate"
    else:
        state = str(manifest.get("default_state") or "included")
    if state not in VALID_STATES:
        raise ValueError(f"Invalid selection state {state!r} for {rel}")
    return state, entry


def _composition_floor(rule: dict[str, Any], n: int) -> int:
    low = float((rule.get("preferred_share") or [0.0, 1.0])[0])
    return max(int(rule.get("minimum_count", 0)), int(math.ceil(low * n)))


def _composition_upper(rule: dict[str, Any], n: int) -> int:
    high = float((rule.get("preferred_share") or [0.0, 1.0])[1])
    by_share = int(math.ceil(high * n))
    soft_cap = int(rule.get("soft_cap_count", by_share))
    return max(_composition_floor(rule, n), min(by_share, soft_cap))


def _axis_health(records: list[dict[str, Any]], tier: str) -> dict[str, Any]:
    # Authority is calibrated against the ACTIVE set, not the candidate pool. This prevents
    # candidate audition from flipping an analyzer axis across a brittle threshold.
    threshold = {"high": 0.95, "medium": 0.90, "low": 0.85}.get(tier, 0.90)
    result: dict[str, Any] = {}
    for axis in AXES:
        values: list[str] = []
        for r in records:
            pose = r.get("facial_pose") or {}
            key = "torso_yaw" if axis == "torso" else axis
            value = pose.get(key)
            if value and value != "unknown":
                values.append(str(value))
        counts = Counter(values)
        dominant, dominant_count = (counts.most_common(1)[0] if counts else (None, 0))
        share = dominant_count / len(values) if values else 0.0
        quarantined = len(values) >= 12 and share >= threshold
        result[axis] = {
            "status": "quarantined" if quarantined else "available_qualified",
            "sample_count": len(values),
            "dominant_value": dominant,
            "dominant_count": dominant_count,
            "dominant_share": round(share, 3),
            "threshold": threshold,
            "basis": "initial_active_dataset",
        }
    result["face_pose"] = {
        "status": (
            "available_qualified"
            if result["head_yaw"]["status"] != "quarantined" and result["head_pitch"]["status"] != "quarantined"
            else "quarantined"
        ),
        "basis": "depends_on_head_yaw_and_head_pitch",
    }
    return result


def _normalize_line_angle(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"raw_deg": None, "normalized_deg": None, "abs_cant_deg": None, "status": "unavailable"}
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return {"raw_deg": raw, "normalized_deg": None, "abs_cant_deg": None, "status": "invalid"}
    normalized = ((value + 90.0) % 180.0) - 90.0
    abs_cant = abs(normalized)
    # Very large projected shoulder cants are possible, but in this harness they are also a
    # useful detector/keypoint-assignment sanity warning. Do not let them provide rarity
    # protection until reviewed.
    status = "sanity_review" if abs_cant > 45.0 else "usable"
    return {
        "raw_deg": round(value, 2),
        "normalized_deg": round(normalized, 2),
        "abs_cant_deg": round(abs_cant, 2),
        "status": status,
    }


def _fine_action(record: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    override = ((entry.get("overrides") or {}).get("fine_action")) if entry else None
    if override:
        return {"value": str(override), "confidence": 1.0, "source": "manual_override", "usable_for_selection": True}

    action = record.get("action_contact") or {}
    evidence = action.get("evidence") or []
    best: tuple[str | None, float] = (None, 0.0)
    for ev in evidence:
        tag = str(ev.get("tag") or "")
        conf = float(ev.get("confidence") or 0.0)
        if tag == "hands_in_pockets" and conf > best[1]:
            best = ("hands_in_pockets", conf)
        elif tag == "hand_at_hip" and conf > best[1]:
            best = ("hands_on_hips", conf)
    value, confidence = best
    return {
        "value": value,
        "confidence": round(confidence, 3),
        "source": "cached_action_evidence" if value else "unavailable",
        "usable_for_selection": bool(value and confidence >= 0.85),
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

    action = record.get("action_contact") or {}
    semantic_action_allowed = tier == "high"
    action_signature = str(action.get("action_signature") or "none") if semantic_action_allowed else None
    if action_signature == "none":
        action_signature = None
    fine_action = _fine_action(record, entry)

    geom = ((record.get("pose_evidence") or {}).get("target_2d_geometry") or {})
    shoulder = _normalize_line_angle(geom.get("shoulder_line_angle_from_horizontal_deg"))
    geometry_class = None
    if shoulder["status"] == "usable" and shoulder["abs_cant_deg"] is not None and shoulder["abs_cant_deg"] >= 15.0:
        geometry_class = "strong_shoulder_cant"

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
        "action_signature": action_signature,
        "fine_action": fine_action,
        "geometry_class": geometry_class,
        "shoulder_geometry": shoulder,
        "environment": environment,
        "illumination": illumination,
    }


def _novelty_score(features: dict[str, Any], seen: dict[str, set[str]], weights: dict[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    fields = [
        ("head_yaw", float(weights.get("new_head_yaw", 0.4))),
        ("head_pitch", float(weights.get("new_head_pitch", 0.2))),
        ("action_signature", float(weights.get("new_action_signature", 0.2))),
        ("geometry_class", float(weights.get("new_geometry_class", 0.12))),
        ("environment", 0.08),
        ("illumination", 0.05),
    ]
    for field, weight in fields:
        value = features.get(field)
        if value and value not in seen[field]:
            score += weight
            reasons.append(f"new {field}={value}")
    fine = features.get("fine_action") or {}
    if fine.get("usable_for_selection") and fine.get("value") and fine["value"] not in seen["fine_action"]:
        score += 0.25
        reasons.append(f"new fine_action={fine['value']}")
    return score, reasons


def _add_seen(features: dict[str, Any], seen: dict[str, set[str]]) -> None:
    for field in seen:
        if field == "fine_action":
            fine = features.get("fine_action") or {}
            value = fine.get("value") if fine.get("usable_for_selection") else None
        else:
            value = features.get(field)
        if value:
            seen[field].add(str(value))


def _blank_seen() -> dict[str, set[str]]:
    return {k: set() for k in ("head_yaw", "head_pitch", "action_signature", "fine_action", "geometry_class", "environment", "illumination")}


def _profile(records: list[dict[str, Any]], selected_paths: set[str], guidance: dict[str, Any], target_n: int) -> dict[str, Any]:
    selected = [r for r in records if str(r.get("relative_path")) in selected_paths]
    counts = Counter(str(((r.get("guidance_composition") or {}).get("class")) or "unknown") for r in selected)
    composition: dict[str, Any] = {}
    for cls, rule in (guidance.get("composition") or {}).items():
        floor = _composition_floor(rule, target_n)
        upper = _composition_upper(rule, target_n)
        current = int(counts.get(cls, 0))
        composition[cls] = {
            "count": current,
            "floor": floor,
            "preferred_upper": upper,
            "debt": max(0, floor - current),
            "status": "deficit" if current < floor else ("surplus" if current > upper else "within_guidance"),
        }
    return {"image_count": len(selected), "composition": composition}


def _select_portfolio(
    records: list[dict[str, Any]],
    states: dict[str, str],
    features: dict[str, dict[str, Any]],
    guidance: dict[str, Any],
    target_size: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    available = [r for r in records if states[str(r.get("relative_path"))] in {"included", "candidate"}]
    if target_size > len(available):
        target_size = len(available)
    weights = guidance.get("selection_weights") or {}
    selected: list[str] = []
    trace: list[dict[str, Any]] = []
    seen = _blank_seen()
    class_counts: Counter[str] = Counter()
    rules = guidance.get("composition") or {}

    def choose_one(pool: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
        best: tuple[float, dict[str, Any], list[str]] | None = None
        for r in pool:
            path = str(r.get("relative_path"))
            if path in selected:
                continue
            f = features[path]
            novelty, reasons = _novelty_score(f, seen, weights)
            stability = 0.03 if states[path] == "included" else 0.0
            score = float(f["quality"]) * float(weights.get("measured_quality", 1.0)) + novelty + stability
            cls = str(f["class"])
            if phase == "fill" and cls in rules:
                upper = _composition_upper(rules[cls], target_size)
                if class_counts[cls] >= upper:
                    score -= 0.45 + 0.12 * (class_counts[cls] - upper)
            if best is None or score > best[0]:
                best = (score, r, reasons)
        if best is None:
            return None
        score, r, reasons = best
        path = str(r.get("relative_path"))
        selected.append(path)
        class_counts[str(features[path]["class"])] += 1
        _add_seen(features[path], seen)
        trace.append({
            "selection_step": len(selected),
            "phase": phase,
            "image": path,
            "original_state": states[path],
            "score": round(score, 4),
            "quality": round(float(features[path]["quality"]), 4),
            "novelty_reasons": reasons,
        })
        return r

    # Phase 1: meet guidance floors using the best diversity-aware set available in each class.
    for cls, rule in rules.items():
        floor = _composition_floor(rule, target_size)
        pool = [r for r in available if features[str(r.get("relative_path"))]["class"] == cls]
        while class_counts[cls] < floor:
            if choose_one(pool, f"floor:{cls}") is None:
                break

    # Phase 2: fill remaining slots by marginal quality/diversity while discouraging avoidable surplus.
    while len(selected) < target_size:
        if choose_one(available, "fill") is None:
            break

    return selected, trace


def _build_swaps(selected_candidates: list[str], removed_included: list[str], features: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = list(removed_included)
    swaps: list[dict[str, Any]] = []
    for cand in selected_candidates:
        same = [p for p in remaining if features[p]["class"] == features[cand]["class"]]
        pool = same if same else remaining
        if not pool:
            break
        donor = min(pool, key=lambda p: float(features[p]["quality"]))
        remaining.remove(donor)
        swaps.append({
            "add": cand,
            "remove": donor,
            "same_composition_class": features[donor]["class"] == features[cand]["class"],
            "add_class": features[cand]["class"],
            "remove_class": features[donor]["class"],
            "add_quality": round(float(features[cand]["quality"]), 4),
            "remove_quality": round(float(features[donor]["quality"]), 4),
        })
    return swaps


def _render_report(payload: dict[str, Any]) -> str:
    ws = payload["selection_workspace"]
    baseline = payload["baseline_profile"]
    proposed = payload["proposed_profile"]
    lines = [
        "# Dataset evidence report — selection workspace v8",
        "",
        f"> Analysis source: **{payload['analysis_model']}** · mode **{ws['mode']}**.",
        f"> Guidance profile: **{payload['guidance_profile']['profile_id']}** · authority **{payload['guidance_profile']['authority']}**.",
        "> Candidate evaluation does not change the active-dataset denominator until a selection is applied.",
        "",
        "## Workspace",
        "",
        f"- Included: **{ws['state_counts'].get('included', 0)}**",
        f"- Candidates: **{ws['state_counts'].get('candidate', 0)}**",
        f"- Excluded: **{ws['state_counts'].get('excluded', 0)}**",
        f"- Superseded: **{ws['state_counts'].get('superseded', 0)}**",
        f"- Target portfolio size: **{ws['target_size']}**",
        "",
        "## Composition — active vs proposed",
        "",
        "| class | active | floor | proposed | proposed status |",
        "|---|---:|---:|---:|---|",
    ]
    for cls in payload["guidance_profile"]["composition_classes"]:
        b = baseline["composition"].get(cls, {})
        p = proposed["composition"].get(cls, {})
        lines.append(f"| {cls} | {b.get('count', 0)} | {p.get('floor', 0)} | {p.get('count', 0)} | {p.get('status', 'n/a')} |")

    lines += ["", "## Proposed changes", ""]
    added = payload["proposal"]["add_candidates"]
    removed = payload["proposal"]["remove_included"]
    lines.append(f"- Add candidates: **{len(added)}** — {', '.join(added) if added else 'none'}")
    lines.append(f"- Remove/exclude included: **{len(removed)}** — {', '.join(removed) if removed else 'none'}")
    lines.append("")
    if payload["proposal"]["swaps"]:
        lines += ["### Suggested swap pairing", "", "| add | remove | same class |", "|---|---|---|"]
        for swap in payload["proposal"]["swaps"]:
            lines.append(f"| {swap['add']} | {swap['remove']} | {swap['same_composition_class']} |")
        lines.append("")

    lines += ["## Candidate decisions", "", "| image | class | quality | decision | fine action | shoulder geometry |", "|---|---|---:|---|---|---|"]
    for row in payload["candidate_decisions"]:
        shoulder = row["shoulder_geometry"]
        shoulder_text = "n/a" if shoulder.get("abs_cant_deg") is None else f"{shoulder['abs_cant_deg']:.2f}° ({shoulder['status']})"
        fine = row["fine_action"]
        fine_text = fine.get("value") or "unassessed"
        lines.append(f"| {row['image']} | {row['class']} | {row['quality']:.3f} | {row['decision']} | {fine_text} | {shoulder_text} |")

    lines += ["", "## Analyzer authority anchored to active dataset", ""]
    for axis, info in payload["axis_health"].items():
        if axis == "face_pose":
            lines.append(f"- {axis}: **{info['status']}**")
        else:
            lines.append(f"- {axis}: **{info['status']}** (dominant={info.get('dominant_value')}, share={info.get('dominant_share')})")

    if payload["geometry_sanity_reviews"]:
        lines += ["", "## Geometry sanity reviews", ""]
        for item in payload["geometry_sanity_reviews"]:
            lines.append(f"- **{item['image']}**: raw shoulder angle {item['raw_deg']}° → undirected normalized {item['normalized_deg']}°; excluded from geometry novelty until reviewed.")

    lines += [
        "",
        "## V8 interpretation notes",
        "",
        "- `included`, `candidate`, `excluded`, and `superseded` are workspace states, not intrinsic image labels.",
        "- The proposal is a what-if portfolio; v8 never mutates the dataset or selection manifest.",
        "- Guidance floors are calculated from the requested final portfolio size, not from the number of images auditioning in the candidate pool.",
        "- Analyzer-axis health is anchored to the initial active set so adding candidates cannot accidentally rehabilitate a previously degenerate axis.",
        "- Shoulder/hip line orientation is treated as undirected geometry; extreme normalized cant is flagged for review rather than used as rarity protection.",
        "- Fine action/contact is hierarchical and confidence-gated. Manual overrides are supported in the selection manifest; Analyze v2 should provide these fields directly.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    guidance = _load_guidance(args.guidance_profile)
    manifest = _load_selection_manifest(args.selection_manifest)
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    base, temp_paths = _run_v7_base(args, run_dir, slug)
    records = list(base.get("records") or [])
    if not records:
        print("V7 base contains no records", file=sys.stderr)
        return 2

    entries = _manifest_entry_map(manifest)
    states: dict[str, str] = {}
    record_entries: dict[str, dict[str, Any]] = {}
    for record in records:
        path = str(record.get("relative_path") or "")
        state, entry = _assign_selection_state(record, manifest, entries, args.candidate_glob)
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
    axis_health = _axis_health(active_records, tier)
    features: dict[str, dict[str, Any]] = {}
    geometry_reviews: list[dict[str, Any]] = []
    for record in records:
        path = str(record.get("relative_path") or "")
        f = _record_features(record, record_entries[path], axis_health, guidance, tier)
        features[path] = f
        if f["shoulder_geometry"]["status"] == "sanity_review":
            geometry_reviews.append({"image": path, **f["shoulder_geometry"]})

    if args.selection_mode == "flexible":
        # Flexible mode never forces an incumbent removal. Add candidates only when they pay an
        # active composition debt or introduce high-confidence novel evidence with adequate quality.
        baseline_profile = _profile(records, included_paths, guidance, len(included_paths))
        seen = _blank_seen()
        for path in included_paths:
            _add_seen(features[path], seen)
        add: list[str] = []
        for path in sorted(candidate_paths, key=lambda p: float(features[p]["quality"]), reverse=True):
            cls = features[path]["class"]
            debt = int((baseline_profile["composition"].get(cls) or {}).get("debt", 0))
            novelty, _ = _novelty_score(features[path], seen, guidance.get("selection_weights") or {})
            if debt > 0 or (float(features[path]["quality"]) >= 1.0 and novelty >= 0.2):
                add.append(path)
                _add_seen(features[path], seen)
        selected = sorted(included_paths | set(add))
        trace: list[dict[str, Any]] = []
        proposal_target_size = len(selected)
    else:
        selected, trace = _select_portfolio(records, states, features, guidance, target_size)
        proposal_target_size = target_size

    selected_set = set(selected)
    add_candidates = sorted(selected_set & candidate_paths)
    remove_included = sorted(included_paths - selected_set)
    rejected_candidates = sorted(candidate_paths - selected_set)
    swaps = _build_swaps(add_candidates, remove_included, features)

    baseline_profile = _profile(records, included_paths, guidance, target_size)
    proposed_profile = _profile(records, selected_set, guidance, proposal_target_size)

    candidate_decisions: list[dict[str, Any]] = []
    for path in sorted(candidate_paths):
        f = features[path]
        candidate_decisions.append({
            "image": path,
            "class": f["class"],
            "quality": round(float(f["quality"]), 4),
            "decision": "proposed_add" if path in selected_set else "not_selected",
            "fine_action": f["fine_action"],
            "shoulder_geometry": f["shoulder_geometry"],
        })

    for record in records:
        path = str(record.get("relative_path") or "")
        record["selection_workspace_v8"] = {
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
        "schema_version": "dataset-evidence-8.0",
        "analysis_model": base.get("analysis_model") or model_id,
        "analysis_source": base.get("analysis_source"),
        "dwpose_source": base.get("dwpose_source"),
        "guidance_profile": {
            "profile_id": guidance.get("profile_id"),
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
        "axis_health": axis_health,
        "baseline_profile": baseline_profile,
        "proposed_profile": proposed_profile,
        "proposal": {
            "selected_paths": selected,
            "add_candidates": add_candidates,
            "remove_included": remove_included,
            "reject_candidates": rejected_candidates,
            "swaps": swaps,
            "selection_trace": trace,
        },
        "candidate_decisions": candidate_decisions,
        "geometry_sanity_reviews": geometry_reviews,
        "records": records,
        "method_notes": [
            "V8 models dataset membership as workspace state: included, candidate, excluded, or superseded.",
            "Candidate images are evaluated against the active included dataset; they do not change guidance denominators until selected.",
            "Preserve-size and target-size modes create a non-destructive proposed portfolio; flexible mode can recommend additive candidates without forced donor removal.",
            "Analyzer-axis degeneracy is calibrated on the initial active set so candidate audition cannot cross a brittle distribution threshold and silently change evidence authority.",
            "DWPose shoulder line angles are normalized as undirected line geometry into [-90, 90); extreme projected cants are sanity-reviewed and excluded from novelty protection.",
            "Fine hand/action semantics are hierarchical and confidence-gated; optional manifest overrides support human correction without changing cached image analysis.",
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
    print(f"Proposed adds: {add_candidates}")
    print(f"Proposed removals: {remove_included}")
    print(f"Rejected candidates: {rejected_candidates}")
    print("Axis health: " + str({k: v.get("status") for k, v in axis_health.items()}))
    if geometry_reviews:
        print("Geometry sanity reviews: " + str([x["image"] for x in geometry_reviews]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
