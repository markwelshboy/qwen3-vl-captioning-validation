from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset_evidence_v3 import _present_recommendation
from .runner import model_slug, resolve_model_id


DEFAULT_GUIDANCE = Path(__file__).resolve().parents[1] / "guidance_profiles" / "identity_lora_balanced_v1.json"
STANDARD_GUIDANCE_CLASSES = ("identity_close", "upper_body", "full_body")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "Guidance-aware dataset optimizer layered on source-calibrated v5 evidence. "
            "Separates measured image quality from heuristic composition guidance, diversity debt, and quality debt."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence_v6", help="Output basename inside run_dir.")
    parser.add_argument(
        "--guidance-profile",
        type=Path,
        default=DEFAULT_GUIDANCE,
        help=f"Guidance profile JSON (default: {DEFAULT_GUIDANCE}).",
    )
    parser.add_argument(
        "--base-v5-json",
        type=Path,
        help="Optional existing v5 JSON to post-process instead of regenerating the deterministic v5 base.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_guidance(path: Path) -> dict[str, Any]:
    payload = _read_json(path.expanduser().resolve())
    if payload.get("schema_version") != "dataset-guidance-profile-1.0":
        raise ValueError(f"Unsupported guidance schema: {payload.get('schema_version')!r}")
    for cls in STANDARD_GUIDANCE_CLASSES:
        if cls not in (payload.get("composition") or {}):
            raise ValueError(f"Guidance profile is missing composition class {cls!r}")
    return payload


def _run_v5_base(args: argparse.Namespace, run_dir: Path, slug: str) -> tuple[dict[str, Any], list[Path]]:
    if args.base_v5_json:
        path = args.base_v5_json.expanduser().resolve()
        return _read_json(path), []

    internal_prefix = f".__v6_base_{args.output_prefix}"
    cmd = [
        sys.executable,
        "-m",
        "qwen_caption_validate.dataset_evidence_v5",
        str(run_dir),
        "--model",
        args.model,
        "--output-prefix",
        internal_prefix,
    ]
    if args.dwpose_dir:
        cmd.extend(["--dwpose-dir", str(args.dwpose_dir.expanduser().resolve())])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"v5 base profiler failed with exit code {proc.returncode}")

    base_json = run_dir / f"{internal_prefix}_{slug}.json"
    base_md = run_dir / f"{internal_prefix}_{slug}.md"
    if not base_json.exists():
        raise RuntimeError(f"v5 base profiler did not create expected file: {base_json}")
    return _read_json(base_json), [base_json, base_md]


def _guidance_class(record: dict[str, Any], capability: dict[str, Any]) -> dict[str, str]:
    framing = record.get("framing_fusion") or {}
    signal = record.get("signal") or {}
    extent = str(framing.get("dwpose_extent_hint") or "unknown")
    shot = str(signal.get("effective_shot_scale") or "unknown")
    tier = str(capability.get("policy_tier") or "medium")

    if extent == "full_length" or signal.get("trusted_composition_class") == "full_body":
        return {"class": "full_body", "authority": "high_visible_extent", "source": "dwpose_extent"}
    if extent in {"waist_or_upper_body", "three_quarter_or_long"}:
        return {"class": "upper_body", "authority": "high_visible_extent", "source": "dwpose_extent"}

    if extent == "close_or_medium_close":
        if tier == "low":
            return {
                "class": "close_or_medium_close_unresolved",
                "authority": "unassessed_split",
                "source": "dwpose_extent_cannot_resolve_close_vs_medium_close",
            }
        if shot in {"medium_close_up", "medium", "three_quarter"}:
            return {"class": "upper_body", "authority": "qualified_framing_fusion", "source": "qwen+dwpose_framing"}
        if shot in {"close_up", "extreme_close_up"}:
            return {"class": "identity_close", "authority": "qualified_framing_fusion", "source": "qwen+dwpose_framing"}

    if shot in {"medium_close_up", "medium", "three_quarter"}:
        return {"class": "upper_body", "authority": "qualified_qwen_framing", "source": "qwen_framing"}
    if shot in {"close_up", "extreme_close_up"}:
        return {"class": "identity_close", "authority": "qualified_qwen_framing", "source": "qwen_framing"}
    return {"class": "unknown", "authority": "unknown", "source": "unresolved"}


def _target_window(rule: dict[str, Any], n: int) -> dict[str, int | float]:
    shares = rule.get("preferred_share") or [0.0, 1.0]
    low_share = float(shares[0])
    high_share = float(shares[1])
    minimum = int(rule.get("minimum_count") or 0)
    soft_cap = int(rule.get("soft_cap_count") or max(n, minimum))

    share_floor = int(math.ceil(low_share * n))
    share_ceiling = int(math.ceil(high_share * n))
    floor = max(minimum, min(share_floor, soft_cap))
    ceiling = max(floor, min(max(share_ceiling, minimum), soft_cap))
    return {
        "preferred_share_low": low_share,
        "preferred_share_high": high_share,
        "minimum_count": minimum,
        "soft_cap_count": soft_cap,
        "guidance_floor_count": floor,
        "guidance_ceiling_count": ceiling,
    }


def _status_for_count(cls: str, count: int, unresolved_count: int, window: dict[str, Any]) -> dict[str, Any]:
    floor = int(window["guidance_floor_count"])
    ceiling = int(window["guidance_ceiling_count"])
    eligible_unresolved = unresolved_count if cls in {"identity_close", "upper_body"} else 0
    possible_max = count + eligible_unresolved

    if count > ceiling:
        status = "surplus"
    elif possible_max < floor:
        status = "deficit"
    elif count < floor <= possible_max:
        status = "partially_unassessed"
    elif count < floor:
        status = "deficit"
    else:
        status = "within_guidance"

    return {
        "status": status,
        "coverage_debt_count": max(0, floor - possible_max) if status == "deficit" else 0,
        "known_shortfall_count": max(0, floor - count),
        "surplus_count": max(0, count - ceiling),
        "possible_count_including_unresolved": possible_max,
    }


def _feature_availability(base: dict[str, Any], capability: dict[str, Any]) -> dict[str, bool]:
    availability = base.get("coverage_availability") or {}
    semantic_actions = str(capability.get("semantic_action_protection") or "") != "report_only_no_dataset_protection"
    return {
        "head_yaw": str((availability.get("head_yaw") or {}).get("status") or "").startswith("available"),
        "head_pitch": str((availability.get("head_pitch") or {}).get("status") or "").startswith("available"),
        "action_signature": semantic_actions,
        "geometry_class": True,
    }


def _feature_value(record: dict[str, Any], feature: str, available: dict[str, bool]) -> str | None:
    if not available.get(feature, False):
        return None
    if feature == "head_yaw":
        value = str((record.get("coverage_signature") or {}).get("head_yaw") or "unknown")
        return None if value == "unknown" else value
    if feature == "head_pitch":
        value = str((record.get("coverage_signature") or {}).get("head_pitch") or "unknown")
        return None if value == "unknown" else value
    if feature == "action_signature":
        value = str((record.get("action_contact") or {}).get("action_signature") or "none")
        return None if value == "none" else value
    if feature == "geometry_class":
        components = set((record.get("action_contact") or {}).get("signature_components") or [])
        return "strong_shoulder_cant" if "strong_shoulder_cant" in components else None
    return None


def _base_candidate_score(record: dict[str, Any], weights: dict[str, Any]) -> float:
    measured = float((record.get("measured_signal_density") or {}).get("score") or 0.0)
    score = measured * float(weights.get("measured_quality", 1.0))
    sig = record.get("coverage_signature") or {}
    yaw = str(sig.get("head_yaw") or "unknown")
    pitch = str(sig.get("head_pitch") or "unknown")
    if yaw in {"strong_left", "strong_right"}:
        score += float(weights.get("strong_yaw_salience", 0.0))
    if pitch in {"strong_up", "strong_down"}:
        score += float(weights.get("strong_pitch_salience", 0.0))
    geom = (record.get("pose_evidence") or {}).get("target_2d_geometry") or {}
    angle = geom.get("shoulder_line_angle_from_horizontal_deg")
    if angle is not None:
        try:
            magnitude = min(abs(float(angle)) / 30.0, 1.0)
            score += magnitude * float(weights.get("strong_shoulder_cant_salience", 0.0))
        except (TypeError, ValueError):
            pass
    return score


def _novelty_score(
    record: dict[str, Any],
    selected: list[dict[str, Any]],
    available: dict[str, bool],
    weights: dict[str, Any],
) -> tuple[float, list[str]]:
    if not selected:
        return 0.0, []
    mapping = {
        "head_yaw": "new_head_yaw",
        "head_pitch": "new_head_pitch",
        "action_signature": "new_action_signature",
        "geometry_class": "new_geometry_class",
    }
    bonus = 0.0
    reasons: list[str] = []
    for feature, weight_key in mapping.items():
        value = _feature_value(record, feature, available)
        if value is None:
            continue
        seen = {_feature_value(other, feature, available) for other in selected}
        if value not in seen:
            bonus += float(weights.get(weight_key, 0.0))
            reasons.append(f"new {feature}={value}")
    return bonus, reasons


def _select_representatives(
    candidates: list[dict[str, Any]],
    slots: int,
    available: dict[str, bool],
    weights: dict[str, Any],
) -> list[dict[str, Any]]:
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < slots:
        scored: list[tuple[float, float, list[str], dict[str, Any]]] = []
        for record in remaining:
            base = _base_candidate_score(record, weights)
            novelty, novelty_reasons = _novelty_score(record, selected, available, weights)
            scored.append((base + novelty, base, novelty_reasons, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        total, base, novelty_reasons, winner = scored[0]
        winner.setdefault("guidance_selection_trace", []).append(
            {
                "selection_step": len(selected) + 1,
                "total_score": round(total, 4),
                "base_quality_salience_score": round(base, 4),
                "novelty_reasons": novelty_reasons,
            }
        )
        selected.append(winner)
        remaining.remove(winner)
    return selected


def _diversity_assessment(
    selected: list[dict[str, Any]],
    rule: dict[str, Any],
    available: dict[str, bool],
) -> dict[str, Any]:
    requested = ((rule.get("diversity") or {}).get("minimum_distinct") or {})
    out: dict[str, Any] = {}
    for feature, minimum in requested.items():
        minimum = int(minimum)
        if not available.get(feature, False):
            out[feature] = {
                "status": "unassessed_from_source",
                "minimum_distinct": minimum,
                "distinct_count": None,
                "values": [],
                "diversity_debt": None,
            }
            continue
        values = sorted({v for r in selected if (v := _feature_value(r, feature, available)) is not None})
        effective_minimum = min(minimum, len(selected))
        debt = max(0, effective_minimum - len(values))
        out[feature] = {
            "status": "debt" if debt else "satisfied",
            "minimum_distinct": minimum,
            "effective_minimum_for_selected_count": effective_minimum,
            "distinct_count": len(values),
            "values": values,
            "diversity_debt": debt,
        }
    return out


def _build_guidance_layer(base: dict[str, Any], guidance: dict[str, Any]) -> dict[str, Any]:
    records = base.get("records") or []
    capability = base.get("analysis_capability_policy") or {}
    n = len(records)
    available = _feature_availability(base, capability)
    weights = guidance.get("selection_weights") or {}

    class_counts: Counter[str] = Counter()
    for record in records:
        classification = _guidance_class(record, capability)
        record["guidance_composition"] = classification
        class_counts[classification["class"]] += 1
        record["guidance_protection"] = []
        record["guidance_swap_candidate"] = False
        record.pop("guidance_selection_trace", None)

    unresolved = class_counts.get("close_or_medium_close_unresolved", 0)
    summaries: dict[str, Any] = {}
    all_selected: set[int] = set()

    for cls in STANDARD_GUIDANCE_CLASSES:
        rule = (guidance.get("composition") or {})[cls]
        window = _target_window(rule, n)
        current = class_counts.get(cls, 0)
        status = _status_for_count(cls, current, unresolved, window)
        candidates = [r for r in records if (r.get("guidance_composition") or {}).get("class") == cls]
        slots = min(current, int(window["guidance_floor_count"]))
        selected = _select_representatives(candidates, slots, available, weights)
        selected_ids = {id(r) for r in selected}
        all_selected.update(selected_ids)

        for rank, record in enumerate(selected, 1):
            record["guidance_protection"].append(
                {
                    "dimension": "guidance_composition",
                    "value": cls,
                    "profile_id": guidance.get("profile_id"),
                    "rank": rank,
                    "protected_slots": slots,
                    "guidance_floor_count": int(window["guidance_floor_count"]),
                    "class_status": status["status"],
                    "authority": guidance.get("authority"),
                }
            )

        low_quality = [
            r["relative_path"]
            for r in selected
            if (r.get("measured_signal_density") or {}).get("label") == "low"
        ]
        diversity_rule = {"diversity": (guidance.get("diversity") or {}).get(cls, {})}
        diversity = _diversity_assessment(selected, diversity_rule, available)
        summaries[cls] = {
            "label": rule.get("label", cls),
            "current_count": current,
            "current_share": round(current / n, 4) if n else 0.0,
            **window,
            **status,
            "selected_representatives": [r["relative_path"] for r in selected],
            "selected_representative_count": len(selected),
            "quality_debt_count": len(low_quality),
            "quality_debt_images": low_quality,
            "diversity": diversity,
            "notes": rule.get("notes"),
        }

    # A surplus class can fund swaps into a deficit class. Existing v5 coverage
    # protection and v6 guidance protection both veto automatic swap candidacy.
    for record in records:
        cls = (record.get("guidance_composition") or {}).get("class")
        cls_summary = summaries.get(str(cls))
        if not cls_summary or cls_summary["status"] != "surplus":
            continue
        if record.get("protected_dimensions") or record.get("guidance_protection"):
            continue
        record["guidance_swap_candidate"] = True

    deficits = [
        {
            "class": cls,
            "label": summary["label"],
            "coverage_debt_count": summary["coverage_debt_count"],
            "known_shortfall_count": summary["known_shortfall_count"],
        }
        for cls, summary in summaries.items()
        if summary["status"] == "deficit"
    ]
    swap_candidates = sorted(
        [r for r in records if r.get("guidance_swap_candidate")],
        key=lambda r: float((r.get("measured_signal_density") or {}).get("score") or 0.0),
    )
    swaps: list[dict[str, Any]] = []
    cursor = 0
    for deficit in deficits:
        for _ in range(int(deficit["coverage_debt_count"])):
            if cursor >= len(swap_candidates):
                break
            donor = swap_candidates[cursor]
            cursor += 1
            swaps.append(
                {
                    "replace_or_retire": donor["relative_path"],
                    "from_class": (donor.get("guidance_composition") or {}).get("class"),
                    "add_class": deficit["class"],
                    "reason": "rebalance a guidance-surplus class toward a guidance-deficit class while keeping dataset size roughly fixed",
                }
            )

    return {
        "profile_id": guidance.get("profile_id"),
        "profile_version": guidance.get("version"),
        "authority": guidance.get("authority"),
        "training_objective": guidance.get("training_objective"),
        "description": guidance.get("description"),
        "policy_notes": guidance.get("policy_notes") or [],
        "image_count": n,
        "raw_guidance_class_counts": dict(sorted(class_counts.items())),
        "unresolved_close_vs_medium_count": unresolved,
        "feature_availability_for_diversity": available,
        "composition": summaries,
        "coverage_debts": deficits,
        "fixed_size_rebalance_suggestions": swaps,
    }


def _recompute_recommendations(base: dict[str, Any], guidance_layer: dict[str, Any]) -> None:
    capability = base.get("analysis_capability_policy") or {}
    for record in base.get("records") or []:
        measured = record.get("measured_signal_density") or {}
        low = measured.get("label") == "low"
        protected_v5 = bool(record.get("protected_dimensions"))
        protected_guidance = bool(record.get("guidance_protection"))
        reasons: list[str] = []
        replacement: list[str] = []

        if low and (protected_v5 or protected_guidance):
            base_label = "keep_until_cleaner_equivalent"
            reasons.append("low model-independent measured signal-density proxy")
            if protected_guidance:
                reasons.append("guidance policy currently needs this image as a protected composition representative")
            if protected_v5:
                reasons.append("existing trusted evidence coverage also selects this image as a representative")
            replacement.extend(
                [
                    "replace only with an image that preserves or improves the protected guidance/evidence coverage",
                    "prefer higher measured signal density and equal-or-better within-bucket diversity",
                ]
            )
        elif low:
            base_label = "replace_candidate"
            reasons.extend(
                [
                    "low model-independent measured signal-density proxy",
                    "not currently required by trusted coverage or the active guidance floor",
                ]
            )
            replacement.append("prefer a cleaner image that fills a guidance debt or improves trusted diversity")
        else:
            base_label = "keep"
            reasons.append("model-independent measured signal density is not currently low enough to justify replacement")
            if protected_guidance:
                reasons.append("selected as a guidance composition representative")
            if protected_v5:
                reasons.append("selected as a trusted evidence representative")

        record["base_recommendation"] = base_label
        record["presented_recommendation"] = _present_recommendation(base_label, capability)
        record["recommendation_reasons"] = reasons
        record["replacement_target"] = replacement

        cls = (record.get("guidance_composition") or {}).get("class")
        cls_summary = (guidance_layer.get("composition") or {}).get(str(cls)) or {}
        if record.get("guidance_swap_candidate"):
            record["guidance_action"] = {
                "label": "eligible_swap_candidate",
                "reason": f"{cls} is above the active guidance band and this image is not required as a protected representative",
            }
        elif protected_guidance and low:
            record["guidance_action"] = {
                "label": "retain_until_guidance_equivalent",
                "reason": "the dataset needs this composition slot even though the current representative has quality debt",
            }
        elif protected_guidance:
            record["guidance_action"] = {
                "label": "retain_for_guidance",
                "reason": "selected as part of the diversity-aware representative set for the active guidance floor",
            }
        elif cls_summary.get("status") == "deficit":
            record["guidance_action"] = {
                "label": "retain_while_class_is_thin",
                "reason": "its composition class remains below the active guidance floor",
            }
        else:
            record["guidance_action"] = {"label": "neutral", "reason": "no additional guidance-specific protection or swap pressure"}


def _guidance_counts(base: dict[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter((r.get("guidance_action") or {}).get("label", "unknown") for r in base.get("records") or []).items()))


def _make_markdown(payload: dict[str, Any]) -> str:
    base = payload
    g = payload["guidance_policy"]
    cap = payload.get("analysis_capability_policy") or {}
    s = payload.get("dataset_summary") or {}
    lines = [
        "# Dataset evidence report — guidance-aware v6",
        "",
        f"> Analysis source: **{cap.get('model_id')}** · policy tier **{cap.get('policy_tier')}**.",
        f"> Guidance profile: **{g['profile_id']}** · authority **{g['authority']}**.",
        "> Guidance bands are policy, not measured facts. Image quality evidence remains separate from coverage/debt reasoning.",
        "",
        "## Dataset summary",
        "",
        f"- Images profiled: **{g['image_count']}**",
        f"- Measured signal density: `{json.dumps(s.get('measured_signal_density_counts', {}), sort_keys=True)}`",
        f"- Dataset actions: `{json.dumps(s.get('presented_recommendation_counts_v6', {}), sort_keys=True)}`",
        f"- Guidance actions: `{json.dumps(s.get('guidance_action_counts', {}), sort_keys=True)}`",
        f"- Quarantined VLM axes: `{json.dumps(sorted((payload.get('dynamic_axis_quarantine') or {}).keys()))}`",
        "",
        "## Active guidance policy",
        "",
        f"- Objective: **{g['training_objective']}**",
        f"- {g['description']}",
        "",
        "| class | current | share | guidance count band | status | coverage debt | quality debt |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for cls in STANDARD_GUIDANCE_CLASSES:
        item = g["composition"][cls]
        lines.append(
            f"| {item['label']} | {item['current_count']} | {item['current_share']:.1%} | "
            f"{item['guidance_floor_count']}–{item['guidance_ceiling_count']} | **{item['status']}** | "
            f"{item['coverage_debt_count']} | {item['quality_debt_count']} |"
        )

    if g.get("unresolved_close_vs_medium_count"):
        lines.extend(
            [
                "",
                f"> **{g['unresolved_close_vs_medium_count']}** image(s) could not be safely split between close-up and medium-close guidance classes from this analyzer; those images are not silently forced into a precise percentage bucket.",
            ]
        )

    lines.extend(["", "## Guidance debt", ""])
    any_debt = False
    for cls in STANDARD_GUIDANCE_CLASSES:
        item = g["composition"][cls]
        if item["status"] == "deficit":
            any_debt = True
            lines.append(
                f"- **Coverage debt — {item['label']}**: add approximately {item['coverage_debt_count']} image(s) to reach the lower guidance floor."
            )
        if item["quality_debt_count"]:
            any_debt = True
            lines.append(
                f"- **Quality debt — {item['label']}**: protected representative(s) with low measured density: {', '.join(item['quality_debt_images'])}."
            )
        for feature, div in item["diversity"].items():
            if div["status"] == "debt":
                any_debt = True
                lines.append(
                    f"- **Diversity debt — {item['label']} / {feature}**: {div['distinct_count']} distinct value(s) across the protected set; effective target {div['effective_minimum_for_selected_count']}."
                )
            elif div["status"] == "unassessed_from_source":
                lines.append(f"- **Diversity unassessed — {item['label']} / {feature}**: source evidence is not authoritative enough for this axis.")
    if not any_debt:
        lines.append("- No count/quality/diversity debt triggered by the active profile.")

    lines.extend(["", "## Diversity-aware guidance representatives", ""])
    for cls in STANDARD_GUIDANCE_CLASSES:
        item = g["composition"][cls]
        reps = ", ".join(item["selected_representatives"]) or "—"
        lines.append(f"- **{item['label']}**: {reps}")

    lines.extend(["", "## Fixed-size rebalance suggestions", ""])
    if g["fixed_size_rebalance_suggestions"]:
        for item in g["fixed_size_rebalance_suggestions"]:
            lines.append(
                f"- Consider replacing/retiring **{item['replace_or_retire']}** ({item['from_class']}) when adding a stronger **{item['add_class']}** image."
            )
    else:
        lines.append("- No specific surplus-to-deficit swap is asserted from the currently assessable classes.")

    lines.extend(
        [
            "",
            "## Per-image optimizer view",
            "",
            "| image | guidance class | measured density | guidance protection | guidance action | dataset action |",
            "|---|---|---|---|---|---|",
        ]
    )
    severity = {"replace_candidate": 0, "keep_until_cleaner_equivalent": 1, "keep": 2}
    records = sorted(payload.get("records") or [], key=lambda r: (severity.get(r.get("base_recommendation"), 9), float((r.get("measured_signal_density") or {}).get("score") or 0.0)))
    for record in records:
        measured = record.get("measured_signal_density") or {}
        protect = ", ".join(p["value"] for p in record.get("guidance_protection") or []) or "—"
        action = (record.get("guidance_action") or {}).get("label", "neutral")
        presented = (record.get("presented_recommendation") or {}).get("label", record.get("base_recommendation"))
        lines.append(
            f"| {record.get('relative_path')} | {(record.get('guidance_composition') or {}).get('class')} | "
            f"{measured.get('label')} ({float(measured.get('score') or 0.0):.3f}) | {protect} | {action} | **{presented}** |"
        )

    lines.extend(["", "## Policy notes", ""])
    for note in g.get("policy_notes") or []:
        lines.append(f"- {note}")
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
        base, cleanup = _run_v5_base(args, run_dir, slug)
        guidance_layer = _build_guidance_layer(base, guidance)
        _recompute_recommendations(base, guidance_layer)

        base["schema_version"] = "dataset-evidence-6.0"
        base["guidance_policy"] = guidance_layer
        base.setdefault("method_notes", []).extend(
            [
                "V6 treats composition guidance as a versioned heuristic policy separate from measured evidence.",
                "Protection is based on the lower guidance floor, not an exact percentage target; broad bands and soft caps intentionally create tolerance and diminishing returns.",
                "Guidance representative selection is diversity-aware: a lower-quality image may remain strategically valuable if it contributes a facial/action/geometry view not supplied by cleaner peers.",
                "Coverage debt, diversity debt, and quality debt are reported separately so a dataset can have enough images of a class while still needing a cleaner or more diverse representative.",
                "Surplus classes produce swap opportunities rather than automatic deletion commands; the optimizer prefers trading redundant surplus evidence for a current guidance debt.",
            ]
        )
        summary = base.setdefault("dataset_summary", {})
        summary["guidance_class_counts"] = guidance_layer["raw_guidance_class_counts"]
        summary["guidance_action_counts"] = _guidance_counts(base)
        summary["base_recommendation_counts_v6"] = dict(sorted(Counter(r["base_recommendation"] for r in base.get("records") or []).items()))
        summary["presented_recommendation_counts_v6"] = dict(
            sorted(Counter((r.get("presented_recommendation") or {}).get("label", "unknown") for r in base.get("records") or []).items())
        )

        out_json = run_dir / f"{args.output_prefix}_{slug}.json"
        out_md = run_dir / f"{args.output_prefix}_{slug}.md"
        out_json.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
        out_md.write_text(_make_markdown(base), encoding="utf-8")

        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        print(f"Done. JSON:   {out_json}")
        print(f"      Report: {out_md}")
        print(f"Analysis policy: {(base.get('analysis_capability_policy') or {}).get('policy_tier')} / {(base.get('analysis_capability_policy') or {}).get('judgement_breadth')}")
        print(f"Guidance profile: {guidance_layer['profile_id']} / {guidance_layer['authority']}")
        print(f"Guidance classes: {guidance_layer['raw_guidance_class_counts']}")
        print(f"Guidance status: { {k: v['status'] for k, v in guidance_layer['composition'].items()} }")
        print(f"Coverage debt: { {k: v['coverage_debt_count'] for k, v in guidance_layer['composition'].items()} }")
        print(f"Quality debt: { {k: v['quality_debt_count'] for k, v in guidance_layer['composition'].items()} }")
        print(f"Dataset actions: {summary['presented_recommendation_counts_v6']}")
        print(f"Guidance actions: {summary['guidance_action_counts']}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
