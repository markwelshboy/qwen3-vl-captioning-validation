from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .pose_evidence import build_pose_evidence
from .runner import model_slug, resolve_model_id


COVERAGE_SIGNAL = {
    "face_dominant": 4.5,
    "large": 4.0,
    "medium": 2.8,
    "small": 1.5,
    "unknown": 2.2,
}
SHOT_IDENTITY_MODIFIER = {
    "extreme_close_up": 1.00,
    "close_up": 1.15,
    "medium_close_up": 1.10,
    "medium": 1.00,
    "three_quarter": 0.85,
    "full_length": 0.65,
    "unknown": 0.90,
}
POSE_SIGNAL = {
    "extreme_close_up": 0.8,
    "close_up": 1.1,
    "medium_close_up": 1.7,
    "medium": 2.4,
    "three_quarter": 3.2,
    "full_length": 4.0,
    "unknown": 1.5,
}
REGION_COVERAGE = {"small": 1.0, "medium": 2.0, "large": 4.0}
REGION_COMPLEXITY = {"low": 1.0, "medium": 1.5, "high": 2.0}
RELEVANCE_DISCOUNT = {"none": 1.0, "low": 0.85, "medium": 0.55, "high": 0.25}
RELEVANCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-dataset-evidence",
        description=(
            "Join cached Qwen analysis with cached DWPose geometry and produce a transparent "
            "dataset-level coverage / visual-burden / active-SNR heuristic report."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory.")
    parser.add_argument("--model", default="32b", help="Analysis model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--output-prefix", default="dataset_evidence", help="Output basename inside run_dir.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _orientation(analysis: dict[str, Any]) -> tuple[str, str, str, str]:
    orientation = ((analysis.get("target_subject") or {}).get("orientation") or {})
    torso = orientation.get("torso_yaw") or {}
    head = orientation.get("head_yaw") or {}
    return (
        str(torso.get("direction") or "unknown"),
        str(torso.get("magnitude") or "unknown"),
        str(head.get("direction") or "unknown"),
        str(head.get("magnitude") or "unknown"),
    )


def _nuisance_burden(analysis: dict[str, Any]) -> dict[str, Any]:
    regions = analysis.get("nuisance_regions") or []
    weighted: list[dict[str, Any]] = []
    total = 0.0
    entropy_focus_count = 0

    for region in regions:
        if not isinstance(region, dict):
            continue
        coverage = str(region.get("frame_coverage") or "small")
        complexity = str(region.get("visual_complexity") or "low")
        identity_rel = str(region.get("identity_relevance") or "none")
        pose_rel = str(region.get("pose_relevance") or "none")
        relevance = identity_rel
        if RELEVANCE_RANK.get(pose_rel, 0) > RELEVANCE_RANK.get(identity_rel, 0):
            relevance = pose_rel

        score = (
            REGION_COVERAGE.get(coverage, 1.0)
            * REGION_COMPLEXITY.get(complexity, 1.0)
            * RELEVANCE_DISCOUNT.get(relevance, 1.0)
        )
        total += score
        candidate = bool(region.get("entropy_focus_candidate"))
        entropy_focus_count += int(candidate)
        weighted.append(
            {
                "description": region.get("description"),
                "frame_coverage": coverage,
                "visual_complexity": complexity,
                "identity_relevance": identity_rel,
                "pose_relevance": pose_rel,
                "entropy_focus_candidate": candidate,
                "irrelevant_burden_points": round(score, 3),
            }
        )

    weighted.sort(key=lambda r: float(r["irrelevant_burden_points"]), reverse=True)
    return {
        "irrelevant_visual_burden": round(total, 3),
        "region_count": len(weighted),
        "entropy_focus_region_count": entropy_focus_count,
        "regions": weighted,
    }


def _signal_profile(analysis: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    framing = analysis.get("framing") or {}
    coverage = str(framing.get("subject_frame_coverage") or "unknown")
    shot = str(framing.get("shot_scale") or "unknown")
    identity_score = COVERAGE_SIGNAL.get(coverage, 2.2) * SHOT_IDENTITY_MODIFIER.get(shot, 0.9)

    target_pose = pose.get("target_2d_geometry") or {}
    connectivity = target_pose.get("connectivity") or {}
    complete_chains = sum(bool((connectivity.get(name) or {}).get("complete")) for name in ("left_arm", "right_arm", "left_leg", "right_leg"))
    pose_score = POSE_SIGNAL.get(shot, 1.5) + 0.15 * complete_chains
    if target_pose.get("pose_extent_hint") == "full_length":
        pose_score += 0.35

    clipped_bbox = target_pose.get("clipped_in_frame_keypoint_bbox") or {}
    return {
        "subject_frame_coverage": coverage,
        "shot_scale": shot,
        "identity_signal_score": round(identity_score, 3),
        "pose_signal_score": round(pose_score, 3),
        "clipped_keypoint_bbox_height_fraction": clipped_bbox.get("height_fraction"),
        "clipped_keypoint_bbox_area_fraction": clipped_bbox.get("area_fraction"),
        "complete_limb_chains": complete_chains,
    }


def _confound_burden(analysis: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    non_target = analysis.get("non_target_entities") or []
    depictions = analysis.get("embedded_depictions") or []
    people = pose.get("person_evidence") or {}
    significant = int(people.get("significant_secondary_people") or 0)
    small = int(people.get("small_secondary_people") or 0)
    score = len(non_target) * 1.0 + len(depictions) * 0.8 + significant * 1.2 + small * 0.35
    return {
        "score": round(score, 3),
        "non_target_entity_count": len(non_target),
        "embedded_depiction_count": len(depictions),
        "significant_secondary_people": significant,
        "small_secondary_people": small,
    }


def _snr_label(identity_signal: float, nuisance: float, confound: float, coverage: str) -> tuple[str, float]:
    # This is deliberately an interpretable training-signal-density heuristic,
    # not physical image SNR and not a learned quality score.
    score = identity_signal / (1.0 + 0.55 * nuisance + 0.60 * confound)
    if coverage == "small" and nuisance >= 3.0:
        return "low", score
    if score < 0.75:
        return "low", score
    if score < 1.45:
        return "medium", score
    return "high", score


def _rarity(count: int) -> str:
    if count <= 1:
        return "unique"
    if count <= 2:
        return "rare"
    return "common"


def _recommendation(item: dict[str, Any]) -> tuple[str, list[str]]:
    snr = item["active_snr"]["label"]
    shot = item["signal"]["shot_scale"]
    rarity = item["coverage_value"]["view_rarity"]
    burden = float(item["nuisance"]["irrelevant_visual_burden"])
    pose_signal = float(item["signal"]["pose_signal_score"])
    entropy_regions = int(item["nuisance"]["entropy_focus_region_count"])
    reasons: list[str] = []

    if snr == "low":
        reasons.append("low active subject-to-irrelevant-detail signal density")
        if rarity in {"unique", "rare"} and pose_signal >= 2.5:
            reasons.append("the image still contributes rare pose/framing coverage")
            if shot == "full_length":
                reasons.append("cropping would destroy the full-length coverage that makes the image useful")
            reasons.append("prefer sourcing a cleaner equivalent rather than immediately deleting it")
            return "keep_until_cleaner_equivalent", reasons
        reasons.append("similar coverage is represented elsewhere in the dataset")
        return "replace_candidate", reasons

    if burden >= 3.0 or entropy_regions:
        reasons.append("useful subject signal is present but irrelevant visual burden is substantial")
        reasons.append("a subject/entropy-focus mask is worth testing before replacement")
        return "consider_entropy_focus", reasons

    reasons.append("useful signal is not heavily diluted by currently identified nuisance burden")
    return "keep", reasons


def _make_markdown(payload: dict[str, Any]) -> str:
    coverage = payload["dataset_summary"]["shot_scale_counts"]
    snr_counts = payload["dataset_summary"]["active_snr_counts"]
    recommendations = payload["dataset_summary"]["recommendation_counts"]
    lines = [
        "# Dataset evidence report",
        "",
        "> `active_snr` is a transparent training-signal-density heuristic, not photometric SNR. It uses subject coverage, shot scale, Qwen nuisance-region burden, confound evidence, and DWPose geometry. It does **not** yet include measured per-image training loss.",
        "",
        "## Dataset summary",
        "",
        f"- Images profiled: **{payload['dataset_summary']['image_count']}**",
        f"- Shot scale counts: `{json.dumps(coverage, sort_keys=True)}`",
        f"- Active SNR counts: `{json.dumps(snr_counts, sort_keys=True)}`",
        f"- Recommendations: `{json.dumps(recommendations, sort_keys=True)}`",
        "",
        "## Per-image evidence",
        "",
        "| image | shot | coverage | active SNR | nuisance burden | pose signal | view rarity | recommendation |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]

    rank = {"replace_candidate": 0, "keep_until_cleaner_equivalent": 1, "consider_entropy_focus": 2, "keep": 3}
    records = sorted(
        payload["records"],
        key=lambda r: (rank.get(r["recommendation"], 9), -float(r["nuisance"]["irrelevant_visual_burden"])),
    )
    for r in records:
        lines.append(
            "| {image} | {shot} | {coverage} | {snr} | {burden:.2f} | {pose:.2f} | {rarity} | **{rec}** |".format(
                image=r["relative_path"],
                shot=r["signal"]["shot_scale"],
                coverage=r["signal"]["subject_frame_coverage"],
                snr=r["active_snr"]["label"],
                burden=float(r["nuisance"]["irrelevant_visual_burden"]),
                pose=float(r["signal"]["pose_signal_score"]),
                rarity=r["coverage_value"]["view_rarity"],
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
        top = (r["nuisance"].get("regions") or [])[:3]
        if top:
            lines.append("- Largest currently identified irrelevant-burden regions:")
            for region in top:
                desc = region.get("description") or "unnamed region"
                lines.append(
                    f"  - {desc} ({region.get('frame_coverage')}, {region.get('visual_complexity')} complexity; burden {region.get('irrelevant_burden_points')})"
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

    provisional: list[dict[str, Any]] = []
    shot_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()

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
        signal = _signal_profile(analysis, pose)
        nuisance = _nuisance_burden(analysis)
        confound = _confound_burden(analysis, pose)
        torso_dir, torso_mag, head_dir, head_mag = _orientation(analysis)
        shot = signal["shot_scale"]
        view_key = f"{shot}|torso:{torso_dir}"
        shot_counts[shot] += 1
        view_counts[view_key] += 1

        snr_label, snr_score = _snr_label(
            float(signal["identity_signal_score"]),
            float(nuisance["irrelevant_visual_burden"]),
            float(confound["score"]),
            signal["subject_frame_coverage"],
        )
        provisional.append(
            {
                "relative_path": record.get("relative_path"),
                "result_key": key,
                "image_summary": analysis.get("image_summary"),
                "signal": signal,
                "nuisance": nuisance,
                "confound": confound,
                "pose_evidence": pose,
                "orientation": {
                    "torso_direction": torso_dir,
                    "torso_magnitude": torso_mag,
                    "head_direction": head_dir,
                    "head_magnitude": head_mag,
                },
                "view_key": view_key,
                "active_snr": {
                    "label": snr_label,
                    "heuristic_score": round(snr_score, 3),
                    "definition": "estimated useful identity signal relative to currently identified irrelevant visual/confound burden; not photometric SNR",
                },
            }
        )

    if not provisional:
        print("No matched Qwen analysis + DWPose records found.", file=sys.stderr)
        return 2

    recommendations: Counter[str] = Counter()
    snr_counts: Counter[str] = Counter()
    for item in provisional:
        shot_count = shot_counts[item["signal"]["shot_scale"]]
        view_count = view_counts[item["view_key"]]
        item["coverage_value"] = {
            "shot_scale_dataset_count": shot_count,
            "view_dataset_count": view_count,
            "shot_scale_rarity": _rarity(shot_count),
            "view_rarity": _rarity(view_count),
        }
        recommendation, reasons = _recommendation(item)
        item["recommendation"] = recommendation
        item["recommendation_reasons"] = reasons
        recommendations[recommendation] += 1
        snr_counts[item["active_snr"]["label"]] += 1

    payload = {
        "schema_version": "dataset-evidence-1.0",
        "analysis_model": model_id,
        "analysis_source": str(model_dir),
        "dwpose_source": str(dwpose_dir),
        "method_notes": [
            "active_snr is a heuristic for training signal density, not physical image SNR.",
            "The heuristic intentionally keeps identity signal, pose signal, nuisance burden, confound burden, and dataset rarity as separate inspectable components.",
            "Per-image training loss is not included yet; when available it should be joined as empirical learning-difficulty evidence rather than folded invisibly into this score.",
            "A low-SNR image can still be worth retaining when it uniquely covers a pose/view; the preferred action is then often to source a cleaner equivalent.",
        ],
        "dataset_summary": {
            "image_count": len(provisional),
            "shot_scale_counts": dict(sorted(shot_counts.items())),
            "view_counts": dict(sorted(view_counts.items())),
            "active_snr_counts": dict(sorted(snr_counts.items())),
            "recommendation_counts": dict(sorted(recommendations.items())),
            "thin_coverage": [shot for shot, count in sorted(shot_counts.items()) if count <= 2],
        },
        "records": provisional,
    }

    json_path = run_dir / f"{args.output_prefix}_{slug}.json"
    md_path = run_dir / f"{args.output_prefix}_{slug}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_make_markdown(payload), encoding="utf-8")

    print(f"Done. JSON:   {json_path}")
    print(f"      Report: {md_path}")
    print(f"Active SNR: {dict(sorted(snr_counts.items()))}")
    print(f"Recommendations: {dict(sorted(recommendations.items()))}")
    if payload["dataset_summary"]["thin_coverage"]:
        print(f"Thin shot-scale coverage: {payload['dataset_summary']['thin_coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
