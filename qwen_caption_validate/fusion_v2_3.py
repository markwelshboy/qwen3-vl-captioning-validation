from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .analysis_v2_normalize import normalize_analysis_v2
from .fusion_v2 import fuse_analysis_v2
from .pose_evidence import build_pose_evidence
from .runner import model_slug, resolve_model_id, validate_analysis
from .sam3d_support import qualify_sam3d_geometry


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_SCHEMAS = {
    "2.0": PACKAGE_ROOT / "schemas" / "analysis_v2.schema.json",
    "2.1": PACKAGE_ROOT / "schemas" / "analysis_v2_1.schema.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-fusion-v23",
        description=(
            "Fusion v2.3: combine Analyze-v2/v2.1 semantics, DWPose 2-D evidence, "
            "and optional SAM 3D Body geometry with landmark-visibility support gating."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing Analyze-v2/v2.1 validation run directory.")
    parser.add_argument("--model", default="32b-fp8", help="Analysis model alias or Hugging Face model ID.")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument(
        "--sam3d-dir",
        type=Path,
        help="Optional directory containing <image>.sam3d.json files from qwen-sam3d-probe.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: <run_dir>/fusion-v2.3/<model-slug>).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _schema_for(version: str) -> dict[str, Any] | None:
    path = ANALYSIS_SCHEMAS.get(version)
    return _read_json(path) if path else None


def _enrich_v23(base_fusion: dict[str, Any], analysis: dict[str, Any], sam3d_record: dict[str, Any] | None) -> dict[str, Any]:
    fused = deepcopy(base_fusion)
    fused["schema_version"] = "analysis-fusion-2.3"

    if sam3d_record is None:
        sam3d_audit: dict[str, Any] = {
            "schema_version": "sam3d-support-audit-0.1",
            "status": "unavailable",
            "authority": "no_sam3d_record",
            "selection_usable": False,
        }
    else:
        sam3d_audit = qualify_sam3d_geometry(analysis, sam3d_record)
        sam3d_audit["status"] = "available"

    fused["sam3d_geometry_audit"] = sam3d_audit
    fused.setdefault("caption_authority", {})["sam3d_torso_depth_rotation"] = (
        "caption_authoritative_only_when_sam3d_geometry_audit.torso_depth_rotation.authority_is_qualified_3d_geometry"
    )
    fused.setdefault("selection_policy", {})["sam3d_geometry"] = (
        "qualified_evidence_available_but_report_only_for_portfolio_selection_until_later_dataset_policy_version"
    )
    fused["selection_policy"]["note"] = (
        "Fusion-v2.3 qualifies supported SAM3D geometry but still does not alter V8.1 portfolio weights. "
        "A later dataset-policy version must explicitly opt into this evidence."
    )

    authority = ((sam3d_audit.get("torso_depth_rotation") or {}).get("authority"))
    if authority == "qualified_3d_geometry":
        fused.setdefault("fusion_warnings", []).append(
            "SAM3D unsigned torso-depth rotation is image-supported and qualified as 3-D geometry; direction remains unsigned"
        )
    elif authority in {"report_only_partial_image_support", "reconstructed_prior_only"}:
        fused.setdefault("fusion_warnings", []).append(
            "SAM3D torso geometry is not fully image-supported; reconstructed anatomy must not be treated as observed evidence"
        )
    elif authority == "qualified_geometry_pending_target_provenance":
        fused.setdefault("fusion_warnings", []).append(
            "SAM3D geometry is landmark-supported but target-bbox provenance requires review because other/depicted people are present"
        )

    return fused


def _bump(counter: dict[str, int], key: Any) -> None:
    name = str(key or "unavailable")
    counter[name] = counter.get(name, 0) + 1


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    model_dir = run_dir / slug
    if not model_dir.is_dir():
        raise SystemExit(f"Analysis model directory does not exist: {model_dir}")

    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    if not dwpose_dir.is_dir():
        raise SystemExit(f"DWPose directory does not exist: {dwpose_dir}")

    sam3d_dir = args.sam3d_dir.expanduser().resolve() if args.sam3d_dir else None
    if sam3d_dir is not None and not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory does not exist: {sam3d_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "fusion-v2.3" / slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_paths = sorted(model_dir.glob("*.analysis.json"))
    written = 0
    skipped = 0
    invalid = 0
    missing_dwpose = 0
    missing_sam3d = 0
    normalized_count = 0
    v21_count = 0
    qualified_sam3d = 0
    qualified_sam3d_shoulders = 0
    qualified_sam3d_hips = 0
    provenance_review = 0
    shoulder_authority_counts: dict[str, int] = {}
    hip_authority_counts: dict[str, int] = {}
    torso_authority_counts: dict[str, int] = {}
    torso_support_counts: dict[str, int] = {}
    index: list[dict[str, Any]] = []

    for analysis_path in analysis_paths:
        key = analysis_path.name.removesuffix(".analysis.json")
        out_path = output_dir / f"{key}.fused_v2_3.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        result = _read_json(analysis_path)
        analysis = result.get("analysis")
        if not isinstance(analysis, dict):
            invalid += 1
            index.append({"image_key": key, "status": "skipped_missing_analysis"})
            continue

        version = str(analysis.get("schema_version") or "")
        schema = _schema_for(version)
        if schema is None:
            invalid += 1
            index.append({"image_key": key, "status": "skipped_unsupported_analysis_schema", "analysis_schema_version": version})
            continue

        normalized, normalization_actions = normalize_analysis_v2(analysis)
        schema_errors = validate_analysis(normalized, schema)
        if schema_errors:
            invalid += 1
            index.append(
                {
                    "image_key": key,
                    "status": "skipped_schema_invalid_after_normalization",
                    "analysis_schema_version": version,
                    "normalization_actions": normalization_actions,
                    "schema_errors": schema_errors,
                }
            )
            continue
        if normalization_actions:
            normalized_count += 1
        if version == "2.1":
            v21_count += 1

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        if not dwpose_path.exists():
            missing_dwpose += 1
            index.append({"image_key": key, "status": "missing_dwpose"})
            continue

        pose = build_pose_evidence(_read_json(dwpose_path))
        sam3d_path = sam3d_dir / f"{key}.sam3d.json" if sam3d_dir is not None else None
        sam3d_record = None
        if sam3d_path is not None:
            if sam3d_path.exists():
                sam3d_record = _read_json(sam3d_path)
            else:
                missing_sam3d += 1

        base_fused = fuse_analysis_v2(normalized, pose)
        fused = _enrich_v23(base_fused, normalized, sam3d_record)
        sam3d = fused.get("sam3d_geometry_audit") or {}
        shoulder = sam3d.get("shoulder_depth_rotation") or {}
        hip = sam3d.get("hip_depth_rotation") or {}
        torso = sam3d.get("torso_depth_rotation") or {}
        provenance = sam3d.get("target_provenance") or {}

        shoulder_authority = shoulder.get("authority")
        hip_authority = hip.get("authority")
        torso_authority = torso.get("authority")
        _bump(shoulder_authority_counts, shoulder_authority)
        _bump(hip_authority_counts, hip_authority)
        _bump(torso_authority_counts, torso_authority)
        _bump(torso_support_counts, torso.get("support_state"))

        if shoulder_authority == "qualified_component_geometry":
            qualified_sam3d_shoulders += 1
        if hip_authority == "qualified_component_geometry":
            qualified_sam3d_hips += 1
        if torso_authority == "qualified_3d_geometry":
            qualified_sam3d += 1
        if provenance.get("context_risk") == "requires_review":
            provenance_review += 1

        payload = {
            "image": result.get("image"),
            "model": result.get("model"),
            "analysis_path": str(analysis_path),
            "dwpose_path": str(dwpose_path),
            "sam3d_path": str(sam3d_path) if sam3d_path and sam3d_path.exists() else None,
            "analysis_schema_valid_after_normalization": True,
            "analysis_normalization_actions": normalization_actions,
            "fusion": fused,
        }
        _write_json(out_path, payload)
        written += 1
        index.append(
            {
                "image_key": key,
                "status": "written",
                "analysis_schema_version": version,
                "normalization_actions": normalization_actions,
                "fusion_warnings": fused.get("fusion_warnings") or [],
                "camera": fused.get("camera_audit"),
                "framing": fused.get("framing_audit"),
                "body_axis": fused.get("projected_body_axis_audit"),
                "sam3d": sam3d,
            }
        )

    summary = {
        "schema_version": "analysis-fusion-2.3-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "analysis_model_slug": slug,
        "dwpose_dir": str(dwpose_dir),
        "sam3d_dir": str(sam3d_dir) if sam3d_dir else None,
        "output_dir": str(output_dir),
        "written": written,
        "skipped_existing": skipped,
        "normalized_records": normalized_count,
        "analyze_v2_1_records": v21_count,
        "invalid_or_unsupported_analysis": invalid,
        "missing_dwpose": missing_dwpose,
        "missing_sam3d": missing_sam3d,
        "qualified_sam3d_shoulder_depth_records": qualified_sam3d_shoulders,
        "qualified_sam3d_hip_depth_records": qualified_sam3d_hips,
        "qualified_sam3d_torso_depth_records": qualified_sam3d,
        "sam3d_target_provenance_review_records": provenance_review,
        "sam3d_authority_counts": {
            "shoulder_depth_rotation": shoulder_authority_counts,
            "hip_depth_rotation": hip_authority_counts,
            "torso_depth_rotation": torso_authority_counts,
            "torso_support_state": torso_support_counts,
        },
        "records": index,
    }
    _write_json(output_dir / "fusion_v2_3.index.json", summary)

    print(f"Fusion-v2.3 output: {output_dir}")
    print(
        f"Written: {written}; reused: {skipped}; v2.1: {v21_count}; normalized: {normalized_count}; "
        f"invalid: {invalid}; missing DWPose: {missing_dwpose}; missing SAM3D: {missing_sam3d}; "
        f"qualified SAM3D shoulder-depth: {qualified_sam3d_shoulders}; hip-depth: {qualified_sam3d_hips}; "
        f"torso-depth: {qualified_sam3d}; provenance review: {provenance_review}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
