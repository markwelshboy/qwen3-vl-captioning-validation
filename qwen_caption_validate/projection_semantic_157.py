from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import projection_semantic_153 as v153
from .analysis_v2_normalize import normalize_analysis_v2
from .caption_projection_157 import build_caption_projection
from .runner import model_slug, resolve_model_id


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = v153.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        print(f"Run manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    analysis_model_id = resolve_model_id(args.analysis_model)
    slug = model_slug(analysis_model_id)
    analysis_dir = run_dir / slug
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.7" / slug)).expanduser().resolve()
    pose_dir = (args.pose_semantics_dir or (run_dir / "pose-semantics-v0.10" / slug)).expanduser().resolve()
    subject_dir = (args.subject_geometry_dir or (run_dir / "subject-geometry-semantics-v0.2" / slug)).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "projection-v1.5.7" / slug)).expanduser().resolve()

    for path, label in ((analysis_dir, "Analyze"), (fusion_dir, "Fusion"), (pose_dir, "Pose Semantics")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2
    if not subject_dir.is_dir():
        print(f"WARNING: Subject Geometry directory not found: {subject_dir}; replay will preserve pre-subject-geometry orientation behavior", file=sys.stderr)

    subject_token = args.subject_token or manifest.get("subject_token") or "sH1Vx"
    caption_policy = v153._caption_policy(args, manifest, subject_token)
    records = list(manifest.get("images") or [])
    if args.only:
        needles = tuple(args.only)
        records = [r for r in records if any(n in str(r.get("result_key") or "") for n in needles)]
    if args.limit is not None:
        records = records[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    index_records: list[dict[str, Any]] = []
    written = reused = missing = invalid_pose = invalid_subject = subject_missing = 0

    for record in records:
        key = str(record.get("result_key") or "")
        evidence_path = output_dir / f"{key}.caption_evidence.json"
        audit_path = output_dir / f"{key}.projection_audit.json"
        if evidence_path.exists() and audit_path.exists() and not args.overwrite:
            reused += 1
            continue

        analysis_record = v153._read_json(analysis_dir / f"{key}.analysis.json")
        raw_analysis = (analysis_record or {}).get("analysis")
        fusion = v153._read_json(fusion_dir / f"{key}.fused_v2_3.json")
        pose_semantics = v153._read_json(pose_dir / f"{key}.pose_semantics.json")
        subject_semantics = v153._read_json(subject_dir / f"{key}.subject_geometry_semantics.json") if subject_dir.is_dir() else None
        if not isinstance(raw_analysis, dict) or fusion is None or pose_semantics is None:
            missing += 1
            index_records.append({"result_key": key, "status": "missing_required_source"})
            continue
        if v153._pose_semantics_root(pose_semantics).get("schema_version") != "pose-semantics-0.10":
            invalid_pose += 1
            index_records.append({"result_key": key, "status": "invalid_pose_semantics"})
            continue

        subject_root = v153._subject_semantics_root(subject_semantics)
        if subject_semantics is None:
            subject_missing += 1
        elif subject_root.get("schema_version") != "subject-geometry-semantics-0.2":
            invalid_subject += 1
            subject_semantics = None

        analysis, _ = normalize_analysis_v2(raw_analysis)
        evidence, audit = build_caption_projection(
            fusion,
            analysis,
            pose_semantics=pose_semantics,
            subject_geometry_semantics=subject_semantics,
            caption_policy=caption_policy,
        )
        _write(evidence_path, evidence)
        _write(audit_path, audit)

        pose = evidence.get("pose_orientation") or {}
        orientation = pose.get("subject_geometry_orientation") or {}
        semantic_orientation = pose.get("semantic_orientation") or {}
        projection = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
        economy = projection.get("projection_157_structural_economy") or {}
        visibility = evidence.get("visibility_constraints") or {}
        index_records.append(
            {
                "result_key": key,
                "status": "written",
                "subject_geometry_integrated": bool(subject_semantics),
                "body_orientation": orientation.get("body_orientation") if isinstance(orientation, dict) else None,
                "face_yaw_orientation": orientation.get("face_yaw_orientation") if isinstance(orientation, dict) else None,
                "head_body_relation": orientation.get("head_body_relation") if isinstance(orientation, dict) else None,
                "retained_semantic_orientation_fields": sorted(semantic_orientation) if isinstance(semantic_orientation, dict) else [],
                "remaining_3d_geometry_keys": sorted((pose.get("qualified_3d_geometry") or {}).keys()) if isinstance(pose.get("qualified_3d_geometry"), dict) else [],
                "removed_depth_geometry_keys": sorted((economy.get("component_depth_geometry_audit_only") or {}).keys()),
                "removed_depth_claim_ids": economy.get("component_depth_required_claims_removed") or [],
                "caption_visibility_keys": sorted(visibility) if isinstance(visibility, dict) else [],
                "positive_visibility_quarantined": bool(economy.get("positive_visibility_audit_only")),
                "camera_subject_caption_facing": False,
            }
        )
        written += 1

    index = {
        "schema_version": "caption-projection-1.5.7-run",
        "run_dir": str(run_dir),
        "analysis_model": analysis_model_id,
        "analysis_source": str(analysis_dir),
        "fusion_source": str(fusion_dir),
        "pose_semantics_source": str(pose_dir),
        "subject_geometry_source": str(subject_dir),
        "output_dir": str(output_dir),
        "written": written,
        "reused": reused,
        "missing_required_sources": missing,
        "invalid_pose_semantics": invalid_pose,
        "missing_subject_geometry": subject_missing,
        "invalid_subject_geometry": invalid_subject,
        "records": index_records,
    }
    index_path = output_dir / "projection_157.index.json"
    _write(index_path, index)
    print(f"Projection 1.5.7 output: {output_dir}")
    print(
        f"Written: {written}; reused: {reused}; missing required: {missing}; "
        f"subject missing: {subject_missing}; invalid subject: {invalid_subject}"
    )
    print(f"Index: {index_path}")
    return 0 if written or reused else 2


if __name__ == "__main__":
    raise SystemExit(main())
