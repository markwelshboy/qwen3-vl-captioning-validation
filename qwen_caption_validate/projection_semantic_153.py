from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analysis_v2_normalize import normalize_analysis_v2
from .caption_projection_153 import build_caption_projection
from .compose_fusion_compare import _caption_policy, _read_json
from .runner import model_slug, resolve_model_id


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pose_semantics_root(value: dict[str, Any]) -> dict[str, Any]:
    root = value.get("pose_semantics") if isinstance(value.get("pose_semantics"), dict) else value
    return root if isinstance(root, dict) else {}


def _subject_semantics_root(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    root = value.get("subject_geometry_semantics") if isinstance(value.get("subject_geometry_semantics"), dict) else value
    return root if isinstance(root, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-projection-semantic-153",
        description=(
            "Replay Projection 1.5.3 from cached Fusion 2.3.x + Pose Semantics v0.10 + optional Subject Geometry Semantics v0.2."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis-model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--pose-semantics-dir", type=Path)
    parser.add_argument("--subject-geometry-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subject-token")
    parser.add_argument("--gender-grammar")
    parser.add_argument("--subject-pronoun")
    parser.add_argument("--object-pronoun")
    parser.add_argument("--possessive-pronoun")
    parser.add_argument("--protected-trait", action="append", default=[])
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    output_dir = (args.output_dir or (run_dir / "projection-v1.5.3" / slug)).expanduser().resolve()

    for path, label in ((analysis_dir, "Analyze"), (fusion_dir, "Fusion"), (pose_dir, "Pose Semantics")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2
    if not subject_dir.is_dir():
        print(f"WARNING: Subject Geometry directory not found: {subject_dir}; replay will preserve Projection 1.5.2 orientation behavior", file=sys.stderr)

    subject_token = args.subject_token or manifest.get("subject_token") or "sH1Vx"
    caption_policy = _caption_policy(args, manifest, subject_token)
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

        analysis_record = _read_json(analysis_dir / f"{key}.analysis.json")
        raw_analysis = (analysis_record or {}).get("analysis")
        fusion = _read_json(fusion_dir / f"{key}.fused_v2_3.json")
        pose_semantics = _read_json(pose_dir / f"{key}.pose_semantics.json")
        subject_semantics = _read_json(subject_dir / f"{key}.subject_geometry_semantics.json") if subject_dir.is_dir() else None
        if not isinstance(raw_analysis, dict) or fusion is None or pose_semantics is None:
            missing += 1
            index_records.append({"result_key": key, "status": "missing_required_source"})
            continue
        if _pose_semantics_root(pose_semantics).get("schema_version") != "pose-semantics-0.10":
            invalid_pose += 1
            index_records.append({"result_key": key, "status": "invalid_pose_semantics"})
            continue

        subject_root = _subject_semantics_root(subject_semantics)
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
        integration = ((audit.get("projection") or audit).get("subject_geometry_semantics_integration") or {})
        claim_ids = [
            str(item.get("id") or "")
            for item in evidence.get("required_claims") or []
            if isinstance(item, dict) and str(item.get("id") or "").startswith("subject_geometry_")
        ]
        index_records.append(
            {
                "result_key": key,
                "status": "written",
                "subject_geometry_integrated": bool(subject_semantics),
                "body_orientation": orientation.get("body_orientation"),
                "face_orientation": orientation.get("face_orientation"),
                "head_body_relation": orientation.get("head_body_relation"),
                "subject_geometry_required_claims": claim_ids,
                "legacy_required_claims_removed": integration.get("legacy_required_claims_removed") or [],
                "cross_source_conflict_count": len(integration.get("cross_source_conflicts_audit_only") or []),
                "camera_subject_caption_facing": False,
            }
        )
        written += 1

    index = {
        "schema_version": "caption-projection-1.5.3-run",
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
    index_path = output_dir / "projection_153.index.json"
    _write(index_path, index)
    print(f"Projection 1.5.3 output: {output_dir}")
    print(
        f"Written: {written}; reused: {reused}; missing required: {missing}; "
        f"subject missing: {subject_missing}; invalid subject: {invalid_subject}"
    )
    print(f"Index: {index_path}")
    return 0 if written or reused else 2


if __name__ == "__main__":
    raise SystemExit(main())
