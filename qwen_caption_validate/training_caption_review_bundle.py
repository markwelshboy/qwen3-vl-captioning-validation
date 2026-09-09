from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .runner import model_slug, resolve_model_id


SCHEMA_VERSION = "training-caption-review-bundle-0.1"
ANNOTATION_SCHEMA_VERSION = "training-caption-review-annotations-0.1"
DEFAULT_OUTPUT_SUBDIR = "training-caption-review-v0.1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_captions_dir(run_dir: Path, supplied: Path | None, source_model: str) -> Path:
    if supplied is not None:
        path = supplied.expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"Final-caption directory not found: {path}")
        return path

    slug = model_slug(resolve_model_id(source_model))
    semantic = run_dir / "semantic-v3"
    preferred = [
        semantic / "compact-renderer-semantic-repair-v0.3-final87" / slug,
        semantic / "compact-renderer-semantic-repair-v0.3-full87" / slug,
        semantic / "compact-renderer-semantic-repair-v0.3" / slug,
    ]
    for path in preferred:
        if path.is_dir() and any(path.glob("*.semantic_repaired_caption.json")):
            return path.resolve()

    candidates = sorted(
        (
            path / slug
            for path in semantic.glob("compact-renderer-semantic-repair-v0.3*")
            if path.is_dir()
        ),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for path in candidates:
        if path.is_dir() and any(path.glob("*.semantic_repaired_caption.json")):
            return path.resolve()
    raise SystemExit(
        "Could not find final Semantic V3 training captions; pass --captions-dir explicitly."
    )


def _resolve_pose_review_dir(run_dir: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        path = supplied.expanduser().resolve()
    else:
        path = run_dir / "semantic-v3" / "pose-review-v0.11"
    if not path.is_dir() or not (path / "pose_review.index.json").is_file():
        raise SystemExit(
            "Pose Review v0.11 bundle not found. Build it first, e.g.:\n"
            f"  bash ./run_pose_review_bundle_workspace.sh {run_dir} --overwrite"
        )
    return path.resolve()


def _safe_source(root: Path, rel: str) -> Path | None:
    try:
        candidate = (root / rel).resolve()
        candidate.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return candidate


def _copy_relative(source_root: Path, output: Path, rel: str, overwrite: bool) -> bool:
    source = _safe_source(source_root, rel)
    if source is None or not source.is_file():
        return False
    target = output / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not target.exists():
        shutil.copy2(source, target)
    return True


def _caption_summary(raw: dict[str, Any]) -> dict[str, Any]:
    audit = raw.get("quality_audit") or {}
    return {
        "text": str(raw.get("final_caption") or "").strip(),
        "word_count": audit.get("rendered_word_count"),
        "repair_action": raw.get("repair_action"),
        "full_gate": bool(audit.get("passes_basic_gate")),
        "semantic_warnings": raw.get("final_semantic_warnings") or [],
    }


def build_bundle(
    *,
    run_dir: Path,
    captions_dir: Path,
    pose_review_dir: Path,
    pose_language_dir: Path,
    output: Path,
    wanted: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    pose_index = _read_json(pose_review_dir / "pose_review.index.json")
    pose_records = pose_index.get("records") or []
    if not isinstance(pose_records, list) or not pose_records:
        raise SystemExit(f"Pose Review index contains no records: {pose_review_dir}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "records").mkdir(parents=True, exist_ok=True)
    wanted_lower = {value.lower() for value in (wanted or set())}

    records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for pose_record in pose_records:
        if not isinstance(pose_record, dict):
            continue
        key = str(pose_record.get("image_key") or "").strip()
        if not key or (wanted_lower and key.lower() not in wanted_lower):
            continue

        compact_path = captions_dir / f"{key}.compact.semantic_repaired_caption.json"
        medium_path = captions_dir / f"{key}.medium.semantic_repaired_caption.json"
        pose_language_path = pose_language_dir / f"{key}.pose_language.json"
        if not compact_path.is_file() or not medium_path.is_file():
            missing.append({"image_key": key, "reason": "missing_final_compact_or_medium_caption"})
            continue

        original_rel = str(pose_record.get("original") or "")
        overlay_rel = str(pose_record.get("overlay") or "")
        if not _copy_relative(pose_review_dir, output, original_rel, overwrite):
            missing.append({"image_key": key, "reason": "missing_pose_review_original"})
            continue
        if not _copy_relative(pose_review_dir, output, overlay_rel, overwrite):
            missing.append({"image_key": key, "reason": "missing_pose_review_overlay"})
            continue

        compact_raw = _read_json(compact_path)
        medium_raw = _read_json(medium_path)
        pose_language = _read_json(pose_language_path) if pose_language_path.is_file() else {}
        caption_ready = [
            str(value).strip()
            for value in (pose_language.get("caption_ready_phrases") or [])
            if str(value).strip()
        ]

        raw_rel = f"records/{key}.json"
        raw_payload = {
            "image_key": key,
            "pose_review_record": pose_record,
            "pose_language": pose_language,
            "compact": compact_raw,
            "medium": medium_raw,
        }
        _write_json(output / raw_rel, raw_payload)

        modifiers = pose_record.get("posture_modifier_diagnostic") or {}
        records.append(
            {
                "image_key": key,
                "original": original_rel,
                "overlay": overlay_rel,
                "raw_json": raw_rel,
                "pose": pose_record.get("pose"),
                "best_candidate_pose": pose_record.get("best_candidate_pose"),
                "crop_support_percent": pose_record.get("crop_support_percent"),
                "winner_margin_percent": pose_record.get("winner_margin_percent"),
                "reconstruction_match_percent": pose_record.get("reconstruction_match_percent"),
                "support_class": pose_record.get("support_class"),
                "caption_ready_phrases": caption_ready,
                "pose_modifiers": {
                    "lean_severity": modifiers.get("lean_severity"),
                    "shoulder_line_tilt_severity": modifiers.get("shoulder_line_tilt_severity"),
                    "body_orientation": (pose_record.get("body_orientation_diagnostic") or {}).get("orientation"),
                },
                "captions": {
                    "compact": _caption_summary(compact_raw),
                    "medium": _caption_summary(medium_raw),
                },
            }
        )
        print(
            f"{key}: compact={records[-1]['captions']['compact']['word_count']}w "
            f"medium={records[-1]['captions']['medium']['word_count']}w pose={pose_record.get('pose')}"
        )

    index = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "captions_dir": str(captions_dir),
        "pose_review_dir": str(pose_review_dir),
        "pose_language_dir": str(pose_language_dir),
        "record_count": len(records),
        "missing": missing,
        "legend": pose_index.get("legend") or {},
        "records": records,
    }
    _write_json(output / "training_caption_review.index.json", index)

    annotations_path = output / "training_caption_review_annotations.json"
    if not annotations_path.exists():
        _write_json(
            annotations_path,
            {"schema_version": ANNOTATION_SCHEMA_VERSION, "records": {}},
        )

    (output / "README.txt").write_text(
        "Semantic V3 Training Caption Review Bundle v0.1\n\n"
        "Run the local review server from the repository root:\n"
        f"  bash ./run_training_caption_review_server.sh {output}\n",
        encoding="utf-8",
    )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-training-caption-review-bundle",
        description=(
            "Build a self-contained human-review bundle for final compact/medium "
            "Semantic V3 training captions using the existing Pose Review original + "
            "DWPose/SAM3D overlay assets. No model inference occurs."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--captions-dir", type=Path)
    parser.add_argument("--pose-review-dir", type=Path)
    parser.add_argument("--pose-language-dir", type=Path)
    parser.add_argument("--source-model", default="32b-fp8")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    captions_dir = _resolve_captions_dir(run_dir, args.captions_dir, args.source_model)
    pose_review_dir = _resolve_pose_review_dir(run_dir, args.pose_review_dir)
    pose_language_dir = (
        args.pose_language_dir or (run_dir / "semantic-v3" / "pose-language-v0.1")
    ).expanduser().resolve()
    if not pose_language_dir.is_dir():
        raise SystemExit(f"Pose-language directory not found: {pose_language_dir}")
    output = (
        args.output or (run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR)
    ).expanduser().resolve()

    index = build_bundle(
        run_dir=run_dir,
        captions_dir=captions_dir,
        pose_review_dir=pose_review_dir,
        pose_language_dir=pose_language_dir,
        output=output,
        wanted=set(args.only),
        overwrite=args.overwrite,
    )
    print(f"Training caption review bundle: {output}")
    print(f"Records: {index['record_count']}; missing={len(index['missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
