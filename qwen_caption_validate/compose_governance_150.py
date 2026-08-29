from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import compose_fusion_compare as _base
from .analysis_v2_normalize import normalize_analysis_v2
from .caption_projection_150 import build_caption_projection, lint_caption
from .runner import generate_text, load_model, model_slug, resolve_model_id, unload_model


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FUSION_PROMPT = PACKAGE_ROOT / "prompts" / "compose_identity_fusion_safe_v1.txt"

_GOVERNANCE_ADDENDUM = r"""

PROJECTION 1.5.0 POSE-SEMANTIC GOVERNANCE
- `pose_orientation.semantic_pose.posture`, when non-null, is a qualified FACT from Pose Semantics v0.10. State it once in natural language.
- If `semantic_pose.posture` is null, do NOT infer standing, seated, squatting, reclining, kneeling, or lying from lower-level geometry, furniture, crop, or context. A candidate posture is deliberately absent from caption-facing evidence.
- `pose_orientation.semantic_pose.gestures` contains caption-preferred recognizable gesture/support primitives. Prefer those phrases over serializing the component joint/contact fields that established them.
- Semantic economy is mandatory: when a whole posture or gesture primitive already expresses the useful visual fact, do not repeat knee angles, arm-chain geometry, hand/forearm duplicates, or support-chain bookkeeping merely because those fields are present elsewhere in the governed evidence.
- Lower-level governed evidence may still be used for visually distinctive information NOT subsumed by the semantic primitive, such as a separate held object, meaningful body orientation, head turn, crop/framing, clothing, scene, or lighting.
- Never turn an absent/candidate posture into a FACT during prose composition. Do not use words like "probably seated" or "appears to be standing" unless such qualification is explicitly present in caption-facing evidence.
- Continue all Projection 1.4.3 laterality, framing, interaction, contact, scene, and identity-firewall rules.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-compose-governance-150",
        description=(
            "Generate governed text-only captions from Fusion 2.3.x plus Pose Semantics v0.10. "
            "Only qualified semantic pose FACTS reach Compose; candidates remain audit-only."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis-model", default="32b-fp8")
    parser.add_argument("--compose-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--pose-semantics-dir", type=Path)
    parser.add_argument("--fusion-prompt", type=Path, default=DEFAULT_FUSION_PROMPT)
    parser.add_argument("--subject-token")
    parser.add_argument("--gender-grammar")
    parser.add_argument("--subject-pronoun")
    parser.add_argument("--object-pronoun")
    parser.add_argument("--possessive-pronoun")
    parser.add_argument("--protected-trait", action="append", default=[])
    parser.add_argument("--detail", choices=["concise", "balanced", "detailed"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--run-label")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-caption-tokens", type=int, default=450)
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    return _base._read_json(path)


def _render_prompt(template: str, evidence: dict[str, Any], subject_token: str, detail: str) -> str:
    return _base._render_fusion_prompt(template, evidence, subject_token, detail) + _GOVERNANCE_ADDENDUM


def _pose_semantics_ok(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    root = payload.get("pose_semantics") if isinstance(payload.get("pose_semantics"), dict) else payload
    return root.get("schema_version") == "pose-semantics-0.10"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        print(f"Run manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    analysis_model_id = resolve_model_id(args.analysis_model)
    analysis_slug = model_slug(analysis_model_id)
    analysis_dir = run_dir / analysis_slug
    if not analysis_dir.is_dir():
        print(f"Analysis directory not found: {analysis_dir}", file=sys.stderr)
        return 2

    fusion_dir = (
        args.fusion_dir.expanduser().resolve()
        if args.fusion_dir
        else run_dir / "fusion-v2.3.7" / analysis_slug
    )
    semantics_dir = (
        args.pose_semantics_dir.expanduser().resolve()
        if args.pose_semantics_dir
        else run_dir / "pose-semantics-v0.10" / analysis_slug
    )
    for path, label in ((fusion_dir, "Fusion"), (semantics_dir, "Pose Semantics v0.10")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    compose_model_id = resolve_model_id(args.compose_model)
    compose_slug = model_slug(compose_model_id)
    run_label = _base._safe_label(args.run_label or f"semantic150-{args.backend}-{args.quantization}-{args.dtype}")
    subject_token = args.subject_token or manifest.get("subject_token") or "sH1Vx"
    detail = args.detail or manifest.get("detail") or "balanced"
    caption_policy = _base._caption_policy(args, manifest, subject_token)
    fusion_template = args.fusion_prompt.read_text(encoding="utf-8")

    image_records = list(manifest.get("images") or [])
    if args.only:
        needles = tuple(args.only)
        image_records = [
            record
            for record in image_records
            if any(needle in str(record.get("result_key") or "") for needle in needles)
        ]
    if args.limit is not None:
        image_records = image_records[: args.limit]

    output_dir = run_dir / "compose_semantic_150" / f"{analysis_slug}__compose__{compose_slug}__{run_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    missing = 0
    invalid_semantics = 0
    pending = 0
    for record in image_records:
        key = str(record.get("result_key") or "")
        analysis_record = _read_json(analysis_dir / f"{key}.analysis.json")
        raw_analysis = (analysis_record or {}).get("analysis")
        fusion_payload = _read_json(fusion_dir / f"{key}.fused_v2_3.json")
        semantics = _read_json(semantics_dir / f"{key}.pose_semantics.json")
        if not isinstance(raw_analysis, dict) or fusion_payload is None or semantics is None:
            missing += 1
            print(f"WARNING: missing Analyze/Fusion/Pose Semantics source for {key}; skipping", file=sys.stderr)
            continue
        if not _pose_semantics_ok(semantics):
            invalid_semantics += 1
            print(f"WARNING: {key} is not pose-semantics-0.10; skipping", file=sys.stderr)
            continue

        analysis, normalization_actions = normalize_analysis_v2(raw_analysis)
        evidence, audit = build_caption_projection(
            fusion_payload,
            analysis,
            pose_semantics=semantics,
            caption_policy=caption_policy,
        )
        txt = output_dir / f"{key}.fusion-safe.txt"
        meta = output_dir / f"{key}.fusion-safe.json"
        need = args.overwrite or not txt.exists() or not meta.exists()
        pending += int(need)
        prepared.append(
            {
                **record,
                "result_key": key,
                "analysis": analysis,
                "normalization_actions": normalization_actions,
                "pose_semantics": semantics,
                "caption_evidence": evidence,
                "projection_audit": audit,
                "txt": txt,
                "meta": meta,
                "need": need,
            }
        )

    if not prepared:
        print("No complete semantic Compose records found.", file=sys.stderr)
        return 2

    loaded = None
    load_seconds: float | None = None
    if pending:
        print(
            f"Loading {compose_model_id} for Projection 1.5.0 semantic Compose "
            f"({pending} generation(s); run={run_label}) ..."
        )
        loaded = load_model(
            compose_model_id,
            backend=args.backend,
            dtype=args.dtype,
            quantization=args.quantization,
            attn_implementation=args.attn,
            cache_dir=args.cache_dir,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        load_seconds = loaded.load_seconds
        print(f"Loaded in {load_seconds:.1f}s. Compose remains text-only.")

    try:
        for item in tqdm(prepared, desc="Semantic governed Compose 1.5.0"):
            if not item["need"]:
                continue
            prompt = _render_prompt(fusion_template, item["caption_evidence"], subject_token, detail)
            caption, seconds = generate_text(loaded, prompt, max_new_tokens=args.max_caption_tokens)
            metadata = {
                "schema_version": "compose-semantic-1.5.0",
                "image": item.get("relative_path"),
                "analysis_model": analysis_model_id,
                "compose_model": compose_model_id,
                "compose_seconds": seconds,
                "subject_token": subject_token,
                "detail": detail,
                "image_conditioned": False,
                "run_label": run_label,
                "backend_requested": args.backend,
                "dtype_requested": args.dtype,
                "quantization_requested": args.quantization,
                "caption_policy": caption_policy,
                "normalization_actions": item["normalization_actions"],
                "caption_evidence": item["caption_evidence"],
                "projection_audit": item["projection_audit"],
                "pose_semantics_audit_only": item["pose_semantics"],
                "caption_lint": lint_caption(caption, item["caption_evidence"]),
            }
            _base._write_caption(item["txt"], item["meta"], caption, metadata)
    finally:
        if loaded is not None:
            print(f"Unloading {compose_model_id} ...")
            unload_model(loaded)

    records: list[dict[str, Any]] = []
    aggregate_lines: list[str] = []
    for item in prepared:
        caption = _base._read_text(item["txt"])
        meta = _read_json(item["meta"])
        semantic_pose = ((item["caption_evidence"].get("pose_orientation") or {}).get("semantic_pose") or {})
        lint = (meta or {}).get("caption_lint") or {}
        records.append(
            {
                "result_key": item["result_key"],
                "relative_path": item.get("relative_path"),
                "semantic_pose": semantic_pose,
                "caption": caption,
                "word_count": (meta or {}).get("word_count"),
                "compose_seconds": (meta or {}).get("compose_seconds"),
                "caption_lint": lint,
            }
        )
        aggregate_lines.extend(
            [
                f"===== {item['result_key']} =====",
                f"semantic_pose: {json.dumps(semantic_pose, ensure_ascii=False)}",
                f"lint: violations={lint.get('violation_count', 0)} warnings={lint.get('warning_count', 0)}",
                caption or "<missing>",
                "",
            ]
        )

    summary = {
        "schema_version": "compose-semantic-1.5.0-run",
        "projection_revision": "1.5.0",
        "analysis_model": analysis_model_id,
        "compose_model": compose_model_id,
        "compose_model_load_seconds": load_seconds,
        "analysis_source": str(analysis_dir),
        "fusion_source": str(fusion_dir),
        "pose_semantics_source": str(semantics_dir),
        "output_dir": str(output_dir),
        "run_label": run_label,
        "record_count": len(records),
        "missing_sources": missing,
        "invalid_pose_semantics": invalid_semantics,
        "records": records,
    }
    summary_path = output_dir / "compose_semantic_150.index.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    aggregate_path = output_dir / "compose_semantic_150.txt"
    aggregate_path.write_text("\n".join(aggregate_lines), encoding="utf-8")

    print(f"Done. Captions: {output_dir}")
    print(f"      JSON:     {summary_path}")
    print(f"      Text:     {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
