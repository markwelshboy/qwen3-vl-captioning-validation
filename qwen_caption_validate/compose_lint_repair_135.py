from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .caption_projection_135 import lint_caption
from .runner import generate_text, load_model, model_slug, resolve_model_id, unload_model


REPAIR_PROMPT = """You are repairing an existing text-only identity-training caption against a deterministic governance contract.

Return ONLY the revised caption. Do not explain the repair.

Rules:
- Use only facts present in CAPTION_EVIDENCE_JSON. Do not add or reconstruct visual facts.
- Preserve the trigger token as the grammatical opening.
- Preserve useful supported scene, appearance, pose, framing, lighting, and every required claim unless the lint finding specifically proves the wording is invalid.
- Resolve every listed lint violation. If an anatomical side is not qualified, rewrite side-neutrally or use safe bilateral wording such as \"both shoulders\" when the evidence supports bilateral visibility. Never infer the complementary anatomical side.
- Do not transfer side-bound geometry such as nearer/farther/forward/retracted when laterality is not explicitly qualified for that relation.
- Preserve required support relations and signed-depth claims exactly in meaning.
- Keep the caption natural and dense; make the minimum semantic change needed for compliance.

CAPTION_EVIDENCE_JSON:
{{CAPTION_EVIDENCE_JSON}}

ORIGINAL_CAPTION:
{{ORIGINAL_CAPTION}}

LINT_FINDINGS_JSON:
{{LINT_FINDINGS_JSON}}
"""


def _safe_label(value: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return text.strip("-._") or "default"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _needs_repair(lint: dict[str, Any], include_warnings: bool) -> bool:
    if int(lint.get("violation_count") or 0) > 0:
        return True
    return include_warnings and int(lint.get("warning_count") or 0) > 0


def _render_repair_prompt(caption: str, evidence: dict[str, Any], lint: dict[str, Any]) -> str:
    return (
        REPAIR_PROMPT.replace(
            "{{CAPTION_EVIDENCE_JSON}}",
            json.dumps(evidence, indent=2, ensure_ascii=False),
        )
        .replace("{{ORIGINAL_CAPTION}}", caption.strip())
        .replace("{{LINT_FINDINGS_JSON}}", json.dumps(lint, indent=2, ensure_ascii=False))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-compose-lint-repair-135",
        description="One-shot text-only repair of Projection 1.3.5 governed captions that fail authority lint.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis-model", default="32b-fp8")
    parser.add_argument("--compose-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--source-run-label", required=True)
    parser.add_argument("--run-label", help="Output run label; default: <source>-repair1")
    parser.add_argument("--only", nargs="+", help="Only repair/copy result keys containing one of these strings.")
    parser.add_argument("--include-warnings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-caption-tokens", type=int, default=450)
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    analysis_id = resolve_model_id(args.analysis_model)
    compose_id = resolve_model_id(args.compose_model)
    analysis_slug = model_slug(analysis_id)
    compose_slug = model_slug(compose_id)
    source_label = _safe_label(args.source_run_label)
    target_label = _safe_label(args.run_label or f"{source_label}-repair1")

    source_dir = run_dir / "compose_fusion_compare" / f"{analysis_slug}__compose__{compose_slug}__{source_label}"
    target_dir = run_dir / "compose_fusion_compare" / f"{analysis_slug}__compose__{compose_slug}__{target_label}"
    if not source_dir.is_dir():
        print(f"Source Compose directory not found: {source_dir}", file=sys.stderr)
        return 2
    target_dir.mkdir(parents=True, exist_ok=True)

    source_meta_paths = sorted(source_dir.glob("*.fusion-safe.json"))
    if args.only:
        needles = tuple(args.only)
        source_meta_paths = [p for p in source_meta_paths if any(n in p.name for n in needles)]
    if not source_meta_paths:
        print("No governed captions matched.", file=sys.stderr)
        return 2

    work: list[tuple[Path, dict[str, Any], bool]] = []
    repair_count = 0
    for source_meta in source_meta_paths:
        meta = _read_json(source_meta)
        if meta is None:
            continue
        lint = meta.get("caption_lint") or {}
        needs = _needs_repair(lint, bool(args.include_warnings))
        work.append((source_meta, meta, needs))
        repair_count += int(needs)

    loaded = None
    if repair_count:
        print(f"Loading {compose_id} for {repair_count} one-shot lint repair(s) ...")
        loaded = load_model(
            compose_id,
            backend=args.backend,
            dtype=args.dtype,
            quantization=args.quantization,
            attn_implementation=args.attn,
            cache_dir=args.cache_dir,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(f"Loaded in {loaded.load_seconds:.1f}s. Repair remains text-only.")

    records: list[dict[str, Any]] = []
    repaired = repaired_clean = copied_clean = still_failed = 0
    try:
        for source_meta, source, needs in work:
            key = source_meta.name.removesuffix(".fusion-safe.json")
            source_txt = source_dir / f"{key}.fusion-safe.txt"
            target_meta = target_dir / source_meta.name
            target_txt = target_dir / source_txt.name
            if target_meta.exists() and target_txt.exists() and not args.overwrite:
                existing = _read_json(target_meta) or {}
                records.append({
                    "image_key": key,
                    "status": "reused",
                    "repair_attempted": bool(existing.get("repair_attempted")),
                    "passed": bool((existing.get("caption_lint") or {}).get("passed")),
                })
                continue

            original_caption = str(source.get("caption") or "").strip()
            original_lint = copy.deepcopy(source.get("caption_lint") or {})
            evidence = copy.deepcopy(source.get("caption_evidence") or {})

            if not needs:
                shutil.copy2(source_txt, target_txt)
                out = copy.deepcopy(source)
                out.update({
                    "run_label": target_label,
                    "repair_attempted": False,
                    "repair_source_run_label": source_label,
                })
                _write_json(target_meta, out)
                copied_clean += 1
                records.append({"image_key": key, "status": "copied_clean", "repair_attempted": False, "passed": True})
                continue

            prompt = _render_repair_prompt(original_caption, evidence, original_lint)
            revised, repair_seconds = generate_text(loaded, prompt, max_new_tokens=args.max_caption_tokens)
            revised_lint = lint_caption(revised, evidence)
            out = copy.deepcopy(source)
            initial_seconds = float(source.get("compose_seconds") or 0.0)
            out.update({
                "run_label": target_label,
                "caption": revised.strip(),
                "word_count": len(revised.strip().split()),
                "caption_lint": revised_lint,
                "repair_attempted": True,
                "repair_source_run_label": source_label,
                "repair_source_caption": original_caption,
                "repair_source_lint": original_lint,
                "initial_compose_seconds": initial_seconds,
                "repair_seconds": repair_seconds,
                "total_compose_seconds": initial_seconds + float(repair_seconds),
            })
            target_txt.write_text(revised.strip() + "\n", encoding="utf-8")
            _write_json(target_meta, out)
            repaired += 1
            if revised_lint.get("passed") and not revised_lint.get("warning_count"):
                repaired_clean += 1
            else:
                still_failed += 1
            records.append({
                "image_key": key,
                "status": "repaired",
                "repair_attempted": True,
                "passed": bool(revised_lint.get("passed")),
                "warning_count": int(revised_lint.get("warning_count") or 0),
                "violation_count": int(revised_lint.get("violation_count") or 0),
                "repair_seconds": repair_seconds,
            })
    finally:
        if loaded is not None:
            unload_model(loaded)

    index = {
        "schema_version": "compose-lint-repair-1.0",
        "governance_revision": "1.3.5",
        "run_dir": str(run_dir),
        "analysis_model": analysis_id,
        "compose_model": compose_id,
        "source_run_label": source_label,
        "run_label": target_label,
        "source_dir": str(source_dir),
        "output_dir": str(target_dir),
        "include_warnings": bool(args.include_warnings),
        "matched": len(work),
        "copied_clean": copied_clean,
        "repaired": repaired,
        "repaired_clean": repaired_clean,
        "still_failed": still_failed,
        "records": records,
    }
    _write_json(target_dir / "lint_repair.index.json", index)
    print(f"Repair output: {target_dir}")
    print(f"Matched: {len(work)}; copied clean: {copied_clean}; repaired: {repaired}; repaired clean: {repaired_clean}; still failed: {still_failed}")
    return 0 if work else 2


if __name__ == "__main__":
    raise SystemExit(main())
