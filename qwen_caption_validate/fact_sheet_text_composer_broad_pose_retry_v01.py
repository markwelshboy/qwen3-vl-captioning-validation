from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v01 as engine
from . import fact_sheet_text_composer_v04 as composer

DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.4"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.4-broad-pose-retry"
SCHEMA_VERSION = "fact-sheet-text-composer-broad-pose-retry-0.1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _broad_pose_violations(record: dict[str, Any]) -> list[str]:
    audit = record.get("caption_audit") if isinstance(record.get("caption_audit"), dict) else {}
    return [
        str(value) for value in (audit.get("violations") or [])
        if str(value).startswith("unauthorized_broad_pose:")
    ]


def _forbidden_groups(violations: list[str]) -> list[str]:
    values: list[str] = []
    for violation in violations:
        _, _, suffix = violation.partition(":")
        values.extend(part.strip() for part in suffix.split(",") if part.strip())
    return sorted(set(values))


def _retry_prompt(record: dict[str, Any]) -> str:
    projection = record.get("evidence_projection") if isinstance(record.get("evidence_projection"), dict) else {}
    previous = str(record.get("caption") or "").strip()
    forbidden = _forbidden_groups(_broad_pose_violations(record))
    forbidden_text = ", ".join(forbidden) if forbidden else "unauthorized broad-pose labels"
    return f"""You are a TEXT-ONLY caption composer performing one narrow policy retry. You cannot inspect the source image.

Return only the corrected caption as plain text.

The previous caption was rejected because it introduced broad-pose language that is not authorized by the governed evidence. Forbidden broad-pose group(s) for this retry: {forbidden_text}.

RULES
- Preserve the supported content, trigger binding, pronoun grammar, framing, local body relationships, torso geometry, head/gaze, appearance, objects, scene, and anatomical laterality from COMPOSER EVIDENCE.
- Do NOT use standing/stand/stands, sitting/sit/seated, lying/lie/lies/reclining/reclined, crouching/crouched, kneeling/kneel, or equivalent broad-pose claims unless `authoritative_facts.body.broad_pose` explicitly authorizes that category.
- If broad pose is not authorized, describe the visible relationship directly instead, e.g. "is shown in an upper-body view", "with the torso oriented horizontally relative to the frame", or "inside a stainless steel elevator".
- Do not remove a precise local body relationship merely to avoid a broad-pose word.
- Do not invent any new fact.
- Preserve the exact trigger as the first grammatical subject when one is supplied. Continue to use the configured pronouns naturally. Trigger re-mention or possessive trigger form is allowed only if needed for attachment clarity.
- Never replace the primary subject with "a person", "a woman", "a man", or "the subject" when a trigger is supplied.

COMPOSER EVIDENCE
{json.dumps(projection, indent=2, ensure_ascii=False)}

PREVIOUS REJECTED CAPTION
{previous}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retry only Phase-5.3 captions rejected for unauthorized broad pose.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--model", default="32b-fp8")
    p.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--vllm-max-model-len", type=int, default=8192)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else run_dir / DEFAULT_INPUT_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    if not input_dir.is_dir():
        print(f"Composer directory not found: {input_dir}", file=sys.stderr)
        return 2

    wanted = set(args.only)
    paths = sorted(input_dir.glob("*.composed.json"))
    if wanted:
        paths = [p for p in paths if p.name.removesuffix(".composed.json") in wanted]
    records = [_read_json(path) for path in paths]
    retry_records = [record for record in records if _broad_pose_violations(record)]
    if not retry_records:
        print("No matching broad-pose violations require retry.")
        return 0

    from .runner import generate_text, load_model, resolve_backend, resolve_model_id, unload_model

    os.environ.setdefault("QWEN_VLLM_TEXT_ONLY_PROFILE", "1")
    model_id = resolve_model_id(args.model)
    backend = resolve_backend(model_id, args.backend)
    print(f"Loading {model_id} once for {len(retry_records)} broad-pose retry call(s) ...")
    loaded = load_model(
        model_id,
        backend=backend,
        dtype=args.dtype,
        quantization="none",
        cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    try:
        for index, source in enumerate(retry_records, start=1):
            key = str(source.get("image_key") or "unknown")
            out_path = output_dir / f"{key}.composed.json"
            caption_path = output_dir / f"{key}.caption.txt"
            if out_path.is_file() and not args.overwrite:
                results.append(_read_json(out_path))
                continue

            violations_before = _broad_pose_violations(source)
            prompt = _retry_prompt(source)
            print(f"\n[{index}/{len(retry_records)}] {key}: broad-pose retry")
            try:
                raw, seconds = generate_text(loaded, prompt, max_new_tokens=args.max_tokens)
                caption = engine._normalize_caption(raw)
                if not caption:
                    raise ValueError("empty retry response")
                projection = source.get("evidence_projection") if isinstance(source.get("evidence_projection"), dict) else {}
                audit = composer._caption_audit(caption, projection)
                status = "ok" if not audit.get("violations") else "needs_review"
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "status": status,
                    "image_key": key,
                    "source_composed_record": source.get("input_fact_sheet"),
                    "model": model_id,
                    "backend": backend,
                    "inference_seconds": seconds,
                    "composer_has_image_access": False,
                    "retry_reason": violations_before,
                    "evidence_projection": projection,
                    "previous_caption": source.get("caption"),
                    "caption": caption,
                    "caption_audit": audit,
                    "raw_response": raw,
                    "retry_count": 1,
                }
                _write_json(out_path, result)
                caption_path.write_text(caption + "\n", encoding="utf-8")
                results.append(result)
                print(f"  {status} | {audit.get('word_count')} words")
                print(f"  {caption}")
                if audit.get("violations"):
                    print("  VIOLATIONS: " + ", ".join(audit["violations"]))
            except Exception as exc:
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "image_key": key,
                    "retry_reason": violations_before,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _write_json(out_path, result)
                results.append(result)
                print(f"ERROR: {result['error']}", file=sys.stderr)
    finally:
        unload_model(loaded)

    counts = Counter(str(record.get("status") or "unknown") for record in results)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "record_count": len(results),
        "status_counts": dict(sorted(counts.items())),
        "retry_policy": "one_text_only_retry_for_unauthorized_broad_pose_only",
        "records": results,
    }
    _write_json(output_dir / "broad_pose_retry.index.json", index)
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
