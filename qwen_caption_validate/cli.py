from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from . import runner as runner_module
from .analysis_v2_normalize import normalize_analysis_v2
from .report import build_report
from .runner import (
    discover_images,
    generate,
    generate_text,
    load_model,
    model_slug,
    parse_json_response,
    render_compose_prompt,
    resolve_backend,
    resolve_model_id,
    unload_model,
    validate_analysis,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS_PROMPT = PACKAGE_ROOT / "prompts" / "analysis_v1.txt"
DEFAULT_COMPOSE_PROMPT = PACKAGE_ROOT / "prompts" / "compose_identity_pose_v1.txt"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "analysis_v1.schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-vl-validate",
        description="Run Qwen3-VL models over an image dataset and compare structured visual analyses/captions.",
    )
    parser.add_argument("dataset", type=Path, help="Folder containing training images (and optional .txt sidecars).")
    parser.add_argument("--models", nargs="+", default=["8b", "32b"], help="Model aliases (8b, 32b, 8b-fp8, 32b-fp8) or Hugging Face model IDs.")
    parser.add_argument("--output", type=Path, default=Path("runs"), help="Root output folder.")
    parser.add_argument("--run-name", help="Stable run name. Reusing it resumes/skips completed outputs.")
    parser.add_argument("--recursive", action="store_true", help="Scan dataset recursively.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help=(
            "Optional glob for relative paths or basenames to process. Repeatable. "
            "Useful for targeted regression reruns, e.g. --include 'jQTv_720x1280_00002.png'."
        ),
    )
    parser.add_argument("--limit", type=int, help="Process only the first N images after include filtering (useful for prompt iteration).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing per-model outputs.")
    parser.add_argument("--analysis-prompt", type=Path, default=DEFAULT_ANALYSIS_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--compose", action="store_true", help="After analysis, generate and cache a training caption from the structured JSON.")
    parser.add_argument("--compose-prompt", type=Path, default=DEFAULT_COMPOSE_PROMPT)
    parser.add_argument("--subject-token", default="sH1Vx")
    parser.add_argument("--detail", choices=["concise", "balanced", "detailed"], default="balanced")
    parser.add_argument("--max-analysis-tokens", type=int, default=1800)
    parser.add_argument("--max-caption-tokens", type=int, default=450)
    parser.add_argument(
        "--analysis-batch-size",
        type=int,
        default=1,
        help="Number of independent image-analysis requests to submit together to vLLM (default: 1).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "transformers", "vllm"],
        default="auto",
        help="Inference backend. auto routes official *-FP8 checkpoints to vLLM and ordinary checkpoints to Transformers.",
    )
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Optional bitsandbytes quantization for the Transformers backend. Prefer native *-FP8 checkpoints with vLLM on an L40S.",
    )
    parser.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"], help="Optional Transformers attention implementation (ignored by vLLM).")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face/vLLM cache directory.")
    parser.add_argument("--min-pixels", type=int, help="Optional Qwen processor minimum image pixels.")
    parser.add_argument("--max-pixels", type=int, help="Optional Qwen processor maximum image pixels.")
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory vLLM may use (default: 0.92; useful for 32B FP8 on a 48 GB L40S).",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=8192,
        help="Maximum vLLM context length. 8192 is ample for this validation workload and avoids wasting KV-cache budget.",
    )
    return parser.parse_args()


def _result_key(relative_path: Path) -> str:
    text = str(relative_path.with_suffix(""))
    return text.replace("/", "__").replace("\\", "__")


def _read_sidecar(image: Path) -> str | None:
    sidecar = image.with_suffix(".txt")
    if not sidecar.exists():
        return None
    try:
        return sidecar.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return sidecar.read_text(encoding="utf-8", errors="replace").strip()


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _matches_include(image: Path, dataset: Path, patterns: list[str]) -> bool:
    if not patterns:
        return True
    relative = str(image.relative_to(dataset)).replace("\\", "/")
    return any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(image.name, pattern) for pattern in patterns)


def _analysis_result(
    *,
    raw: str,
    seconds: float,
    record: dict,
    model_id: str,
    backend: str,
    schema: dict,
) -> dict:
    parsed, parse_error = parse_json_response(raw)
    normalization_actions = []
    if isinstance(parsed, dict) and parsed.get("schema_version") == "2.0":
        parsed, normalization_actions = normalize_analysis_v2(parsed)
    schema_errors = validate_analysis(parsed, schema) if parsed is not None else []
    return {
        "image": record["relative_path"],
        "model": model_id,
        "backend": backend,
        "inference_seconds": seconds,
        "analysis_seconds": seconds,
        "analysis": parsed,
        "raw_response": raw,
        "parse_error": parse_error,
        "normalization_actions": normalization_actions,
        "schema_valid": parsed is not None and not schema_errors,
        "schema_errors": schema_errors,
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        print(f"Dataset folder does not exist: {dataset}", file=sys.stderr)
        return 2

    if not 0.0 < args.vllm_gpu_memory_utilization <= 1.0:
        print("--vllm-gpu-memory-utilization must be > 0 and <= 1", file=sys.stderr)
        return 2
    if args.analysis_batch_size < 1:
        print("--analysis-batch-size must be >= 1", file=sys.stderr)
        return 2

    images = discover_images(dataset, recursive=args.recursive)
    if args.include:
        images = [image for image in images if _matches_include(image, dataset, args.include)]
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print(f"No supported images found in {dataset} for the requested filters.", file=sys.stderr)
        return 2

    analysis_prompt = args.analysis_prompt.read_text(encoding="utf-8")
    compose_prompt = args.compose_prompt.read_text(encoding="utf-8") if args.compose else None
    schema = json.loads(args.schema.read_text(encoding="utf-8"))

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output.expanduser().resolve() / run_name
    report_images = run_dir / "images"
    report_images.mkdir(parents=True, exist_ok=True)

    model_ids = [resolve_model_id(name) for name in args.models]
    model_slugs = [model_slug(model_id) for model_id in model_ids]

    image_records = []
    for image in images:
        rel = image.relative_to(dataset)
        key = _result_key(rel)
        report_copy = report_images / f"{key}{image.suffix.lower()}"
        if not report_copy.exists() or args.overwrite:
            shutil.copy2(image, report_copy)
        image_records.append(
            {
                "relative_path": str(rel),
                "result_key": key,
                "report_image": str(report_copy.relative_to(run_dir)),
                "existing_caption": _read_sidecar(image),
            }
        )

    manifest_path = run_dir / "run.json"
    existing_manifest = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}

    model_map = dict(existing_manifest.get("models", {}))
    model_runtime = dict(existing_manifest.get("model_runtime", {}))
    for slug, model_id in zip(model_slugs, model_ids):
        model_map[slug] = model_id

    manifest = {
        "run_name": run_name,
        "dataset": str(dataset),
        "models": model_map,
        "model_runtime": model_runtime,
        "analysis_prompt": str(args.analysis_prompt),
        "schema": str(args.schema),
        "compose": bool(args.compose),
        "compose_prompt": str(args.compose_prompt) if args.compose else None,
        "subject_token": args.subject_token,
        "detail": args.detail,
        "backend": args.backend,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "attention": args.attn,
        "analysis_batch_size": args.analysis_batch_size,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_model_len": args.vllm_max_model_len,
        "include_patterns": args.include,
        "images": image_records,
    }
    _write_manifest(manifest_path, manifest)

    for model_id, slug in zip(model_ids, model_slugs):
        model_dir = run_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        pending = []
        for image, record in zip(images, image_records):
            result_path = model_dir / f"{record['result_key']}.analysis.json"
            caption_path = model_dir / f"{record['result_key']}.caption.txt"
            caption_meta_path = model_dir / f"{record['result_key']}.caption.json"
            needs_analysis = args.overwrite or not result_path.exists()
            needs_caption = args.compose and (args.overwrite or not caption_path.exists() or not caption_meta_path.exists())
            if needs_analysis or needs_caption:
                pending.append((image, record, needs_analysis, needs_caption))

        if not pending:
            print(f"[{model_id}] nothing to do; all requested outputs already exist.")
            continue

        resolved_backend = resolve_backend(model_id, args.backend)
        print(f"\nLoading {model_id} with {resolved_backend} ...")
        loaded = load_model(
            model_id,
            backend=args.backend,
            dtype=args.dtype,
            quantization=args.quantization,
            attn_implementation=args.attn,
            cache_dir=args.cache_dir,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(f"Loaded in {loaded.load_seconds:.1f}s. Processing {len(pending)} image(s).")
        effective_batch_size = args.analysis_batch_size if loaded.backend == "vllm" else 1
        if effective_batch_size > 1:
            print(f"Analyze request batch size: {effective_batch_size}")
        model_runtime[slug] = {
            "load_seconds": loaded.load_seconds,
            "backend": loaded.backend,
            "quantization": loaded.quantization,
            "dtype": args.dtype,
            "attention": args.attn if loaded.backend == "transformers" else None,
            "analysis_batch_size": effective_batch_size,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization if loaded.backend == "vllm" else None,
            "vllm_max_model_len": args.vllm_max_model_len if loaded.backend == "vllm" else None,
        }
        manifest["model_runtime"] = model_runtime
        _write_manifest(manifest_path, manifest)

        try:
            jsonl_path = model_dir / "results.jsonl"
            with jsonl_path.open("a", encoding="utf-8") as jsonl:
                if effective_batch_size > 1:
                    batch_generate = getattr(runner_module, "generate_batch", None)
                    if not callable(batch_generate):
                        raise RuntimeError(
                            "Analyze batching requested but the active runner does not provide generate_batch"
                        )

                    results_by_key: dict[str, dict] = {}
                    analysis_pending = [item for item in pending if item[2]]
                    for offset in range(0, len(analysis_pending), effective_batch_size):
                        batch = analysis_pending[offset : offset + effective_batch_size]
                        batch_images = [item[0] for item in batch]
                        generated = batch_generate(
                            loaded,
                            batch_images,
                            analysis_prompt,
                            max_new_tokens=args.max_analysis_tokens,
                        )
                        if len(generated) != len(batch):
                            raise RuntimeError(
                                f"Analyze batch returned {len(generated)} result(s) for {len(batch)} request(s)"
                            )
                        for (_, record, _, _), (raw, seconds) in zip(batch, generated):
                            result = _analysis_result(
                                raw=raw,
                                seconds=seconds,
                                record=record,
                                model_id=model_id,
                                backend=loaded.backend,
                                schema=schema,
                            )
                            result_path = model_dir / f"{record['result_key']}.analysis.json"
                            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
                            jsonl.flush()
                            results_by_key[record["result_key"]] = result

                    for image, record, needs_analysis, needs_caption in tqdm(pending, desc=slug):
                        result = results_by_key.get(record["result_key"])
                        result_path = model_dir / f"{record['result_key']}.analysis.json"
                        caption_path = model_dir / f"{record['result_key']}.caption.txt"
                        caption_meta_path = model_dir / f"{record['result_key']}.caption.json"
                        if result is None:
                            result = json.loads(result_path.read_text(encoding="utf-8"))

                        if needs_caption:
                            analysis = result.get("analysis") if result else None
                            if analysis is None:
                                caption_path.with_suffix(".caption.error.txt").write_text(
                                    "Caption skipped because analysis JSON could not be parsed.\n",
                                    encoding="utf-8",
                                )
                            else:
                                prompt = render_compose_prompt(compose_prompt, analysis, args.subject_token, args.detail)
                                caption, compose_seconds = generate_text(
                                    loaded,
                                    prompt,
                                    max_new_tokens=args.max_caption_tokens,
                                )
                                caption = caption.strip()
                                caption_path.write_text(caption + "\n", encoding="utf-8")
                                caption_meta = {
                                    "image": record["relative_path"],
                                    "model": model_id,
                                    "backend": loaded.backend,
                                    "compose_seconds": compose_seconds,
                                    "detail": args.detail,
                                    "subject_token": args.subject_token,
                                    "caption": caption,
                                }
                                caption_meta_path.write_text(
                                    json.dumps(caption_meta, indent=2, ensure_ascii=False),
                                    encoding="utf-8",
                                )
                else:
                    for image, record, needs_analysis, needs_caption in tqdm(pending, desc=slug):
                        result_path = model_dir / f"{record['result_key']}.analysis.json"
                        caption_path = model_dir / f"{record['result_key']}.caption.txt"
                        caption_meta_path = model_dir / f"{record['result_key']}.caption.json"
                        result = None

                        if needs_analysis:
                            raw, seconds = generate(
                                loaded,
                                image,
                                analysis_prompt,
                                max_new_tokens=args.max_analysis_tokens,
                            )
                            result = _analysis_result(
                                raw=raw,
                                seconds=seconds,
                                record=record,
                                model_id=model_id,
                                backend=loaded.backend,
                                schema=schema,
                            )
                            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
                            jsonl.flush()
                        else:
                            result = json.loads(result_path.read_text(encoding="utf-8"))

                        if needs_caption:
                            analysis = result.get("analysis") if result else None
                            if analysis is None:
                                caption_path.with_suffix(".caption.error.txt").write_text(
                                    "Caption skipped because analysis JSON could not be parsed.\n",
                                    encoding="utf-8",
                                )
                            else:
                                prompt = render_compose_prompt(compose_prompt, analysis, args.subject_token, args.detail)
                                caption, compose_seconds = generate_text(
                                    loaded,
                                    prompt,
                                    max_new_tokens=args.max_caption_tokens,
                                )
                                caption = caption.strip()
                                caption_path.write_text(caption + "\n", encoding="utf-8")
                                caption_meta = {
                                    "image": record["relative_path"],
                                    "model": model_id,
                                    "backend": loaded.backend,
                                    "compose_seconds": compose_seconds,
                                    "detail": args.detail,
                                    "subject_token": args.subject_token,
                                    "caption": caption,
                                }
                                caption_meta_path.write_text(
                                    json.dumps(caption_meta, indent=2, ensure_ascii=False),
                                    encoding="utf-8",
                                )
        finally:
            print(f"Unloading {model_id} ...")
            unload_model(loaded)

        build_report(run_dir, list(model_map.keys()))

    report = build_report(run_dir, list(model_map.keys()))
    print(f"\nDone. Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
