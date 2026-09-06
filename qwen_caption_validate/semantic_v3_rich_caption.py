from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import extract_v3
from .extract_v3_wire import _install_image_only_vllm
from .runner import (
    generate,
    load_model,
    model_slug,
    resolve_backend,
    resolve_model_id,
    unload_model,
)


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_caption_v01.txt"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 800
ARTIFACT_VERSION = "semantic-v3-rich-caption-0.1"
RUN_VERSION = "semantic-v3-rich-caption-0.1-run"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _caption_stats(text: str) -> dict[str, int]:
    words = re.findall(r"\S+", text)
    sentences = [piece for piece in re.split(r"(?<=[.!?])\s+", text.strip()) if piece.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
    }


def _record_summary(payload: dict[str, Any]) -> dict[str, Any]:
    performance = payload.get("performance") if isinstance(payload.get("performance"), dict) else {}
    stats = payload.get("caption_stats") if isinstance(payload.get("caption_stats"), dict) else {}
    return {
        "image_key": payload.get("image_key"),
        "finish_reason": performance.get("finish_reason"),
        "output_tokens": performance.get("output_tokens"),
        "words": stats.get("words"),
        "sentences": stats.get("sentences"),
        "characters": stats.get("characters"),
    }


def _generate_vllm_batch(
    loaded,
    image_paths: list[Path],
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import SamplingParams

    requests: list[dict[str, Any]] = []
    prepare_seconds: list[float] = []
    prepare_total_started = time.perf_counter()
    for image_path in image_paths:
        item_started = time.perf_counter()
        requests.append(extract_v3.runner_module._prepare_vllm_multimodal(loaded, image_path, prompt))
        prepare_seconds.append(time.perf_counter() - item_started)
    prepare_total = time.perf_counter() - prepare_total_started

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    generation_started = time.perf_counter()
    outputs = loaded.model.generate(requests, sampling_params=sampling, use_tqdm=False)
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != len(image_paths):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(image_paths)} rich-caption requests")

    items: list[dict[str, Any]] = []
    for image_path, prepared_seconds, output in zip(image_paths, prepare_seconds, outputs):
        perf = extract_v3._request_perf_fields(output, max_new_tokens)
        perf["image"] = image_path
        perf["prepare_seconds"] = prepared_seconds
        items.append(perf)

    output_tokens = sum(int(item.get("output_tokens") or 0) for item in items)
    return items, {
        "prepare_seconds": prepare_total,
        "generation_seconds": generation_seconds,
        "output_tokens": output_tokens,
        "aggregate_output_tokens_per_second": output_tokens / generation_seconds if generation_seconds > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a rich natural-language visual caption in one image-conditioned VLM call. "
            "This is the primary semantic draft; Pose/Gestalt correction happens downstream."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2
    if args.max_tokens < 1:
        print("--max-tokens must be >= 1", file=sys.stderr)
        return 2

    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    try:
        images = [
            path
            for path in extract_v3._discover_images(run_dir, images_dir)
            if extract_v3._matches(path, args.only)
        ]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not images:
        print("No matching images found for rich captioning.", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    backend = resolve_backend(model_id, args.backend)
    slug = model_slug(model_id)
    output_dir = (
        args.output_dir or (run_dir / "semantic-v3" / "rich-caption-v0.1" / slug)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")

    pending: list[tuple[Path, str, Path]] = []
    reused: list[dict[str, Any]] = []
    for image in images:
        key = extract_v3._result_key(image)
        out_path = output_dir / f"{key}.rich_caption.json"
        if args.overwrite or not out_path.exists():
            pending.append((image, key, out_path))
        else:
            reused.append(_record_summary(_read_json(out_path)))

    if pending and backend == "vllm":
        _install_image_only_vllm()

    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    loaded = None
    if pending:
        print(f"Loading {model_id} for rich semantic captions ...")
        loaded = load_model(
            model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(
            f"Loaded in {loaded.load_seconds:.2f}s. Captioning {len(pending)} image(s). "
            f"batch_size={args.batch_size} max_tokens={args.max_tokens}"
        )

    try:
        if loaded is not None and loaded.backend == "vllm":
            total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
            for batch_index, offset in enumerate(range(0, len(pending), args.batch_size), start=1):
                batch_started = time.perf_counter()
                batch = pending[offset : offset + args.batch_size]
                items, generation_perf = _generate_vllm_batch(
                    loaded,
                    [item[0] for item in batch],
                    prompt,
                    max_new_tokens=args.max_tokens,
                )
                for (image, key, out_path), item in zip(batch, items):
                    caption = str(item.get("text") or "").strip()
                    stats = _caption_stats(caption)
                    performance = {
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "prepare_seconds": item.get("prepare_seconds"),
                        "shared_batch_generation_seconds": generation_perf["generation_seconds"],
                        "prompt_tokens": item.get("prompt_tokens"),
                        "output_tokens": item.get("output_tokens"),
                        "max_new_tokens": args.max_tokens,
                        "finish_reason": item.get("finish_reason"),
                        "ttft_seconds": item.get("ttft_seconds"),
                        "queue_seconds": item.get("queue_seconds"),
                        "decode_seconds": item.get("decode_seconds"),
                        "decode_tokens_per_second": item.get("decode_tokens_per_second"),
                        "engine_e2e_seconds": item.get("engine_e2e_seconds"),
                    }
                    payload = {
                        "schema_version": ARTIFACT_VERSION,
                        "image_key": key,
                        "image": str(image),
                        "model": model_id,
                        "backend": loaded.backend,
                        "prompt": str(prompt_path),
                        "caption": caption,
                        "raw_response": caption,
                        "caption_stats": stats,
                        "performance": performance,
                    }
                    _write_json(out_path, payload)
                    generated.append(_record_summary(payload))
                    print(
                        "RICH_CAPTION_PERF "
                        f"image={image.name} batch={batch_index}/{total_batches} "
                        f"tokens={item.get('output_tokens')}/{args.max_tokens} "
                        f"words={stats['words']} sentences={stats['sentences']} "
                        f"finish={item.get('finish_reason')}"
                    )

                batch_wall = time.perf_counter() - batch_started
                batch_runtime.append({
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "images": [item[1] for item in batch],
                    "generation_seconds": generation_perf["generation_seconds"],
                    "wall_seconds": batch_wall,
                    "amortized_seconds_per_image": batch_wall / len(batch) if batch else 0.0,
                    "output_tokens": generation_perf["output_tokens"],
                    "aggregate_output_tokens_per_second": generation_perf["aggregate_output_tokens_per_second"],
                })
        elif loaded is not None:
            for batch_index, (image, key, out_path) in enumerate(pending, start=1):
                caption, inference_seconds = generate(
                    loaded,
                    image,
                    prompt,
                    max_new_tokens=args.max_tokens,
                )
                caption = caption.strip()
                stats = _caption_stats(caption)
                payload = {
                    "schema_version": ARTIFACT_VERSION,
                    "image_key": key,
                    "image": str(image),
                    "model": model_id,
                    "backend": loaded.backend,
                    "prompt": str(prompt_path),
                    "caption": caption,
                    "raw_response": caption,
                    "caption_stats": stats,
                    "performance": {
                        "batch_index": batch_index,
                        "batch_size": 1,
                        "inference_seconds": inference_seconds,
                        "max_new_tokens": args.max_tokens,
                        "finish_reason": None,
                        "output_tokens": None,
                    },
                }
                _write_json(out_path, payload)
                generated.append(_record_summary(payload))
                print(
                    f"RICH_CAPTION_PERF image={image.name} words={stats['words']} "
                    f"sentences={stats['sentences']} inference={inference_seconds:.3f}s"
                )
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(reused + generated, key=lambda item: str(item.get("image_key") or ""))
    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "images_dir": str(images_dir or (run_dir / "images")),
        "model": model_id,
        "backend": backend,
        "prompt": str(prompt_path),
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "vllm_max_model_len": args.vllm_max_model_len,
        "record_count": len(records),
        "generated": len(generated),
        "reused": len(reused),
        "batch_runtime": batch_runtime,
        "records": records,
    }
    _write_json(output_dir / "rich_caption.index.json", index)
    print(f"Rich semantic captions: {output_dir}")
    print(f"Records: {len(records)}; generated: {len(generated)}; reused: {len(reused)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
