from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from . import runner as runner_module
from .extract_v3_contract import audit_extract_contract
from .runner import (
    generate,
    load_model,
    model_slug,
    parse_json_response,
    resolve_backend,
    resolve_model_id,
    unload_model,
    validate_analysis,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "extract_v3.txt"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "extract_v3.schema.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_BATCH_SIZE = 2


def _install_image_only_vllm() -> None:
    try:
        import vllm
    except ImportError:
        return

    original_llm = vllm.LLM
    if getattr(original_llm, "_extract_v3_image_only", False):
        return

    def image_only_llm(*args, **kwargs):
        requested = kwargs.get("limit_mm_per_prompt")
        if requested is not None and requested != {"image": 1, "video": 0}:
            raise RuntimeError(
                "Extract v3 requires image-only vLLM limits {'image': 1, 'video': 0}; "
                f"refusing conflicting limits {requested!r}"
            )
        kwargs["limit_mm_per_prompt"] = {"image": 1, "video": 0}
        return original_llm(*args, **kwargs)

    image_only_llm._extract_v3_image_only = True  # type: ignore[attr-defined]
    vllm.LLM = image_only_llm


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _discover_images(run_dir: Path, images_dir: Path | None) -> list[Path]:
    root = images_dir or (run_dir / "images")
    if not root.is_dir():
        raise FileNotFoundError(f"Extract image directory not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _matches(path: Path, only: list[str]) -> bool:
    if not only:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) or pattern in path.name for pattern in only)


def _result_key(path: Path) -> str:
    return path.stem


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_number(metrics: Any, name: str) -> float | None:
    if metrics is None:
        return None
    return _finite_number(getattr(metrics, name, None))


def _duration(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None or later < earlier:
        return None
    return later - earlier


def _fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def _request_perf_fields(request_output: Any, max_new_tokens: int) -> dict[str, Any]:
    completion = request_output.outputs[0]
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None) or []
    output_token_ids = getattr(completion, "token_ids", None) or []
    output_tokens = len(output_token_ids)

    metrics = getattr(request_output, "metrics", None)
    arrival = _metric_number(metrics, "arrival_time")
    scheduled = _metric_number(metrics, "first_scheduled_time")
    first_token = _metric_number(metrics, "first_token_time")
    finished = _metric_number(metrics, "finished_time")
    ttft = _duration(first_token, arrival)
    queue = _duration(scheduled, arrival)
    engine_e2e = _duration(finished, arrival)
    decode = _duration(finished, first_token)
    decode_tok_s = None
    if decode is not None and decode > 0 and output_tokens > 1:
        decode_tok_s = (output_tokens - 1) / decode

    return {
        "text": completion.text.strip(),
        "prompt_tokens": len(prompt_token_ids),
        "output_tokens": output_tokens,
        "finish_reason": getattr(completion, "finish_reason", None),
        "ttft_seconds": ttft,
        "queue_seconds": queue,
        "decode_seconds": decode,
        "decode_tokens_per_second": decode_tok_s,
        "engine_e2e_seconds": engine_e2e,
        "max_new_tokens": max_new_tokens,
    }


def _generate_vllm_batch_profiled(
    loaded,
    image_paths: list[Path],
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import SamplingParams

    requests = []
    prepare_seconds_by_image: list[float] = []
    prepare_started = time.perf_counter()
    for image_path in image_paths:
        item_started = time.perf_counter()
        requests.append(runner_module._prepare_vllm_multimodal(loaded, image_path, prompt))
        prepare_seconds_by_image.append(time.perf_counter() - item_started)
    prepare_total = time.perf_counter() - prepare_started

    generation_started = time.perf_counter()
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = loaded.model.generate(
        requests,
        sampling_params=sampling,
        use_tqdm=False,
    )
    generation_seconds = time.perf_counter() - generation_started

    if len(outputs) != len(image_paths):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(image_paths)} Extract requests"
        )

    items: list[dict[str, Any]] = []
    for image_path, item_prepare, output in zip(image_paths, prepare_seconds_by_image, outputs):
        fields = _request_perf_fields(output, max_new_tokens)
        fields["image"] = image_path
        fields["prepare_seconds"] = item_prepare
        fields["shared_generation_seconds"] = generation_seconds
        items.append(fields)

    total_output_tokens = sum(int(item["output_tokens"]) for item in items)
    batch = {
        "batch_size": len(image_paths),
        "prepare_seconds": prepare_total,
        "generation_seconds": generation_seconds,
        "output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": (
            total_output_tokens / generation_seconds if generation_seconds > 0 else 0.0
        ),
    }
    return items, batch


def _record_summary(payload: dict[str, Any]) -> dict[str, Any]:
    extract = payload.get("extract") if isinstance(payload.get("extract"), dict) else {}
    hypotheses = extract.get("hypotheses") if isinstance(extract.get("hypotheses"), dict) else {}
    posture = hypotheses.get("posture") if isinstance(hypotheses.get("posture"), dict) else {}
    orientation = hypotheses.get("torso_orientation") if isinstance(hypotheses.get("torso_orientation"), dict) else {}
    perf = payload.get("performance") if isinstance(payload.get("performance"), dict) else {}
    return {
        "image_key": payload.get("image_key"),
        "schema_valid": bool(payload.get("schema_valid")),
        "contract": payload.get("contract") or {},
        "entity_count": len(extract.get("entities") or []) if isinstance(extract, dict) else 0,
        "relation_count": len(extract.get("relations") or []) if isinstance(extract, dict) else 0,
        "posture_candidate": posture.get("value"),
        "torso_orientation_candidate": orientation.get("orientation_band"),
        "inference_seconds": payload.get("inference_seconds"),
        "output_tokens": perf.get("output_tokens"),
        "finish_reason": perf.get("finish_reason"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-visual-extract-v3",
        description=(
            "Run the single image-conditioned V3 visual Extract over cached run images. "
            "Later Analyze/Gestalt stages should reason from these records without re-reading pixels."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Validation run containing images/, used as the persistent output root.")
    parser.add_argument("--images-dir", type=Path, help="Optional image directory override; outputs still live under run_dir.")
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4200)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of independent image Extract requests submitted together to vLLM (default: 2).",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def _build_payload(
    *,
    image: Path,
    key: str,
    model_id: str,
    backend: str,
    raw: str,
    schema: dict[str, Any],
    inference_seconds: float,
    performance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    post_started = time.perf_counter()

    parse_started = time.perf_counter()
    parsed, parse_error = parse_json_response(raw)
    parse_seconds = time.perf_counter() - parse_started

    schema_started = time.perf_counter()
    schema_errors = validate_analysis(parsed, schema) if isinstance(parsed, dict) else []
    schema_seconds = time.perf_counter() - schema_started

    contract_started = time.perf_counter()
    contract = audit_extract_contract(parsed) if isinstance(parsed, dict) else {
        "schema_version": "visual-extract-contract-audit-0.1",
        "analyze_reconstructable": False,
        "gestalt_reconstructable": False,
        "analyze_missing_paths": ["unparsed_extract"],
        "gestalt_missing_paths": ["unparsed_extract"],
    }
    contract_seconds = time.perf_counter() - contract_started
    post_seconds = time.perf_counter() - post_started

    payload = {
        "schema_version": "visual-extract-artifact-3.0",
        "image_key": key,
        "image": str(image),
        "model": model_id,
        "backend": backend,
        "inference_seconds": inference_seconds,
        "performance": performance,
        "extract": parsed,
        "raw_response": raw,
        "parse_error": parse_error,
        "schema_valid": isinstance(parsed, dict) and not schema_errors,
        "schema_errors": schema_errors,
        "contract": contract,
    }
    return payload, {
        "parse_seconds": parse_seconds,
        "schema_seconds": schema_seconds,
        "contract_seconds": contract_seconds,
        "post_seconds": post_seconds,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2

    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    try:
        images = [path for path in _discover_images(run_dir, images_dir) if _matches(path, args.only)]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not images:
        print("No matching images found for Extract v3.", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    output_dir = (args.output_dir or (run_dir / "extract-v3.0" / slug)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    schema_path = args.schema.expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = _read_json(schema_path)

    pending: list[tuple[Path, str, Path]] = []
    reused: list[dict[str, Any]] = []
    for image in images:
        key = _result_key(image)
        out_path = output_dir / f"{key}.extract.json"
        if args.overwrite or not out_path.exists():
            pending.append((image, key, out_path))
        else:
            reused.append(_record_summary(_read_json(out_path)))

    if pending and resolve_backend(model_id, args.backend) == "vllm":
        _install_image_only_vllm()

    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    loaded = None
    effective_batch_size = 1
    if pending:
        print(f"Loading {model_id} for V3 visual Extract ...")
        loaded = load_model(
            model_id,
            backend=args.backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        effective_batch_size = args.batch_size if loaded.backend == "vllm" else 1
        print(
            f"Loaded in {loaded.load_seconds:.2f}s. Extracting {len(pending)} image(s). "
            f"batch_size={effective_batch_size} max_tokens={args.max_tokens}"
        )

    try:
        total_batches = (len(pending) + effective_batch_size - 1) // effective_batch_size
        for batch_index, offset in enumerate(range(0, len(pending), effective_batch_size), start=1):
            batch_started = time.perf_counter()
            batch = pending[offset : offset + effective_batch_size]
            batch_images = [item[0] for item in batch]

            if loaded is not None and loaded.backend == "vllm":
                generated_items, generation_perf = _generate_vllm_batch_profiled(
                    loaded,
                    batch_images,
                    prompt,
                    max_new_tokens=args.max_tokens,
                )
            else:
                generated_items = []
                generation_started = time.perf_counter()
                for image in batch_images:
                    raw, seconds = generate(loaded, image, prompt, max_new_tokens=args.max_tokens)
                    generated_items.append({
                        "image": image,
                        "text": raw,
                        "prepare_seconds": 0.0,
                        "shared_generation_seconds": seconds,
                        "prompt_tokens": None,
                        "output_tokens": None,
                        "finish_reason": None,
                        "ttft_seconds": None,
                        "queue_seconds": None,
                        "decode_seconds": None,
                        "decode_tokens_per_second": None,
                        "engine_e2e_seconds": seconds,
                        "max_new_tokens": args.max_tokens,
                    })
                generation_perf = {
                    "batch_size": len(batch),
                    "prepare_seconds": 0.0,
                    "generation_seconds": time.perf_counter() - generation_started,
                    "output_tokens": None,
                    "aggregate_output_tokens_per_second": None,
                }

            post_total = 0.0
            write_total = 0.0
            for (image, key, out_path), item in zip(batch, generated_items):
                raw = str(item["text"])
                request_perf = {
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
                payload, post_perf = _build_payload(
                    image=image,
                    key=key,
                    model_id=model_id,
                    backend=loaded.backend if loaded is not None else "unknown",
                    raw=raw,
                    schema=schema,
                    inference_seconds=float(generation_perf["generation_seconds"]),
                    performance=request_perf,
                )
                post_total += post_perf["post_seconds"]
                request_perf.update(post_perf)

                write_started = time.perf_counter()
                _write_json(out_path, payload)
                write_seconds = time.perf_counter() - write_started
                write_total += write_seconds
                request_perf["write_seconds"] = write_seconds

                record = _record_summary(payload)
                generated.append(record)
                decode_rate = item.get("decode_tokens_per_second")
                decode_rate_text = "n/a" if decode_rate is None else f"{decode_rate:.2f}"
                print(
                    "EXTRACT_PERF "
                    f"image={image.name} batch={batch_index}/{total_batches} "
                    f"prepare={float(item.get('prepare_seconds') or 0.0):.3f}s "
                    f"batch_generate={generation_perf['generation_seconds']:.3f}s "
                    f"ttft={_fmt_seconds(item.get('ttft_seconds'))} "
                    f"queue={_fmt_seconds(item.get('queue_seconds'))} "
                    f"decode={_fmt_seconds(item.get('decode_seconds'))} "
                    f"decode_tok_s={decode_rate_text} "
                    f"engine_e2e={_fmt_seconds(item.get('engine_e2e_seconds'))} "
                    f"tokens={item.get('output_tokens')}/{args.max_tokens} "
                    f"finish={item.get('finish_reason')} "
                    f"parse={post_perf['parse_seconds']:.3f}s "
                    f"schema={post_perf['schema_seconds']:.3f}s "
                    f"contract={post_perf['contract_seconds']:.3f}s "
                    f"write={write_seconds:.3f}s"
                )
                print(
                    f"{key}: schema={'ok' if record['schema_valid'] else 'FAIL'} "
                    f"analyze={'ok' if record['contract'].get('analyze_reconstructable') else 'missing'} "
                    f"gestalt={'ok' if record['contract'].get('gestalt_reconstructable') else 'missing'} "
                    f"entities={record['entity_count']} relations={record['relation_count']} "
                    f"posture={record['posture_candidate']} torso={record['torso_orientation_candidate']}"
                )

            batch_wall = time.perf_counter() - batch_started
            amortized = batch_wall / len(batch) if batch else 0.0
            output_tokens = generation_perf.get("output_tokens")
            aggregate_tok_s = generation_perf.get("aggregate_output_tokens_per_second")
            output_tokens_text = "n/a" if output_tokens is None else str(output_tokens)
            aggregate_tok_s_text = "n/a" if aggregate_tok_s is None else f"{aggregate_tok_s:.2f}"
            batch_record = {
                "batch_index": batch_index,
                "batch_size": len(batch),
                "images": [item[1] for item in batch],
                "prepare_seconds": generation_perf["prepare_seconds"],
                "generation_seconds": generation_perf["generation_seconds"],
                "post_seconds": post_total,
                "write_seconds": write_total,
                "wall_seconds": batch_wall,
                "amortized_seconds_per_image": amortized,
                "output_tokens": output_tokens,
                "aggregate_output_tokens_per_second": aggregate_tok_s,
            }
            batch_runtime.append(batch_record)
            print(
                "EXTRACT_BATCH_PERF "
                f"batch={batch_index}/{total_batches} size={len(batch)} "
                f"prepare={generation_perf['prepare_seconds']:.3f}s "
                f"generate={generation_perf['generation_seconds']:.3f}s "
                f"post={post_total:.3f}s write={write_total:.3f}s "
                f"wall={batch_wall:.3f}s amortized={amortized:.3f}s/image "
                f"output_tokens={output_tokens_text} aggregate_tok_s={aggregate_tok_s_text}"
            )
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(reused + generated, key=lambda item: str(item.get("image_key") or ""))
    index = {
        "schema_version": "visual-extract-run-3.0",
        "run_dir": str(run_dir),
        "images_dir": str(images_dir or (run_dir / "images")),
        "model": model_id,
        "prompt": str(prompt_path),
        "schema": str(schema_path),
        "output_dir": str(output_dir),
        "batch_size": effective_batch_size,
        "max_tokens": args.max_tokens,
        "record_count": len(records),
        "generated": len(generated),
        "reused": len(reused),
        "schema_valid_count": sum(1 for record in records if record.get("schema_valid")),
        "analyze_reconstructable_count": sum(1 for record in records if (record.get("contract") or {}).get("analyze_reconstructable")),
        "gestalt_reconstructable_count": sum(1 for record in records if (record.get("contract") or {}).get("gestalt_reconstructable")),
        "batch_runtime": batch_runtime,
        "records": records,
    }
    _write_json(output_dir / "extract.index.json", index)
    print(f"Extract v3: {output_dir}")
    print(
        f"Generated: {len(generated)}; reused: {len(reused)}; "
        f"schema valid: {index['schema_valid_count']}/{len(records)}; "
        f"Analyze contract: {index['analyze_reconstructable_count']}/{len(records)}; "
        f"Gestalt contract: {index['gestalt_reconstructable_count']}/{len(records)}"
    )
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
