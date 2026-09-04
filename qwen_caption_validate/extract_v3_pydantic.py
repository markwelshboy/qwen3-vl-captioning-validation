from __future__ import annotations

"""Pydantic-first one-pass Visual Extract.

The VLM-facing JSON Schema is generated directly from `ExtractWireV1` and
passed to vLLM/xgrammar. The returned JSON is parsed by Pydantic, expanded
without semantic inference into `VisualExtractV3`, and then checked against the
legacy canonical JSON Schema plus Analyze/Gestalt reconstruction contracts.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import extract_v3
from .extract_v3_contract import audit_extract_contract
from .extract_v3_models import ExtractWireV1, VisualExtractV3
from .extract_v3_wire import _generate_wire_batch, _install_image_only_vllm
from .extract_v3_wire_contract import expand_extract_wire
from .runner import (
    load_model,
    model_slug,
    resolve_backend,
    resolve_model_id,
    unload_model,
    validate_analysis,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "extract_v3_wire.txt"
DEFAULT_CANONICAL_SCHEMA = PACKAGE_ROOT / "schemas" / "extract_v3.schema.json"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 4000
WIRE_VERSION = "x3p2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _wire_schema() -> dict[str, Any]:
    return ExtractWireV1.model_json_schema(by_alias=True)


def _schema_sha256(schema: dict[str, Any]) -> str:
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for item in exc.errors(include_url=False, include_context=False, include_input=False):
        loc = ".".join(str(part) for part in item.get("loc") or []) or "$"
        errors.append(f"{loc}: {item.get('msg')} [{item.get('type')}]")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-visual-extract-v3-pydantic",
        description=(
            "One visual VLM pass constrained by a Pydantic-generated compact JSON Schema, "
            "then deterministically expanded into canonical visual-extract-3.0."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--canonical-schema", type=Path, default=DEFAULT_CANONICAL_SCHEMA)
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


def _empty_contract(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "visual-extract-contract-audit-0.1",
        "analyze_reconstructable": False,
        "gestalt_reconstructable": False,
        "analyze_missing_paths": [reason],
        "gestalt_missing_paths": [reason],
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
        images = [
            path
            for path in extract_v3._discover_images(run_dir, images_dir)
            if extract_v3._matches(path, args.only)
        ]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not images:
        print("No matching images found for Pydantic Extract v3.", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    output_dir = (
        args.output_dir or (run_dir / "extract-v3-pydantic.2" / slug)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    canonical_schema_path = args.canonical_schema.expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    canonical_schema = _read_json(canonical_schema_path)
    wire_schema = _wire_schema()
    wire_schema_hash = _schema_sha256(wire_schema)

    pending: list[tuple[Path, str, Path]] = []
    reused: list[dict[str, Any]] = []
    for image in images:
        key = extract_v3._result_key(image)
        out_path = output_dir / f"{key}.extract.json"
        if args.overwrite or not out_path.exists():
            pending.append((image, key, out_path))
        else:
            reused.append(extract_v3._record_summary(_read_json(out_path)))

    if pending and resolve_backend(model_id, args.backend) != "vllm":
        print("Pydantic Extract currently requires vLLM structured decoding.", file=sys.stderr)
        return 2
    if pending:
        _install_image_only_vllm()

    print(
        f"Extract contract: Pydantic ExtractWireV1({WIRE_VERSION}) "
        f"-> model_json_schema(by_alias=True) sha256={wire_schema_hash[:12]}... "
        "-> xgrammar -> model_validate_json -> VisualExtractV3"
    )

    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    loaded = None
    if pending:
        print(f"Loading {model_id} for Pydantic V3 Extract ...")
        loaded = load_model(
            model_id,
            backend="vllm",
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(
            f"Loaded in {loaded.load_seconds:.2f}s. Extracting {len(pending)} image(s). "
            f"wire={WIRE_VERSION} batch_size={args.batch_size} max_tokens={args.max_tokens}"
        )

    try:
        total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
        for batch_index, offset in enumerate(range(0, len(pending), args.batch_size), start=1):
            batch_started = time.perf_counter()
            batch = pending[offset : offset + args.batch_size]
            batch_images = [item[0] for item in batch]
            items, generation_perf = _generate_wire_batch(
                loaded,
                batch_images,
                prompt,
                wire_schema,
                max_new_tokens=args.max_tokens,
            )

            post_total = 0.0
            write_total = 0.0
            for (image, key, out_path), item in zip(batch, items):
                post_started = time.perf_counter()
                raw = str(item["text"])

                wire_model: ExtractWireV1 | None = None
                wire_errors: list[str] = []
                try:
                    wire_model = ExtractWireV1.model_validate_json(raw)
                except ValidationError as exc:
                    wire_errors = _validation_errors(exc)

                canonical_model: VisualExtractV3 | None = None
                canonical: dict[str, Any] | None = None
                canonical_errors: list[str] = []
                expansion: dict[str, Any] | None = None
                expansion_error: str | None = None
                contract = _empty_contract("invalid_pydantic_wire")

                if wire_model is not None:
                    try:
                        canonical_model, expansion = expand_extract_wire(wire_model)
                        canonical = canonical_model.model_dump(mode="json", by_alias=True)
                        VisualExtractV3.model_validate(canonical)
                        canonical_errors = validate_analysis(canonical, canonical_schema)
                        contract = audit_extract_contract(canonical)
                    except ValidationError as exc:
                        canonical_errors = _validation_errors(exc)
                        expansion_error = "Canonical Pydantic validation failed"
                    except Exception as exc:
                        expansion_error = f"{type(exc).__name__}: {exc}"

                post_seconds = time.perf_counter() - post_started
                post_total += post_seconds

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
                    "post_seconds": post_seconds,
                }
                payload = {
                    "schema_version": "visual-extract-pydantic-artifact-0.2",
                    "image_key": key,
                    "image": str(image),
                    "model": model_id,
                    "backend": loaded.backend if loaded is not None else None,
                    "inference_seconds": generation_perf["generation_seconds"],
                    "performance": request_perf,
                    "wire_schema_version": WIRE_VERSION,
                    "wire_contract": "ExtractWireV1",
                    "wire_schema_sha256": wire_schema_hash,
                    "wire_extract": wire_model.model_dump(mode="json", by_alias=True) if wire_model else None,
                    "wire_model_dump": wire_model.model_dump(mode="json") if wire_model else None,
                    "wire_schema_valid": wire_model is not None,
                    "wire_schema_errors": wire_errors,
                    "expansion": expansion,
                    "expansion_error": expansion_error,
                    "canonical_contract": "VisualExtractV3",
                    "canonical_pydantic_valid": canonical_model is not None and not expansion_error,
                    "extract": canonical,
                    "raw_response": raw,
                    "parse_error": wire_errors[0] if wire_errors else None,
                    "schema_valid": canonical is not None and not canonical_errors and not expansion_error,
                    "schema_errors": canonical_errors,
                    "contract": contract,
                }

                write_started = time.perf_counter()
                extract_v3._write_json(out_path, payload)
                write_seconds = time.perf_counter() - write_started
                write_total += write_seconds
                request_perf["write_seconds"] = write_seconds

                record = extract_v3._record_summary(payload)
                generated.append(record)
                print(
                    "EXTRACT_PYDANTIC_PERF "
                    f"image={image.name} batch={batch_index}/{total_batches} "
                    f"prepare={float(item.get('prepare_seconds') or 0.0):.3f}s "
                    f"batch_generate={generation_perf['generation_seconds']:.3f}s "
                    f"tokens={item.get('output_tokens')}/{args.max_tokens} "
                    f"finish={item.get('finish_reason')} post={post_seconds:.3f}s write={write_seconds:.3f}s"
                )
                print(
                    f"{key}: wire={'ok' if payload['wire_schema_valid'] else 'FAIL'} "
                    f"canonical={'ok' if payload['schema_valid'] else 'FAIL'} "
                    f"analyze={'ok' if contract.get('analyze_reconstructable') else 'missing'} "
                    f"gestalt={'ok' if contract.get('gestalt_reconstructable') else 'missing'} "
                    f"entities={record['entity_count']} relations={record['relation_count']} "
                    f"posture={record['posture_candidate']} torso={record['torso_orientation_candidate']}"
                )
                if wire_errors:
                    print(f"  wire_error: {wire_errors[0]}")
                if expansion_error:
                    print(f"  expansion_error: {expansion_error}")
                if canonical_errors:
                    print(f"  canonical_error: {canonical_errors[0]}")

            batch_wall = time.perf_counter() - batch_started
            amortized = batch_wall / len(batch) if batch else 0.0
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
                "output_tokens": generation_perf["output_tokens"],
                "aggregate_output_tokens_per_second": generation_perf["aggregate_output_tokens_per_second"],
            }
            batch_runtime.append(batch_record)
            print(
                "EXTRACT_PYDANTIC_BATCH_PERF "
                f"batch={batch_index}/{total_batches} size={len(batch)} "
                f"generate={generation_perf['generation_seconds']:.3f}s "
                f"wall={batch_wall:.3f}s amortized={amortized:.3f}s/image "
                f"output_tokens={generation_perf['output_tokens']} "
                f"aggregate_tok_s={generation_perf['aggregate_output_tokens_per_second']:.2f}"
            )
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(reused + generated, key=lambda item: str(item.get("image_key") or ""))
    index = {
        "schema_version": "visual-extract-pydantic-run-0.2",
        "run_dir": str(run_dir),
        "images_dir": str(images_dir or (run_dir / "images")),
        "model": model_id,
        "prompt": str(prompt_path),
        "wire_schema_version": WIRE_VERSION,
        "wire_contract": "ExtractWireV1",
        "wire_schema_source": "ExtractWireV1.model_json_schema(by_alias=True)",
        "wire_schema_sha256": wire_schema_hash,
        "canonical_contract": "VisualExtractV3",
        "canonical_schema": str(canonical_schema_path),
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "record_count": len(records),
        "generated": len(generated),
        "reused": len(reused),
        "schema_valid_count": sum(1 for record in records if record.get("schema_valid")),
        "analyze_reconstructable_count": sum(
            1 for record in records if (record.get("contract") or {}).get("analyze_reconstructable")
        ),
        "gestalt_reconstructable_count": sum(
            1 for record in records if (record.get("contract") or {}).get("gestalt_reconstructable")
        ),
        "batch_runtime": batch_runtime,
        "records": records,
    }
    extract_v3._write_json(output_dir / "extract.index.json", index)
    print(f"Pydantic Extract v3: {output_dir}")
    print(
        f"Generated: {len(generated)}; reused: {len(reused)}; "
        f"canonical valid: {index['schema_valid_count']}/{len(records)}; "
        f"Analyze contract: {index['analyze_reconstructable_count']}/{len(records)}; "
        f"Gestalt contract: {index['gestalt_reconstructable_count']}/{len(records)}"
    )
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
