from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from . import extract_v3
from .extract_v3_wire import _generate_wire_batch, _install_image_only_vllm
from .runner import load_model, model_slug, resolve_backend, resolve_model_id, unload_model


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_kiss_packet.txt"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 1800
PACKET_VERSION = "kiss1"
ARTIFACT_VERSION = "semantic-v3-kiss-packet-0.2"
RUN_VERSION = "semantic-v3-kiss-packet-0.2-run"

ShortFact = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=140),
]


class SemanticPacketKISS1(BaseModel):
    """Sparse image-semantic transport packet.

    Every semantic bin is optional. The schema deliberately avoids confidence,
    evidence, counterevidence, hypotheses, landmark questionnaires, and nested
    clarification structures. Empty bins are valid observations.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, title="SemanticPacketKISS1")

    version: Literal["kiss1"] = Field(alias="v")
    subject: list[ShortFact] = Field(default_factory=list, alias="s", max_length=8)
    body: list[ShortFact] = Field(default_factory=list, alias="b", max_length=6)
    interactions: list[ShortFact] = Field(default_factory=list, alias="i", max_length=6)
    scene: list[ShortFact] = Field(default_factory=list, alias="sc", max_length=8)
    composition: list[ShortFact] = Field(default_factory=list, alias="co", max_length=6)
    uncertainties: list[ShortFact] = Field(default_factory=list, alias="u", max_length=4)


def _validation_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for item in exc.errors(include_url=False, include_context=False, include_input=False):
        loc = ".".join(str(part) for part in item.get("loc") or []) or "$"
        errors.append(f"{loc}: {item.get('msg')} [{item.get('type')}]")
    return errors


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-semantic-v3-kiss-packet",
        description=(
            "One image-conditioned VLM call emits a sparse KISS semantic packet. "
            "No Analyze or Gestalt reconstruction pass is involved."
        ),
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


def _record_summary(payload: dict[str, Any]) -> dict[str, Any]:
    packet = payload.get("packet") if isinstance(payload.get("packet"), dict) else {}
    return {
        "image_key": payload.get("image_key"),
        "schema_valid": bool(payload.get("schema_valid")),
        "finish_reason": ((payload.get("performance") or {}).get("finish_reason")),
        "output_tokens": ((payload.get("performance") or {}).get("output_tokens")),
        "subject_count": len(packet.get("subject") or []),
        "body_count": len(packet.get("body") or []),
        "interaction_count": len(packet.get("interactions") or []),
        "scene_count": len(packet.get("scene") or []),
        "composition_count": len(packet.get("composition") or []),
        "uncertainty_count": len(packet.get("uncertainties") or []),
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
        print("No matching images found for KISS semantic packet.", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    output_dir = (
        args.output_dir or (run_dir / "semantic-v3" / "kiss-packet-v0.2" / slug)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = SemanticPacketKISS1.model_json_schema(by_alias=True)

    pending: list[tuple[Path, str, Path]] = []
    reused: list[dict[str, Any]] = []
    for image in images:
        key = extract_v3._result_key(image)
        out_path = output_dir / f"{key}.semantic_packet.json"
        if args.overwrite or not out_path.exists():
            pending.append((image, key, out_path))
        else:
            reused.append(_record_summary(_read_json(out_path)))

    if pending and resolve_backend(model_id, args.backend) != "vllm":
        print("KISS semantic packet currently requires vLLM structured decoding.", file=sys.stderr)
        return 2
    if pending:
        _install_image_only_vllm()

    print(
        "Semantic V3 KISS packet: one VLM call; sparse storage bins; "
        "no confidence/evidence/hypothesis interrogation"
    )
    print(
        f"Contract: {PACKET_VERSION}; artifact={ARTIFACT_VERSION}; max_tokens={args.max_tokens}; "
        f"batch_size={args.batch_size}; max_model_len={args.vllm_max_model_len}"
    )

    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    loaded = None
    if pending:
        print(f"Loading {model_id} for KISS semantic packet ...")
        loaded = load_model(
            model_id,
            backend="vllm",
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(f"Loaded in {loaded.load_seconds:.2f}s. Extracting {len(pending)} image(s).")

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
                schema,
                max_new_tokens=args.max_tokens,
            )

            for (image, key, out_path), item in zip(batch, items):
                raw = str(item["text"])
                model: SemanticPacketKISS1 | None = None
                errors: list[str] = []
                try:
                    model = SemanticPacketKISS1.model_validate_json(raw)
                except ValidationError as exc:
                    errors = _validation_errors(exc)

                packet = model.model_dump(mode="json") if model is not None else None
                wire_packet = model.model_dump(mode="json", by_alias=True) if model is not None else None
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
                    "backend": loaded.backend if loaded is not None else None,
                    "packet_schema_version": PACKET_VERSION,
                    "packet": packet,
                    "wire_packet": wire_packet,
                    "schema_valid": model is not None,
                    "schema_errors": errors,
                    "raw_response": raw,
                    "performance": performance,
                }
                extract_v3._write_json(out_path, payload)
                record = _record_summary(payload)
                generated.append(record)
                print(
                    "KISS_PACKET_PERF "
                    f"image={image.name} batch={batch_index}/{total_batches} "
                    f"tokens={item.get('output_tokens')}/{args.max_tokens} "
                    f"finish={item.get('finish_reason')} "
                    f"valid={'yes' if model is not None else 'NO'}"
                )
                if errors:
                    print(f"  schema_error: {errors[0]}")

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
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(reused + generated, key=lambda item: str(item.get("image_key") or ""))
    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "images_dir": str(images_dir or (run_dir / "images")),
        "model": model_id,
        "prompt": str(prompt_path),
        "packet_schema_version": PACKET_VERSION,
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "vllm_max_model_len": args.vllm_max_model_len,
        "record_count": len(records),
        "generated": len(generated),
        "reused": len(reused),
        "schema_valid_count": sum(1 for record in records if record.get("schema_valid")),
        "batch_runtime": batch_runtime,
        "records": records,
    }
    extract_v3._write_json(output_dir / "semantic_packet.index.json", index)
    print(f"KISS semantic packet: {output_dir}")
    print(f"Records: {len(records)}; valid: {index['schema_valid_count']}")
    return 0 if index["schema_valid_count"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
