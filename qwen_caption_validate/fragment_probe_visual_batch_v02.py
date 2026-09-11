from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .extract_v3_wire import _install_image_only_vllm
from .runner import generate, load_model, resolve_backend, resolve_model_id, unload_model


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fragment_probe_v02_visual.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SCHEMA_VERSION = "fragment-probe-visual-0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch EDICO-style visual fragment probe; loads Qwen once and evaluates multiple images"
    )
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=[], help="Optional image stems to include")
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _discover_images(source: Path, only: set[str]) -> list[Path]:
    if source.is_file():
        images = [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    elif source.is_dir():
        images = sorted(
            p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        images = []

    if only:
        images = [p for p in images if p.stem in only]

    return images


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not prompt_path.is_file():
        print(f"Prompt not found: {prompt_path}", file=sys.stderr)
        return 2

    requested = set(args.only)
    images = _discover_images(source, requested)
    if not images:
        print(f"No matching images found under {source}", file=sys.stderr)
        return 2

    found = {p.stem for p in images}
    missing = sorted(requested - found)
    if missing:
        print("WARNING: requested image keys not found: " + ", ".join(missing), file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    model_id = resolve_model_id(args.model)
    backend = resolve_backend(model_id, args.backend)

    if backend == "vllm":
        _install_image_only_vllm()

    print(f"Prompt: {prompt_path}")
    print(f"Prompt SHA256: {prompt_sha256}")
    print(f"Images: {len(images)}")
    print(f"Loading {model_id} once for batch visual fragment probe ...")
    loaded = load_model(
        model_id,
        backend=backend,
        dtype=args.dtype,
        quantization="none",
        cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
    )
    print(f"Loaded in {loaded.load_seconds:.2f}s")

    records: list[dict[str, Any]] = []
    try:
        for index, image in enumerate(images, start=1):
            key = image.stem
            print(f"\n[{index}/{len(images)}] {key}")
            try:
                text, inference_seconds = generate(
                    loaded,
                    image,
                    prompt,
                    max_new_tokens=args.max_tokens,
                )
                text = text.strip()
                text_path = output_dir / f"{key}.fragments.txt"
                json_path = output_dir / f"{key}.fragment_probe.json"
                text_path.write_text(text + "\n", encoding="utf-8")
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "ok",
                    "image_key": key,
                    "image": str(image),
                    "prompt": str(prompt_path),
                    "prompt_sha256": prompt_sha256,
                    "model": model_id,
                    "backend": backend,
                    "inference_seconds": inference_seconds,
                    "fragments_text": text,
                    "fragments_file": str(text_path),
                }
                _write_json(json_path, record)
                records.append(record)
                print(text)
            except Exception as exc:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "image_key": key,
                    "image": str(image),
                    "prompt": str(prompt_path),
                    "prompt_sha256": prompt_sha256,
                    "model": model_id,
                    "backend": backend,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _write_json(output_dir / f"{key}.fragment_probe.json", record)
                records.append(record)
                print(f"ERROR: {record['error']}", file=sys.stderr)
    finally:
        unload_model(loaded)

    index_record = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "output_dir": str(output_dir),
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "model": model_id,
        "backend": backend,
        "record_count": len(records),
        "ok_count": sum(1 for r in records if r.get("status") == "ok"),
        "error_count": sum(1 for r in records if r.get("status") == "error"),
        "records": records,
    }
    _write_json(output_dir / "fragment_probe_v02.index.json", index_record)

    print(
        f"\nBatch complete: {index_record['ok_count']} ok, "
        f"{index_record['error_count']} errors -> {output_dir}"
    )
    return 0 if index_record["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
