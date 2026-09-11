from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .caption_refiner_text_fusion_v14_textonly import _install_text_only_vllm
from .runner import generate_text, load_model, resolve_backend, resolve_model_id, unload_model


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fragment_probe_v01_compose.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-only prose composer from an approved fragment sheet")
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    fragments_path = args.fragments.expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not fragments_path.is_file():
        print(f"Fragment sheet not found: {fragments_path}", file=sys.stderr)
        return 2
    if not prompt_path.is_file():
        print(f"Prompt not found: {prompt_path}", file=sys.stderr)
        return 2

    fragments = fragments_path.read_text(encoding="utf-8").strip()
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.replace("{{FRAGMENTS}}", fragments)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(args.model)
    backend = resolve_backend(model_id, args.backend)
    if backend == "vllm":
        _install_text_only_vllm()

    print(f"Loading {model_id} for text-only fragment composition ...")
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

    try:
        text, inference_seconds = generate_text(
            loaded,
            prompt,
            max_new_tokens=args.max_tokens,
        )
    finally:
        unload_model(loaded)

    key = fragments_path.stem
    caption_path = output_dir / f"{key}.caption.txt"
    json_path = output_dir / f"{key}.compose_probe.json"
    caption_path.write_text(text.strip() + "\n", encoding="utf-8")
    _write_json(
        json_path,
        {
            "schema_version": "fragment-probe-compose-0.1",
            "fragments": str(fragments_path),
            "prompt": str(prompt_path),
            "model": model_id,
            "backend": backend,
            "inference_seconds": inference_seconds,
            "caption": text.strip(),
            "caption_file": str(caption_path),
        },
    )

    print("\n===== QWEN TEXT-ONLY CAPTION =====\n")
    print(text.strip())
    print(f"\nSaved: {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
