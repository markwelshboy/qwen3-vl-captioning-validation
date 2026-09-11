from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from .extract_v3_wire import _install_image_only_vllm
from .runner import load_model, resolve_backend, resolve_model_id, unload_model

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "edico_pose_only_v01.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SCHEMA_VERSION = "edico-pose-only-0.1"
USER_PROMPT = "Analyze this image according to your instructions."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDICO-faithful pose-only batch probe using system instruction + generic image request"
    )
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=[], help="Optional image stems to include")
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "vllm"], default="vllm")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
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


def _prepare_request(loaded, image_path: Path, system_instruction: str) -> dict[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError("EDICO pose probe requires qwen-vl-utils>=0.0.14") from exc

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    text = loaded.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=loaded.processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    request: dict[str, Any] = {"prompt": text, "multi_modal_data": mm_data}
    if video_kwargs:
        request["mm_processor_kwargs"] = video_kwargs
    return request


def _normalize_caption(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^\s*,\s*|\s*,\s*$", "", value)
    value = re.sub(r"\.$", "", value)
    value = re.sub(r",\s*,+", ", ", value)
    return value.strip()


def _split_terms(caption: str) -> list[str]:
    return [part.strip() for part in caption.split(",") if part.strip()]


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

    system_instruction = prompt_path.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(system_instruction.encode("utf-8")).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(args.model)
    backend = resolve_backend(model_id, args.backend)
    if backend != "vllm":
        raise SystemExit("EDICO pose probe currently uses vLLM so system-role + temperature behavior stays explicit")

    _install_image_only_vllm()

    print(f"Prompt: {prompt_path}")
    print(f"Prompt SHA256: {prompt_sha256}")
    print(f"Images: {len(images)}")
    print(f"Temperature: {args.temperature}")
    print(f"User prompt: {USER_PROMPT}")
    print(f"Loading {model_id} once for EDICO pose-only probe ...")
    loaded = load_model(
        model_id,
        backend=backend,
        quantization="none",
        cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
    )
    print(f"Loaded in {loaded.load_seconds:.2f}s")

    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=0,
    )

    records: list[dict[str, Any]] = []
    try:
        for index, image in enumerate(images, start=1):
            key = image.stem
            print(f"\n[{index}/{len(images)}] {key}")
            last_error: Exception | None = None
            raw_text = ""
            inference_seconds = 0.0
            for attempt in range(1, 4):
                try:
                    request = _prepare_request(loaded, image, system_instruction)
                    started = time.perf_counter()
                    outputs = loaded.model.generate([request], sampling_params=sampling, use_tqdm=False)
                    inference_seconds = time.perf_counter() - started
                    raw_text = outputs[0].outputs[0].text.strip()
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        print(f"attempt {attempt}/3 failed: {type(exc).__name__}: {exc}", file=sys.stderr)

            if last_error is not None:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "image_key": key,
                    "image": str(image),
                    "error": f"{type(last_error).__name__}: {last_error}",
                }
                _write_json(output_dir / f"{key}.edico_pose.json", record)
                records.append(record)
                print(f"ERROR: {record['error']}", file=sys.stderr)
                continue

            caption = _normalize_caption(raw_text)
            terms = _split_terms(caption)
            caption_path = output_dir / f"{key}.edico_pose.txt"
            terms_path = output_dir / f"{key}.edico_pose_terms.txt"
            json_path = output_dir / f"{key}.edico_pose.json"
            caption_path.write_text(caption + "\n", encoding="utf-8")
            terms_path.write_text("".join(f"- {term}\n" for term in terms), encoding="utf-8")

            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "image_key": key,
                "image": str(image),
                "system_instruction": str(prompt_path),
                "prompt_sha256": prompt_sha256,
                "user_prompt": USER_PROMPT,
                "model": model_id,
                "backend": backend,
                "temperature": args.temperature,
                "inference_seconds": inference_seconds,
                "raw_text": raw_text,
                "caption": caption,
                "terms": terms,
                "caption_file": str(caption_path),
                "terms_file": str(terms_path),
            }
            _write_json(json_path, record)
            records.append(record)

            print(caption)
            print("split terms:")
            for term in terms:
                print(f"  - {term}")
    finally:
        unload_model(loaded)

    index_record = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "output_dir": str(output_dir),
        "system_instruction": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "user_prompt": USER_PROMPT,
        "model": model_id,
        "backend": backend,
        "temperature": args.temperature,
        "record_count": len(records),
        "ok_count": sum(1 for r in records if r.get("status") == "ok"),
        "error_count": sum(1 for r in records if r.get("status") == "error"),
        "records": records,
    }
    _write_json(output_dir / "edico_pose_probe_v01.index.json", index_record)
    print(
        f"\nBatch complete: {index_record['ok_count']} ok, "
        f"{index_record['error_count']} errors -> {output_dir}"
    )
    return 0 if index_record["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
