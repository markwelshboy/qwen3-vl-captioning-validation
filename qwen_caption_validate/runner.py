from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ALIASES = {
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
    "32b": "Qwen/Qwen3-VL-32B-Instruct",
    "8b-fp8": "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "32b-fp8": "Qwen/Qwen3-VL-32B-Instruct-FP8",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    processor: Any
    load_seconds: float
    quantization: str
    backend: str


def resolve_model_id(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def resolve_backend(model_id: str, requested: str) -> str:
    if requested != "auto":
        return requested
    # Qwen currently recommends vLLM/SGLang for the official FP8 VL weights;
    # Transformers can load some variants but not all (notably 32B FP8).
    if model_id.upper().endswith("-FP8"):
        return "vllm"
    return "transformers"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def discover_images(dataset: Path, recursive: bool = False) -> list[Path]:
    iterator = dataset.rglob("*") if recursive else dataset.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _dtype_value(dtype_name: str):
    dtype_name = dtype_name.lower()
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _quantization_config(quantization: str, dtype: str):
    if quantization == "none":
        return None
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "4bit":
        compute_dtype = _dtype_value(dtype)
        if compute_dtype == "auto" or compute_dtype == torch.float32:
            compute_dtype = torch.bfloat16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unsupported quantization mode: {quantization}")


def _processor_kwargs(
    cache_dir: Path | None,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    return kwargs


def _load_transformers(
    model_id: str,
    *,
    dtype: str,
    quantization: str,
    attn_implementation: str | None,
    cache_dir: Path | None,
    min_pixels: int | None,
    max_pixels: int | None,
) -> LoadedModel:
    started = time.perf_counter()

    model_kwargs: dict[str, Any] = {
        "dtype": _dtype_value(dtype),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    quantization_config = _quantization_config(quantization, dtype)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)

    processor = AutoProcessor.from_pretrained(
        model_id,
        **_processor_kwargs(cache_dir, min_pixels, max_pixels),
    )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    model.eval()

    return LoadedModel(
        model_id=model_id,
        model=model,
        processor=processor,
        load_seconds=time.perf_counter() - started,
        quantization=quantization,
        backend="transformers",
    )


def _load_vllm(
    model_id: str,
    *,
    cache_dir: Path | None,
    min_pixels: int | None,
    max_pixels: int | None,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> LoadedModel:
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM backend requires a CUDA GPU for this validator")

    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested but vllm is not installed. "
            "Install the optional stack described in README.md."
        ) from exc

    # Qwen's offline vLLM examples use spawn for the worker process.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_id,
        **_processor_kwargs(cache_dir, min_pixels, max_pixels),
    )

    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": False,
        "tensor_parallel_size": 1,
        "seed": 0,
        "max_model_len": max_model_len,
    }
    if cache_dir:
        llm_kwargs["download_dir"] = str(cache_dir)

    model = LLM(**llm_kwargs)
    return LoadedModel(
        model_id=model_id,
        model=model,
        processor=processor,
        load_seconds=time.perf_counter() - started,
        quantization="checkpoint-native",
        backend="vllm",
    )


def load_model(
    model_id: str,
    *,
    backend: str = "auto",
    dtype: str = "auto",
    quantization: str = "none",
    attn_implementation: str | None = None,
    cache_dir: Path | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    vllm_gpu_memory_utilization: float = 0.92,
    vllm_max_model_len: int = 8192,
) -> LoadedModel:
    resolved = resolve_backend(model_id, backend)
    if resolved == "vllm":
        if quantization != "none":
            raise ValueError(
                "--quantization 4bit/8bit is a Transformers fallback and cannot be combined "
                "with --backend vllm. Use a native quantized checkpoint such as *-FP8 instead."
            )
        return _load_vllm(
            model_id,
            cache_dir=cache_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
        )
    if resolved == "transformers":
        return _load_transformers(
            model_id,
            dtype=dtype,
            quantization=quantization,
            attn_implementation=attn_implementation,
            cache_dir=cache_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    raise ValueError(f"Unsupported backend: {resolved}")


def unload_model(loaded: LoadedModel) -> None:
    # Modern vLLM V1 engines have finalizers that tear down workers and release
    # model/KV memory when the LLM object is collected. Keep cleanup deliberately
    # conservative here; if a specific vLLM release leaks across sequential models,
    # the CLI can be run once per model while reusing the same --run-name.
    del loaded.model
    del loaded.processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _vllm_messages_for_image(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _prepare_vllm_multimodal(loaded: LoadedModel, image_path: Path, prompt: str) -> dict[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "vLLM Qwen3-VL image preparation requires qwen-vl-utils>=0.0.14"
        ) from exc

    messages = _vllm_messages_for_image(image_path, prompt)
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

    request: dict[str, Any] = {
        "prompt": text,
        "multi_modal_data": mm_data,
    }
    if video_kwargs:
        request["mm_processor_kwargs"] = video_kwargs
    return request


def _generate_vllm(
    loaded: LoadedModel,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, float]:
    from vllm import SamplingParams

    started = time.perf_counter()
    request = _prepare_vllm_multimodal(loaded, image_path, prompt)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = loaded.model.generate([request], sampling_params=sampling, use_tqdm=False)
    text = outputs[0].outputs[0].text.strip()
    return text, time.perf_counter() - started


def _generate_transformers(
    loaded: LoadedModel,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, float]:
    started = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = loaded.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(loaded.model.device)

    with torch.inference_mode():
        generated_ids = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]
    trimmed = generated_ids[:, input_length:]
    text = loaded.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text, time.perf_counter() - started


def generate(
    loaded: LoadedModel,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, float]:
    if loaded.backend == "vllm":
        return _generate_vllm(
            loaded,
            image_path,
            prompt,
            max_new_tokens=max_new_tokens,
        )
    return _generate_transformers(
        loaded,
        image_path,
        prompt,
        max_new_tokens=max_new_tokens,
    )


def _chat_text(loaded: LoadedModel, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    return loaded.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_text_vllm(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, float]:
    from vllm import SamplingParams

    started = time.perf_counter()
    text_prompt = _chat_text(loaded, prompt)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = loaded.model.generate([text_prompt], sampling_params=sampling, use_tqdm=False)
    text = outputs[0].outputs[0].text.strip()
    return text, time.perf_counter() - started


def _generate_text_transformers(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, float]:
    started = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    inputs = loaded.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(loaded.model.device)
    with torch.inference_mode():
        generated_ids = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    input_length = inputs["input_ids"].shape[-1]
    trimmed = generated_ids[:, input_length:]
    text = loaded.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text, time.perf_counter() - started


def generate_text(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, float]:
    """Generate from text only. Used for Compose so it cannot re-interpret the image."""
    if loaded.backend == "vllm":
        return _generate_text_vllm(loaded, prompt, max_new_tokens=max_new_tokens)
    return _generate_text_transformers(loaded, prompt, max_new_tokens=max_new_tokens)


_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    fence = _JSON_FENCE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()

    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            return None, "Top-level JSON value was not an object"
        return parsed, None
    except json.JSONDecodeError as first_error:
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first != -1 and last > first:
            try:
                parsed = json.loads(candidate[first : last + 1])
                if isinstance(parsed, dict):
                    return parsed, None
            except json.JSONDecodeError:
                pass
        return None, str(first_error)


def validate_analysis(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda err: list(err.path))
    messages: list[str] = []
    for err in errors:
        path = ".".join(str(part) for part in err.path) or "$"
        messages.append(f"{path}: {err.message}")
    return messages


def render_compose_prompt(template: str, analysis: dict[str, Any], subject_token: str, detail: str) -> str:
    return (
        template.replace("{{SUBJECT_TOKEN}}", subject_token)
        .replace("{{DETAIL_PROFILE}}", detail)
        .replace("{{ANALYSIS_JSON}}", json.dumps(analysis, indent=2, ensure_ascii=False))
    )
