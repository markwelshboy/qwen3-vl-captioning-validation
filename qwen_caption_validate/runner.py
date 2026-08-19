from __future__ import annotations

import gc
import json
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
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    processor: Any
    load_seconds: float
    quantization: str


def resolve_model_id(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


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


def load_model(
    model_id: str,
    *,
    dtype: str = "auto",
    quantization: str = "none",
    attn_implementation: str | None = None,
    cache_dir: Path | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
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

    processor_kwargs: dict[str, Any] = {}
    if cache_dir:
        processor_kwargs["cache_dir"] = str(cache_dir)
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    model.eval()

    return LoadedModel(
        model_id=model_id,
        model=model,
        processor=processor,
        load_seconds=time.perf_counter() - started,
        quantization=quantization,
    )


def unload_model(loaded: LoadedModel) -> None:
    del loaded.model
    del loaded.processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def generate(
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


def generate_text(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str, float]:
    """Generate from text only. Used for Compose so it cannot re-interpret the image."""
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
