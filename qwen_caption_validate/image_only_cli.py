from __future__ import annotations

"""Analyze-v2.1 launcher with the actual multimodal workload declared to vLLM.

The caption validator processes still-image requests and never submits video.
Some vLLM releases otherwise profile the largest supported video shape during
engine startup, which can consume the KV-cache headroom on 48 GB GPUs even
though that workload can never occur here.

This launcher also emits compact per-image and per-batch performance diagnostics
for the vLLM Analyze path. The diagnostics do not change prompts, sampling,
token limits, or output files; they only split request preparation from
generation and report token/timing metadata returned by vLLM.
"""

import math
import time
from pathlib import Path
from typing import Any


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


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def _request_perf_fields(request_output: Any, generation_seconds: float, max_new_tokens: int) -> dict[str, Any]:
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
        "wall_tok_s": output_tokens / generation_seconds if generation_seconds > 0 else 0.0,
        "ttft": ttft,
        "queue": queue,
        "decode": decode,
        "decode_tok_s": decode_tok_s,
        "engine_e2e": engine_e2e,
        "metrics": metrics,
        "max_new_tokens": max_new_tokens,
    }


def _print_request_perf(image_path: Path, prepare_seconds: float, generation_seconds: float, fields: dict[str, Any]) -> None:
    width, height = _image_size(image_path)
    size_text = f"{width}x{height}" if width and height else "unknown"
    decode_rate_text = "n/a" if fields["decode_tok_s"] is None else f"{fields['decode_tok_s']:.2f}"
    total_seconds = prepare_seconds + generation_seconds
    print(
        "ANALYZE_PERF "
        f"image={image_path.name} size={size_text} "
        f"prepare={prepare_seconds:.3f}s generate={generation_seconds:.3f}s total={total_seconds:.3f}s "
        f"prompt_tokens={fields['prompt_tokens']} "
        f"output_tokens={fields['output_tokens']}/{fields['max_new_tokens']} "
        f"finish={fields['finish_reason']} wall_tok_s={fields['wall_tok_s']:.2f} "
        f"ttft={_fmt_seconds(fields['ttft'])} queue={_fmt_seconds(fields['queue'])} "
        f"decode={_fmt_seconds(fields['decode'])} decode_tok_s={decode_rate_text} "
        f"engine_e2e={_fmt_seconds(fields['engine_e2e'])}"
    )

    metrics = fields["metrics"]
    metric_dict = getattr(metrics, "__dict__", None) if metrics is not None else None
    if isinstance(metric_dict, dict):
        extras = []
        for name, value in sorted(metric_dict.items()):
            number = _finite_number(value)
            if number is not None and name not in {
                "arrival_time",
                "first_scheduled_time",
                "first_token_time",
                "finished_time",
            }:
                extras.append(f"{name}={number:.6g}")
        if extras:
            print(f"ANALYZE_VLLM_METRICS image={image_path.name} " + " ".join(extras))


def main() -> int:
    import vllm

    original_llm = vllm.LLM

    def image_only_llm(*args, **kwargs):
        requested = kwargs.get("limit_mm_per_prompt")
        if requested is not None and requested != {"image": 1, "video": 0}:
            raise RuntimeError(
                "Analyze v2.1 requires vLLM multimodal limits "
                "{'image': 1, 'video': 0}; refusing conflicting limits "
                f"{requested!r}"
            )

        kwargs["limit_mm_per_prompt"] = {"image": 1, "video": 0}
        print("vLLM workload: image-only; max 1 image/prompt; video disabled")
        return original_llm(*args, **kwargs)

    # runner._load_vllm imports LLM lazily with `from vllm import LLM`, so
    # patching the package export before importing the runner keeps all of the
    # established runtime knobs intact.
    vllm.LLM = image_only_llm

    from . import runner

    original_generate = runner.generate

    def profiled_generate(
        loaded,
        image_path: Path,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> tuple[str, float]:
        if loaded.backend != "vllm":
            return original_generate(
                loaded,
                image_path,
                prompt,
                max_new_tokens=max_new_tokens,
            )

        from vllm import SamplingParams

        prepare_started = time.perf_counter()
        request = runner._prepare_vllm_multimodal(loaded, image_path, prompt)
        prepare_seconds = time.perf_counter() - prepare_started

        generation_started = time.perf_counter()
        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = loaded.model.generate(
            [request],
            sampling_params=sampling,
            use_tqdm=False,
        )
        generation_seconds = time.perf_counter() - generation_started

        fields = _request_perf_fields(outputs[0], generation_seconds, max_new_tokens)
        _print_request_perf(image_path, prepare_seconds, generation_seconds, fields)
        return fields["text"], prepare_seconds + generation_seconds

    def profiled_generate_batch(
        loaded,
        image_paths: list[Path],
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> list[tuple[str, float]]:
        if not image_paths:
            return []
        if loaded.backend != "vllm" or len(image_paths) == 1:
            return [
                profiled_generate(
                    loaded,
                    image_path,
                    prompt,
                    max_new_tokens=max_new_tokens,
                )
                for image_path in image_paths
            ]

        from vllm import SamplingParams

        requests = []
        prepare_seconds_by_image = []
        prepare_started = time.perf_counter()
        for image_path in image_paths:
            item_started = time.perf_counter()
            requests.append(runner._prepare_vllm_multimodal(loaded, image_path, prompt))
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
                f"vLLM returned {len(outputs)} outputs for {len(image_paths)} Analyze requests"
            )

        fields_by_image = [
            _request_perf_fields(output, generation_seconds, max_new_tokens)
            for output in outputs
        ]
        total_output_tokens = sum(fields["output_tokens"] for fields in fields_by_image)
        aggregate_tok_s = total_output_tokens / generation_seconds if generation_seconds > 0 else 0.0
        print(
            "ANALYZE_BATCH_PERF "
            f"batch={len(image_paths)} prepare={prepare_total:.3f}s "
            f"generate={generation_seconds:.3f}s output_tokens={total_output_tokens} "
            f"aggregate_tok_s={aggregate_tok_s:.2f} images_per_s={len(image_paths) / generation_seconds:.5f}"
        )

        results = []
        for image_path, item_prepare, fields in zip(
            image_paths,
            prepare_seconds_by_image,
            fields_by_image,
        ):
            _print_request_perf(image_path, item_prepare, generation_seconds, fields)
            results.append((fields["text"], item_prepare + generation_seconds))
        return results

    # cli imports `generate` from runner during module import. Install the
    # profiling wrappers first so only Analyze-v2.1 gets these diagnostics.
    runner.generate = profiled_generate
    runner.generate_batch = profiled_generate_batch

    from .cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
