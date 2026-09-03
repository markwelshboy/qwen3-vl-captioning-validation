from __future__ import annotations

"""Structured-output launcher for Extract v3.

The base Extract runner owns batching, timing, artifact persistence and contract
checks. This launcher constrains vLLM generation to the configured Extract JSON
schema and configures xgrammar to emit compact JSON without structural
whitespace. The latter matters materially for this large persistent record:
pretty-printed JSON can add thousands of decode tokens without adding semantic
information.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

from . import extract_v3


def _schema_path_from_argv() -> Path:
    args = sys.argv[1:]
    for index, value in enumerate(args):
        if value == "--schema" and index + 1 < len(args):
            return Path(args[index + 1]).expanduser().resolve()
        if value.startswith("--schema="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return extract_v3.DEFAULT_SCHEMA.resolve()


def _install_compact_structured_vllm() -> None:
    """Declare the Extract-only multimodal and structured-output engine policy."""
    try:
        import vllm
    except ImportError:
        return

    original_llm = vllm.LLM
    if getattr(original_llm, "_extract_v3_compact_structured", False):
        return

    def compact_structured_llm(*args, **kwargs):
        requested_mm = kwargs.get("limit_mm_per_prompt")
        if requested_mm is not None and requested_mm != {"image": 1, "video": 0}:
            raise RuntimeError(
                "Extract v3 requires image-only vLLM limits {'image': 1, 'video': 0}; "
                f"refusing conflicting limits {requested_mm!r}"
            )
        kwargs["limit_mm_per_prompt"] = {"image": 1, "video": 0}

        requested_so = kwargs.get("structured_outputs_config")
        if requested_so is not None:
            raise RuntimeError(
                "Extract v3 owns vLLM structured_outputs_config; refusing a "
                f"conflicting caller value {requested_so!r}"
            )
        kwargs["structured_outputs_config"] = {
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        }
        print(
            "vLLM structured-output engine: "
            "backend=xgrammar disable_any_whitespace=true"
        )
        return original_llm(*args, **kwargs)

    compact_structured_llm._extract_v3_compact_structured = True  # type: ignore[attr-defined]
    vllm.LLM = compact_structured_llm


def _generate_vllm_batch_structured(
    loaded,
    image_paths: list[Path],
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema_path = _schema_path_from_argv()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    requests = []
    prepare_seconds_by_image: list[float] = []
    prepare_started = time.perf_counter()
    for image_path in image_paths:
        item_started = time.perf_counter()
        requests.append(
            extract_v3.runner_module._prepare_vllm_multimodal(
                loaded,
                image_path,
                prompt,
            )
        )
        prepare_seconds_by_image.append(time.perf_counter() - item_started)
    prepare_total = time.perf_counter() - prepare_started

    structured = StructuredOutputsParams(json=schema)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        structured_outputs=structured,
    )

    generation_started = time.perf_counter()
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
    for image_path, item_prepare, output in zip(
        image_paths,
        prepare_seconds_by_image,
        outputs,
    ):
        fields = extract_v3._request_perf_fields(output, max_new_tokens)
        fields["image"] = image_path
        fields["prepare_seconds"] = item_prepare
        fields["shared_generation_seconds"] = generation_seconds
        fields["structured_json"] = True
        fields["structural_whitespace_disabled"] = True
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
        "structured_json": True,
        "structural_whitespace_disabled": True,
        "schema": str(schema_path),
    }
    return items, batch


def main() -> int:
    extract_v3._install_image_only_vllm = _install_compact_structured_vllm
    extract_v3._generate_vllm_batch_profiled = _generate_vllm_batch_structured
    print(
        f"Extract structured JSON: enabled ({_schema_path_from_argv()}); "
        "xgrammar structural whitespace disabled"
    )
    return extract_v3.main()


if __name__ == "__main__":
    raise SystemExit(main())
