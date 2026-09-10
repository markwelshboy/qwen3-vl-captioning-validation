from __future__ import annotations

"""Text-only vLLM bootstrap for caption refiner v0.14.

The underlying checkpoint is Qwen3-VL, so vLLM otherwise profiles its multimodal
encoder using the model's default limits.  For this stage that is both unnecessary
and expensive: v0.14 sends text prompts only.  Install the limits before the generic
runner imports/constructs vllm.LLM.
"""


def _install_text_only_vllm() -> None:
    try:
        import vllm
    except ImportError:
        return

    original_llm = vllm.LLM
    if getattr(original_llm, "_caption_refiner_v14_text_only", False):
        return

    def text_only_llm(*args, **kwargs):
        requested = kwargs.get("limit_mm_per_prompt")
        expected = {"image": 0, "video": 0}
        if requested is not None and requested != expected:
            raise RuntimeError(
                "Caption refiner v0.14 is text-only and requires vLLM multimodal "
                f"limits {expected!r}; refusing conflicting limits {requested!r}"
            )

        kwargs["limit_mm_per_prompt"] = expected
        # This run never submits multimodal data, so there is no reason to reserve
        # processor cache or profile the vision/video encoder at engine startup.
        kwargs["mm_processor_cache_gb"] = 0
        kwargs["skip_mm_profiling"] = True
        # Keep compile/profile concurrency proportional to this tiny offline job
        # instead of vLLM's much larger generic serving default.
        kwargs.setdefault("max_num_seqs", 8)

        print(
            "vLLM v0.14 text-only guard: "
            "limit_mm_per_prompt={'image': 0, 'video': 0} "
            "mm_processor_cache_gb=0 skip_mm_profiling=True max_num_seqs="
            f"{kwargs['max_num_seqs']}"
        )
        return original_llm(*args, **kwargs)

    text_only_llm._caption_refiner_v14_text_only = True  # type: ignore[attr-defined]
    vllm.LLM = text_only_llm


def main() -> int:
    _install_text_only_vllm()
    from . import caption_refiner_text_fusion_v14 as impl

    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
