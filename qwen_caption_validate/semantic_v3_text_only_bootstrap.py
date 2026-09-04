from __future__ import annotations

import sys
from typing import Any


def install_text_only_vllm() -> None:
    """Force a multimodal vLLM checkpoint into text-only runtime mode.

    Qwen3-VL is still the checkpoint architecture, but Semantic V3 reasoning stages
    must never reserve encoder capacity for image/video inputs. vLLM 0.11 supports
    disabling modalities with a zero per-prompt limit. skip_mm_profiling prevents
    startup profiling of maximum-size multimodal inputs that can otherwise consume
    the KV-cache budget even though generation is text-only.
    """
    try:
        import vllm
    except ImportError:
        return

    original_llm = vllm.LLM
    if getattr(original_llm, "_semantic_v3_text_only", False):
        return

    def text_only_llm(*args: Any, **kwargs: Any):
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        kwargs["skip_mm_profiling"] = True
        print(
            "vLLM Semantic V3 text-only mode: "
            "limit_mm_per_prompt={image:0,video:0} skip_mm_profiling=true"
        )
        return original_llm(*args, **kwargs)

    text_only_llm._semantic_v3_text_only = True  # type: ignore[attr-defined]
    vllm.LLM = text_only_llm


def main() -> int:
    install_text_only_vllm()
    from .semantic_v3_gestalt_v02_runtime import main as gestalt_main

    return gestalt_main()


if __name__ == "__main__":
    raise SystemExit(main())
