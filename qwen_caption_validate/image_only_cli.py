from __future__ import annotations

"""Analyze-v2.1 launcher with the actual multimodal workload declared to vLLM.

The caption validator processes exactly one still image per request and never
submits video. Some vLLM releases otherwise profile the largest supported video
shape during engine startup, which can consume the KV-cache headroom on 48 GB
GPUs even though that workload can never occur here.

Keep this declaration at the launcher boundary so it remains explicit and
stable across vLLM release changes.
"""


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
    # patching the package export before importing the CLI keeps the existing
    # runner implementation and all of its established runtime knobs intact.
    vllm.LLM = image_only_llm

    from .cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
