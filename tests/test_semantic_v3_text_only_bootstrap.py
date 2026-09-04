from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from qwen_caption_validate.semantic_v3_text_only_bootstrap import install_text_only_vllm


class SemanticV3TextOnlyBootstrapTests(unittest.TestCase):
    def test_disables_image_video_capacity_and_mm_profiling(self) -> None:
        captured: dict[str, object] = {}

        class FakeLLM:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            install_text_only_vllm()
            fake_vllm.LLM(model="Qwen/Qwen3-VL-32B-Instruct-FP8", max_model_len=8192)  # type: ignore[attr-defined]

        kwargs = captured["kwargs"]
        self.assertIsInstance(kwargs, dict)
        assert isinstance(kwargs, dict)
        self.assertEqual(kwargs["limit_mm_per_prompt"], {"image": 0, "video": 0})
        self.assertIs(kwargs["skip_mm_profiling"], True)
        self.assertEqual(kwargs["max_model_len"], 8192)

    def test_text_only_policy_overrides_accidental_multimodal_kwargs(self) -> None:
        captured: dict[str, object] = {}

        class FakeLLM:
            def __init__(self, *args, **kwargs):
                captured["kwargs"] = kwargs

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            install_text_only_vllm()
            fake_vllm.LLM(  # type: ignore[attr-defined]
                limit_mm_per_prompt={"image": 1, "video": 1},
                skip_mm_profiling=False,
            )

        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        self.assertEqual(kwargs["limit_mm_per_prompt"], {"image": 0, "video": 0})
        self.assertIs(kwargs["skip_mm_profiling"], True)

    def test_install_is_idempotent(self) -> None:
        class FakeLLM:
            pass

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            install_text_only_vllm()
            first = fake_vllm.LLM  # type: ignore[attr-defined]
            install_text_only_vllm()
            second = fake_vllm.LLM  # type: ignore[attr-defined]

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
