from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from qwen_caption_validate.face_authority_uniface_v2 import _providers_for_mode


def test_provider_modes_cpu_and_auto() -> None:
    assert _providers_for_mode("cpu") == ["CPUExecutionProvider"]
    assert _providers_for_mode("auto") is None


def test_cuda_mode_uses_cuda_then_cpu_and_ignores_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    assert _providers_for_mode("cuda") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_cuda_mode_fails_closed_without_cuda_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(SystemExit, match="CUDAExecutionProvider is unavailable"):
        _providers_for_mode("cuda")
