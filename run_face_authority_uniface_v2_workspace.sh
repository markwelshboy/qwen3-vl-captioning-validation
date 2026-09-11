#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 IMAGE_OR_DIR --output-dir DIR [--only KEY ...] [--provider cuda|cpu|auto] [--dwpose-dir DIR]" >&2
  exit 2
fi

ROOT="${UNIFACE_WORKSPACE_ROOT:-/workspace/uniface-probe}"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "UniFace probe environment not found at $ROOT/.venv" >&2
  echo "Run ./bootstrap_face_authority_uniface_workspace.sh first." >&2
  exit 2
fi

# Prefer an explicit CUDA -> CPU fallback provider list so TensorRT does not
# become the implicit first choice merely because ORT registered it.  Callers
# can still request --provider cpu or --provider auto deliberately.
args=("$@")
has_provider=0
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--provider" || "${args[$i]}" == --provider=* ]]; then
    has_provider=1
    break
  fi
done

if (( ! has_provider )); then
  args+=(--provider cuda)
fi

# IMPORTANT: CUDA/cuDNN must be preloaded in the SAME Python process that
# constructs the UniFace ORT sessions.  Doing this in a diagnostic subprocess
# is insufficient because the loaded shared libraries disappear when that
# subprocess exits.
exec "$PY" - "${args[@]}" <<'PY'
from __future__ import annotations

import sys

import onnxruntime as ort


def provider_mode(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--provider" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--provider="):
            return arg.split("=", 1)[1]
    return "cuda"


mode = provider_mode(sys.argv[1:])

if mode in {"cuda", "auto"} and hasattr(ort, "preload_dlls"):
    try:
        # Empty string explicitly searches the NVIDIA CUDA/cuDNN pip packages
        # installed by onnxruntime-gpu[cuda,cudnn].
        ort.preload_dlls(cuda=True, cudnn=True, msvc=False, directory="")
    except TypeError:
        # Compatibility fallback for older ORT preload_dlls signatures.
        ort.preload_dlls()

providers = ort.get_available_providers()
print("=== UniFace runtime ===")
print("onnxruntime:", ort.__version__)
print("device:", ort.get_device())
print("registered providers:", providers)
print("requested UniFace mode:", mode)
if mode == "cuda":
    print("session providers: [CUDAExecutionProvider, CPUExecutionProvider]")

if mode == "cuda" and "CUDAExecutionProvider" not in providers:
    raise SystemExit(
        "ERROR: CUDAExecutionProvider is not registered in this UniFace environment."
    )

from qwen_caption_validate.face_authority_uniface_v2 import main

raise SystemExit(main())
PY
