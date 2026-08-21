#!/usr/bin/env bash
set -euo pipefail

# Reproducible vLLM workspace for the Qwen3-VL FP8 validation path.
#
# This environment is intentionally separate from build_workspace.sh. The base
# workspace owns the Transformers/bitsandbytes/DWPose stack; this workspace is
# for native Qwen3-VL FP8 inference through vLLM.
#
# IMPORTANT COMPATIBILITY PIN:
#   vLLM >= 0.24 removed Transformers-v4 support. This project currently uses
#   Transformers 4.57.x APIs and declares transformers<5, so the reproducible
#   compatibility pair is pinned here to:
#       vllm==0.23.0
#       transformers==4.57.6
#
# Usage:
#   bash ./build_vllm_workspace.sh --clean
#
# Runtime:
#   QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm bash ./run_analysis_v2_1_workspace.sh ...
#
# Optional overrides (use together if intentionally testing a different pair):
#   QWEN_VLLM_WORKSPACE_ROOT=/workspace/qwen3-vllm
#   QWEN_VLLM_PYTHON_VERSION=3.12
#   QWEN_VLLM_VERSION=0.23.0
#   QWEN_VLLM_TRANSFORMERS_VERSION=4.57.6

WORK_ROOT="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}"
PYTHON_VERSION="${QWEN_VLLM_PYTHON_VERSION:-3.12}"
VLLM_VERSION="${QWEN_VLLM_VERSION:-0.23.0}"
TRANSFORMERS_VERSION="${QWEN_VLLM_TRANSFORMERS_VERSION:-4.57.6}"
CLEAN=0

usage() {
    cat <<EOF
Usage: bash ./build_vllm_workspace.sh [--clean]

Options:
  --clean    Remove and recreate ${WORK_ROOT}/.venv before installing.
  -h,--help  Show this help.

Environment overrides:
  QWEN_VLLM_WORKSPACE_ROOT         Workspace root (default: /workspace/qwen3-vllm)
  QWEN_VLLM_PYTHON_VERSION        Python version (default: 3.12)
  QWEN_VLLM_VERSION               vLLM version (default: 0.23.0)
  QWEN_VLLM_TRANSFORMERS_VERSION  Transformers version (default: 4.57.6)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$WORK_ROOT" != /workspace && "$WORK_ROOT" != /workspace/* ]]; then
    echo "ERROR: QWEN_VLLM_WORKSPACE_ROOT must be /workspace or beneath it." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$WORK_ROOT/.venv"
BIN_DIR="$WORK_ROOT/bin"

mkdir -p \
    "$WORK_ROOT/tmp" \
    "$WORK_ROOT/uv-cache" \
    "$WORK_ROOT/pip-cache" \
    "$WORK_ROOT/huggingface/hub" \
    "$WORK_ROOT/huggingface/xet" \
    "$WORK_ROOT/.cache/nv" \
    "$WORK_ROOT/torch-cache" \
    "$BIN_DIR"

export TMPDIR="$WORK_ROOT/tmp"
export UV_CACHE_DIR="$WORK_ROOT/uv-cache"
export PIP_CACHE_DIR="$WORK_ROOT/pip-cache"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export TORCH_HOME="$WORK_ROOT/torch-cache"
export CUDA_CACHE_PATH="$WORK_ROOT/.cache/nv"
export HF_HOME="$WORK_ROOT/huggingface"
export HF_HUB_CACHE="$WORK_ROOT/huggingface/hub"
export HF_XET_CACHE="$WORK_ROOT/huggingface/xet"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER || true

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    if [[ -x "/workspace/qwen3/bin/uv" ]]; then
        echo "/workspace/qwen3/bin/uv"
        return 0
    fi
    if [[ -x "$BIN_DIR/uv" ]]; then
        echo "$BIN_DIR/uv"
        return 0
    fi
    return 1
}

if ! UV_BIN="$(find_uv)"; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: uv is not installed and curl is unavailable." >&2
        exit 1
    fi
    echo "Installing uv under $BIN_DIR ..."
    export UV_INSTALL_DIR="$BIN_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="$BIN_DIR/uv"
fi

echo "=== vLLM workspace build ==="
echo "Repo:              $REPO_ROOT"
echo "Workspace root:    $WORK_ROOT"
echo "Venv:              $VENV"
echo "Python:            $PYTHON_VERSION"
echo "vLLM:              $VLLM_VERSION"
echo "Transformers:      $TRANSFORMERS_VERSION"
echo "HF cache:          $HF_HUB_CACHE"
echo

df -h "$WORK_ROOT" / 2>/dev/null || true

if (( CLEAN )) && [[ -e "$VENV" ]]; then
    echo
    echo "Removing existing vLLM venv: $VENV"
    rm -rf "$VENV"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    echo
    echo "Creating Python $PYTHON_VERSION venv ..."
    "$UV_BIN" venv "$VENV" --python "$PYTHON_VERSION" --seed
else
    echo
    echo "Reusing existing venv: $VENV"
fi

PY="$VENV/bin/python"

echo
echo "Installing pinned vLLM / Transformers compatibility pair ..."
"$UV_BIN" pip install \
    --python "$PY" \
    --torch-backend=auto \
    "vllm==$VLLM_VERSION" \
    "transformers==$TRANSFORMERS_VERSION"

echo
echo "Installing Qwen VL image utilities ..."
"$UV_BIN" pip install \
    --python "$PY" \
    'qwen-vl-utils>=0.0.14'

echo
echo "Installing validation harness into the vLLM environment ..."
(
    cd "$REPO_ROOT"
    "$UV_BIN" pip install --python "$PY" -e .
)

echo
echo "=== vLLM runtime preflight ==="
"$PY" - <<'PY'
import importlib.metadata as md
import sys

import torch
import transformers
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", md.version("vllm"))
print("qwen-vl-utils:", md.version("qwen-vl-utils"))
print("CUDA available:", torch.cuda.is_available())
print("vLLM LLM import: OK")
print("vLLM SamplingParams import: OK")
print("qwen_vl_utils.process_vision_info import: OK")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available to PyTorch; refusing to accept this vLLM build.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

echo
echo "Checking dependency consistency ..."
"$UV_BIN" pip check --python "$PY"

echo
echo "Writing environment freeze ..."
"$UV_BIN" pip freeze --python "$PY" > "$WORK_ROOT/environment.freeze.txt"

echo
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader || true
fi

echo
echo "=== vLLM workspace build complete ==="
echo "Python:       $PY"
echo "Freeze:       $WORK_ROOT/environment.freeze.txt"
echo "Runner usage: QWEN_WORKSPACE_ROOT=$WORK_ROOT bash $REPO_ROOT/run_analysis_v2_1_workspace.sh ..."
