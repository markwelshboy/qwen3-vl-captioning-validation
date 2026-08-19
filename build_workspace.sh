#!/usr/bin/env bash
set -euo pipefail

# Reproducible /workspace build for the Qwen3-VL validation harness.
#
# Goals:
#   * never touch /opt/venv (Fizgig may own it)
#   * keep venv, model caches, temp files, and package caches under /workspace
#   * install a CUDA 12.8 PyTorch build suitable for the L40S/driver-570 host
#   * install the Transformers + bitsandbytes/NF4 validation stack
#   * fail immediately if CUDA cannot actually initialize
#
# Usage:
#   bash ./build_workspace.sh
#   bash ./build_workspace.sh --clean
#
# Override the workspace root if desired:
#   QWEN_WORKSPACE_ROOT=/workspace/my-qwen bash ./build_workspace.sh --clean

WORK_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
PYTHON_VERSION="${QWEN_PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${QWEN_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CLEAN=0

usage() {
    cat <<EOF
Usage: bash ./build_workspace.sh [--clean]

Options:
  --clean    Remove and recreate ${WORK_ROOT}/.venv before installing.
  -h,--help  Show this help.

Environment overrides:
  QWEN_WORKSPACE_ROOT   Workspace root (default: /workspace/qwen3)
  QWEN_PYTHON_VERSION   Python version for uv venv (default: 3.12)
  QWEN_TORCH_INDEX_URL  PyTorch wheel index (default: CUDA 12.8)
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
    echo "ERROR: QWEN_WORKSPACE_ROOT must be /workspace or beneath it." >&2
    echo "       Refusing to put the large environment/caches on the container overlay." >&2
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

# Keep all heavyweight/transient state off the container overlay.
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

# A parent image may set this globally. It causes huggingface_hub to demand the
# optional hf_transfer package. Current downloads can use Xet instead.
unset HF_HUB_ENABLE_HF_TRANSFER || true

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
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

echo "=== Workspace build ==="
echo "Repo:            $REPO_ROOT"
echo "Workspace root:  $WORK_ROOT"
echo "Venv:            $VENV"
echo "HF cache:        $HF_HUB_CACHE"
echo "Temp:            $TMPDIR"
echo "uv cache:        $UV_CACHE_DIR"
echo "Python:          $PYTHON_VERSION"
echo "Torch index:     $TORCH_INDEX_URL"
echo

df -h "$WORK_ROOT" / 2>/dev/null || true

echo
if (( CLEAN )) && [[ -e "$VENV" ]]; then
    echo "Removing existing venv: $VENV"
    rm -rf "$VENV"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Creating Python $PYTHON_VERSION venv ..."
    "$UV_BIN" venv "$VENV" --python "$PYTHON_VERSION" --seed
else
    echo "Reusing existing venv: $VENV"
fi

PY="$VENV/bin/python"

# Install CUDA-12.8 PyTorch explicitly. Do not allow a CUDA-13 vLLM wheel to
# silently replace this stack; vLLM is intentionally not part of this build.
echo
echo "Installing/updating CUDA 12.8 PyTorch ..."
"$UV_BIN" pip install \
    --python "$PY" \
    --index-url "$TORCH_INDEX_URL" \
    torch torchvision

# Install this repo and the bitsandbytes/NF4 extra. Running from the repo makes
# the editable path deterministic regardless of the caller's current directory.
echo
echo "Installing validator + bitsandbytes ..."
(
    cd "$REPO_ROOT"
    "$UV_BIN" pip install --python "$PY" -e '.[bnb]'
)

# Hard preflight: a build that imports Torch but cannot initialize CUDA is not
# considered successful. This catches wrong-driver/wrong-wheel environments
# before a model starts loading on CPU.
echo
echo "=== CUDA preflight ==="
"$PY" - <<'PY'
import sys
import torch
import transformers
import bitsandbytes as bnb

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("bitsandbytes:", bnb.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available; refusing to accept this build.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
fi

echo
echo "=== Build complete ==="
echo "Validator: $VENV/bin/qwen-vl-validate"
echo "Runner:    $REPO_ROOT/run_workspace.sh"
echo
cat <<EOF
Example:
  bash "$REPO_ROOT/run_workspace.sh" /data/sh1vx \\
    --models 8b 32b --backend transformers --quantization 4bit \\
    --dtype bfloat16 --attn sdpa --run-name analysis-v1-nf4
EOF
