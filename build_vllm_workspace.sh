#!/usr/bin/env bash
set -euo pipefail

# Reproducible vLLM workspace for the Qwen3-VL FP8 validation path.
#
# This environment is intentionally separate from build_workspace.sh. The base
# workspace owns the Transformers/bitsandbytes/DWPose stack; this workspace is
# for native Qwen3-VL FP8 inference through vLLM.
#
# IMPORTANT COMPATIBILITY PIN:
#   The rented L40S fleet can expose NVIDIA 570-series host drivers. Those are
#   compatible with CUDA 12.8, but not CUDA 12.9. vLLM 0.12+ moved its standard
#   CUDA build to 12.9, so using --torch-backend=auto can silently install a
#   cu129 stack that imports correctly and then dies during native CUDA engine
#   initialization with "CUDA driver version is insufficient".
#
#   vLLM 0.11.0 is the last release whose standard NVIDIA build is CUDA 12.8.1.
#   It already contains Qwen3-VL support and accepts Transformers >=4.55.2, so
#   the reproducible compatibility stack is pinned here to:
#       vllm==0.11.0
#       torch==2.8.0+cu128 (via vLLM dependency + uv torch backend)
#       transformers==4.57.6
#
#   The known-good validation environment did NOT contain the optional
#   FlashInfer sampler stack. Reusing a venv that previously held a newer vLLM
#   can leave flashinfer-python / flashinfer-cubin behind even after vLLM is
#   downgraded. On 570.124.06 those stale CUDA extensions can both fail with
#   cudaErrorInsufficientDriver and alter startup memory profiling. Therefore
#   the default compatibility profile removes and rejects FlashInfer entirely.
#
# Usage:
#   bash ./build_vllm_workspace.sh --clean
#
# Runtime:
#   QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm bash ./run_analysis_v2_1_workspace.sh ...
#
# Optional overrides (use together only when intentionally testing a different stack):
#   QWEN_VLLM_WORKSPACE_ROOT=/workspace/qwen3-vllm
#   QWEN_VLLM_PYTHON_VERSION=3.12
#   QWEN_VLLM_VERSION=0.11.0
#   QWEN_VLLM_TRANSFORMERS_VERSION=4.57.6
#   QWEN_VLLM_TORCH_BACKEND=cu128
#   QWEN_VLLM_EXPECTED_CUDA=12.8
#   QWEN_VLLM_ALLOW_FLASHINFER=0

WORK_ROOT="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}"
PYTHON_VERSION="${QWEN_VLLM_PYTHON_VERSION:-3.12}"
VLLM_VERSION="${QWEN_VLLM_VERSION:-0.11.0}"
TRANSFORMERS_VERSION="${QWEN_VLLM_TRANSFORMERS_VERSION:-4.57.6}"
TORCH_BACKEND="${QWEN_VLLM_TORCH_BACKEND:-cu128}"
EXPECTED_CUDA="${QWEN_VLLM_EXPECTED_CUDA:-12.8}"
ALLOW_FLASHINFER="${QWEN_VLLM_ALLOW_FLASHINFER:-0}"
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
  QWEN_VLLM_VERSION               vLLM version (default: 0.11.0)
  QWEN_VLLM_TRANSFORMERS_VERSION  Transformers version (default: 4.57.6)
  QWEN_VLLM_TORCH_BACKEND         uv torch backend (default: cu128)
  QWEN_VLLM_EXPECTED_CUDA         required torch CUDA runtime prefix (default: 12.8)
  QWEN_VLLM_ALLOW_FLASHINFER      allow optional FlashInfer stack (default: 0)
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

if [[ "$ALLOW_FLASHINFER" != "0" && "$ALLOW_FLASHINFER" != "1" ]]; then
    echo "ERROR: QWEN_VLLM_ALLOW_FLASHINFER must be 0 or 1." >&2
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
echo "Torch backend:     $TORCH_BACKEND"
echo "Expected CUDA:     $EXPECTED_CUDA"
echo "Allow FlashInfer:  $ALLOW_FLASHINFER"
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

if [[ "$ALLOW_FLASHINFER" == "0" ]]; then
    echo
    echo "Removing optional FlashInfer packages from compatibility workspace, if present ..."
    "$UV_BIN" pip uninstall --python "$PY" flashinfer-python flashinfer-cubin >/dev/null 2>&1 || true
fi

echo
echo "Installing pinned CUDA-compatible vLLM / Transformers stack ..."
"$UV_BIN" pip install \
    --python "$PY" \
    --torch-backend="$TORCH_BACKEND" \
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
QWEN_VLLM_EXPECTED_VERSION="$VLLM_VERSION" \
QWEN_VLLM_EXPECTED_CUDA="$EXPECTED_CUDA" \
QWEN_VLLM_ALLOW_FLASHINFER="$ALLOW_FLASHINFER" \
"$PY" - <<'PY'
import importlib.metadata as md
import importlib.util
import os
import sys

import torch
import transformers
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

expected_vllm = os.environ["QWEN_VLLM_EXPECTED_VERSION"]
expected_cuda = os.environ["QWEN_VLLM_EXPECTED_CUDA"]
allow_flashinfer = os.environ["QWEN_VLLM_ALLOW_FLASHINFER"] == "1"
actual_vllm = md.version("vllm")
actual_cuda = str(torch.version.cuda or "")
flashinfer_importable = importlib.util.find_spec("flashinfer") is not None

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", actual_cuda)
print("transformers:", transformers.__version__)
print("vllm:", actual_vllm)
print("qwen-vl-utils:", md.version("qwen-vl-utils"))
print("CUDA available:", torch.cuda.is_available())
print("FlashInfer importable:", flashinfer_importable)
for package in ("flashinfer-python", "flashinfer-cubin", "apache-tvm-ffi"):
    try:
        print(f"{package}:", md.version(package))
    except md.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
print("vLLM LLM import: OK")
print("vLLM SamplingParams import: OK")
print("qwen_vl_utils.process_vision_info import: OK")

if actual_vllm != expected_vllm:
    raise SystemExit(f"ERROR: expected vLLM {expected_vllm}, got {actual_vllm}")
if not actual_cuda.startswith(expected_cuda):
    raise SystemExit(
        f"ERROR: expected torch CUDA runtime {expected_cuda}.x, got {actual_cuda or 'none'}. "
        "Refusing a workspace that may exceed the host-driver CUDA capability."
    )
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available to PyTorch; refusing to accept this vLLM build.")
if flashinfer_importable and not allow_flashinfer:
    raise SystemExit(
        "ERROR: FlashInfer is importable in the default cu128 compatibility workspace. "
        "This usually means the venv retained optional native packages from a newer vLLM stack. "
        "Rebuild with --clean."
    )

# Force a real CUDA runtime call; import-only checks are insufficient because a
# newer native extension can import successfully and fail only when initialized.
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print("CUDA smoke tensor:", float(x.item()))
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

echo
echo "Checking dependency consistency ..."
"$UV_BIN" pip check --python "$PY"

echo
echo "Writing environment freeze ..."
"$UV_BIN" pip freeze --python "$PY" > "$WORK_ROOT/environment.freeze.txt"
printf '%s\n' \
    "vllm=$VLLM_VERSION" \
    "transformers=$TRANSFORMERS_VERSION" \
    "torch_backend=$TORCH_BACKEND" \
    "expected_cuda=$EXPECTED_CUDA" \
    "allow_flashinfer=$ALLOW_FLASHINFER" \
    > "$WORK_ROOT/environment.spec.txt"

echo
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader || true
fi

echo
echo "=== vLLM workspace build complete ==="
echo "Python:       $PY"
echo "Freeze:       $WORK_ROOT/environment.freeze.txt"
echo "Spec:         $WORK_ROOT/environment.spec.txt"
echo "Runner usage: QWEN_WORKSPACE_ROOT=$WORK_ROOT bash $REPO_ROOT/run_analysis_v2_1_workspace.sh ..."
