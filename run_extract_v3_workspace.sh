#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$WORK_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: vLLM workspace venv not found at $WORK_ROOT/.venv" >&2
  echo "Run: bash $REPO_ROOT/build_vllm_workspace.sh --clean" >&2
  exit 1
fi

mkdir -p \
  "$WORK_ROOT/tmp" \
  "$WORK_ROOT/uv-cache" \
  "$WORK_ROOT/pip-cache" \
  "$WORK_ROOT/huggingface/hub" \
  "$WORK_ROOT/huggingface/xet" \
  "$WORK_ROOT/.cache/nv" \
  "$WORK_ROOT/torch-cache"

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

# vLLM V1 enables FlashInfer sampling automatically when FlashInfer is
# importable.  The dedicated workspace is intentionally torch cu128 for rented
# L40S hosts whose 570-series driver cannot execute CUDA 12.9 runtime code, but
# FlashInfer may carry/JIT a newer CUDA runtime independently of torch.  The
# native vLLM/PyTorch sampler is fully adequate for our deterministic
# temperature=0 extraction workload and avoids that optional compatibility
# hazard.  Explicitly setting 1 still allows targeted testing on newer hosts.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "vLLM FlashInfer sampler: $VLLM_USE_FLASHINFER_SAMPLER (0 = PyTorch-native sampler)"

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.extract_v3 "$@"
