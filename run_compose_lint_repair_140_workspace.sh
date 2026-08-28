#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$WORK_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: workspace venv not found at $WORK_ROOT/.venv" >&2
    exit 1
fi

mkdir -p \
    "$WORK_ROOT/tmp" \
    "$WORK_ROOT/pip-cache" \
    "$WORK_ROOT/huggingface/hub" \
    "$WORK_ROOT/huggingface/xet" \
    "$WORK_ROOT/.cache/nv" \
    "$WORK_ROOT/torch-cache"

export TMPDIR="$WORK_ROOT/tmp"
export PIP_CACHE_DIR="$WORK_ROOT/pip-cache"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export TORCH_HOME="$WORK_ROOT/torch-cache"
export CUDA_CACHE_PATH="$WORK_ROOT/.cache/nv"
export HF_HOME="$WORK_ROOT/huggingface"
export HF_HUB_CACHE="$WORK_ROOT/huggingface/hub"
export HF_XET_CACHE="$WORK_ROOT/huggingface/xet"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER || true

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.compose_lint_repair_140 "$@"
