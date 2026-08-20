#!/usr/bin/env bash
set -euo pipefail

# Run the SAM 3D Body geometry probe from its isolated workspace. The upstream
# source tree is placed on PYTHONPATH rather than installed into the Qwen venv.

WORK_ROOT="${SAM3D_WORKSPACE_ROOT:-/workspace/sam3d-body}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="$WORK_ROOT/src/sam-3d-body"
PY="$WORK_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: SAM3D workspace venv not found at $WORK_ROOT/.venv" >&2
    echo "Run: bash $REPO_ROOT/build_sam3d_workspace.sh" >&2
    exit 1
fi
if [[ ! -d "$UPSTREAM_DIR/sam_3d_body" ]]; then
    echo "ERROR: SAM 3D Body upstream checkout not found at $UPSTREAM_DIR" >&2
    echo "Run: bash $REPO_ROOT/build_sam3d_workspace.sh" >&2
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

export PYTHONPATH="$UPSTREAM_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.sam3d_probe "$@"
