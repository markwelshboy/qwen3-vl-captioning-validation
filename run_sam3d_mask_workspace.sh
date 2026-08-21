#!/usr/bin/env bash
set -euo pipefail

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
    exit 1
fi

mkdir -p "$WORK_ROOT/tmp" "$WORK_ROOT/.cache"
export TMPDIR="$WORK_ROOT/tmp"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$UPSTREAM_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.sam3d_mask "$@"
