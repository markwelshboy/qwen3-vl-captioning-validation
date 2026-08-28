#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$WORK_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: workspace venv not found at $WORK_ROOT/.venv" >&2
  exit 1
fi

mkdir -p "$WORK_ROOT/tmp" "$WORK_ROOT/.cache" "$WORK_ROOT/pip-cache"
export TMPDIR="$WORK_ROOT/tmp"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export PIP_CACHE_DIR="$WORK_ROOT/pip-cache"

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.pose_semantics_07 "$@"
