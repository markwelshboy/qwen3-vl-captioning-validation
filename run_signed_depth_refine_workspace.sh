#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
REPO_ROOT="$WORK_ROOT/qwen3-vl-captioning-validation"
PY="$WORK_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Python environment not found: $PY" >&2
  exit 2
fi

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.signed_depth_refine "$@"
