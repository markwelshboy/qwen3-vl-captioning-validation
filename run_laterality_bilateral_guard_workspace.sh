#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
REPO_ROOT="${QWEN_REPO_ROOT:-$WORK_ROOT/qwen3-vl-captioning-validation}"
PY="${QWEN_PYTHON:-$WORK_ROOT/.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "Python not found or not executable: $PY" >&2
  exit 2
fi

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.laterality_bilateral_guard "$@"
