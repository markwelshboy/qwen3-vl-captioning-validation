#!/usr/bin/env bash
set -euo pipefail

ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Python environment not found: ${PY}" >&2
  echo "Run ./build_workspace.sh first or set QWEN_WORKSPACE_ROOT." >&2
  exit 2
fi

exec "${PY}" -m qwen_caption_validate.pose_library_census "$@"
