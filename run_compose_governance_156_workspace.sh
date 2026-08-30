#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$WORK_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: workspace venv not found at $PY" >&2
  exit 1
fi

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.compose_governance_156 "$@"
