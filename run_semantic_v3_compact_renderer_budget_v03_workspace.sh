#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$WORK_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: workspace venv not found at $WORK_ROOT/.venv" >&2
  exit 1
fi

export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
mkdir -p "$TMPDIR"

echo "Semantic V3 compact renderer budget v0.3: v0.2 semantic-family selection plus renderer-truncation/completeness gating."
echo "No model/GPU inference occurs. finish=length sources must be rerendered before a caption can be certified final."

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.semantic_v3_compact_renderer_budget_v03 "$@"
