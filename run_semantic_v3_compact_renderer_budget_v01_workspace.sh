#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$WORK_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: vLLM workspace venv not found at $WORK_ROOT/.venv" >&2
  exit 1
fi

echo "Semantic V3 compact renderer budget v0.1: deterministic sentence selection under hard compact/medium word budgets."
echo "No model/GPU load occurs; complete renderer sentences are selected without rewriting text."

cd "$REPO_ROOT"
exec "$PY" -m qwen_caption_validate.semantic_v3_compact_renderer_budget_v01 "$@"
