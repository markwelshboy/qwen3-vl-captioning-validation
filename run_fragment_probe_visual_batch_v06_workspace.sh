#!/usr/bin/env bash
set -euo pipefail

PY="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}/.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "Could not find qwen workspace Python." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT="$REPO_ROOT/prompts/fragment_probe_v06_reduced_typed_pose.txt"

has_prompt=0
for arg in "$@"; do
  case "$arg" in
    --prompt|--prompt=*) has_prompt=1 ;;
  esac
done

if (( has_prompt )); then
  exec "$PY" -m qwen_caption_validate.fragment_probe_visual_batch_v02 "$@"
else
  exec "$PY" -m qwen_caption_validate.fragment_probe_visual_batch_v02 \
    --prompt "$PROMPT" \
    "$@"
fi
