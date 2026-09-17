#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [routing-shadow options...]" >&2
  exit 2
fi

PY="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Could not find qwen workspace Python at $PY" >&2
  exit 2
fi

exec "$PY" -m qwen_caption_validate.framing_routing_policy_shadow_v01 "$@"
