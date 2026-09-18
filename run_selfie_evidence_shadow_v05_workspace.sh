#!/usr/bin/env bash
set -euo pipefail

ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Python environment not found: $PY" >&2
  exit 2
fi

exec "$PY" -m qwen_caption_validate.selfie_evidence_shadow_v05 "$@"
