#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${QWEN_PYTHON:-/workspace/qwen3/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_DIR [framing-semantics-shadow-v06 args...]" >&2
  exit 2
fi

exec "$PYTHON" -m qwen_caption_validate.framing_semantics_shadow_v06 "$@"
