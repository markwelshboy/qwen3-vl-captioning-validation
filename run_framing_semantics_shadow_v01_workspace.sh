#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [framing-semantics-shadow args...]" >&2
  exit 2
fi

RUN_DIR=$1
shift

PYTHON=${QWEN_WORKSPACE_PYTHON:-/workspace/qwen3/.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

exec "$PYTHON" -m qwen_caption_validate.framing_semantics_shadow_v01 \
  "$RUN_DIR" \
  "$@"
