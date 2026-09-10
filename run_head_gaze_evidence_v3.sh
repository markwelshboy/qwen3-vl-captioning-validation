#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -m qwen_caption_validate.head_gaze_evidence_v3 "$@"
