#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_TOKENS="${ANALYZE_V2_1_MAX_TOKENS:-3200}"

exec bash "$REPO_ROOT/run_workspace.sh" \
  "$@" \
  --analysis-prompt "$REPO_ROOT/prompts/analysis_v2_1.txt" \
  --schema "$REPO_ROOT/schemas/analysis_v2_1.schema.json" \
  --max-analysis-tokens "$MAX_TOKENS"
