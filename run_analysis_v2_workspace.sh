#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_TOKENS="${ANALYZE_V2_MAX_TOKENS:-3000}"

exec "$REPO_ROOT/run_workspace.sh" \
  "$@" \
  --analysis-prompt "$REPO_ROOT/prompts/analysis_v2.txt" \
  --schema "$REPO_ROOT/schemas/analysis_v2.schema.json" \
  --max-analysis-tokens "$MAX_TOKENS"
