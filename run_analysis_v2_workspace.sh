#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_TOKENS="${ANALYZE_V2_MAX_TOKENS:-3000}"

# Invoke through bash rather than relying on the executable bit. GitHub's
# contents API may create/update shell scripts as mode 0644, and the workspace
# runner is intentionally a source-tree script rather than an installed binary.
exec bash "$REPO_ROOT/run_workspace.sh" \
  "$@" \
  --analysis-prompt "$REPO_ROOT/prompts/analysis_v2.txt" \
  --schema "$REPO_ROOT/schemas/analysis_v2.schema.json" \
  --max-analysis-tokens "$MAX_TOKENS"
