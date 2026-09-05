#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python venv not found at $PY" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "Semantic V3 Fusion 3.0.1: Extract + Analyze + Gestalt + governed Pose v0.16 -> canonical truth"
echo "Model/GPU load: NONE"
echo "Evidence policy: Extract+Analyze+Gestalt are one semantic family; Pose v0.16 is governed physical authority"
echo "Authority corrections: pose joint authority != confidence; withheld reconstruction cannot leak through modifiers; rejected proximal hand chains lose whole-hand specificity"
exec "$PY" -m qwen_caption_validate.semantic_v3_fusion_v301 "$@"
