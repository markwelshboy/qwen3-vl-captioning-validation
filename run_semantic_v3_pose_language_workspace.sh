#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [--pose-dir PATH] [--output PATH] [--only KEY ...] [--overwrite] [--tar]" >&2
  exit 2
fi
shift

RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
POSE_DIR="${RUN_DIR}/sam3d-pose-discovery-01/relational-pose-profile-v0.16"
OUTPUT="${RUN_DIR}/semantic-v3/pose-language-v0.1"

PASS_ARGS=()
while (($#)); do
  case "$1" in
    --pose-dir)
      [[ $# -ge 2 ]] || { echo "--pose-dir requires a path" >&2; exit 2; }
      POSE_DIR="$2"; shift 2 ;;
    --pose-dir=*) POSE_DIR="${1#--pose-dir=}"; shift ;;
    --output)
      [[ $# -ge 2 ]] || { echo "--output requires a path" >&2; exit 2; }
      OUTPUT="$2"; shift 2 ;;
    --output=*) OUTPUT="${1#--output=}"; shift ;;
    --only)
      PASS_ARGS+=("--only")
      shift
      while (($#)) && [[ "$1" != --* ]]; do PASS_ARGS+=("$1"); shift; done ;;
    --overwrite|--tar) PASS_ARGS+=("$1"); shift ;;
    -h|--help)
      cat <<'EOF'
Usage:
  run_semantic_v3_pose_language_workspace.sh RUN_DIR [options]

Zero-GPU translation of frozen relational-pose-profile-0.16 records into short,
natural caption language. Raw geometry remains in the Pose artifact; this layer
emits only human-scale posture/orientation/relations and explicit crop framing.

Withheld reconstruction is never promoted directly. Strong reconstruction hints
remain conditional and require independent semantic corroboration later.

Options:
  --pose-dir PATH   Pose v0.16 profile directory
  --output PATH     Output directory (default: RUN_DIR/semantic-v3/pose-language-v0.1)
  --only KEY ...    Process only exact image keys
  --overwrite       Regenerate existing summaries
  --tar             Tar the output directory
EOF
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "Python environment not found: ${PY}" >&2; exit 2; }
[[ -d "${POSE_DIR}" ]] || { echo "Pose v0.16 directory not found: ${POSE_DIR}" >&2; exit 2; }

cd "${REPO_ROOT}"
exec "${PY}" -m qwen_caption_validate.semantic_v3_pose_language \
  "${POSE_DIR}" \
  --output "${OUTPUT}" \
  "${PASS_ARGS[@]}"
