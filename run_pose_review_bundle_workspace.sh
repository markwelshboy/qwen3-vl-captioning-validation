#!/usr/bin/env bash
set -euo pipefail

ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Python environment not found: ${PY}" >&2
  echo "Run ./build_workspace.sh first or set QWEN_WORKSPACE_ROOT." >&2
  exit 2
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_DIR [pose_review_bundle options] [--tar]" >&2
  exit 2
fi

RUN_DIR="$1"
shift
MAKE_TAR=0
OUTPUT=""
ARGS=("${RUN_DIR}")

while (($#)); do
  case "$1" in
    --tar)
      MAKE_TAR=1
      shift
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "--output requires a path" >&2; exit 2; }
      OUTPUT="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --output=*)
      OUTPUT="${1#--output=}"
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

"${PY}" -m qwen_caption_validate.pose_review_bundle_04 "${ARGS[@]}"

if [[ "${MAKE_TAR}" -eq 1 ]]; then
  RUN_DIR_ABS="$(${PY} -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${RUN_DIR}")"
  if [[ -z "${OUTPUT}" ]]; then
    OUTPUT="${RUN_DIR_ABS}/semantic-v3/pose-review-v0.4"
  else
    OUTPUT="$(${PY} -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${OUTPUT}")"
  fi
  [[ -d "${OUTPUT}" ]] || { echo "Review bundle output not found: ${OUTPUT}" >&2; exit 2; }
  TAR_PATH="${OUTPUT}.tar"
  tar -C "$(dirname "${OUTPUT}")" -cf "${TAR_PATH}" "$(basename "${OUTPUT}")"
  echo "Tar: ${TAR_PATH}"
fi
