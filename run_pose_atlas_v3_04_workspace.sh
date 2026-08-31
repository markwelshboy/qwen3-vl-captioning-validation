#!/usr/bin/env bash
set -euo pipefail

ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Python environment not found: ${PY}" >&2
  echo "Run ./build_workspace.sh first or set QWEN_WORKSPACE_ROOT." >&2
  exit 2
fi

MAKE_TAR=0
ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == "--tar" ]]; then
    MAKE_TAR=1
  else
    ARGS+=("${arg}")
  fi
done

"${PY}" -m qwen_caption_validate.pose_atlas_v3_04_hotfix "${ARGS[@]}"

if [[ "${MAKE_TAR}" -eq 1 ]]; then
  RUN_DIR=""
  OUTPUT_DIR=""

  # Resolve the positional run directory and an optional explicit --output.
  # Keep this parser deliberately small and aligned with pose_atlas_v3_04.py.
  for ((i=0; i<${#ARGS[@]}; i++)); do
    arg="${ARGS[$i]}"
    case "${arg}" in
      --output)
        if (( i + 1 < ${#ARGS[@]} )); then
          OUTPUT_DIR="${ARGS[$((i + 1))]}"
          ((i+=1))
        fi
        ;;
      --output=*)
        OUTPUT_DIR="${arg#--output=}"
        ;;
      --images-dir|--dwpose-dir|--sam3d-dir|--annotations|--only|--quality)
        ((i+=1))
        ;;
      --overwrite)
        ;;
      --*)
        ;;
      *)
        if [[ -z "${RUN_DIR}" ]]; then
          RUN_DIR="${arg}"
        fi
        ;;
    esac
  done

  if [[ -z "${OUTPUT_DIR}" ]]; then
    if [[ -z "${RUN_DIR}" ]]; then
      echo "Cannot determine atlas output directory for --tar." >&2
      exit 2
    fi
    OUTPUT_DIR="${RUN_DIR%/}/semantic-v3/pose-atlas-v0.4"
  fi

  # Match pathlib's expanduser().resolve() behavior used by the Python atlas.
  OUTPUT_DIR="$(${PY} -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${OUTPUT_DIR}")"

  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Atlas output directory not found for --tar: ${OUTPUT_DIR}" >&2
    exit 2
  fi

  TAR_PATH="${OUTPUT_DIR}.tar"
  tar -C "$(dirname "${OUTPUT_DIR}")" -cf "${TAR_PATH}" "$(basename "${OUTPUT_DIR}")"
  echo "Tar: ${TAR_PATH}"
fi
