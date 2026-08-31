#!/usr/bin/env bash
set -euo pipefail

ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
PY="${ROOT}/.venv/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Python environment not found: ${PY}" >&2
  echo "Run ./build_workspace.sh first or set QWEN_WORKSPACE_ROOT." >&2
  exit 2
fi

# Historical calibration runs contain more than one serialized shape for
# raw_pose.bodies.  The compatibility entry point normalizes those cached
# DWPose records, then delegates to the normal V3 atlas implementation.
exec "${PY}" -m qwen_caption_validate.pose_atlas_v3_compat "$@"
