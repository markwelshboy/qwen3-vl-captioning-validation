#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [--sam3d-dir PATH] [--profile-output PATH] [--census-output PATH] [--tar]" >&2
  exit 2
fi
shift

RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
DWPOSE_DIR="${RUN_DIR}/dwpose"
IMAGES_DIR="${RUN_DIR}/images"
SAM3D_DIR="${RUN_DIR}/sam3d-pose-discovery-01"
PROFILE_OUTPUT=""
CENSUS_OUTPUT=""
MAKE_TAR=0

while (($#)); do
  case "$1" in
    --sam3d-dir)
      [[ $# -ge 2 ]] || { echo "--sam3d-dir requires a path" >&2; exit 2; }
      SAM3D_DIR="$2"; shift 2 ;;
    --sam3d-dir=*) SAM3D_DIR="${1#--sam3d-dir=}"; shift ;;
    --profile-output)
      [[ $# -ge 2 ]] || { echo "--profile-output requires a path" >&2; exit 2; }
      PROFILE_OUTPUT="$2"; shift 2 ;;
    --profile-output=*) PROFILE_OUTPUT="${1#--profile-output=}"; shift ;;
    --census-output)
      [[ $# -ge 2 ]] || { echo "--census-output requires a path" >&2; exit 2; }
      CENSUS_OUTPUT="$2"; shift 2 ;;
    --census-output=*) CENSUS_OUTPUT="${1#--census-output=}"; shift ;;
    --tar) MAKE_TAR=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage:
  run_pose_profile_recalibration_workspace.sh RUN_DIR [options]

Rebuilds only the SAM3D/DWPose relational profile and pose-library census from
existing caches. It does not load or run SAM3D inference.

v0.7 keeps the v0.6 posture classifier frozen and adds report-only support /
balance, recline, and kneeling-context diagnostics for calibration.

Options:
  --sam3d-dir PATH      Existing SAM3D array cache (default: RUN_DIR/sam3d-pose-discovery-01)
  --profile-output PATH Profile v0.7 output directory
  --census-output PATH  Census v0.2 output directory
  --tar                 Tar the profile directory (including default census)
EOF
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "Python environment not found: ${PY}" >&2; exit 2; }
[[ -d "${SAM3D_DIR}" ]] || { echo "SAM3D cache not found: ${SAM3D_DIR}" >&2; exit 2; }
[[ -d "${DWPOSE_DIR}" ]] || { echo "DWPose cache not found: ${DWPOSE_DIR}" >&2; exit 2; }
[[ -d "${IMAGES_DIR}" ]] || { echo "Images directory not found: ${IMAGES_DIR}" >&2; exit 2; }

SAM3D_DIR="$(${PY} -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${SAM3D_DIR}")"
if [[ -z "${PROFILE_OUTPUT}" ]]; then PROFILE_OUTPUT="${SAM3D_DIR}/relational-pose-profile-v0.7"; fi
PROFILE_OUTPUT="$(${PY} -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${PROFILE_OUTPUT}")"
if [[ -z "${CENSUS_OUTPUT}" ]]; then CENSUS_OUTPUT="${PROFILE_OUTPUT}/pose-library-census"; fi
CENSUS_OUTPUT="$(${PY} -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${CENSUS_OUTPUT}")"

echo "=== 1/2 Relational pose profile v0.7 (v0.6 scores frozen) ==="
QWEN_WORKSPACE_ROOT="${ROOT}" bash "${REPO_ROOT}/run_sam3d_relational_pose_profile_07_workspace.sh" \
  "${SAM3D_DIR}" \
  --dwpose-dir "${DWPOSE_DIR}" \
  --images-dir "${IMAGES_DIR}" \
  --output "${PROFILE_OUTPUT}"

echo
echo "=== 2/2 Pose-library census v0.2 ==="
QWEN_WORKSPACE_ROOT="${ROOT}" bash "${REPO_ROOT}/run_pose_library_census_02_workspace.sh" \
  "${PROFILE_OUTPUT}" \
  --output "${CENSUS_OUTPUT}"

if [[ "${MAKE_TAR}" -eq 1 ]]; then
  TAR_PATH="${PROFILE_OUTPUT}.tar"
  tar -C "$(dirname "${PROFILE_OUTPUT}")" -cf "${TAR_PATH}" "$(basename "${PROFILE_OUTPUT}")"
  echo
  echo "Tar: ${TAR_PATH}"
fi

echo
echo "Pose profile recalibration complete."
echo "Profile: ${PROFILE_OUTPUT}"
echo "Census: ${CENSUS_OUTPUT}/pose_library_census.md"
