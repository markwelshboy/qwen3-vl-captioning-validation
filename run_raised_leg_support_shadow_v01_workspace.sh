#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-runs}"
INPUT="${ROOT}/semantic-v3/caption-fact-sheet-v0.2.12"
OUTPUT="${ROOT}/semantic-v3/raised-leg-support-shadow-v0.1"

python -m qwen_caption_validate.raised_leg_support_shadow_v01 \
  --input "${INPUT}" \
  --output "${OUTPUT}"
