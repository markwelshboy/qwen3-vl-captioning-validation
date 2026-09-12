# Routed Gestalt Acquisition v0.1

This is **Phase 3** of the crop-aware caption-refiner experiment.

Phase 1 chooses the semantic body-resolution policy. Phase 2 acquires the smallest useful body fact sheet for that route. Phase 3 now asks Qwen for the domains it is strongest at **without asking it to solve pose**:

- transient appearance
- expression / visible action
- objects
- scene / environment
- secondary people
- overall image gestalt

The same gestalt prompt is used for every Phase-1 route. The Phase-2 body output is referenced in metadata for audit but is **not injected into the Qwen prompt**. This keeps the acquisitions independent before normalization/composition.

## Ownership contract

The Phase-3 prompt explicitly does not own:

- broad posture
- body configuration
- framing / shot scale / crop extent
- head pose
- eye gaze
- anatomical laterality

Those domains remain with the routed body branch and specialists.

## Calibration run

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git pull --ff-only

RUN_DIR=/workspace/qwen3/qwen3-vl-captioning-validation/runs/SemanticV3-Blind-20260906-160740
IMAGES=/workspace/qwen3/images

KEYS=(
  imageblind-01_00001
  imageblind-01_00014
  imageblind-01_00020
  imageblind-01_00021
  imageblind-01_00023
  imageblind-01_00031
  imageblind-01_00049
  imageblind-01_00051
  imageblind-01_00064
)

QWEN_VLLM_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_routed_gestalt_v01_workspace.sh "$RUN_DIR" \
  --images-dir "$IMAGES" \
  --only "${KEYS[@]}" \
  --model 32b-fp8 \
  --backend vllm \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-max-model-len 8192 \
  --max-tokens 450 \
  --overwrite
```

All nine images should receive one gestalt call, including `framing_only` images. That is intentional: `framing_only` suppresses the **pose/body** question, not ordinary visual-semantic captioning.

## Compact review

```bash
OUT="$RUN_DIR/semantic-v3/routed-gestalt-v0.1/routed_gestalt.index.json"

jq -r '
  .records[] |
  [
    .image_key,
    .policy_mode,
    .status,
    ((.acquisition.gestalt // "-")),
    ((.acquisition.appearance // []) | join("; ")),
    ((.acquisition.expression_action // []) | join("; ")),
    ((.acquisition.objects // []) | join("; ")),
    ((.acquisition.scene // []) | join("; ")),
    ((.acquisition.secondary_people // []) | join("; ")),
    ((.parse.unexpected_keys // []) | join(","))
  ] | @tsv
' "$OUT"
```

Raw Qwen responses:

```bash
jq -r '
  .records[] |
  "\n===== \(.image_key) | \(.policy_mode) | \(.status) =====\n" +
  (.raw_response // "[NO RESPONSE]")
' "$OUT"
```

## Gate to clear before normalization/composition

The purpose of this run is not to create final prose yet. We want to establish that ordinary Qwen visual semantics remain rich when pose has been removed from its job.

Specific calibration questions:

- `00021`: does it give a useful ordinary selfie description without needing a standing/seated claim?
- `00023` / `00051`: do the tight crops still receive complete-looking appearance/scene semantics while the body branch stays empty?
- `00049`: does it capture the social/scene/action content while leaving shoulder/head geometry to the other branch?
- `00001` / `00064`: does the gestalt branch stay complementary to the useful broad-pose branch rather than redundantly solving pose again?
- `00020`: under-description of the unusual bed/selfie posture is acceptable; the architecture should not be tuned around this outlier.

Extra top-level JSON fields are dropped from the normalized acquisition and reported under `parse.unexpected_keys`. Free-text domain leakage inside allowed fields remains visible in the raw response for human review; this phase deliberately does not add another semantic correction machine.

If this gate passes, Phase 4 should normalize the independent body, framing, laterality, head/gaze, and gestalt records into one typed fact sheet before any text-only composition step.
