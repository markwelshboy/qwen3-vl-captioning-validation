# Full governed caption pipeline — Fusion 2.3.3 / Projection 1.3.5

This is the current exploratory end-to-end path for turning an image dataset into governed training-caption sidecars.

The pipeline is intentionally staged so expensive visual inference is cached and later governance/projection changes can be replayed without rerunning Analyze.

```text
images
  -> Analyze v2.1 (Qwen3-VL 32B FP8, image-conditioned)
  -> DWPose (deterministic 2-D observations)
  -> SAM 3D Body (3-D geometry; not visibility authority)
  -> Fusion 2.3
  -> laterality refinement 2.3.1
  -> bilateral collision guard 2.3.2
  -> signed-depth refinement 2.3.3
  -> Projection 1.3.5 + Qwen3-VL 8B BF16 text-only Compose
  -> authority lint
  -> one-shot text-only lint repair when needed
  -> final lint
  -> optional training-caption export
```

## 0. Variables

Adjust these first.

```bash
REPO=/workspace/qwen3/qwen3-vl-captioning-validation
DATASET=/data/pseudo-dataset
RUN_NAME=pseudo-01
RUN_DIR="$REPO/runs/$RUN_NAME"
SUBJECT_TOKEN=sH1Vx

ANALYSIS_MODEL=32b-fp8
ANALYSIS_SLUG=Qwen__Qwen3-VL-32B-Instruct-FP8
COMPOSE_MODEL=Qwen/Qwen3-VL-8B-Instruct
COMPOSE_LABEL=8b-bf16-governance135
REPAIR_LABEL=${COMPOSE_LABEL}-repair1

CAPTION_EXPORT=/data/pseudo-governed-captions
```

Use the same `SUBJECT_TOKEN` that will be used in the companion stock-caption training run.

## 1. Fresh pod / repository

```bash
mkdir -p /workspace/qwen3
cd /workspace/qwen3

if [[ -d qwen3-vl-captioning-validation/.git ]]; then
  git -C qwen3-vl-captioning-validation pull --ff-only
else
  git clone https://github.com/markwelshboy/qwen3-vl-captioning-validation.git
fi

cd "$REPO"
git rev-parse HEAD
```

## 2. Build the three isolated runtimes

### Transformers / DWPose / deterministic governance

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
  bash ./build_workspace.sh --clean
```

### vLLM for native Qwen3-VL 32B FP8 Analyze

```bash
cd "$REPO"
QWEN_VLLM_WORKSPACE_ROOT=/workspace/qwen3-vllm \
  bash ./build_vllm_workspace.sh --clean
```

### SAM 3D Body

SAM 3D Body uses a separate environment and a gated Hugging Face checkpoint.

```bash
cd "$REPO"
SAM3D_WORKSPACE_ROOT=/workspace/sam3d-body \
  bash ./build_sam3d_workspace.sh --clean --download
```

If checkpoint access is approved but the pod is not authenticated, authenticate locally in that workspace and rerun the build. Never put the token in scripts or logs.

## 3. Analyze v2.1 — 32B FP8

This is the expensive image-conditioned semantic pass.

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_analysis_v2_1_workspace.sh "$DATASET" \
  --models "$ANALYSIS_MODEL" \
  --backend vllm \
  --run-name "$RUN_NAME" \
  --recursive \
  --subject-token "$SUBJECT_TOKEN" \
  --detail balanced
```

Outputs are cached beneath:

```text
$RUN_DIR/$ANALYSIS_SLUG/
```

## 4. DWPose

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_dwpose_workspace.sh "$DATASET" \
  --output "$RUN_DIR/dwpose" \
  --recursive \
  --device auto
```

## 5. SAM 3D Body

Meshes are unnecessary for a dataset run unless visual mesh inspection is desired.

```bash
cd "$REPO"
SAM3D_WORKSPACE_ROOT=/workspace/sam3d-body \
bash ./run_sam3d_probe_workspace.sh "$DATASET" \
  --dwpose-dir "$RUN_DIR/dwpose" \
  --output "$RUN_DIR/sam3d" \
  --bbox-source dwpose \
  --inference-type body \
  --no-save-mesh
```

## 6. Fusion 2.3 — base Analyze + DWPose + visibility-gated SAM3D

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_fusion_v2_3_workspace.sh "$RUN_DIR" \
  --model "$ANALYSIS_MODEL" \
  --dwpose-dir "$RUN_DIR/dwpose" \
  --sam3d-dir "$RUN_DIR/sam3d"
```

## 7. Fusion 2.3.1 — laterality refinement

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_laterality_refine_workspace.sh "$RUN_DIR" \
  --model "$ANALYSIS_MODEL"
```

## 8. Fusion 2.3.2 — bilateral collision guard

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_laterality_bilateral_guard_workspace.sh "$RUN_DIR" \
  --model "$ANALYSIS_MODEL"
```

## 9. Fusion 2.3.3 — signed depth

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_signed_depth_refine_workspace.sh "$RUN_DIR" \
  --model "$ANALYSIS_MODEL"
```

The final deterministic Fusion input to Compose is now:

```text
$RUN_DIR/fusion-v2.3.3/$ANALYSIS_SLUG/
```

## 10. Projection 1.3.5 + 8B BF16 text-only Compose

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_compose_governance_135_workspace.sh "$RUN_DIR" \
  --analysis-model "$ANALYSIS_MODEL" \
  --fusion-dir "$RUN_DIR/fusion-v2.3.3/$ANALYSIS_SLUG" \
  --compose-model "$COMPOSE_MODEL" \
  --backend transformers \
  --quantization none \
  --dtype bfloat16 \
  --detail balanced \
  --subject-token "$SUBJECT_TOKEN" \
  --variants fusion-safe \
  --run-label "$COMPOSE_LABEL"
```

This writes the first-pass governed captions and authority-lint results.

## 11. One-shot lint repair + final training-caption export

Only captions with a lint violation/warning receive another text-only generation. Clean captions are copied unchanged.

`--export-caption-dir` writes the final post-repair captions using the **original image-relative path/stem**:

```text
images/foo.png          -> <export>/images/foo.txt
portrait_001.webp       -> <export>/portrait_001.txt
```

```bash
cd "$REPO"
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_compose_lint_repair_135_workspace.sh "$RUN_DIR" \
  --analysis-model "$ANALYSIS_MODEL" \
  --compose-model "$COMPOSE_MODEL" \
  --source-run-label "$COMPOSE_LABEL" \
  --run-label "$REPAIR_LABEL" \
  --backend transformers \
  --quantization none \
  --dtype bfloat16 \
  --export-caption-dir "$CAPTION_EXPORT"
```

The export contains:

```text
$CAPTION_EXPORT/
  <matching image stems>.txt
  caption_export.index.json
```

`caption_export.index.json` records accepted versus `review_required` captions, repair state, and final warning/violation counts. The current exporter writes all final captions so exploratory training can proceed; the index preserves review status instead of silently dropping examples.

## 12. Sanity checks before training

```bash
jq '{matched,written,review_required,missing}' \
  "$CAPTION_EXPORT/caption_export.index.json"

find "$DATASET" -type f \( \
  -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' \
\) | wc -l

find "$CAPTION_EXPORT" -type f -name '*.txt' | wc -l
```

Inspect any remaining review candidates:

```bash
jq -r '.records[] | select(.review_required) | [.image,.caption_path,.warning_count,.violation_count] | @tsv' \
  "$CAPTION_EXPORT/caption_export.index.json"
```

## 13. Prepare a controlled stock-vs-governed caption A/B

The clean experiment changes **captions only**. Keep image files, trigger token, repeats, optimizer, LR, rank/alpha, batch/accumulation, augmentation, resolution policy, total steps, checkpoint cadence, validation prompts and validation seeds identical.

If the source `DATASET` already contains the stock `.txt` captions, it is the baseline caption source. The governed alternative is `CAPTION_EXPORT`.

If the trainer requires captions beside images, create two staging datasets. A simple copy-based version is:

```bash
STOCK_TRAIN=/data/pseudo-train-stock
GOV_TRAIN=/data/pseudo-train-governed

rm -rf "$STOCK_TRAIN" "$GOV_TRAIN"
cp -a "$DATASET" "$STOCK_TRAIN"
cp -a "$DATASET" "$GOV_TRAIN"

# Replace only captions in the governed clone.
find "$GOV_TRAIN" -type f -name '*.txt' -delete
rsync -a --exclude='caption_export.index.json' "$CAPTION_EXPORT/" "$GOV_TRAIN/"
```

Before training, confirm both trees have the same image count and the governed tree has the expected caption count.

## 14. Experimental interpretation

For the first exploratory A/B, do not optimize around a single final loss number. Compare the same checkpoints and validation seeds for:

- identity fidelity;
- pose/laterality fidelity;
- interaction/contact fidelity;
- background/scene leakage into identity;
- clothing/accessory disentanglement;
- generalization to poses not present in the dataset;
- per-image or individual-loss behavior if available;
- whether difficult images become less dominant in the loss trajectory.

The immediate question is only whether the governed captions move the training trajectory or qualitative output enough to justify deeper tuning.
