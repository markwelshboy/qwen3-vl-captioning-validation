# Qwen3-VL Captioning Validation

A small standalone harness for testing Qwen3-VL visual analysis and training-caption methodology **before** plumbing it into Fizgig-Web.

The first target is the question we actually care about: given the same training crop and the same structured analysis prompt, what does **Qwen3-VL-8B-Instruct** see versus **Qwen3-VL-32B-Instruct**?

## What it does

- Scans a dataset folder for `.png`, `.jpg`, `.jpeg`, `.webp`, and `.bmp` images.
- Reads matching `.txt` sidecars so existing captions are visible in the report.
- Runs the same structured **Analysis v1** prompt against 8B, 32B, or arbitrary compatible Hugging Face model IDs.
- Saves raw response, parsed JSON, schema-validation state, backend, and inference time for every image/model pair.
- Optionally performs a second **Compose** call from cached Analysis JSON and pre-caches the resulting training caption.
- Records model-load, Analysis, and Compose runtime separately.
- Runs models sequentially so only one model is intended to be resident at a time.
- Supports resume: reuse `--run-name` and completed image/model outputs are skipped.
- Writes a local side-by-side `report.html` with per-image and aggregate runtime summaries.

## Important FP8 backend note

The official Qwen3-VL FP8 VL checkpoints are not all loadable directly through Transformers. In particular, Qwen's 32B FP8 model card currently says to use **vLLM or SGLang** rather than Transformers.

The validator therefore supports two backends:

- `transformers` for ordinary BF16/FP16 checkpoints and optional bitsandbytes fallback testing.
- `vllm` for native Qwen FP8 checkpoints.
- `auto` (default) routes model IDs ending in `-FP8` to vLLM and everything else to Transformers.

This avoids the `weight_scale_inv is on the meta device` failure seen when `Qwen/Qwen3-VL-32B-Instruct-FP8` is dispatched through Transformers/Accelerate.

## Recommended L40S stack

An L40S has 48 GB VRAM and is a very good target for the official FP8 checkpoints. For a clean 8B-vs-32B comparison, use the official FP8 variants for **both** models:

```text
Qwen/Qwen3-VL-8B-Instruct-FP8
Qwen/Qwen3-VL-32B-Instruct-FP8
```

For a fresh inference-only machine, vLLM recommends a fresh environment because its wheel is tied to a compatible PyTorch/CUDA stack. Their current recommended install path is `uv` with automatic torch-backend selection.

Example clean setup:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install 'qwen-vl-utils>=0.0.14'
uv pip install -e .
```

If you already have an otherwise disposable CUDA venv on the pod, installing the optional project extra is also convenient:

```bash
pip install -e '.[vllm]'
```

If vLLM replaces the existing PyTorch build, that is expected; vLLM recommends using the PyTorch version bundled/selected for its wheel rather than forcing an unrelated existing torch build.

## Fastest recovery from a partial run

If 8B already completed under Transformers and 32B failed to load, pull the latest repo and run just 32B using the same run name:

```bash
git pull
pip install -e '.[vllm]'

qwen-vl-validate /data/sh1vx \
  --models 32b-fp8 \
  --run-name analysis-v1-fp8
```

The existing 8B results remain in the report and the missing 32B column is filled in. This is useful for an immediate qualitative look, although the runtimes are not a clean backend-to-backend comparison.

For the clean comparison, rerun 8B through vLLM as well:

```bash
qwen-vl-validate /data/sh1vx \
  --models 8b-fp8 \
  --backend vllm \
  --run-name analysis-v1-fp8 \
  --overwrite
```

Or start a fresh all-vLLM run:

```bash
qwen-vl-validate /data/sh1vx \
  --models 8b-fp8 32b-fp8 \
  --run-name analysis-v1-vllm
```

`--backend auto` is the default, so the explicit `--backend vllm` is normally unnecessary for `*-fp8` aliases.

## L40S vLLM defaults

The validator defaults to:

```text
--vllm-gpu-memory-utilization 0.92
--vllm-max-model-len 8192
```

The workload has short prompts and one image per request, so an 8K context is ample and avoids reserving capacity for Qwen's much larger maximum context. If 32B reports insufficient GPU memory, try `0.95`; if the pod needs headroom for another process, reduce it.

## Analyze only

The first experiment should inspect visual understanding before judging generated captions:

```bash
qwen-vl-validate /data/sh1vx \
  --models 8b-fp8 32b-fp8 \
  --run-name analysis-v1-vllm
```

For prompt iteration, use a small diagnostic subset or `--limit`:

```bash
qwen-vl-validate /data/sh1vx \
  --models 8b-fp8 32b-fp8 \
  --run-name prompt-test \
  --limit 6
```

Open:

```text
runs/analysis-v1-vllm/report.html
```

The report shows the source crop, existing caption, structured JSON, backend, and runtime. Runtime is separated into:

- one-off model load time;
- per-image Analysis time (`A`);
- per-image Compose time (`C`) when enabled;
- total generation time and averages by model.

## Analyze -> Compose and pre-cache captions

Once Analysis JSON is behaving sensibly:

```bash
qwen-vl-validate /data/sh1vx \
  --models 8b-fp8 32b-fp8 \
  --run-name analyze-compose-v1-vllm \
  --compose \
  --subject-token sH1Vx \
  --detail balanced
```

Each model directory then contains:

```text
<image>.analysis.json
<image>.caption.txt
<image>.caption.json
```

The `.caption.txt` is the pre-cached caption candidate. The `.caption.json` also records Compose runtime and provenance. This means a future Fizgig Auto-rewrite strategy can prepare caption candidates **before training** and consume them later without paying VLM inference latency during the training run.

## Output layout

```text
runs/analysis-v1-vllm/
├── run.json
├── report.html
├── images/
├── Qwen__Qwen3-VL-8B-Instruct-FP8/
│   ├── results.jsonl
│   ├── image_001.analysis.json
│   └── ...
└── Qwen__Qwen3-VL-32B-Instruct-FP8/
    ├── results.jsonl
    ├── image_001.analysis.json
    └── ...
```

An analysis result preserves the raw model text even when JSON parsing/schema validation fails. Malformed structured output is useful validation data.

## Prompt/schema iteration

Nothing important is hard-coded into the Python implementation:

```text
prompts/analysis_v1.txt
prompts/compose_identity_pose_v1.txt
schemas/analysis_v1.schema.json
```

Override them at runtime:

```bash
qwen-vl-validate /data/sh1vx \
  --models 32b-fp8 \
  --analysis-prompt ./my_analysis_v2.txt \
  --schema ./my_analysis_v2.schema.json \
  --run-name analysis-v2
```

That makes this repo a prompt/schema laboratory rather than an early commitment to Fizgig's eventual architecture.

## Coordinate convention

Analysis v1 deliberately makes coordinate ownership explicit:

- **Body left/right is anatomical**: `left` means the subject's left.
- **Image position is frame-relative**: `image_left`, `image_right`, etc.
- Camera elevation, torso yaw, head yaw, head pitch, head roll, shoulder depth, gaze, limb ownership, contact/support, and foreshortening are kept separate.
- Unknown/uncertain is a valid answer and is preferred over an invented explanation.

The prompt targets the failure modes found in natural identity-training poses: frame/anatomical left-right swaps, hidden-limb invention, false low-angle claims, torso/head conflation, foreign hands/hair assigned to the target, weak contact language, and high-entropy nuisance regions.

## Useful options

```text
--models 8b 32b 8b-fp8 32b-fp8
--backend auto|transformers|vllm
--run-name NAME
--overwrite
--limit N
--recursive
--compose
--detail concise|balanced|detailed
--vllm-gpu-memory-utilization 0.92
--vllm-max-model-len 8192
--dtype auto|bfloat16|float16|float32
--quantization none|8bit|4bit   # Transformers only
--attn sdpa|flash_attention_2|eager   # Transformers only
--cache-dir PATH
--min-pixels N / --max-pixels N
```

## Initial validation strategy

The cleanest first comparison is:

1. Same image crop.
2. Same Analysis v1 prompt/schema.
3. Same vLLM backend.
4. Official FP8 8B versus official FP8 32B.
5. Review structured facts rather than prose quality.

Pay particular attention to anatomical laterality, torso/head counter-rotation, head-pitch magnitude, camera-angle restraint, body-part ownership, support/contact relationships, foreshortening, non-target people/body fragments, reflected/depicted faces, framing/archetype, and nuisance-region complexity versus identity relevance.

If 32B materially improves those fields, then test Analyze -> Compose. Only after that is it worth deciding whether/how to integrate the pipeline into Fizgig-Web.
