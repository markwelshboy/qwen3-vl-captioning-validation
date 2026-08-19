# Qwen3-VL Captioning Validation

A deliberately small standalone harness for testing Qwen3-VL visual analysis and training-caption methodology **before** plumbing it into Fizgig-Web.

The first target is the question we actually care about: given the same training crop and the same structured analysis prompt, what does **Qwen3-VL-8B-Instruct** see versus **Qwen3-VL-32B-Instruct**?

The app runs models sequentially, so only one model is resident in accelerator memory at a time. Model weights remain in the Hugging Face cache unless you remove them separately.

## What it does

- Scans a dataset folder for `.png`, `.jpg`, `.jpeg`, `.webp`, and `.bmp` images.
- Reads matching `.txt` sidecars when present so the existing caption is visible in the report.
- Runs the same structured **Analysis v1** prompt against 8B, 32B, or arbitrary compatible Hugging Face model IDs.
- Saves the raw response, parsed JSON, schema-validation state, and analysis inference time for every image/model pair.
- Optionally performs a second **Compose** call using the structured analysis and pre-caches the resulting training caption plus its compose runtime.
- Records model-load time separately from per-image Analysis and Compose time.
- Unloads each model before loading the next.
- Supports resume: reuse `--run-name` and completed image/model outputs are skipped.
- Writes a local side-by-side `report.html` with per-image and aggregate runtime summaries.

## Installation

Use a CUDA-enabled PyTorch build appropriate for the machine first, then install this project:

```bash
python -m venv .venv
source .venv/bin/activate

# Install the correct CUDA PyTorch build for the host first.
# Then:
pip install -e .
```

Qwen3-VL support requires `transformers >= 4.57.0`. The harness uses the standard Transformers image-text generation path with `device_map="auto"`.

## Recommended single-L40S setup

An L40S has 48 GB VRAM and native FP8 support. A 32B checkpoint in BF16 is too large to be a comfortable single-GPU target, so for this validator prefer Qwen's official FP8 checkpoints rather than adding an unrelated 4-bit quantization variable.

Use the official FP8 variants for **both** sizes when making the cleanest 8B-vs-32B capacity comparison:

```text
Qwen/Qwen3-VL-8B-Instruct-FP8
Qwen/Qwen3-VL-32B-Instruct-FP8
```

A reasonable clean host stack is Python 3.11, current stable CUDA-enabled PyTorch, Transformers 4.x with Qwen3-VL support, Accelerate, and the dependencies installed by this project. Start with PyTorch SDPA; Flash Attention 2 is optional and can be tested after the baseline works.

Example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Install the current CUDA-enabled PyTorch build appropriate for the host/driver.
pip install torch

pip install -e .
```

Optional Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Then add `--attn flash_attention_2`. For the first correctness comparison, `--attn sdpa` is a perfectly sensible baseline and avoids making Flash Attention installation part of the experiment.

The harness also has a `--quantization 8bit|4bit` fallback path for ordinary checkpoints. That requires `bitsandbytes`, but it is **not** the preferred L40S experiment while official FP8 checkpoints exist.

## Fastest first experiment

Put the dataset in a folder and run both official FP8 models sequentially:

```bash
qwen-vl-validate /data/sh1vx \
  --models \
    Qwen/Qwen3-VL-8B-Instruct-FP8 \
    Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --attn sdpa \
  --run-name analysis-v1-fp8
```

For initial prompt iteration, do only a handful of the difficult images:

```bash
qwen-vl-validate /data/sh1vx \
  --models \
    Qwen/Qwen3-VL-8B-Instruct-FP8 \
    Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --attn sdpa \
  --run-name prompt-test \
  --limit 6
```

Open:

```text
runs/analysis-v1-fp8/report.html
```

The report shows the source crop, existing sidecar caption, each model's structured JSON, and the runtime cost. The report distinguishes:

- one-off model load time;
- per-image Analysis time (`A`);
- per-image Compose time (`C`) when enabled;
- total generation time and averages by model.

## Test Analyze -> Compose and pre-cache captions

Once the Analysis JSON is behaving sensibly:

```bash
qwen-vl-validate /data/sh1vx \
  --models \
    Qwen/Qwen3-VL-8B-Instruct-FP8 \
    Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --attn sdpa \
  --run-name analyze-compose-v1-fp8 \
  --compose \
  --subject-token sH1Vx \
  --detail balanced
```

Each model directory will then contain:

```text
<image>.analysis.json
<image>.caption.txt
<image>.caption.json
```

`caption.txt` is the directly reusable pre-cached training caption. `caption.json` stores the caption plus Compose runtime and provenance metadata used by the report.

This lets us answer separately:

1. Did the VLM perceive the crop correctly?
2. Given correct perception, did the training-caption policy produce useful language?
3. How much wall-clock inference did Analysis and Compose each cost?

The same pattern can later support pre-cached rewrite candidates: generate alternative captions from the cached Analysis under named Compose/rewrite prompts ahead of training, then let Fizgig choose among those candidates without invoking the VLM during the training run.

## Output layout

```text
runs/analyze-compose-v1-fp8/
├── run.json
├── report.html
├── images/
├── Qwen__Qwen3-VL-8B-Instruct-FP8/
│   ├── results.jsonl
│   ├── image_001.analysis.json
│   ├── image_001.caption.txt
│   ├── image_001.caption.json
│   └── ...
└── Qwen__Qwen3-VL-32B-Instruct-FP8/
    ├── results.jsonl
    ├── image_001.analysis.json
    ├── image_001.caption.txt
    ├── image_001.caption.json
    └── ...
```

An analysis result preserves the model's raw text even when JSON parsing or schema validation fails. That is intentional: malformed structured output is itself useful validation data.

## Prompt/schema iteration

Nothing important is hard-coded into the Python implementation.

```text
prompts/analysis_v1.txt
prompts/compose_identity_pose_v1.txt
schemas/analysis_v1.schema.json
```

Override them at runtime:

```bash
qwen-vl-validate /data/sh1vx \
  --models Qwen/Qwen3-VL-32B-Instruct-FP8 \
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

This is aimed directly at the failure modes found in natural identity-training poses: frame/anatomical left-right swaps, hidden-limb invention, false low-angle claims, torso/head conflation, foreign hands/hair assigned to the target, weak contact language, and high-entropy nuisance regions.

## Useful options

```text
--models MODEL [MODEL ...]    Run one or more models sequentially
--run-name NAME               Stable run folder; rerun to resume
--overwrite                   Recompute existing results
--limit N                     Quick prompt testing
--recursive                   Scan nested dataset folders
--compose                     Generate/cache the second-stage caption
--detail concise|balanced|detailed
--dtype auto|bfloat16|float16|float32
--quantization none|8bit|4bit
--attn sdpa|flash_attention_2|eager
--cache-dir PATH              Choose Hugging Face cache location
--min-pixels N / --max-pixels N
```

## Initial validation strategy

The first useful comparison is intentionally simple:

1. Same image crop.
2. Same Analysis v1 prompt/schema.
3. Same official FP8 precision for 8B and 32B on the L40S.
4. Review the structured fields rather than judging prose quality.
5. Compare runtime as well as correctness.

Pay particular attention to:

- anatomical laterality;
- torso yaw versus head counter-rotation;
- head pitch magnitude;
- camera-angle restraint;
- body-part ownership;
- support/contact relationships;
- foreshortening;
- non-target people/body fragments;
- reflected/depicted faces;
- framing/photographic archetype;
- nuisance-region complexity versus identity relevance.

If 32B materially improves those fields, then test Analyze -> Compose. Only after that is it worth deciding whether/how to integrate the pipeline into Fizgig-Web.
