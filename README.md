# Qwen3-VL Captioning Validation

A deliberately small standalone harness for testing Qwen3-VL visual analysis and training-caption methodology **before** plumbing it into Fizgig-Web.

The first target is the question we actually care about: given the same training crop and the same structured analysis prompt, what does **Qwen3-VL-8B-Instruct** see versus **Qwen3-VL-32B-Instruct**?

The app runs models sequentially, so only one model is resident in accelerator memory at a time. Model weights remain in the Hugging Face cache unless you remove them separately.

## What it does

- Scans a dataset folder for `.png`, `.jpg`, `.jpeg`, `.webp`, and `.bmp` images.
- Reads matching `.txt` sidecars when present so the existing caption is visible in the report.
- Runs the same structured **Analysis v1** prompt against 8B, 32B, or arbitrary compatible Hugging Face model IDs.
- Saves the raw response, parsed JSON, schema-validation state, and inference time for every image/model pair.
- Optionally performs a second **Compose** call using the structured analysis to produce an identity/pose-aware training caption.
- Unloads each model before loading the next.
- Supports resume: reuse `--run-name` and completed image/model outputs are skipped.
- Writes a local side-by-side `report.html` for visual comparison.

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

For the 32B model, use a sufficiently large GPU or multi-GPU machine. The harness intentionally does **not** quantize by default because the initial experiment is meant to compare model capacity, not model capacity plus a quantization variable.

## Fastest first experiment

Put the 28-image dataset in a folder and run both models sequentially:

```bash
qwen-vl-validate /data/sh1vx \
  --models 8b 32b \
  --run-name analysis-v1
```

For initial prompt iteration, do only a handful of the difficult images:

```bash
qwen-vl-validate /data/sh1vx \
  --models 8b 32b \
  --run-name prompt-test \
  --limit 6
```

Open:

```text
runs/analysis-v1/report.html
```

The report shows the source crop, existing sidecar caption, and each model's structured JSON side by side.

## Test Analyze -> Compose too

Once the Analysis JSON is behaving sensibly:

```bash
qwen-vl-validate /data/sh1vx \
  --models 8b 32b \
  --run-name analyze-compose-v1 \
  --compose \
  --subject-token sH1Vx \
  --detail balanced
```

Each model directory will then contain both:

```text
<image>.analysis.json
<image>.caption.txt
```

This lets us answer separately:

1. Did the VLM perceive the crop correctly?
2. Given correct perception, did the training-caption policy produce useful language?

## Output layout

```text
runs/analysis-v1/
├── run.json
├── report.html
├── images/
├── Qwen__Qwen3-VL-8B-Instruct/
│   ├── results.jsonl
│   ├── image_001.analysis.json
│   └── ...
└── Qwen__Qwen3-VL-32B-Instruct/
    ├── results.jsonl
    ├── image_001.analysis.json
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
  --models 32b \
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
--models 8b 32b             Run one or more models sequentially
--run-name NAME             Stable run folder; rerun to resume
--overwrite                 Recompute existing results
--limit N                   Quick prompt testing
--recursive                 Scan nested dataset folders
--compose                   Run the second caption-composition call
--detail concise|balanced|detailed
--dtype auto|bfloat16|float16|float32
--attn sdpa|flash_attention_2|eager
--cache-dir PATH            Choose Hugging Face cache location
--min-pixels N / --max-pixels N
```

## Initial validation strategy

The first useful comparison is intentionally simple:

1. Same image crop.
2. Same Analysis v1 prompt/schema.
3. 8B versus 32B.
4. Review the structured fields rather than judging prose quality.

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
