# Visual Extract v3.0

Visual Extract v3.0 is the V3 pipeline's single image-conditioned semantic observation pass.

Its governing rule is:

```text
Observe once. Reason many times.
```

The Extract is intentionally persistent. Analyze and Composition/Gestalt should be reconstructable from the Extract record without seeing the source image again. DWPose and SAM3D remain independent geometry evidence families.

## Why this replaces repeated VLM image passes

Analyze v2.1 and Composition Gestalt v1.4 proved useful semantic questions, but repeatedly showing the same image to separate VLM stages is expensive and creates semantic regression: a later cautious pass can weaken an earlier specific observation (for example, a `red car` becoming a generic `red object`).

V3 therefore separates the image-conditioned work from downstream interpretation:

```text
IMAGE
  |
  +--> Visual Extract v3.0   (one VLM image pass)
  |       |
  |       +--> Analyze intelligence   (later; text/structured evidence only)
  |       +--> Gestalt intelligence   (later; text/structured evidence only)
  |
  +--> DWPose
  +--> SAM3D

              -> deterministic Fusion
```

## Record structure

The Extract deliberately contains two epistemic layers.

### Observations

These preserve what the VLM actually noticed in the pixels:

- crop/framing observations;
- transient clothing/accessory inventory;
- visible body fragments and connectivity;
- landmark-region visibility;
- torso/head/body-axis cues;
- gaze cues;
- stable scene entities with ids;
- spatial/contact relations referencing those ids;
- scene/background/lighting observations;
- nuisance/entropy regions;
- foreground and visual-thrust cues.

Specificity is persistent. Once an entity is defensibly identified as a `car` with descriptor `red`, later stages should refer to that entity id rather than re-identify it from scratch.

### Hypotheses

The same image pass may also preserve useful semantic interpretations:

- broad posture;
- torso orientation;
- head orientation;
- head/body relation;
- camera elevation/pitch;
- capture mode;
- support context;
- human-level actions.

These are explicitly hypotheses, each retaining confidence and cues/limitations. They are not governed facts.

## Authority boundary

Extract is not final authority for:

- anatomical laterality;
- ownership through occlusion gaps;
- exact camera geometry;
- exact 3-D body orientation;
- hidden support topology;
- invisible anatomy.

DWPose/SAM3D and later deterministic Fusion govern those claims.

## Completeness contract

Each generated artifact contains a small deterministic contract audit:

```json
{
  "analyze_reconstructable": true,
  "gestalt_reconstructable": true,
  "analyze_missing_paths": [],
  "gestalt_missing_paths": []
}
```

This is a structural test, not a correctness score. The empirical calibration question is stronger:

> Given only the Extract JSON, can downstream intelligence reproduce the useful conclusions of Analyze v2.1 and Composition Gestalt v1.4 without receiving the image?

## Run on the current calibration set

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git switch agent/semantic-fusion-v3
git pull --ff-only

RUN_DIR=/workspace/qwen3/qwen3-vl-captioning-validation/runs/Caption02-02

QWEN_VLLM_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_extract_v3_workspace.sh "$RUN_DIR" \
  --model 32b-fp8 \
  --backend vllm
```

For a targeted first smoke test, use `--only` with representative controls before running all 93 images.

Outputs:

```text
RUN_DIR/extract-v3.0/Qwen__Qwen3-VL-32B-Instruct-FP8/
  <image>.extract.json
  extract.index.json
```

Raw VLM responses are retained alongside parsed Extract records for audit.
