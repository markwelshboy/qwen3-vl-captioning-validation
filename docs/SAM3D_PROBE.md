# SAM 3D Body geometry probe

This is an **experimental, report-only** third evidence source for the gap that remains after Analyze-v2/Fusion-v2.2:

```text
Qwen semantics         DWPose projected 2-D         SAM 3D Body 3-D
      \                        |                         /
       \_______________________|________________________/
                               |
                            Fusion
```

The first experiment is deliberately small. It asks only whether single-image 3-D reconstruction provides useful, defensible evidence for:

- torso rotation in depth;
- backward/forward torso depth tilt or recline;
- cases where Qwen reports a neutral/upright torso but DWPose can only tell us that the projected body is oblique.

Nothing from this probe is selection-authoritative yet.

## Why SAM 3D Body

The upstream model exposes 3-D body keypoints, mesh vertices, global rotation, joint coordinates, and joint global rotations. The first probe uses only simple shoulder/hip/torso vectors from `pred_keypoints_3d`, plus saved meshes for human review.

We intentionally do **not** interpret rotation-vector sign or anatomical direction in the first pass. The initial metrics are unsigned magnitudes.

## Frozen four-image probe

Use the same four images before changing any thresholds or downstream policy:

| image | role |
|---|---|
| `jQTv_720x1280_00008.png` | recline / oblique-body torture case |
| `jQTv_720x1280_00015.png` | torso depth-rotation torture case |
| `jQTv_720x1280_00011.png` | lying/reclined positive control |
| `jQTv_720x1280_00013.png` | upright standing full-body control |

We are looking for **relative separation**, not a magical absolute angle threshold.

A promising result would look qualitatively like:

```text
                         depth rotation     torso depth tilt
00008                          ?                  high
00015                         high                 ?
00011                          ?                  high
00013                         low                 low
```

The meshes must also look geometrically defensible. A plausible-looking but wrong reconstruction is a failure even if the numeric ordering happens to look useful.

## Upstream pin

The build script pins the upstream SAM 3D Body repository to:

```text
b5c765a0d89d789985e186d396315e7590887b94
```

This freezes the experiment against upstream changes.

The default gated checkpoint is:

```text
facebook/sam-3d-body-dinov3
```

Override it with `SAM3D_HF_REPO` or `--model-repo`.

## Build the isolated environment

SAM 3D Body is intentionally kept out of both the Qwen vLLM environment and the original Qwen/DWPose validation venv.

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git pull

SAM3D_WORKSPACE_ROOT=/workspace/sam3d-body \
bash ./build_sam3d_workspace.sh
```

The builder uses Python 3.11 and installs CUDA PyTorch under the isolated workspace. It follows the upstream core dependency list but deliberately omits Detectron2, SAM3 and MoGe for this first experiment.

### Why no Detectron2?

We already have a deterministic DWPose target-person choice and normalized target keypoint bbox. The probe converts that bbox back to image pixels, pads it, and passes it directly to `SAM3DBodyEstimator.process_one_image(...)`.

That means Qwen, DWPose and SAM 3D Body are all analyzing the same intended person without adding another detector to the experiment.

### Why no MoGe/FOV estimator yet?

The first question is whether SAM 3D Body adds useful 3-D body-orientation evidence at all. The first probe uses the model's default camera configuration and marks camera-dependent evidence as diagnostic-only.

If the four meshes/metrics look useful, a later matched experiment can rerun the same images with MoGe2 FOV estimation and compare stability.

## Gated Hugging Face access

The build performs a small access test by downloading only `model_config.yaml` from the gated checkpoint repository.

If access has already been approved but the workspace is not authenticated, authenticate **locally on the pod**:

```bash
HF_HOME=/workspace/sam3d-body/huggingface \
HF_HUB_CACHE=/workspace/sam3d-body/huggingface/hub \
/workspace/sam3d-body/.venv/bin/hf auth login
```

Do not paste a Hugging Face token into chat or commit one to the repository.

To pre-download the complete model during build:

```bash
SAM3D_WORKSPACE_ROOT=/workspace/sam3d-body \
bash ./build_sam3d_workspace.sh --download
```

Otherwise the first probe run downloads it automatically.

## Run the four-image experiment

```bash
SAM3D_WORKSPACE_ROOT=/workspace/sam3d-body \
bash ./run_sam3d_probe_workspace.sh /data/jQTv \
  --dwpose-dir runs/blind-validation-01/dwpose \
  --output runs/blind-validation-01-v2-1/sam3d-probe \
  --include jQTv_720x1280_00008.png \
  --include jQTv_720x1280_00015.png \
  --include jQTv_720x1280_00011.png \
  --include jQTv_720x1280_00013.png
```

Defaults:

```text
model             facebook/sam-3d-body-dinov3
inference type    body
bbox source       DWPose target bbox
bbox padding      20% on each side
save mesh         yes
external FOV      none
```

## Outputs

For each image:

```text
<image>.sam3d.json
<image>.sam3d_arrays.npz
<image>.sam3d.obj
```

and one run index:

```text
sam3d_probe.index.json
```

The compact JSON exposes:

- shoulder left-to-right 3-D vector;
- hip left-to-right 3-D vector;
- hip-midpoint to shoulder-midpoint 3-D vector;
- shoulder out-of-image-plane angle;
- hip out-of-image-plane angle;
- unsigned torso depth-rotation proxy = mean of shoulder/hip depth angles;
- unsigned torso depth-tilt angle;
- raw signed depth fractions as **diagnostic-only** values;
- camera translation/focal length diagnostics;
- raw global rotation without semantic interpretation.

The compressed NPZ preserves the useful raw model arrays for later analysis without embedding the full mesh in JSON.

## Bbox sensitivity check

If any reconstruction looks suspicious, rerun that image with the full image instead of the DWPose crop:

```bash
SAM3D_WORKSPACE_ROOT=/workspace/sam3d-body \
bash ./run_sam3d_probe_workspace.sh /data/jQTv \
  --dwpose-dir runs/blind-validation-01/dwpose \
  --output runs/blind-validation-01-v2-1/sam3d-probe-full-image \
  --bbox-source full \
  --include jQTv_720x1280_00008.png
```

If the inferred orientation changes radically merely from bbox choice, the evidence is not ready for Fusion authority.

## Acceptance criteria

Do not integrate SAM 3D Body into dataset scoring simply because it returns 3-D numbers.

Require all of the following first:

1. The standing `00013` mesh is anatomically plausible and produces low depth/recline magnitudes relative to the torture cases.
2. The lying `00011` case clearly separates from `00013` on torso depth tilt/recline.
3. `00008` captures meaningful recline/obliqueness rather than flattening it to the standing control.
4. `00015` captures substantial out-of-plane torso/shoulder/hip orientation if the mesh visually supports it.
5. The result is reasonably stable to padded-DWPose-bbox versus full-image input.
6. No directional/anatomical sign is given downstream authority until the coordinate convention is verified.
7. The meshes themselves agree with human visual review.

If those conditions pass, the next step is not to change V8.1 weights. It is to add a **SAM3D audit block** to Fusion and validate it report-only on the permanent regression suite.
