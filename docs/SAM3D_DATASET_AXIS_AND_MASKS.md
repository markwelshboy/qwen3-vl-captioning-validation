# SAM3D dataset axis and Entropy Focus mask prototype

This stage intentionally does two things without changing V8.1 selection weights:

1. expose qualified SAM 3D Body shoulder-girdle depth rotation as a **report-only dataset coverage axis**;
2. reuse cached SAM3D meshes to generate **prototype loss-weight masks** for Entropy Focus.

## 1. Shoulder-girdle depth rotation profile

Input is the Fusion-v2.3 model directory containing `*.fused_v2_3.json` files.

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_sam3d_dataset_profile_workspace.sh \
  runs/blind-validation-01-v2-1-full/fusion-v2.3/Qwen__Qwen3-VL-32B-Instruct-FP8
```

The command writes JSON and Markdown reports under the parent `fusion-v2.3` directory.

The axis is named `shoulder_girdle_depth_rotation`, not torso yaw. It is an unsigned, camera-relative 3-D measurement of how far the left-to-right shoulder axis rotates out of the image plane.

Presentation bands are currently:

- low: `[0,15)` degrees
- moderate: `[15,30)`
- high: `[30,50)`
- very high: `>=50`

These bands are **presentation only**. They do not affect Dataset Evidence / V8.1 scores.

Only shoulder measurements with `qualified_component_geometry` and no unresolved human-target provenance risk enter the qualified histogram. Geometry pending target provenance is reported separately rather than silently counted.

## 2. Entropy Focus mask prototype

SAM 3D Body's upstream renderer can project the cached body mesh back onto the source image. The alpha channel of that render gives a body-core silhouette without another model inference.

The prototype produces four artifacts per image:

```text
<image>.body_core.png
<image>.subject_zone.png
<image>.entropy_weight.png
<image>.entropy_preview.png
```

- `body_core`: alpha silhouette of the projected parametric body mesh.
- `subject_zone`: dilated body core; deliberately generous to include nearby hair/clothing pixels.
- `entropy_weight`: grayscale loss-weight proposal. White is full subject weight; background defaults to 0.35 with a soft transition halo.
- `entropy_preview`: source image multiplied by the weight mask for quick visual inspection.

This is **not a segmentation matte**. SAM3D reconstructs a complete parametric body and can project anatomy hidden by furniture or otherwise inferred from the body prior. The mask is therefore intended only as a loss-weighting proposal.

### Four-image visual test

Use the already cached 36-image SAM3D run; no Qwen or SAM3D inference is required:

```bash
SAM3D_WORKSPACE_ROOT=/workspace/sam3d-body \
bash ./run_sam3d_mask_workspace.sh /data/jQTv \
  --sam3d-dir runs/blind-validation-01-v2-1-full/sam3d \
  --fusion-dir runs/blind-validation-01-v2-1-full/fusion-v2.3/Qwen__Qwen3-VL-32B-Instruct-FP8 \
  --output runs/blind-validation-01-v2-1-full/entropy-mask-prototype \
  --include jQTv_720x1280_00013.png \
  --include jQTv_512x512_00015.png \
  --include jQTv_720x1280_00015.png \
  --include jQTv_720x1280_00011.png
```

Defaults:

```text
body-core alpha threshold: 0.01
subject-zone dilation:     4% of rendered body size
soft halo:                 4% of rendered body size
background loss weight:    0.35
subject-zone loss weight:  1.00
```

All are adjustable:

```text
--dilate-frac 0.04
--feather-frac 0.04
--background-weight 0.35
--alpha-threshold 0.01
```

The upstream SAM 3D Body `Renderer` is used directly, matching the model's own visualization path for `pred_vertices`, `pred_cam_t` and `focal_length` rather than reimplementing the camera projection.

## Validation criteria

The mask prototype is useful if it behaves as a forgiving subject-neighborhood weight map:

- forest / foliage / patterned backgrounds should be strongly downweighted;
- the body and immediate subject neighborhood should remain near full weight;
- hair and loose clothing should usually be protected by dilation;
- occluded or inferred body geometry must be treated as a known limitation, not as segmentation truth;
- images where the projected body obviously crosses foreground objects should be candidates for later dedicated segmentation/matting refinement.

The natural product split is therefore:

```text
Generate Entropy Mask
    -> fast SAM3D projected subject zone

Refine Mask
    -> dedicated segmentation / matting for difficult images
```
