# Analyze v2.1 + Fusion v2.3 SAM3D geometry support

Blind Validation 01 showed that SAM 3D Body provides stable unsigned torso-depth geometry, but also reconstructs plausible anatomy outside the visible crop. Fusion v2.3 therefore separates **reconstructed geometry** from **image-supported geometry**.

## Analyze v2.1: landmark visibility provenance

Analyze v2.1 adds `target_subject.geometry_landmark_visibility` for:

- head
- left/right shoulder
- left/right hip
- left/right knee
- left/right ankle

Each landmark region is classified as:

- `visible` — directly represented well enough to constrain geometry
- `partial` — partly visible or materially occluded
- `not_visible` — outside crop or not visually established
- `unknown` — ambiguous

This is deliberately not a keypoint estimator. It answers only whether the anatomical region is actually supported by visible pixels.

Important rule: body continuity does not upgrade missing anatomy. Visible thighs do not imply visible hip joints; a headless crop does not imply a visible head; reconstructed knees/ankles outside the crop remain `not_visible`.

Run:

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_analysis_v2_1_workspace.sh /data/<dataset> \
  --models 32b-fp8 \
  --backend vllm \
  --quantization none \
  --run-name <run-name> \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-max-model-len 8192
```

Analyze v2.0 remains supported and unchanged for existing cached runs. Legacy v2.0 analyses cannot grant SAM3D geometry authority because they do not contain explicit landmark-visibility provenance.

## SAM3D support states

Fusion v2.3 qualifies each torso-axis component independently.

### Shoulder depth axis

Requires both left and right shoulder regions.

### Hip depth axis

Requires both left and right hip regions.

Support states:

- `observed_supported` — both required regions are `visible` with confidence >= 0.75
- `partially_supported` — both have some visible evidence but at least one is partial / lower-confidence
- `prior_reconstructed` — at least one required region is explicitly `not_visible`
- `unknown` — visibility evidence is unresolved

If both shoulder and hip axes are `observed_supported`, unsigned `torso_depth_rotation` receives authority:

```text
qualified_3d_geometry
```

If one axis is observed while the other depends on invisible reconstructed anatomy, the aggregate becomes:

```text
report_only_partial_image_support
```

This preserves genuinely useful shoulder-only evidence without pretending the model actually saw hidden hips.

## Metric names

Fusion exposes:

- `shoulder_depth_rotation.magnitude_deg`
- `hip_depth_rotation.magnitude_deg`
- `torso_depth_rotation.magnitude_deg`
- `torso_axis_out_of_image_plane.magnitude_deg`

The final name replaces the experimental `torso_depth_tilt_deg`. It is camera-relative geometry, not world-relative recline/gravity.

The SAM3D probe currently writes the legacy key for backwards compatibility; Fusion v2.3 reads both names and exposes the corrected name.

Signed depth direction remains diagnostic only.

## Target provenance

SAM 3D Body reconstructs whichever bbox it receives; it does not prove that bbox belongs to the intended identity. Fusion therefore reports target-provenance risk when Analyze sees non-target entities or embedded depictions.

A geometry result can be landmark-supported yet remain:

```text
qualified_geometry_pending_target_provenance
```

when another person/photo/painting is present and bbox identity has not been independently reconciled.

## Run Fusion v2.3

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_fusion_v2_3_workspace.sh \
  runs/<analyze-v2.1-run> \
  --model 32b-fp8 \
  --dwpose-dir runs/<dwpose-run> \
  --sam3d-dir runs/<sam3d-run> \
  --overwrite
```

Outputs:

```text
runs/<analyze-v2.1-run>/fusion-v2.3/<model-slug>/
  <image>.fused_v2_3.json
  fusion_v2_3.index.json
```

## Selection authority

Fusion v2.3 qualifies SAM3D evidence, but **V8.1 portfolio scoring remains unchanged**.

`qualified_3d_geometry` means the evidence is trusted enough to be consumed by a later caption/dataset-policy layer. It does not automatically mean the current V8.1 optimizer uses it.

## Regression evidence

The initial four-image SAM3D probe was stable across DWPose bbox padding 0.10, 0.20, 0.35 and full-image input:

- strongly depth-rotated torso cases remained around 62–72 degrees
- overhead/lying control remained around 21 degrees
- near-frontal standing control remained around 14–17 degrees

The partial-body experiment then demonstrated why the support governor is necessary:

- headless crop with shoulders + hips visible: SAM3D torso geometry is legitimately image-supported
- headless crop with hips absent: SAM3D still reconstructs hips and returns a precise combined torso angle; hip contribution must be marked `prior_reconstructed`
- crop with head visible but hips obscured: visible thighs must not automatically upgrade hip-joint support

These cases should remain permanent regressions.
