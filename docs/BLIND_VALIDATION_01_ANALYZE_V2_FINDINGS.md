# Blind Validation 01 — Analyze-v2 / Fusion-v2 findings

This document records the first 36-image Analyze-v2 run on the `jQTv` blind subject and the follow-up Fusion-v2 review. The candidate images were intentionally adversarial, while several ordinary incumbent images were used as controls.

## Run-level result

- 36/36 Analyze-v2 responses parsed as JSON.
- No output truncation occurred with the 3000-token Analyze-v2 budget.
- The longest response was the difficult full-body woodland image (`jQTv_720x1280_00013.png`) and it completed successfully.
- Only 4/36 raw model responses were schema-valid before lexical normalization.
- Most schema failures were non-semantic vocabulary drift:
  - `camera` instead of `camera_lens`
  - `center` instead of `image_center`
  - `phone screen` instead of `object`
  - `near-horizontal` instead of `near_horizontal`
  - one plural body part using `both` for anatomical side

Analyze-v2.1 therefore adds deterministic vocabulary normalization while preserving the raw model response for audit. Fusion-v2.1 validates the normalized analysis before consuming it.

## Regression results

| image | result | notes |
|---|---|---|
| `jQTv_512x512_00018.png` | strong pass | Structured body-part evidence records two visible fingers, unknown ownership, no palm/wrist/arm, disconnected in crop, and observed neck contact. Fusion makes the fragment non-selection-authoritative. The prose summary still said “a hand”, so the prompt was tightened to make summaries obey fragment-before-whole too. |
| `jQTv_720x1280_00019.png` | strong pass | Analyze-v2 reports a target cup hand/wrist. DWPose hand-root association lands 0.024 from the compatible visible wrist and the arm chain is complete. The old centroid failure is eliminated. |
| `jQTv_720x1280_00002.png` | partial/fail | The previous catastrophic `low` inversion disappeared, but 32B still reports `eye_level` at 0.90 despite the camera being above the subject. Its cited “eyes at same height as camera” evidence is not a valid observable 3-D cue. Analyze-v2.1 explicitly bans this evidence pattern. |
| `jQTv_720x1280_00011.png` | strong pass | Correctly reports high/overhead camera, backward/reclined torso, and strong near-horizontal image-plane body axis. |
| `jQTv_720x1280_00008.png` | partial/fail | Still flattens the reclined/oblique body into neutral torso pitch and upright body axis. Qwen also assigns the cup hand to the right while DWPose hand-root evidence associates it to the left target wrist/arm chain. Fusion-v2.1 now separates action ownership from laterality and explicitly downgrades side conflicts. |
| `jQTv_720x1280_00015.png` | partial | Structural/specular elevator evidence is good, but torso yaw remains substantially under-described. DWPose reports a strong projected torso-axis cant; Fusion-v2.1 now reports semantic-vs-projected-axis conflicts separately without pretending 2-D cant is torso yaw. |
| `jQTv_512x512_00015.png` | strong pass | Correctly distinguishes low texture from high structural/specular complexity and detects the reflective elevator background. |
| `jQTv_720x1280_00013.png` | strong pass | Full-body evidence and extreme woodland texture/structure are both preserved; nuisance regions are correctly marked as entropy-focus candidates. No truncation. |

## Ordinary controls

The ordinary control images remained broadly defensible:

- `jQTv_512x512_00010.png` — straightforward close portrait remained straightforward.
- `jQTv_720x1280_00001.png` — ordinary contextual cup portrait remained coherent.
- `jQTv_512x512_00013.png` — clean full-body incumbent remained correctly identified as full length.

This supports treating the adversarial failures as targeted evidence gaps rather than general analyzer instability.

## Fusion-v2.1 changes after review

Fusion-v2.1 is still audit-first and does not alter V8.1 portfolio weights.

Changes:

1. Normalize safe lexical aliases and reject analyses that remain schema-invalid after normalization.
2. Preserve hand/contact actions separately from anatomical laterality.
3. On direct images, a Qwen hand-side claim that conflicts with DWPose hand-root/wrist association is downgraded to unknown laterality rather than silently accepted.
4. Mirror selfies do not use DWPose side labels to validate true anatomical laterality.
5. Camera audit rejects non-geometric evidence such as “eyes at the same height as the camera” from establishing authority.
6. Add a report-only projected-body-axis audit for conflicts between semantic `image_plane_body_axis` and DWPose projected torso/shoulder geometry.

## Next targeted rerun

Do not rerun all 36 immediately. Re-run the adversarial regression images plus two controls with the tightened prompt.

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_analysis_v2_workspace.sh /data/jQTv \
  --models 32b-fp8 \
  --backend vllm \
  --quantization none \
  --run-name blind-validation-01-v2-1 \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-max-model-len 8192 \
  --include jQTv_512x512_00018.png \
  --include jQTv_720x1280_00019.png \
  --include jQTv_720x1280_00002.png \
  --include jQTv_720x1280_00011.png \
  --include jQTv_720x1280_00008.png \
  --include jQTv_720x1280_00015.png \
  --include jQTv_512x512_00015.png \
  --include jQTv_720x1280_00013.png \
  --include jQTv_512x512_00010.png \
  --include jQTv_720x1280_00001.png
```

Then reuse the original DWPose cache:

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_fusion_v2_workspace.sh \
  runs/blind-validation-01-v2-1 \
  --model 32b-fp8 \
  --dwpose-dir runs/blind-validation-01/dwpose \
  --overwrite
```

Selection integration remains blocked until the camera and body-axis regressions are defensible.