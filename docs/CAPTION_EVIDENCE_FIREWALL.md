# Caption Evidence Firewall / Governed Compose

This experiment returns the validation harness to its original captioning goal after the Analyze-v2.1 + DWPose + SAM3D/Fusion work.

The key question is not whether Fusion can reconstruct more body geometry. It is whether the extra evidence produces a better training caption **without leaking reconstructed anatomy or unqualified laterality into prose**.

## Architecture

```text
Analyze-v2.1 ----\
DWPose -----------+--> Fusion-v2.3 --> Caption Evidence Firewall --> text-only Compose
SAM3D ------------/
```

The firewall is deterministic. Compose never receives the raw SAM3D record or mesh.

## What the safe evidence can expose

- fused/qualified framing;
- semantic orientation, with unqualified anatomical direction replaced by `side_unspecified`;
- gaze and transient expression;
- qualified visible subject parts;
- qualified observed interactions;
- explicit visible / partial / not-visible landmark constraints;
- image-supported unsigned SAM3D depth magnitude only;
- environment/illumination and compact high-burden nuisance context;
- non-target entities / embedded depictions needed for disambiguation.

SAM3D shoulder or pelvis depth is exposed only when the corresponding landmark pair is image-supported and target provenance is clean. Exact degrees and signed direction are not exposed; the Compose model receives only a coarse depth-magnitude band.

## What is deliberately withheld

- raw SAM3D vertices, joints, complete-body reconstruction, camera values, and signed depth diagnostics;
- SAM3D geometry for anatomical regions that are partial/not visible or otherwise unqualified;
- all SAM3D geometry when target-bbox provenance requires review;
- report-only camera elevation;
- report-only deterministic projected-body geometry;
- body parts / interactions that Fusion marked non-usable;
- anatomical side where Fusion did not independently qualify laterality;
- anatomical left/right semantic orientation direction where no independent directional authority exists;
- the raw Analyze `image_summary`.

The last item is intentionally severe. Analyze-v2.1 has no dedicated structured transient-appearance/clothing object, so the safe caption may omit clothing that exists only in the summary. That is useful experimental information: if governed captions become too sparse, the fix should be a better Analyze schema rather than silently reintroducing an ungoverned prose summary.

## Comparison harness

`run_compose_fusion_compare_workspace.sh` produces three text-only captions from the same normalized Analyze record using one Compose model:

1. **Analyze-only** — current baseline Compose prompt.
2. **Analyze + DWPose** — existing secondary-pose Compose policy.
3. **Fusion-safe** — only the deterministic caption-safe evidence view.

The generated HTML report shows the source image for human review, all three captions, the raw Analyze summary (clearly marked as withheld from variant 3), the exact caption-safe JSON, and a firewall audit showing what was allowed or blocked.

No image-conditioned inference occurs during this comparison.

## Recommended first regression subset

Use deliberately difficult and ordinary cases together. For the current jQTv blind-validation dataset a useful first pass is:

```text
jQTv_512x512_00012   partial shoulder support / hair silhouette
jQTv_512x512_00013   environmental full-body / high nuisance
jQTv_512x512_00015   strong shoulder-depth / reflective structure
jQTv_720x1280_00002  camera-angle stress case
jQTv_720x1280_00003  target-provenance review case
jQTv_720x1280_00008  strong body-depth / occlusion
jQTv_720x1280_00011  reclined / overhead composition
jQTv_720x1280_00013  tiny full-body subject / forest entropy
jQTv_720x1280_00015  mirror / strong shoulder-depth
jQTv_720x1280_00019  connected hand + cup positive control
```

Add one or two ordinary portraits so the governed pipeline must also demonstrate that it does not make easy captions needlessly complicated.

## Example: 32B Analyze + 4B NF4 text Compose

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_compose_fusion_compare_workspace.sh \
  runs/entropy-first4 \
  --analysis-model 32b-fp8 \
  --compose-model Qwen/Qwen3-VL-4B-Instruct \
  --backend transformers \
  --quantization 4bit \
  --dtype bfloat16 \
  --only \
    jQTv_512x512_00012 \
    jQTv_512x512_00013 \
    jQTv_512x512_00015 \
    jQTv_720x1280_00002 \
    jQTv_720x1280_00003 \
    jQTv_720x1280_00008 \
    jQTv_720x1280_00011 \
    jQTv_720x1280_00013 \
    jQTv_720x1280_00015 \
    jQTv_720x1280_00019 \
  --overwrite
```

The report is written at the run root with a name containing both the Analyze and Compose model slugs.

## What to judge by eye

Do not reward captions merely for being longer. Look for:

- visible precision;
- visible coverage;
- pose/depth geometry accuracy;
- ownership correctness;
- laterality correctness or appropriate side-neutral wording;
- absence of hidden-anatomy completion;
- training relevance;
- information density;
- whether important transient appearance is missing because Analyze-v2.1 did not structure it.

The last category is intentionally part of the experiment. The governed pipeline should make missing data obvious rather than replacing missing evidence with confident invention.
