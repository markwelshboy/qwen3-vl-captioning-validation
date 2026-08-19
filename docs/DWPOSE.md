# DWPose dataset profiling

The validator includes a deliberately separate DWPose pass so 2D pose evidence can be cached independently of Qwen visual analysis.

DWPose is used here as a **secondary geometric checkpoint**, not as truth. Its useful outputs are body connectedness, visible landmark chains, skeleton extent, multi-person detection, shoulder/hip line geometry, and a coarse independent framing/extent hint. It does **not** independently solve front-vs-back orientation or metric depth, and its predicted anatomical left/right labels can still fail on ambiguous poses.

## Install / update

From the repository on a `/workspace` pod:

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git pull
bash ./build_workspace.sh
```

The workspace build installs `easy-dwpose==1.0.2` through the project's `dwpose` optional dependency and keeps Hugging Face/model caches under `/workspace/qwen3`.

## Profile the current dataset

To place DWPose results beside an existing Qwen validation run:

```bash
bash ./run_dwpose_workspace.sh /data/sh1vx \
  --output runs/analysis-v1-nf4/dwpose
```

Re-running the same command resumes: existing `*.dwpose.json` files are reused unless `--overwrite` is supplied.

Useful options:

```text
--device auto|cuda|cpu
--limit N
--recursive
--overwrite
```

## Outputs

For every image the profiler writes:

```text
<image-key>.dwpose.json
```

Each file contains:

- raw numerical DWPose output (`raw_pose`)
- number of detected person skeletons
- auditable target-person selection
- visible body landmark names/count
- keypoint bounding-box height/width/area fractions
- coarse `pose_extent_hint`
- shoulder-line angle
- hip-line angle
- torso-axis cant relative to image vertical
- left/right arm and leg chain completeness
- explicit limitations describing what DWPose does not establish

The output directory also contains:

```text
dataset.dwpose.json
```

This aggregates person-count and pose-extent distributions plus average target keypoint-bounding-box fractions and runtime. The `pose_extent_hint` is intentionally coarse; it is intended as an independent coverage checkpoint against the Qwen framing analysis rather than a replacement for it.

## Next experiment

First inspect the DWPose JSON against the known difficult images. If the evidence is useful, the next experiment is to feed only the cached **derived DWPose evidence** (not the image) alongside cached Qwen `analysis.json` into Compose. That gives a clean A/B test of whether deterministic 2D geometry materially changes target-model captions without rerunning vision analysis.
