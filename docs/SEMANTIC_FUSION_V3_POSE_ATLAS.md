# Semantic Fusion V3 — Pose Atlas Calibration Phase

This branch starts the V3 architecture as a **parallel research path**. V2 remains a frozen comparison baseline.

The first V3 milestone is deliberately not another caption generator. It is an empirical pose atlas built from already-cached image, DWPose, and SAM3D evidence.

## Why start here?

Recent calibration exposed a structural problem: the caption pipeline increasingly treated missing anatomy as a verification failure. Ordinary human semantics such as `seated` could be withheld because the chair, knees, feet, or complete support chain were outside the crop.

For character-training captions, that is usually the wrong objective.

The new target is:

> Find the highest-level ordinary semantic explanation that is supported, visually compatible, and sufficient for the visible crop.

Key rules:

- missing anatomy outside the crop is **not** counterevidence;
- visible contradiction **is** counterevidence;
- SAM3D reconstruction may support a broad pose class without turning invisible joints into caption facts;
- DWPose is primarily a visible-geometry/laterality constraint, not a requirement to prove every high-level posture;
- before adding a new pose class, ask whether an existing ordinary pose is already a good proxy at the crop;
- semantic economy means collapsing measurements upward into useful human semantics, not deleting them.

## What the pose atlas shows

Each generated WebP card contains six panels:

1. **Original** — the actual training image.
2. **DWPose observed 2D** — the cached target-person 2D skeleton over the image.
3. **SAM3D projected fit** — cached SAM3D projected keypoints over the source image.
4. **SAM3D body-frame front** — a body-frame projection of the saved SAM3D mesh.
5. **SAM3D body-frame side/depth** — the same reconstruction viewed from the side/depth plane.
6. **Semantic/calibration summary** — SAM3D body/face orientation, DWPose extent, and optional human pose annotation.

The mesh views are diagnostic. Their purpose is to answer questions such as:

- Does SAM3D reconstruct the same broad pose family a human sees?
- Is a physically imperfect reconstruction still caption-equivalent because the error lies outside the crop?
- Are there recurring SAM3D failure modes for water occlusion, close crops, reclining bodies, crossed limbs, etc.?
- Which pose distinctions actually matter for captioning at each framing scale?

## No new VLM call

`pose_atlas_v3` uses cached evidence only. This is intentional.

The larger V3 plan is to make the expensive image-conditioned observation atomic (`Extract`) and iterate cheaply over Analyze, Pose, Gestalt, Fusion, and caption policy.

The atlas is the first step because it lets us understand the geometry before designing new semantic heuristics around assumptions.

## Initial pose ontology

See:

```text
semantic_v3/pose_ontology_v0_1.json
```

It intentionally begins with a small compositional set:

- standing
- seated
- reclining
- lying
- crouching/squatting
- kneeling
- walking/stepping
- bending/leaning
- unknown

Orientation, head/body relationships, and common actions are represented as modifiers rather than creating dozens of brittle compound classes.

The ontology is a calibration seed, not a final taxonomy.

## Human annotations

A lightweight example is provided at:

```text
semantic_v3/pose_atlas_annotations.example.json
```

The human target should record only what is useful for semantic calibration:

- broad pose family;
- a small number of modifiers;
- whether exact hidden-body state matters at the crop;
- one short `human_gestalt` phrase describing the visual thrust.

This is explicitly **not** joint-level ground truth.

## Running the atlas

The normal run layout is expected to contain an `images/` directory and cached DWPose/SAM3D outputs. Directory names are auto-discovered where possible; explicit paths are supported when a run uses a different layout.

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation

QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_pose_atlas_v3_workspace.sh \
  /workspace/qwen3/qwen3-vl-captioning-validation/runs/Caption02-02 \
  --overwrite
```

If the caches are not in the auto-discovered locations:

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_pose_atlas_v3_workspace.sh \
  /workspace/qwen3/qwen3-vl-captioning-validation/runs/Caption02-02 \
  --dwpose-dir /path/to/dwpose \
  --sam3d-dir /path/to/sam3d \
  --overwrite
```

To render a small calibration subset:

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_pose_atlas_v3_workspace.sh \
  /workspace/qwen3/qwen3-vl-captioning-validation/runs/Caption02-02 \
  --only sH1Vx_00101 \
  --only sH1Vx_00143 \
  --only sH1Vx_00144 \
  --only sH1Vx_00147 \
  --overwrite
```

Output defaults to:

```text
<run>/semantic-v3/pose-atlas-v0.1/
```

including:

```text
index.html
pose_atlas.index.json
<image-key>.pose_atlas.webp
<image-key>.pose_atlas.json
```

## What to review first

Do not score the reconstruction on invisible-joint perfection.

For each image ask, in order:

1. What broad pose does a human naturally call this?
2. Does SAM3D land in the same broad semantic family?
3. If not, is the difference actually visible in the crop?
4. Does DWPose expose a visible contradiction, or merely lack the cropped anatomy?
5. Which orientation/action modifiers materially define the image?
6. Is an existing base pose already an adequate crop proxy?
7. Only if not, is a new recurring pose heuristic/class needed?

## Next V3 steps after atlas review

Once the 20–30 image calibration set has been inspected:

1. record human pose-family/modifier targets;
2. derive empirical SAM3D reliability notes by crop/pose type;
3. audit current Analyze v2.1 as the raw material for an atomic high-recall `Extract` record;
4. ensure the single Extract contains enough scene/object/framing evidence for both Analyze and Gestalt to reason without a second vision pass;
5. implement crop-aware Pose reasoning using Analyze + DWPose + SAM3D;
6. make Gestalt primarily a relational/compositional reasoning stage over Extract rather than a fresh image re-description;
7. build deterministic Semantic Fusion V3 around support/compatible/unresolved/contradict evidence roles;
8. review the fused semantic graph before reconnecting Compose.

The central V3 development rule is:

> **Failures should improve the semantic ontology or reveal a crop-equivalent proxy, not automatically create another gate.**
