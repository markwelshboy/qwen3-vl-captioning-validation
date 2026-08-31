# Semantic Fusion V3 pose discovery

This pass expands Pose Atlas calibration from a few hand-picked examples to a dataset-wide geometry census. It remains report-only.

## Policy

The geometry layer reports mechanical evidence such as projected standing/sitting, arm flexion, hand proximity, open/closed hand shape, wrist height and current validated composites such as `hands_on_hips` and `head_supported_by_fist`.

It does **not** emit action semantics such as `waving`, `holding`, or `resting on a table`. Those belong to later Fusion/Caption, which can combine geometry with VLM scene/action evidence.

`head_supported_by_fist` support is capped by the weakest required observed component: the broad head/hand relation and the observed fist support.

## End-to-end discovery run

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation-v3
git pull

RUN_DIR=/workspace/qwen3/qwen3-vl-captioning-validation/runs/Caption02-02

SAM3D_WORKSPACE_ROOT=/workspace/sam3d-body \
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_pose_discovery_workspace.sh "$RUN_DIR" --tar
```

Defaults:

- all images in `$RUN_DIR/images`;
- existing DWPose cache in `$RUN_DIR/dwpose`;
- DWPose target bbox with 20% padding;
- SAM3D caches in `$RUN_DIR/sam3d-pose-discovery-01`;
- OBJ meshes disabled for the broad pass;
- relational profiles in `relational-pose-profile-v0.3`;
- pose-library census nested under the profile directory.

Use `--save-mesh` only if full-dataset OBJ retention is desired. Use `--overwrite` to regenerate existing SAM3D cache records.

## Outputs

The relational profile adds `discovery_primitives`, including per-side elbow flexion, hand-anchor proximity, observed hand shape/source/support, wrist height relative to shoulder, image-space wrist offset, and bilateral hand/wrist distances normalized by shoulder width.

The census writes:

```text
pose-library-census/
  pose_library_census.json
  pose_library_census.md
  review_candidates.json
  review_keys.txt
  atlas_only_args.txt
```

The report includes projected-pose counts, current named-relation counts, observed hand-shape counts, repeated mechanical signatures, frequent geometry review buckets, and a bounded list of images selected for visual atlas review.

Review-bucket names are not accepted caption vocabulary. They are only a way to identify recurring geometry before deciding whether an additional deterministic relation deserves to be added.

With `--tar`, the runner prints the path to a tar containing the relational profile directory and the default census output.
