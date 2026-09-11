# Caption Perception Policy v0.1

This is **Phase 1 only** of the crop-aware caption refiner. It is a deterministic routing gate between cached pose evidence and any future Qwen prompt routing.

It does **not** call Qwen, compose captions, merge specialist fragments, or rewrite any existing caption.

## Modes

- `framing_only` — the crop does not expose enough body geometry for broad pose wording and local body configuration is not distinctive enough to matter.
- `configuration` — broad posture is withheld, but visible local relationships (for example head/shoulder/arm configuration) are worth describing.
- `pose_allowed` — observed DWPose crop evidence supports broad pose wording.
- `pose_guided` — broad pose is already supported by observed DWPose evidence, and non-routine observed 2D geometry makes SAM3D relational guidance useful.

## Hard invariant

SAM3D is a reconstructed whole-body hypothesis, not proof that hidden anatomy is visible in the photograph.

Therefore:

> SAM3D availability, confidence, or projected hidden-body completeness can never make `broad_pose_supported` true.

Only DWPose landmarks that are both reported and inside the source photograph contribute to observed visibility. SAM3D is consulted only after the observed crop has independently cleared the broad-pose gate.

This is deliberately conservative. A tight crop with a highly confident full-body SAM3D reconstruction remains `framing_only` or `configuration` unless the source crop itself exposes the lower-body evidence needed for a broad pose claim.

## Output

Each image gets:

```text
semantic-v3/caption-perception-policy-v0.1/<key>.perception_policy.json
```

The record contains region visibility, observed DWPose landmarks and crop extent, deterministic configuration/pose-complexity cues, diagnostic SAM3D projected-inside/outside-frame counts, `pose_relevance`, `policy.mode`, and human-readable reasons.

The run also writes `caption_perception_policy.index.json` with mode counts and the routing invariants.

## Calibration gate

Run only the small calibration set first:

```bash
RUN_DIR=/workspace/qwen3/qwen3-vl-captioning-validation/runs/<RUN>

bash ./run_caption_perception_policy_workspace.sh "$RUN_DIR" \
  --only imageblind-01_00021 \
  --only imageblind-01_00049 \
  --only imageblind-01_00001 \
  --only imageblind-01_00014 \
  --only imageblind-01_00023 \
  --overwrite
```

Inspect compactly:

```bash
jq -r '.records[] | [.image_key,.mode,.pose_relevance,.visibility.extent_hint] | @tsv' \
  "$RUN_DIR/semantic-v3/caption-perception-policy-v0.1/caption_perception_policy.index.json"
```

Expected semantic targets for the first gate:

- `imageblind-01_00021` → `framing_only`
- `imageblind-01_00049` → `configuration`
- `imageblind-01_00001` (tree / non-routine full-body control) → `pose_guided`
- `imageblind-01_00014` (ordinary visible standing control) → `pose_allowed`
- `imageblind-01_00023` (head/shoulder crop control) → `framing_only`

Do not tune Phase 2 prompts until these routes are defensible on the actual cached data.
