# Caption Fact Sheet Normalizer v0.1 — Phase 4A

Phase 4A is a **deterministic, zero-model-call** merge of three already-produced artifacts:

1. Phase-1 perception policy
2. Phase-2 routed body acquisition
3. Phase-3 routed gestalt acquisition

It does **not** run framing, DWPose laterality normalization, SAM3D torso normalization, py-feat head pose, L2CS gaze, or the final text composer. Its only job is to turn the two Qwen acquisitions into one typed record with explicit provenance and authority boundaries.

## Why this exists

The Phase-3 gestalt call is intentionally natural and therefore leaks across domains. For example it may say `lying down`, `looking downward`, `standing barefoot`, or `smartwatch on left wrist` even when those domains are owned elsewhere.

Phase 4A does not try to prompt that language away and does not regex-delete it from the source. Instead it preserves the wording while preventing it from silently becoming authoritative.

Key rule:

> Source wording does not determine authority. Domain ownership determines authority.

## Authority model

- `broad_pose`: only the routed Phase-2 pose candidate may populate it, and only for `pose_allowed` / `pose_guided`. It remains `qwen_pose_hypothesis` until Phase 4B.
- `configuration`: routed Phase-2 body relationships are preserved as `route_scoped_candidate`, with `pending_phase_4b` status.
- `appearance`, `objects`, `scene`, `secondary_people`: preserved from the gestalt Qwen as `qwen_semantic_candidate`.
- any visual candidate containing `left` / `right`: held for DWPose laterality normalization rather than accepted unchanged.
- `expression_action`: kept under `context_only` because the field can mix expression, action, pose, gaze, and capture mechanics.
- holistic `gestalt`: kept under `context_only`; it can help later composition but cannot create pose/gaze/framing/head/laterality authority.
- framing, anatomical laterality, torso geometry, head pose, and gaze are explicitly reserved for Phase 4B.

The fact sheet always has `caption_ready: false` in this phase.

## Run the nine-image calibration set

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git pull --ff-only

RUN_DIR=/workspace/qwen3/qwen3-vl-captioning-validation/runs/SemanticV3-Blind-20260906-160740

KEYS=(
  imageblind-01_00001
  imageblind-01_00014
  imageblind-01_00020
  imageblind-01_00021
  imageblind-01_00023
  imageblind-01_00031
  imageblind-01_00049
  imageblind-01_00051
  imageblind-01_00064
)

bash ./run_fact_sheet_normalizer_v01_workspace.sh "$RUN_DIR" \
  --only "${KEYS[@]}" \
  --overwrite
```

Output:

```text
$RUN_DIR/semantic-v3/caption-fact-sheet-v0.1/
  <key>.fact_sheet.json
  caption_fact_sheet.index.json
```

## Compact review

```bash
OUT="$RUN_DIR/semantic-v3/caption-fact-sheet-v0.1/caption_fact_sheet.index.json"

jq -r '
  .records[] |
  [
    .image_key,
    .policy.mode,
    .status,
    (.facts.body.pose_candidate.text // "-"),
    ([.facts.body.configuration[]?.text] | join("; ")),
    ([.facts.visual.appearance[]?.text] | join("; ")),
    ([.context_only.expression_action[]?.text] | join("; ")),
    (.context_only.gestalt.text // "-"),
    ((.audit.violations // []) | join(","))
  ] | @tsv
' "$OUT"
```

To inspect authority/promotion status rather than only text:

```bash
jq -r '
  .records[] |
  "\n===== \(.image_key) | \(.policy.mode) =====\n" +
  "POSE: " + ((.facts.body.pose_candidate // null) | tostring) + "\n" +
  "CONFIG: " + ((.facts.body.configuration // []) | tostring) + "\n" +
  "VISUAL: " + ((.facts.visual // {}) | tostring) + "\n" +
  "CONTEXT ONLY: " + ((.context_only // {}) | tostring) + "\n" +
  "AUDIT: " + ((.audit // {}) | tostring)
' "$OUT"
```

## Calibration expectations

- `00021`, `00023`, `00051`: no broad pose or configuration facts can appear just because the gestalt prose contains pose-like wording.
- `00020`: `lying down` may remain visible in `context_only`, but `facts.body.pose_candidate` must remain null because the perception policy is `configuration`.
- `00049`: body configuration remains separate from its general smile/scene/secondary-person semantics.
- `00001`, `00064`: broad pose candidate is retained as a hypothesis because their routes allow it.
- Qwen-generated left/right wording in appearance/object facts must be held for laterality normalization rather than accepted unchanged.
- there should be no authority violations for the nine calibration images.

If this gate passes, Phase 4B can populate the reserved expert domains and decide which candidates are approved for the eventual text-only composer.
