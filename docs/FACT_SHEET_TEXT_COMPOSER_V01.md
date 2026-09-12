# Phase 5 — Fact-Sheet Text Composer v0.1

Phase 5 is the first final-prose gate after the frozen Phase-4B.1 authority sheet. It is deliberately text-only: the composer never receives or reopens the source image.

## Input

Default input directory:

`semantic-v3/caption-fact-sheet-v0.2`

Expected schema:

`caption-fact-sheet-0.2.1`

## Projection contract

The full diagnostic fact sheet is not sent to Qwen. A compact composer projection is built first:

- framing extent from the crop governor;
- broad pose only when the routed Qwen pose hypothesis exists;
- body configuration only from `composer_text` entries;
- SAM3D torso orientation only when `composer_eligible=true`;
- publishable head axes only, with no diagnostic/candidate magnitudes;
- publishable gaze only; `camera_relationship=uncertain` is omitted rather than interpreted;
- accepted appearance, objects, scene, and secondary-person semantics;
- a small filtered expression/action channel that excludes pose, body geometry, head, gaze, framing, and laterality language;
- sanitized holistic context for wording / coarse-subject / overall-scene guidance only.

Held facts, diagnostic-only facts, and unresolved review-conflict domains are absent from the composer projection.

## Caption policy

The prompt prefers semantic ordering rather than a mandatory sentence count:

1. meaningful pose / configuration / torso orientation when present;
2. framing plus useful head/gaze semantics;
3. clothing, hair, accessories, and visible appearance;
4. safe expression/action, objects, environment, secondary people, and overall gestalt.

The default `dense` length profile favors rich Krea2-style supervision but explicitly allows shorter captions for sparse close crops.

The composer accepts optional `--trigger-token` and `--subject-class` arguments. The nine-image calibration should initially run without either so composition quality can be judged independently of identity-caption naming policy.

## Post-generation audit

There is no automatic repair pass. The output is audited only for obvious authority regressions such as:

- broad posture words that were not authorized by a broad-pose fact;
- anatomical left/right claims;
- gaze language when no gaze is publishable;
- camera-facing gaze claims when the camera relationship was absent/uncertain;
- head orientation with no publishable head evidence;
- torso orientation reintroduced for an unresolved torso-orientation conflict;
- model/policy/mechanism language leaking into the caption.

A caption with audit violations is written as `needs_review`; it is not silently rewritten.

## Output

Default output directory:

`semantic-v3/text-composer-v0.1`

Each image produces:

- `<image_key>.caption.txt` — plain final caption;
- `<image_key>.composed.json` — caption, compact evidence projection, provenance, and audit;
- `text_composer.index.json` — run index.

## Workspace command

```bash
bash ./run_fact_sheet_text_composer_v01_workspace.sh "$RUN_DIR" \
  --only imageblind-01_00001 imageblind-01_00014 \
  --model 32b-fp8 \
  --backend vllm \
  --length dense \
  --overwrite
```

The architectural invariant is simple: **the image ends at acquisition; the final writer can only verbalize the governed truth surface.**
