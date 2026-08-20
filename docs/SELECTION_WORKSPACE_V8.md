# Dataset selection workspace v8

V8 replaces the experiment-only language of **incumbent vs challenger** with persistent dataset workspace state.

A prepared image or derivative can be:

- `included` — currently part of the active training dataset.
- `candidate` — available for what-if evaluation but not yet part of the active dataset.
- `excluded` — deliberately inactive.
- `superseded` — an older derivative retained for history/cache reuse but inactive.

The important rule is that **candidate images do not change the guidance denominator until they are selected**. If a 28-image active dataset has seven new candidates, guidance floors are still calculated for 28 images while those candidates audition for slots.

## Fizgig-Web flow

The intended product flow is:

```text
source images
    -> image prep / crops / derivatives
    -> Analyze + DWPose (cached per exact prepared image)
    -> Dataset Intelligence
         active included set
         + candidate pool
         + excluded/superseded history
    -> what-if selection / recrop / add source / remove
    -> recompute from cache + only analyze new derivatives
    -> final included set
    -> Compose captions for the selected target/model policy
```

Changing selection state does not require image inference. Creating a new crop does. Changing only a mask or caption policy should not invalidate the semantic Analyze cache.

## Harness usage

For the current 28 + 7 experiment, candidate state can be supplied without a manifest:

```bash
bash ./run_dataset_evidence_workspace.sh \
  runs/analysis-v1-nf4 \
  --model 32b \
  --output-prefix dataset_evidence_v8_32b_selection \
  --candidate-glob 'challenger_*.png' \
  --selection-mode preserve_size
```

This means:

- all non-matching records default to `included`;
- `challenger_*.png` are `candidate`;
- target size is the original included count (28);
- the output is a **proposal only** and does not mutate the dataset.

The 8B run is identical apart from `--model 8b`.

## Manifest

For persistent/UI-like state, use `schemas/selection_workspace_v1.schema.json`.

Example:

```json
{
  "schema_version": "dataset-selection-workspace-1.0",
  "default_state": "included",
  "selection_mode": "preserve_size",
  "entries": [
    {
      "path": "new_crop_0001.png",
      "state": "candidate",
      "source_id": "source_0042",
      "derivative_id": "crop_v2",
      "parent_derivative": "crop_v1"
    },
    {
      "path": "old_crop_0001.png",
      "state": "included",
      "source_id": "source_0042",
      "derivative_id": "crop_v1"
    }
  ]
}
```

Then run:

```bash
bash ./run_dataset_evidence_workspace.sh \
  runs/analysis-v1-nf4 \
  --model 32b \
  --selection-manifest selection.json \
  --selection-mode preserve_size
```

## Selection modes

### `preserve_size`

Candidates compete with included images for the original active-dataset slot count. This is the right mode for the current validation experiment and for a user who wants to rebalance without growing the dataset.

### `target_size`

Use an explicit desired final size:

```bash
--selection-mode target_size --target-size 32
```

### `flexible`

Included images are not automatically removed. Candidates can be recommended as additive when they pay an active composition debt or add sufficiently strong novel evidence.

## Portfolio selection

V8 first attempts to satisfy the lower guidance floor for each composition class at the **requested final size**. It then fills remaining slots by measured quality plus source-qualified novelty while discouraging avoidable class surplus.

A small stability bonus favors an already-included image when a candidate is only trivially better; candidates should earn churn rather than cause it for noise-level score differences.

The output distinguishes:

- proposed candidate additions;
- proposed incumbent removals/exclusions;
- rejected candidates;
- suggested add/remove pairings;
- active vs proposed composition profile.

## Analyzer authority stability

V7 exposed a brittle behavior: adding a few candidates changed 32B torso output from 27/28 frontal to 33/35 frontal, crossing the old dataset-degeneracy threshold and accidentally making torso yaw look trustworthy again.

V8 anchors axis-health calibration to the **initial active dataset**, not the candidate pool. Candidate audition therefore cannot silently rehabilitate an analyzer axis merely by changing the denominator.

This is still a harness approximation. Longer term, authority should come from model+prompt calibration on golden cases, with dataset-level degeneracy used as a warning rather than the main source of truth.

## Action hierarchy

V8 begins separating broad and fine action semantics:

```text
broad: hands_near_hips
fine:  hands_in_pockets
fine:  hands_on_hips
```

Fine action is only allowed to influence selection when confidence is high. A selection manifest may carry a manual `overrides.fine_action` correction without rewriting the cached Analyze result.

This is transitional. Analyze v2 should emit structured action/contact fields directly.

## DWPose line-angle sanity

Shoulder/hip line orientation is undirected. V8 normalizes raw angles into `[-90, 90)` before computing cant magnitude.

Extremely large normalized shoulder cants (>45 degrees in this harness) are flagged for geometry sanity review and are **not** allowed to create novelty/protection until reviewed. This catches cases such as the challenger image whose raw shoulder angle was about -125 degrees.

## Non-destructive design

V8 never applies the proposed portfolio. Fizgig-Web should present the result as a what-if working tree. The user can accept/reject selection changes, create alternate crops, add sources, or reactivate an excluded derivative. Cached image evidence remains reusable throughout.
