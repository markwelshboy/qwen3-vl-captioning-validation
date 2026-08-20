# Dataset selection workspace v8.1

V8.1 keeps the V8 workspace-state model (`included`, `candidate`, `excluded`, `superseded`) and changes the selection algorithm.

## Why V8.1 exists

The first V8 challenger run exposed an important policy error: the lower edge of a preferred percentage band was being treated like a hard quota. That could force a weak upper-body or full-body candidate into the proposed dataset simply because the active set was below the preferred composition share.

V8.1 separates two concepts:

- `minimum_count`: a hard safety floor. The optimizer should not make this debt worse.
- `preferred_share`: a soft portfolio objective. Moving toward the band is useful, but only by a bounded amount.

A weak candidate therefore cannot automatically defeat a substantially better incumbent just because it belongs to an underrepresented composition class.

## Preserve-size optimization

`preserve_size` starts from the current included dataset and evaluates candidate-for-selected-image swaps.

For every possible swap V8.1 computes a portfolio objective containing:

- measured signal-density quality;
- hard minimum protection;
- soft preferred-composition penalties;
- source-qualified head/action diversity;
- conservative environment/light diversity;
- a small candidate churn penalty.

The best swap is applied only when its gain exceeds `selection_optimizer.min_swap_gain`. The process repeats until no remaining candidate can produce enough positive gain.

This makes the question operationally close to the Fizgig-Web workflow:

> Does this new source/crop/derivative improve the active dataset enough to earn a slot, and if so which current image is cheapest to retire?

It does **not** rebuild the entire dataset from scratch.

## Evidence qualification

V8.1 also closes two authority leaks found in V8.

### Compound action signatures

The cached action signature is rebuilt from qualified components before it can contribute diversity. For example:

- `head_torso_counter_rotation` is dropped when torso yaw or head yaw is quarantined.
- `strong_shoulder_cant` is dropped when normalized DWPose shoulder geometry is in sanity review.
- high-tier semantic components need confidence >= 0.75.
- semantic action protection remains report-only below the high analyzer tier.

This prevents suspect geometry from re-entering selection through a compound semantic label after the dedicated geometry feature has already rejected it.

### Fine actions

Fine actions such as `hands_in_pockets` versus `hands_on_hips` remain visible in the report, but cached fine semantics can affect selection only when:

- analyzer tier is `high`; and
- confidence is >= 0.85.

A manual workspace override remains authoritative at every tier. Analyze v2 should eventually emit structured contact/action evidence directly.

## Guidance profile 1.2

The default identity-LoRA profile now contains `selection_optimizer` weights. The initial values are intentionally modest heuristic defaults:

- quality weight: 1.0
- hard-minimum penalty: 3.0 per missing image
- preferred-shortfall penalty: 0.14 per image
- preferred-surplus penalty: 0.06 per image
- head-yaw diversity: 0.10
- head-pitch diversity: 0.06
- action diversity: 0.08
- fine-action diversity: 0.05
- environment diversity: 0.04
- illumination diversity: 0.025
- candidate churn penalty: 0.04
- minimum accepted swap gain: 0.05

These weights are policy knobs, not empirical truths. The challenger tests are intended to calibrate them against independent human judgments.

## Current validation run

For the existing 28 included + 7 candidate experiment, no image inference is required. Reuse the V7 JSON:

```bash
bash ./run_dataset_evidence_workspace.sh \
  runs/analysis-v1-nf4 \
  --model 32b \
  --base-v7-json runs/analysis-v1-nf4/dataset_evidence_v7_32b_challengers_Qwen__Qwen3-VL-32B-Instruct.json \
  --output-prefix dataset_evidence_v8_1_32b_selection \
  --candidate-glob 'challenger_*.png' \
  --selection-mode preserve_size
```

Then repeat with `--model 8b` and its matching V7 JSON.

The key regression expectations are:

1. workspace remains 28 included + 7 candidate with target size 28;
2. torso remains quarantined for the 32B active set;
3. `challenger_00002` shoulder geometry remains a sanity review and its suspect cant/counter-rotation cannot create action novelty;
4. 8B fine-action semantics remain report-only;
5. weak candidates are no longer admitted solely to satisfy the preferred composition percentage;
6. strong candidates can still replace weak/redundant included images even when they are in an already-surplus class.
