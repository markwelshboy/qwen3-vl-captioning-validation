# Dataset guidance profiles

Dataset Evidence v6 separates **what the evidence says** from **what kind of dataset we would prefer to build**.

The measured/source-calibrated profiler remains responsible for image facts and qualified observations. A guidance profile is a versioned policy layer used to reason about composition balance, diversity, protection, and replacement opportunities. Guidance values are deliberately broad and should never be presented as benchmark truth unless they have actually been calibrated empirically.

## Default profile

`guidance_profiles/identity_lora_balanced_v1.json` is the initial experimental profile for a balanced identity LoRA dataset.

Its broad composition bands are:

- identity close-up: 15–30%
- medium / upper-body: 55–75%
- full-length: 10–20%

The profile also defines minimum counts and soft caps. Minimum counts prevent small datasets from losing important categories merely because a percentage rounds down. Soft caps introduce diminishing returns so a very large dataset does not have to preserve an arbitrary percentage forever when additional examples add no meaningful new evidence.

These numbers are **heuristic defaults**. They are intentionally stored outside the profiler code so they can be revised, compared, overridden, or replaced with target-model-specific / user-specific / empirically calibrated guidance later.

## Three kinds of debt

V6 keeps three concepts separate:

### Coverage debt

The dataset has fewer examples in a broad composition class than the lower guidance floor.

Example: a 20-image dataset with a 10% full-body lower bound has a floor of two full-body images. If only one exists, the dataset has one unit of full-body coverage debt.

### Diversity debt

The broad count is adequate, but the protected representatives are too similar on assessable dimensions such as facial yaw, head pitch, action/contact signature, or deterministic geometry.

Two full-body frontal images can therefore satisfy the count while still having worse diversity than one frontal and one profile/non-frontal example.

If an analyzer axis has been quarantined, diversity on that axis is reported as unassessed rather than guessed.

### Quality debt

The required representative count is satisfied, but one or more protected representatives have low measured signal density.

This is the intended interpretation of a poor but strategically necessary image:

> The dataset currently needs the evidence supplied by this image; that does not make the image itself good. Retain it until a cleaner equivalent is available.

## Diversity-aware representative selection

V6 chooses a protected **set**, not merely the individually highest-scoring images.

Selection starts from model-independent measured quality, then rewards useful novelty among already-selected representatives. The default profile can reward a new head-yaw class, head-pitch class, semantic action signature (only when the analyzer is authoritative enough), or deterministic geometry class.

Consequently, a lower-quality profile full-body image can remain more valuable than a second very clean frontal full-body image if the profile view is otherwise absent.

## Guidance class authority

Full-body and clear waist/upper-body extent can be anchored strongly by DWPose. Splitting a DWPose `close_or_medium_close` extent into close-up versus medium-close may require qualified VLM framing evidence.

For low-tier analyzers, V6 can leave that split unresolved rather than inventing a precise percentage. Unresolved images are represented explicitly in the report and can make composition status partially unassessed.

## Replacement versus rebalance

A guidance surplus is not an instruction to delete images. V6 marks unprotected surplus images as **swap candidates**. When another class has coverage debt, the report can suggest exchanging redundant surplus evidence for a stronger image in the deficient class while keeping the total dataset size roughly fixed.

This is different from an image-level `replace_candidate`, which still depends primarily on low measured signal density plus lack of trusted/guidance protection.

## Running with the default profile

```bash
bash ./run_dataset_evidence_workspace.sh \
  runs/analysis-v1-nf4 \
  --model 32b \
  --output-prefix dataset_evidence_v6_32b
```

To use a custom policy:

```bash
bash ./run_dataset_evidence_workspace.sh \
  runs/analysis-v1-nf4 \
  --model 32b \
  --guidance-profile /path/to/my_guidance.json \
  --output-prefix dataset_evidence_custom
```

The v6 profiler regenerates the deterministic/source-calibrated v5 base internally. It does not load Qwen again; it reuses cached analysis and DWPose output.

## Verification implications

Adding a known-better challenger does not necessarily mean the weakest incumbent must immediately lose protection. If the active guidance floor or diversity requirement says the dataset still benefits from all of them, the correct result may be:

- retain the weak incumbent temporarily,
- record quality debt,
- prefer the challenger,
- and request a cleaner equivalent before retiring the weak image.

That behavior is intentional and should be part of selection-verification tests.
