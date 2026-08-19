# Dataset selection verification

The dataset profiler should be verified against human replacement judgements, not only inspected in isolation.

The useful test case is a pair or small group where a human has already decided that one image is a better training asset than another while preserving some relevant coverage. This is especially valuable for full-body images, difficult facial poses, unusual contact/action poses, and low-signal/high-background-burden scenes.

Use `schemas/selection_verification_v1.schema.json` to record cases. A case names an incumbent image, a challenger image, the coverage that should be preserved, the human judgement, and the profiler effects we expect to observe.

Example:

```json
{
  "schema_version": "selection-verification-1.0",
  "cases": [
    {
      "id": "cleaner-full-body-replacement",
      "incumbent": "old_busy_full_body.png",
      "challenger": "new_clean_full_body.png",
      "coverage_to_preserve": ["full_body"],
      "human_expectation": "challenger_better",
      "human_reason": "The challenger preserves full-body evidence with substantially higher subject signal density and less irrelevant scene structure.",
      "expected_profiler_effects": [
        "challenger_higher_measured_signal_density",
        "challenger_lower_deterministic_background_burden",
        "challenger_selected_as_representative",
        "incumbent_loses_representative_status",
        "preserve_same_composition_coverage"
      ]
    }
  ]
}
```

## Why v5 uses representative quotas

A rare label should not automatically protect every image carrying it. Dataset Evidence v5 defines trusted coverage buckets with a desired representative quota, then ranks candidate images within each bucket primarily by the model-independent measured signal-density proxy. If a cleaner challenger is added to a bucket that is already full, the weakest incumbent can lose protected representative status automatically.

That behavior gives us a direct verification target: known-better images should displace known-worse images when they supply the same useful evidence.

## Verify across analyzer sources

V5 also makes source-calibration testable. Run the same challenger/incumbent dataset through cached 8B and 32B analyses while keeping DWPose and deterministic pixel evidence fixed.

A strong verification result is not that every semantic judgement is identical. It is that conclusions based on deterministic evidence remain stable while weaker analyzers become broader or abstain on unreliable semantic/spatial axes.

Examples:

- Full-body extent should remain stable when it is supported by DWPose.
- A quarantined 8B head-yaw/head-pitch axis should not protect or reject images based on those labels.
- A 32B semantic action signature may protect a representative when the axis is allowed and the evidence is sufficiently confident.
- The same full-body challenger should outrank a noisier incumbent on measured signal density regardless of analyzer size.

## Dynamic axis quarantine

V5 quarantines VLM spatial axes whose dataset-level distribution is too degenerate to distinguish genuine dataset bias from analyzer collapse. A quarantined axis is excluded from coverage buckets, representative protection, and highest-value-addition advice.

This means a weak analyzer can still support strong conclusions from DWPose/pixel evidence without producing confidently wrong facial-pose coverage charts.

## Coarse-first action semantics

Action/contact verification should prefer broad classes when the precise semantics are visually ambiguous. For example, `hands_on_hips`, `hands_in_front_pockets`, and `hands_in_rear_pockets` can all map to the broad coverage class `hands_near_hips` unless the source analysis supports a more exact distinction confidently.

V5 also uses word-boundary matching so scene terms such as `headrest` cannot become anatomical `head` contact, and unsupported-hand pose evidence reduces confidence in hand/contact semantics.

For medium/low-tier analyzers, semantic action classes are report-only and cannot preserve a weak image by themselves. High-tier analyzers may use sufficiently confident coarse/compound action coverage for representative selection.

## Measured signal density versus semantic burden

V5 separates the provisional model-independent core from VLM semantic scene interpretation:

- `measured_signal_density`: DWPose extent/rectangle geometry + deterministic background texture + DWPose secondary-person evidence.
- `semantic_burden`: VLM nuisance/confound observations, qualified by analyzer authority.

Keep/replace action currently uses the measured core. Semantic burden remains secondary evidence until it is better calibrated. An omitted VLM nuisance region is treated as `not reported`, never as proof that the scene is clean.

The measured core is still provisional because a DWPose rectangle is not a subject matte. Subject masks and effective face/subject pixels should eventually replace the rectangle proxy.

## Planned automated verifier

Once challenger images are added to a profiled verification dataset, the next step is a small verifier that joins the case manifest with `dataset_evidence_v5_*.json` and reports pass/fail for the expected profiler effects. The schema exists now so human judgements can be captured before tuning the heuristic against them.
