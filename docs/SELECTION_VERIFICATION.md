# Dataset selection verification

The dataset profiler should be verified against human replacement judgements, not only inspected in isolation.

The useful test case is a pair or small group where a human has already decided that one image is a better training asset than another while preserving some relevant coverage. This is especially valuable for full-body images, difficult facial poses, unusual contact/action poses, and low-SNR scenes.

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
        "challenger_higher_active_snr",
        "challenger_lower_background_entropy",
        "challenger_selected_as_representative",
        "incumbent_loses_representative_status",
        "preserve_same_composition_coverage"
      ]
    }
  ]
}
```

## Why v4 uses representative quotas

A rare label should not automatically protect every image carrying it. Dataset Evidence v4 defines coverage buckets with a desired representative quota, then ranks candidate images within each bucket by useful-signal quality. If a cleaner challenger is added to a bucket that is already full, the weakest incumbent can lose protected representative status automatically.

That behavior gives us a direct verification target: known-better images should displace known-worse images when they supply the same useful evidence.

## Coarse-first action semantics

Action/contact verification should prefer broad classes when the precise semantics are visually ambiguous. For example, `hands_on_hips`, `hands_in_front_pockets`, and `hands_in_rear_pockets` can all map to the broad coverage class `hands_near_hips` unless the source analysis supports a more exact distinction confidently.

This prevents false semantic precision from making a poor image look uniquely valuable.

## Planned automated verifier

Once challenger images are added to a profiled verification dataset, the next step is a small verifier that joins the case manifest with `dataset_evidence_v4_*.json` and reports pass/fail for the expected profiler effects. The schema is intentionally created now so the human judgements can be captured before tuning the heuristic against them.
