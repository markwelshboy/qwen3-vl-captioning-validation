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
        "challenger_higher_measured_signal_density",
        "challenger_selected_or_ranked_ahead_within_full_body",
        "preserve_same_or_better_composition_coverage",
        "do_not_reduce_useful_view_diversity"
      ]
    }
  ]
}
```

## Representative quotas and guidance floors

Dataset Evidence v4 introduced coverage buckets and representative quotas. V6 adds a separate versioned guidance policy with broad composition bands, minimum counts, soft caps, and diversity-aware set selection.

This changes an important verification assumption: a known-better challenger does **not** always imply that the weakest incumbent must immediately lose protection.

For example, if the active guidance profile says a 20-image identity dataset should retain at least two full-body examples, a poor profile full-body image may remain strategically protected beside a clean frontal full-body image. Adding a third clean frontal image should not necessarily displace the poor profile image if doing so would reduce useful view diversity.

The correct v6 expectation can instead be:

- challenger ranks above a weaker same-view incumbent,
- the protected set preserves or improves composition count,
- the protected set preserves or improves view/action diversity,
- a low-quality but strategically unique incumbent records **quality debt** and remains `retain_until_guidance_equivalent`,
- the weak incumbent loses protection only when a cleaner challenger supplies equivalent-or-better coverage/diversity.

This is closer to marginal dataset value than simple pairwise image quality.

## Three verification dimensions

A v6 verification case can test three distinct claims:

1. **Measured quality** — does the independently preferred image have stronger model-independent signal-density evidence?
2. **Coverage policy** — does the optimizer preserve the active guidance floor rather than removing a needed composition class?
3. **Diversity-aware selection** — among enough candidates to satisfy the floor, does the selected set prefer complementary evidence rather than merely the highest individual scores?

A failure in one dimension should not be hidden by success in another.

## Coarse-first action semantics

Action/contact verification should prefer broad classes when the precise semantics are visually ambiguous. For example, `hands_on_hips`, `hands_in_front_pockets`, and `hands_in_rear_pockets` can all map to the broad coverage class `hands_near_hips` unless the source analysis supports a more exact distinction confidently.

This prevents false semantic precision from making a poor image look uniquely valuable.

## Source qualification

The same challenger set should be profiled through multiple cached analyzers when possible. Deterministic/DWPose measurements should remain stable. Analyzer-dependent diversity dimensions may be qualified or quarantined.

A weaker analyzer is allowed to say that facial-view diversity is unassessed; it should not manufacture a precise replacement judgement from an axis that failed calibration.

## Planned automated verifier

Once challenger images are added to a profiled verification dataset, a small verifier should join the case manifest with `dataset_evidence_v6_*.json` and report pass/fail separately for measured quality, guidance-floor preservation, diversity preservation, and expected representative ranking.

The schema exists now so human judgements can be captured before tuning the heuristic against them.
