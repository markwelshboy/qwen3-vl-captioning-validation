# Dataset Evidence v7 — acquisition and context diversity

V7 adds a planning layer on top of the source-calibrated evidence and v6 guidance policy.

The core idea is that a new image should ideally pay several kinds of dataset debt at once. Instead of separately saying that the dataset needs one full-body image, a profile view, an upward head pitch, and more capture-context diversity, v7 can turn those into one acquisition brief when the evidence supports doing so.

## Scene and lighting diversity

A balanced identity dataset should not accidentally teach identity as inseparable from one capture context. Broad indoor/enclosed versus outdoor coverage is therefore treated as a guidance objective, alongside composition and pose diversity.

V7 derives a conservative `capture_context` from cached Analyze-v1 text:

- `environment`: `indoor_enclosed`, `outdoor`, `mixed_or_transitional`, or `unknown`
- `illumination`: broad classes such as `outdoor_daylight`, `indoor_artificial`, `mixed_window_light`, `low_light`, plus lower-confidence unspecified/unknown classes

Environment cues can be fairly reliable when the cached analysis explicitly says things such as `outdoors`, `park`, `airplane cabin`, or `indoors`.

Lighting is deliberately lower-authority. Analyze v1 did not request structured illumination observations, so v7 only counts illumination above a relatively high confidence floor. If too many images are unassessed, the report emits a **lighting-analysis gap** instead of inventing a precise light-diversity deficit.

The intended Analyze-v2 upgrade is to make illumination first-class, for example:

- environment: indoor / outdoor / enclosed vehicle / mixed-transition / unknown
- dominant light source: daylight / artificial / mixed / unknown
- directionality: frontal / side / back / diffuse / mixed / unknown
- contrast: low / medium / high / unknown
- color-temperature impression: warm / neutral / cool / mixed / unknown
- exposure/visibility caveats

Those fields should be calibrated by analyzer source just like spatial axes.

## Context guidance policy

The default identity-LoRA profile now asks for broad anti-homogeneity floors rather than exact environment percentages.

The initial experimental policy requires some confidently assessable indoor/enclosed and outdoor examples, with broad minimum floors. These remain `heuristic_default` guidance, not benchmark truth.

Unassessed context is handled explicitly. If a weaker analyzer leaves enough images unresolved that they could plausibly satisfy a floor, v7 reports the category as `partially_unassessed` rather than claiming a deficit.

## Acquisition targets

V7 builds multi-debt acquisition briefs from:

- composition coverage debt from v6
- qualified global head-yaw/head-pitch gaps
- within-composition diversity debt
- confidently established environment debt
- confidently established lighting diversity needs

A high-value target may therefore look conceptually like:

> clean full-body + strong-left/profile facial view + upward head pitch

rather than three independent recommendations.

Context goals are added only when evidence is sufficiently authoritative. If lighting is mostly unassessed, the planner does not force a specific lighting class merely because the policy lists one.

## Donor/removal cost

V6 already identified surplus-class swap candidates. V7 ranks those donors by estimated removal cost.

Removal cost begins with model-independent measured signal density, then adds penalties if removing an image would threaten scarce context evidence. This means a visually ordinary close-up may be less expendable if it is the only confidently observed example of a rare capture context.

Context evidence is secondary in v7: it can change acquisition preference or donor order, but it cannot override trusted v5/v6 composition/geometry protection.

## Running

```bash
bash ./run_dataset_evidence_workspace.sh \
  runs/analysis-v1-nf4 \
  --model 32b \
  --output-prefix dataset_evidence_v7_32b
```

Then compare a weaker cached analyzer without any new Qwen inference:

```bash
bash ./run_dataset_evidence_workspace.sh \
  runs/analysis-v1-nf4 \
  --model 8b \
  --output-prefix dataset_evidence_v7_8b
```

You can also post-process an existing v6 JSON directly:

```bash
bash ./run_dataset_evidence_workspace.sh \
  runs/analysis-v1-nf4 \
  --model 32b \
  --base-v6-json runs/analysis-v1-nf4/dataset_evidence_v6_32b_Qwen__Qwen3-VL-32B-Instruct.json \
  --output-prefix dataset_evidence_v7_32b
```

## Challenger-image verification

The next validation stage is to add independently selected better images and ask whether the planner behaves as expected without tuning against those results.

Useful assertions include:

- a cleaner challenger enters the appropriate composition representative set;
- a challenger that adds missing yaw/pitch/context diversity outranks a cleaner but redundant candidate;
- a low-quality incumbent stays protected when it still supplies unreplaced coverage;
- once an equivalent or better challenger replaces that evidence, the incumbent loses protection and rises in donor/replacement priority;
- the same broad acquisition direction survives weaker analyzer sources, while unsupported fine-grained goals become unassessed rather than confidently wrong.
