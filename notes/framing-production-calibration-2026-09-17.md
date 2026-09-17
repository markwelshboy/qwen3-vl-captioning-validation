# Framing production calibration — 2026-09-17

This note records the validated framing/routing conclusions from the 87-image SemanticV3 blind run and the two visual boundary reviews.

## Canonical rule

Photographic shot scale is optional presentation language. Canonical crop truth is the coherent observed anatomical span derived from DWPose tiers. Reconstructed SAM3D anatomy cannot promote crop observability.

## Routing

Broad posture requires bilateral observed hips and bilateral observed knees. Failure of the broad-pose gate does not disable local configuration observation.

A directly observed `visible_arm_relationship` is sufficient to permit the narrow `configuration` route. That route may abstain with `NO RELIABLE POSE FACTS` and may not name standing/seated/crouching/squatting/kneeling/reclining/lying.

Validated controls:

- `00021`: configuration observer abstains cleanly.
- `00061`: configuration observer restores fist-under-chin/head-support semantics with no broad pose.
- `00085`: broad pose withheld; local cup/arm configuration retained; unsupported `seated` removed.
- `00087`: broad pose remains observable and `squatting` remains pose-guided.

## Standard shot scale

Observed face geometry comes from the target-bound UniFace RetinaFace bbox intersected with the source image.

Observed person geometry comes from the Easy-DWPose YOLOX person detector associated back to the cached target skeleton. The useful close-family statistic is:

`face_person_height_ratio = visible_face_height_fraction / visible_person_height_fraction`

Absolute YOLOX person occupancy is not used as a shot-scale threshold because the detector box often saturates at crop edges.

### `head -> shoulders`

Final conservative calibration bands:

- ratio `<= 0.36`: `medium`
- `0.36 < ratio < 0.38`: withhold standard scale
- `0.38 <= ratio <= 0.66`: `medium_close_up`
- `0.66 < ratio < 0.68`: withhold standard scale
- ratio `>= 0.68`: `close_up`

Evidence for the upper boundary: the visual review keeps `00081` (0.620), `00080` (0.651), and nearby chest/shoulder crops in the medium-close family, while `00069` (0.685) and tighter examples read as close-ups. The full census contains a natural gap from about 0.655 to 0.685.

### `head -> head`

- ratio `>= 0.55`: `close_up`
- otherwise: withhold pending more calibration

`extreme_close_up` is disabled until positive visual examples are available. Images `00027` and `00036`, previously promoted to ECU by face height alone, are visually ordinary close-ups.

### Other coherent spans

- `head -> hips`: `medium`
- `head -> knees`: `medium_wide`
- `head -> ankles`: `full_body` only when the existing pose-conditioned rule authorizes the conventional term; otherwise retain literal anatomical span.
- cropped-head spans such as `shoulders -> hips` and `shoulders -> knees`: withhold conventional shot scale and publish literal anatomical span.

## Composer rendering

When a standard shot scale is confidently available, use it as the natural framing phrase and keep the anatomical span as internal governed evidence rather than mechanically repeating it.

When standard shot scale is withheld, surface the literal anatomical span.

Examples:

- confident scale: `sH1VX is shown in a medium shot, ...`
- confident close family: `sH1VX is shown in a medium close-up, ...`
- no reliable standard scale: `sH1VX is framed from around the shoulders through the hips, ...`

This avoids awkward combinations such as `medium shot, framed from the head through the shoulders` when weaker partial lower-body evidence is also present.
