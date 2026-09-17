# Framing shadow v0.6 calibration

This shadow revision keeps canonical anatomical span as crop truth and changes only optional photographic shot-scale vocabulary.

## Evidence sources

- DWPose coherent anatomical tiers: literal observed body-span evidence.
- UniFace target-bound RetinaFace bbox: observed face-scale geometry.
- Easy-DWPose YOLOX person detector: observed person bbox already used internally before RTMPose. In the current production profiler this bbox is not persisted, so v0.6 reruns detector-only inference in shadow and binds the selected box to the cached DWPose target skeleton by keypoint-bbox containment.

SAM3D reconstructed geometry is not used for framing occupancy.

## Visual calibration findings

The reviewed `head->shoulders` cases showed that absolute face height overstates tightness for arm's-length/selfie-like compositions. The ratio of observed face height to visible target-person detector height separated the reviewed examples more coherently:

- roughly 0.28-0.35: visually medium / broader arm's-length framing
- 0.393: visually medium close-up

v0.6 therefore uses an abstention gap rather than a knife-edge threshold:

- `head->shoulders`, ratio <= 0.36: `medium`
- `head->shoulders`, 0.36 < ratio < 0.38: withhold standard scale
- `head->shoulders`, ratio >= 0.38: `medium_close_up`

For reviewed `head->head` cases, ratios from 0.587 through 0.709 all still read as ordinary close-ups. v0.6 therefore emits `close_up` for ratio >= 0.55 and disables `extreme_close_up` pending positive visual calibration examples.

These are validation-set shadow bands, not universal photographic definitions.

## Production implication if validated

Do not keep a second YOLOX pass in the final pipeline. Persist the detector bbox already produced by Easy-DWPose and expose it as target-bound observed person geometry. The framing specialist should then consume cached anatomical span, UniFace face geometry, and cached person detector geometry.
