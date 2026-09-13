# Gaze Caption Semantics v0.1

Phase 4B.3 adds a caption-facing semantic layer above the existing head/gaze specialist evidence.

The specialist record still answers whether gaze is observable and preserves the raw L2CS point estimate. The caption semantic layer asks a separate question: is that directional displacement large enough to deserve words in a training caption?

## Horizontal policy

- `|yaw| <= 15°`: `near_center`; retain the measurement, suppress lateral caption language.
- `15° < |yaw| < 25°`: `mild_lateral_uncorroborated`; retain diagnostically, suppress from composition while L2CS is the only directional authority.
- `|yaw| >= 25°`: `clear_lateral`; publish `frame_left` / `frame_right` to the composer.

These are caption-salience thresholds, not claims about gaze-estimation accuracy. They are intentionally conservative and should be calibrated against the representative image set rather than treated as physiological gaze bins.

Centered vertical gaze is likewise omitted as semantically uninteresting; an upstream `up` or `down` classification remains publishable. A resolved `toward_camera` or `off_camera` relationship remains independently publishable, while `uncertain` is omitted.

## Invariants

- Raw gaze measurements remain in `facts.gaze` unchanged.
- UniFace/L2CS observability and authority remain unchanged.
- Caption-facing decisions live under `facts.gaze.caption_semantics`.
- A high-observability face does not imply that every non-zero gaze estimate is caption-worthy.
- The text composer consumes only `caption_semantics`, never raw gaze direction fields.

The motivating calibration case is `imageblind-01_00051`: L2CS reports `yaw=+14.60°` with high observability, but the visible gaze is near-camera/near-center and does not warrant the categorical phrase `gaze directed toward frame left`. Under this layer, the raw `+14.60°` remains inspectable while horizontal caption semantics resolve to `near_center` and are withheld from composition.
