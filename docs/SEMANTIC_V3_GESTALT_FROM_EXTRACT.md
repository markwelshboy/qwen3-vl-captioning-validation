# Semantic V3 Gestalt from Extract

## Purpose

V3 Gestalt is a text-only interpretation stage over the immutable canonical `visual-extract-3.0` record. It must never inspect the image. This preserves the V3 architecture:

> Observe once. Reason many times.

The image-conditioned observation step is Extract. Analyze and Gestalt are sibling reasoning branches over the same semantic evidence family. They are not independent votes.

## Input contract

The runner consumes `<image>.extract.json` wrappers produced by Extract v3 and reads only the canonical `.extract` object.

A deterministic projection exposes composition-relevant fields:

- framing
- target visible-body evidence
- Extract-reported landmark visibility
- orientation cues
- gaze
- interactions
- entities and relations
- environment/background regions
- composition observations
- posture/orientation/camera/capture/support hypotheses as **candidates**
- uncertainties

The projection deliberately excludes:

- the source image
- `raw_response`
- wire-format output
- `image_overview`
- transient appearance/identity detail
- Extract normalization internals

The output wrapper records the SHA-256 of the complete canonical Extract record and stores the exact evidence projection used for the Gestalt call.

## Output contract

The semantic output remains `composition-gestalt-1.4`. V3 changes the evidence source and provenance, not the accumulated Gestalt vocabulary.

The wrapper records:

- source Extract path and canonical SHA-256
- source Extract/wire versions and normalization report
- exact projected evidence
- raw text-only model response
- parsed model output
- governed Gestalt output
- schema validity
- governance audit
- semantic-family authority note

## Authority rules

Gestalt interprets composition; it does not replace Pose geometry or Fusion.

- Extract hypotheses are candidates, not ground truth.
- Missing evidence is not negative evidence.
- Broad posture and exact support are separate.
- Camera elevation is not inferred from gaze/face placement alone.
- Selfie capture is not inferred from direct gaze/close crop alone.
- Torso orientation is judged independently of posture and face direction.
- Exact angles and anatomical laterality are not Gestalt authority.
- Ambiguous human fragments remain ownership-conservative.
- Support is not promoted to external scene support unless its ownership and evidence status justify it.
- No object, surface, scene region, or body part absent from Extract may be invented.

The governance pass intentionally performs representation/authority cleanup only (side-neutral body prose and support eligibility audit). It does not silently repair semantic mistakes; those remain visible for calibration and later Fusion policy.

## Initial calibration controls

Start with:

- `jQTv_512x512_00018`
- `seated_head_no_hips`

The first gate is not whether Gestalt agrees with Pose. It is whether Gestalt responsibly interprets only the information actually transported by Extract, especially camera/capture/support/crop fields.

After the two-image gate is clean, run the remaining Extract calibration controls before freezing the Gestalt-from-Extract contract and proceeding to Analyze-from-Extract and Fusion v3.
