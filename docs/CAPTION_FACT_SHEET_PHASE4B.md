# Caption fact sheet Phase 4B — specialist normalization

Phase 4B is a deterministic, no-model-call pass over the frozen Phase-4A fact sheet. It fills specialist-owned domains without allowing reconstruction or generic Qwen prose to create authority outside their contracts.

Inputs:

- `semantic-v3/caption-fact-sheet-v0.1`: Phase-4A typed Qwen candidates and provenance.
- `semantic-v3/caption-perception-policy-v0.1`: crop visibility/framing and source paths.
- `head-gaze-evidence-v03`: normalized UniFace + py-feat + L2CS head/gaze evidence.
- cached DWPose and SAM3D artifacts referenced by each perception-policy record.

Output defaults to `semantic-v3/caption-fact-sheet-v0.2` with schema `caption-fact-sheet-0.2`.

## Authority rules

**Framing** comes directly from the Phase-1 observed crop/visibility record. It cannot infer camera angle or hidden anatomy.

**Anatomical laterality** comes from DWPose named joint identity/visibility. Qwen `left`/`right` attached to anatomical terms is never swapped or trusted. Safe semantic wording is retained with the anatomical side removed, while DWPose joint-side visibility is stored separately. DWPose visibility alone does not prove that a watch, phone, or other semantic object belongs to a particular side.

**Torso geometry** reuses the Phase-2 SAM3D whitelist only: `torso_camera_orientation` and `torso_yaw_magnitude_deg`. Publication still requires the DWPose body-yaw observation gate. SAM3D reconstruction remains explicitly non-observational and cannot promote a broad-pose hypothesis.

**Head pose and eye gaze** consume `head-gaze-evidence-v03` rather than rebuilding its raw model fusion. Head yaw and pitch are promoted independently by axis authority. `corroborated` and `corroborated_direction` axes are publishable; `reduced` axes remain candidates only. Roll is diagnostic. Eye gaze is published only when v0.3 says the eyes are observable and the L2CS result is publishable. A sunglasses null therefore suppresses gaze without suppressing valid head direction.

Phase 4B does not compose a caption. `caption_ready` remains false.

## Calibration expectations

- `00021`: close crop; head centered/frontal remains valid; gaze resolves null because sunglasses obscure eye gaze.
- `00023`: head yaw to frame-right is publishable; reduced pitch remains unpromoted; gaze can still publish frame-right/down/off-camera when v0.3 grants high authority.
- `00049`: head and gaze publish frame-left/down independently of Qwen configuration wording; Qwen anatomical left/right is neutralized.
- `00001`, `00014`, `00064`: observation-gated SAM3D torso facts may be added without upgrading Qwen broad-pose hypotheses.
- `framing_only` images remain broad-pose/configuration empty.
