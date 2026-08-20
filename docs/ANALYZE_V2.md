# Analyze v2 + Fusion v2

Analyze v2 is an intentionally stricter visual-observation schema for identity-LoRA dataset intelligence. It was introduced after Blind Validation 01 exposed several repeatable weaknesses in Analyze v1 while ordinary incumbent images remained broadly defensible.

The design principle remains:

```text
image
  -> Analyze (semantic observation)
  -> deterministic Fusion/Audit (DWPose + rules)
  -> fused evidence
  -> later Compose / dataset-selection policy
```

V8.1 selection is deliberately **not** changed by Analyze-v2 / Fusion-v2. New camera, body-axis, ownership, framing-reconciliation, and scene-complexity fields must survive regression review before they can affect portfolio selection.

## Analyze v2 observations

### Fragment-before-whole body parts

Analyze v2 explicitly distinguishes fingers, hand, wrist, forearm, upper arm, and shoulder. A visible pair of fingers is not automatically a hand/arm observation.

Each body-part observation records anatomical side, ownership, visibility, visible subparts, connectivity to the target chain, contact, support, foreshortening, frame-relative location, and confidence. Contact, ownership, and support are separate claims.

### Independent body geometry axes

Analyze v2 separates torso yaw, torso pitch/recline, torso roll/lateral lean, image-plane body axis, head yaw, head pitch, head roll, and shoulder depth. This avoids forcing every unusual pose into one generic "angled" label.

### Camera elevation requires evidence

`high`, `eye_level`, and `low` carry explicit evidence/counterevidence. Head pitch, scalp visibility, symmetrical framing, generic selfie perspective, and inferred eye/lens height do not establish camera elevation by themselves.

Fusion keeps camera elevation report-only even when the model is confident.

### Structural/specular scene complexity

Analyze v2 separates texture complexity, structural complexity, reflective/specular burden, repeated geometry, strong lines/angles, and reflection presence. A smooth elevator wall can therefore be low-texture but high structural/specular burden.

## DWPose hand association

`pose_evidence.py` associates a detected hand to a visible target wrist using the whole-body **hand root**, not the centroid of all visible hand keypoints.

The centroid remains diagnostic only. Extended fingers can move the centroid far from the wrist and previously caused false ownership warnings.

DWPose evidence schema:

```text
dwpose-caption-evidence-1.1
```

## Fusion v2.2 authority rules

Fusion v2.2 is audit-first and explicitly separates claims that were previously coupled.

### Ownership/action vs anatomical laterality

A strongly supported target hand interaction can remain valid even when Qwen and DWPose disagree about left/right.

```text
action: holding mug         -> may remain qualified
ownership: target           -> may remain qualified
anatomical side: right/left -> downgraded to unknown on conflict
```

This is important for ordinary cup/selfie images such as `jQTv_720x1280_00001.png` and `jQTv_720x1280_00008.png`. A hand-root association to a complete target arm chain can support target ownership/action without granting DWPose authority over true anatomical laterality.

For the adversarial `jQTv_512x512_00018.png`, there is no hand root / target wrist chain, so the isolated two-finger fragment remains unresolved and non-authoritative.

Mirror selfies remain laterality-sensitive: DWPose detector-side left/right is not used to validate true anatomical side through a reflection.

### Free-text summary is report-only

Analyze-v2 may produce correct structured evidence but still semantically complete a fragment in `image_summary` (for example, calling two isolated fingers "a hand"). Fusion v2.2 therefore exposes:

```text
report_only_image_summary
```

instead of treating the free-text summary as caption-authoritative evidence.

Future Compose-from-fusion should consume the qualified structured fields, not the report-only summary.

### Deterministic full-length framing reconciliation

If all of the following agree:

- DWPose extent is `full_length`,
- at least one complete leg chain is present,
- Analyze-v2 itself reports distal leg/foot/ankle/shoe evidence,

then Fusion may reconcile an internally inconsistent semantic `three_quarter`/`medium` framing to:

```text
framing_audit.qualified_shot_scale = full_length
```

This is intended for cases such as `jQTv_720x1280_00013.png`, where Qwen reports visible feet/shoes but labels the shot `three_quarter`.

### Camera governor remains conservative

Fusion classifies eye/lens-height inference, symmetrical framing, selfie convention, and "no strong view down/up" as weak/non-geometric evidence. These cues cannot make camera elevation authoritative.

Obvious overhead cases can still retain qualified geometric evidence, but camera POV remains report-only for selection.

### Projected 2-D body geometry remains distinct from 3-D orientation

DWPose torso-axis and shoulder-line measurements can flag a strong projected cant or conflict, but they do **not** establish torso yaw or recline.

Fusion v2.2 exposes:

- `strong_torso_axis_cant`
- `strong_shoulder_cant_only`
- `conflict`
- `review_required`

without converting those measurements into unsupported 3-D semantics.

This preserves the useful warning on cases such as `jQTv_720x1280_00008.png` and `jQTv_720x1280_00015.png` while acknowledging that a separate 3-D body-orientation estimator is the proper next experiment.

## Blind Validation 01 regression suite

| image | regression question | expected behavior |
|---|---|---|
| `jQTv_512x512_00018.png` | Isolated two-finger fragment against neck | ownership unresolved; no target hand/arm completion; interaction report-only |
| `jQTv_720x1280_00019.png` | Real mug hand with wrist/forearm continuity | hand-root supports target arm chain; action/ownership survive |
| `jQTv_720x1280_00002.png` | Subtle high camera POV | bogus eye-level cues fail Fusion camera qualification; camera remains report-only |
| `jQTv_720x1280_00011.png` | Obvious overhead POV | high/overhead evidence retained; recline and near-horizontal body axis remain distinct |
| `jQTv_720x1280_00008.png` | Reclined/oblique body + cup hand | target action may survive side conflict; projected geometry triggers review without inventing 3-D yaw |
| `jQTv_720x1280_00015.png` | Elevator pose with body cant/rotation | semantic/deterministic projected disagreement remains visible; structural/specular burden stays separate |
| `jQTv_512x512_00015.png` | Smooth reflective elevator | low texture + high structural/specular burden |
| `jQTv_720x1280_00013.png` | Full body in extreme foliage entropy | deterministic full-length reconciliation + explicit high-entropy nuisance evidence |

Normal controls:

- `jQTv_512x512_00010.png` — ordinary close portrait should stay boring.
- `jQTv_720x1280_00001.png` — ordinary cup interaction; side conflict must not erase a strongly supported target holding action.
- `jQTv_512x512_00013.png` — clean incumbent full-body evidence.
- `jQTv_720x1280_00012.png` / `jQTv_512x512_00012.png` — same-source crop pair for future lineage/redundancy work.

## Run Analyze v2

Analyze v2 uses a larger generation budget because detailed full-body images caused Analyze-v1 32B output to hit the old 1800-token ceiling.

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git pull

QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_analysis_v2_workspace.sh /data/jQTv \
  --models 32b-fp8 \
  --backend vllm \
  --quantization none \
  --run-name blind-validation-01-v2 \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-max-model-len 8192
```

The wrapper supplies:

```text
--analysis-prompt prompts/analysis_v2.txt
--schema schemas/analysis_v2.schema.json
--max-analysis-tokens 3000
```

## Re-run Fusion without VLM inference

Existing Analyze-v2 and DWPose caches can be re-fused after Fusion rule changes:

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_fusion_v2_workspace.sh \
  runs/blind-validation-01-v2-1 \
  --model 32b-fp8 \
  --dwpose-dir runs/blind-validation-01/dwpose \
  --overwrite
```

## Unit regressions

Fusion-v2.2 includes synthetic regression checks for the key authority rules:

```bash
/workspace/qwen3-vllm/.venv/bin/python -m unittest tests.test_fusion_v2 -v
```

These verify:

1. isolated unknown fingers remain non-authoritative;
2. target holding action survives a laterality conflict when hand-root + complete target arm chain support ownership;
3. deterministic full-length evidence reconciles a contradictory semantic framing label;
4. eye-level claims based only on inferred eye/lens height are not qualified.

## Before selection integration

Do not tune V8.1 weights against this regression set. Before any new Fusion-v2 axis affects portfolio scoring, require that ordinary controls remain stable and the corresponding adversarial cases behave defensibly.

The unresolved semantic gap is now primarily **3-D body orientation/recline**. DWPose supplies useful projected 2-D geometry, but a separate 3-D human-pose/body-orientation experiment is the next appropriate source of evidence rather than further prompt tuning.