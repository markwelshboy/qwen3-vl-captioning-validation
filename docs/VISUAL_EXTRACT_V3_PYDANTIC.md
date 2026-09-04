# Visual Extract v3 — Pydantic-first structured methodology

This is the typed implementation of the V3 rule:

```text
Observe once. Reason many times.
```

There is exactly one image-conditioned VLM pass. Pydantic is the source of truth for both the compact VLM-facing wire contract and the descriptive persistent Extract.

## Data path

```text
IMAGE
  |
  v
Qwen VLM
  |
  | constrained by ExtractWireV1.model_json_schema(by_alias=True)
  v
vLLM structured outputs / xgrammar
  |
  v
ExtractWireV1.model_validate_json(raw)
  |
  | deterministic, semantic-free expansion
  v
VisualExtractV3 (Pydantic)
  |
  +--> legacy extract_v3.schema.json compatibility check
  +--> Analyze reconstructability audit
  +--> Gestalt reconstructability audit
  |
  +--> later text-only Analyze / Gestalt / Fusion
```

## Why two typed models

`ExtractWireV1` is optimized for generation. Python field names are descriptive while compact JSON aliases reduce repeated decode syntax.

`VisualExtractV3` is optimized for persistence and downstream reasoning. It keeps descriptive canonical field names expected by the rest of the V3 architecture.

The wire is not a second reasoning stage. Expansion cannot add image semantics. It only:

- expands aliases and entity references;
- assigns deterministic appearance ids;
- maps categorical confidence bands to fixed compatibility values;
- constructs and validates the canonical typed object.

Raw VLM output is retained for provenance.

## Confidence

The VLM emits categorical confidence rather than fake decimal precision:

```text
h -> 0.90
m -> 0.65
l -> 0.35
u -> 0.00
```

The numeric values exist only for compatibility with the current canonical contract. The original confidence band remains in the retained wire record.

## Schema authority

There is no hand-maintained VLM wire JSON Schema in the Pydantic execution path.

```python
wire_schema = ExtractWireV1.model_json_schema(by_alias=True)
```

The run index records a SHA-256 digest of that exact generated schema.

The canonical `VisualExtractV3` object is also checked against the older `schemas/extract_v3.schema.json` during migration. Pydantic is authoritative; the JSON Schema check is a regression/compatibility alarm.

## x3p2 semantic calibration

The initial x3p1 smoke proved the architecture but exposed interface-driven semantic errors. x3p2 keeps the architecture and makes targeted contract changes.

### Anatomy aliases are intentionally explicit

x3p1 used `lh` / `rh` for left/right hip. In calibration Qwen repeatedly populated those slots with hand/finger evidence. x3p2 spends a few extra characters:

```text
head
lshoulder / rshoulder
lhip / rhip
lknee / rknee
lankle / rankle
```

This is a deliberate trade: semantic safety is worth a handful of decode tokens.

### Dedicated semantic channels

The target subject is never duplicated in `entities`.

- `s.cl`: target clothing
- `s.ac`: worn/attached target accessories
- `s.mk`: visible target body markings/tattoos
- `s.ix`: direct holding/contact/reaching/crossing/gesture interactions
- `r`: scene/spatial/visibility relations only
- `h.sup`: support hypotheses only

Generic relations no longer offer wearing, holding, contact or support predicates. This prevents the model from expressing the same fact in several competing graph structures.

### Cross-record Pydantic invariants

Pydantic validation now rejects:

- duplicate entity ids;
- non-contiguous entity ids;
- dangling entity references;
- self-relations;
- explicit target-subject pseudo-entities;
- `full_body` + `face_dominant` framing;
- frontal torso hypotheses that also claim a left/right frame-facing direction.

Visual identity itself cannot be solved mechanically, so prompt/calibration review still guards against a non-target `person` entity that visually duplicates the target.

### Restored evidence rules

x3p2 restores lessons from Analyze 2.1:

- fingers do not become a hand without visible palm/wrist continuity;
- contact does not establish ownership;
- visible hands/thighs do not establish hip visibility;
- ordinary clothing does not make a represented shoulder/hip partial;
- broad seated posture may be contextually supported while hidden hips remain unobserved;
- direct gaze, centered framing, selfie appearance or visible scalp do not establish camera elevation;
- torso orientation is independent of face/head orientation;
- support targets must be locally/contextually implicated, not merely present somewhere in the background;
- ambiguous surfaces remain generic rather than being promoted to a table.

## Output budget

x3p1 calibration showed natural records from roughly 2.1k to 3.45k output tokens. The x3p2 runner therefore uses:

```text
max_tokens = 4000
```

as a safety ceiling, not a target. Easy images should still stop naturally well below it.

## Backend boundary

The observation contract is independent of model weight format. Future FP8 / INT8 / INT4-AWQ-Marlin comparisons should change the model/backend configuration, not the Extract schema, semantic prompt, expansion logic or regression criteria.

```text
model/backend implementation
        |
        v
same ExtractWireV1 schema hash
        |
        v
same VisualExtractV3 canonical record
```

## Experimental output

x3p2 writes separately from x3p1:

```text
RUN_DIR/extract-v3-pydantic.2/<model-slug>/
```

Each artifact retains raw output, alias-form wire data, descriptive wire data, Pydantic validation diagnostics, deterministic expansion metadata, canonical Extract, compatibility-schema diagnostics, reconstruction audits and performance data.

## Calibration gate

Mechanical acceptance requires:

```text
finish=stop
wire=ok
canonical=ok
analyze=ok
gestalt=ok
```

Semantic acceptance is stronger: selected regression images must demonstrate correct fragment/ownership behavior, crop-aware landmark provenance, scene/reflective Gestalt, accessory/marking persistence, contextual posture recovery and conservative support/camera hypotheses.
