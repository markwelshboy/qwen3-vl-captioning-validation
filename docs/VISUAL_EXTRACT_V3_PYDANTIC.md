# Visual Extract v3 — Pydantic-first structured methodology

This is the experimental typed implementation of the V3 rule:

```text
Observe once. Reason many times.
```

It does not change the semantic architecture. There is still exactly one image-conditioned VLM pass. The change is the interface between that VLM pass and the persistent canonical Extract.

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

`ExtractWireV1` is optimized for generation. Python field names are descriptive, while short JSON aliases reduce repeated output syntax. For example the Python model exposes `category`, `descriptors`, `frame_location`, `visibility`, and `confidence`, while the VLM-facing JSON uses `c`, `d`, `l`, `v`, and `q`.

`VisualExtractV3` is optimized for persistence and downstream reasoning. It keeps the descriptive canonical field names expected by the rest of the V3 architecture.

The wire is not a second reasoning stage. Expansion cannot add image semantics. It only:

- expands short aliases and entity references;
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

The numeric values exist only for compatibility with the current canonical contract. The original confidence band remains present in the retained wire record.

## Schema authority

There is no hand-maintained VLM wire JSON Schema in the Pydantic execution path.

The decoder schema is generated directly at runtime:

```python
wire_schema = ExtractWireV1.model_json_schema(by_alias=True)
```

The run index records a SHA-256 digest of that exact generated schema.

The canonical `VisualExtractV3` Pydantic object is also checked against the existing `schemas/extract_v3.schema.json` during migration. This makes divergence visible while Pydantic becomes the typed source of truth.

## Backend boundary

The observation contract is intentionally independent of model weight format. A future FP8 / INT8 / INT4-AWQ-Marlin comparison should change the model/backend configuration, not the Extract schema, prompt semantics, expansion logic, or regression criteria.

That makes it possible to compare capability/speed tradeoffs against identical structured outputs:

```text
model/backend implementation
        |
        v
same ExtractWireV1 contract
        |
        v
same VisualExtractV3 canonical record
```

## Experimental output

The Pydantic path writes separately from the direct-canonical Extract:

```text
RUN_DIR/extract-v3-pydantic.1/<model-slug>/
```

Each artifact retains:

- `raw_response`;
- compact alias-form `wire_extract`;
- descriptive `wire_model_dump`;
- wire Pydantic validation errors, if any;
- deterministic expansion metadata;
- canonical `extract`;
- legacy canonical JSON-Schema errors, if any;
- Analyze/Gestalt structural audits;
- per-request and per-batch performance data.

## Initial acceptance gate

Before a full calibration run, the same two-image pair used for the direct-schema experiment should pass:

```text
finish=stop
wire=ok
canonical=ok
analyze=ok
gestalt=ok
```

The main performance comparison is generated tokens/image and amortized seconds/image. Semantic acceptance still requires inspection of the regression controls; a smaller record is not useful if it loses visual specificity or Gestalt evidence.
