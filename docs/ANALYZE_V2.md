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

V8.1 selection is deliberately **not** changed by this first Analyze-v2 pass. New camera, body-axis, ownership, and scene-complexity fields are report-only until they survive regression review. This prevents a new semantic field from changing portfolio decisions merely because it exists.

## What changed in Analyze v2

### 1. Fragment-before-whole body-part observations

Analyze v1 could still semantically complete a visible fragment into an unseen limb. Analyze v2 explicitly distinguishes:

- fingers
- hand
- wrist
- forearm
- upper arm
- shoulder

A visible pair of fingers is not automatically a hand/arm observation. Each body-part observation records:

- anatomical side
- ownership (`target`, `other`, `unknown`)
- visibility (`full`, `partial`, `fragment`)
- visible subparts
- connectivity to the target limb chain
- contact
- support
- foreshortening
- frame-relative location
- confidence

Contact, ownership, and support are separate claims.

### 2. Torso geometry split into independent axes

Analyze v2 separates:

- torso yaw
- torso pitch / forward-recline relationship
- torso roll / lateral lean
- image-plane body axis
- head yaw
- head pitch
- head roll
- shoulder depth

This is intended to prevent strongly reclined or diagonally posed subjects from being flattened into a single `torso_yaw` label.

### 3. Camera elevation requires evidence

`high`, `eye_level`, and `low` now require an explicit evidence list. Head pitch, visible scalp, ceiling, or generic selfie perspective are not sufficient by themselves.

Fusion v2 keeps camera elevation **report-only** even at high confidence until the regression set shows the axis is trustworthy.

### 4. Structural/specular background complexity

Analyze v1 mainly exposed `visual_complexity`. Analyze v2 separates:

- texture complexity
- structural complexity
- reflective/specular burden
- repeated geometry
- strong lines/angles
- reflection presence

This allows a smooth elevator wall to be low-texture while still being structurally/specularly difficult.

### 5. First-class scene and illumination evidence

Analyze v2 records broad environment and lighting observations directly rather than requiring downstream text-cue heuristics to reconstruct them.

## DWPose hand association change

`pose_evidence.py` now associates a detected hand to a visible target wrist using the **hand root (whole-body hand keypoint 0)** rather than the centroid of all visible hand keypoints.

The old centroid distance remains diagnostic only. Extended fingers can move the centroid far from the wrist and produce a false ownership warning even when the hand is physically connected.

DWPose hand evidence schema is now:

```text
dwpose-caption-evidence-1.1
```

## Fusion v2 authority rules

Fusion v2 is audit-first.

- Raw Analyze-v2 observations are preserved.
- DWPose projected 2D geometry is appended independently.
- A target-owned hand/finger claim without visible-chain or hand-root/wrist support is downgraded for selection authority.
- An isolated finger fragment cannot establish an unseen hand/arm chain.
- Camera elevation is report-only.
- Structural/specular scene burden is report-only.
- DWPose torso-axis cant is reported separately from semantic torso yaw/recline.

This is intentionally conservative. The next selection integration should consume only axes that pass regression review.

## Blind Validation 01 regression cases

The following images should remain a permanent regression suite.

| image | regression question | Analyze-v2 expectation |
|---|---|---|
| `jQTv_512x512_00018.png` | Two visible fingers touch/cross the neck but palm/wrist/arm are absent. Does the model invent a target-owned hand/arm? | Record `fingers` fragment; ownership unresolved unless visible evidence establishes it; no target-owned support action. |
| `jQTv_720x1280_00019.png` | Cup hand has plausible wrist/forearm continuity. Does deterministic hand association reject a real connection because fingers extend away from wrist? | Hand-root/wrist association should support the compatible target arm chain; centroid must not veto it. |
| `jQTv_720x1280_00002.png` | Subtle but real camera-above-subject view. | Camera elevation should be `high` with explicit view-down/floor-plane evidence, or conservatively `unknown`; must not confidently report `low`. |
| `jQTv_720x1280_00011.png` | Obvious overhead/high-angle POV with busy background. | High camera elevation recognized; recline/support geometry represented separately from background burden. |
| `jQTv_720x1280_00008.png` | Reclined/oblique body and partially hidden cup arm. | Distinguish yaw from recline/image-plane body axis; do not invent arm connectivity. |
| `jQTv_720x1280_00015.png` | Strong body orientation/lean amid elevator line geometry. | Preserve meaningful torso/body-axis evidence while separately reporting structural/specular background burden. |
| `jQTv_512x512_00015.png` | Smooth shiny elevator background. | Low texture is allowed, but reflective/specular and line/structural burden should not collapse to `low complexity`. |
| `jQTv_720x1280_00013.png` | Useful full-body evidence in extreme foliage entropy. | Full-body/limb evidence remains visible while large high-texture nuisance regions are explicitly captured. |

Normal controls from the same blind subject:

- `jQTv_512x512_00010.png` — ordinary close portrait should stay boring and defensible.
- `jQTv_720x1280_00001.png` — ordinary contextual portrait/cup interaction.
- `jQTv_512x512_00013.png` — clean incumbent full-body evidence.
- `jQTv_720x1280_00012.png` / `jQTv_512x512_00012.png` — same-source crop pair; useful later for lineage/redundancy work.

## Run Analyze v2

Analyze v2 uses a larger default generation budget because detailed full-body images caused 32B Analyze-v1 output to hit the old 1800-token ceiling.

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git pull

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

Override the budget with:

```bash
ANALYZE_V2_MAX_TOKENS=3600 bash ./run_analysis_v2_workspace.sh ...
```

Do not use `--compose` for the first regression run.

## Reuse existing DWPose evidence

If the images are byte-for-byte the same as Blind Validation 01, the existing DWPose cache can be reused. Fusion v2 accepts the old DWPose output directory directly.

```bash
bash ./run_fusion_v2_workspace.sh \
  runs/blind-validation-01-v2 \
  --model 32b-fp8 \
  --dwpose-dir runs/blind-validation-01/dwpose
```

Outputs are written to:

```text
runs/blind-validation-01-v2/fusion-v2/<model-slug>/
```

with one `*.fused_v2.json` per valid Analyze-v2 image plus `fusion_v2.index.json`.

## Acceptance criteria before selection integration

Do not change V8.1 portfolio weights merely to make the regression set look good. First require:

1. Ordinary control images remain straightforward and schema-valid.
2. `00018` no longer acquires a selection-authoritative target hand/arm through semantic completion.
3. `00019` is not falsely rejected by hand-centroid geometry.
4. `00002` is no longer confidently inverted to a low camera angle.
5. `00011` remains recognized as high/overhead.
6. `00008` and `00015` expose body/recline/image-plane geometry without forcing it into torso yaw.
7. Elevator/specular cases distinguish low texture from structural/specular burden.
8. Full-body images complete without output truncation.

Only after those checks should a later dataset-selection version consider camera POV, body-axis/recline, or structural/specular burden as qualified portfolio evidence.
