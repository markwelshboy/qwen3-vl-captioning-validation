# Caption Refiner — SAM3D Pose-card VLM prototype

This is intentionally a **parallel refinement tool**, not another replacement caption pipeline.

The existing caption is read from the JSON artifact that currently produces the Fizgig caption. It is **not regenerated** here.

The only new VLM call is:

```text
cached SAM3D arrays + OBJ mesh
        ↓
pose-only visual card
        ├── full reconstructed body projected against the real photograph frame
        ├── 3D front/camera-plane mesh view
        └── 3D side/depth mesh view
        ↓
Qwen3-VL
        ↓
exactly two caption-ready sentences
        1. pose / body orientation / meaningful body relation
        2. framing / camera / head direction or gaze
```

The original photograph is shown to the **human reviewer** in the web UI, but is deliberately not shown to the pose VLM. That keeps the second model focused on pose/framing rather than re-captioning clothing, identity, scenery, etc.

Existing deterministic `pose-language-v0.1` can optionally be shown underneath the VLM result as a diagnostic reference. It does not automatically edit the caption.

## Why this prototype exists

The working hypothesis is that SAM3D/Pose is most useful as a **caption refinement reference**. When the normal caption says something vague such as `slightly angled`, the pose card may produce a more useful description such as `seated with the torso turned three-quarter to the camera, turning the head back toward the lens`. If the broad noun is wrong (`seated` vs `crouching`) but the crop makes the distinction minor, the reviewer can edit one word.

The web UI therefore makes the human the final merger.

## Output contract

The pose VLM is prompted to emit exactly two sentences:

1. broad pose/body orientation, meaningful limb configuration, head/body relationship;
2. framing/crop, camera relationship, head direction/gaze when useful.

It must not describe clothing, hair, identity, age, expression, scene, background, or lighting, and must not mention SAM3D/mesh/reconstruction/joints/angles.

The existing JSON-derived caption remains untouched until a reviewer chooses an edit.

## Build a small test first

From the prototype branch:

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git fetch origin
git switch agent/caption-refiner-pose-vlm
git pull --ff-only
```

Choose the existing run and the directory containing the **current caption JSON artifacts**:

```bash
RUN_DIR=/workspace/qwen3/qwen3-vl-captioning-validation/runs/<RUN>
CAPTION_DIR="$RUN_DIR/<CURRENT-CAPTION-JSON-DIR>"
POSE_DIR="$RUN_DIR/semantic-v3/pose-language-v0.1"
OUT_DIR="$RUN_DIR/caption-refiner-pose-vlm-v0.1"
```

If the caption lives at a known JSON path, pass it explicitly, for example `--caption-field caption` or `--caption-field result.caption`. If omitted, the prototype tries common caption fields and then searches caption-named string fields.

First build pose cards with **no model load**:

```bash
QWEN_VLLM_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_caption_refiner_pose_vlm_workspace.sh "$RUN_DIR" \
  --captions-dir "$CAPTION_DIR" \
  --pose-dir "$POSE_DIR" \
  --only <IMAGE_KEY_1> <IMAGE_KEY_2> <IMAGE_KEY_3> \
  --dry-run \
  --overwrite
```

Inspect:

```bash
jq '{record_count,generated_pose_refs,missing}' \
  "$OUT_DIR/caption_refiner.index.json"
```

The generated pose-only visuals are under:

```text
$OUT_DIR/pose-cards/
```

## Generate the two-sentence pose references

Run the same command without `--dry-run`:

```bash
QWEN_VLLM_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_caption_refiner_pose_vlm_workspace.sh "$RUN_DIR" \
  --captions-dir "$CAPTION_DIR" \
  --pose-dir "$POSE_DIR" \
  --only <IMAGE_KEY_1> <IMAGE_KEY_2> <IMAGE_KEY_3> \
  --model 32b-fp8 \
  --backend vllm \
  --batch-size 2 \
  --max-tokens 180 \
  --overwrite
```

The CLI prints each pose reference as it is generated. Each image also gets:

```text
<key>.pose_refiner.json
```

containing:

- source image;
- exact source caption JSON and selected caption field;
- unchanged existing caption;
- image-space left/right terms worth visually checking;
- pose-card path;
- optional existing deterministic Pose language;
- VLM pose text split into sentence 1 and sentence 2;
- `exactly_two_sentences` validation flag.

## Start the web refiner

```bash
QWEN_WORKSPACE_ROOT=/workspace/qwen3 \
bash ./run_caption_refiner_server.sh "$OUT_DIR" \
  --host 0.0.0.0 \
  --port 8767
```

The card shows:

```text
┌─────────────────────────────┬──────────────────────────────────────┐
│ original photograph         │ pose-only SAM3D card                │
│                             │ projection + frame / 3D front / side │
├─────────────────────────────┼──────────────────────────────────────┤
│ existing JSON caption       │ two-sentence pose/framing reference │
│ spatial words highlighted   │ deterministic Pose reference below  │
├─────────────────────────────┴──────────────────────────────────────┤
│ final editable caption                                             │
│ [replace first 2] [replace 1] [replace 2] [append] [restore]       │
└────────────────────────────────────────────────────────────────────┘
```

Spatial words such as `left`, `right`, `frame left`, `frame right`, `behind`, and `foreground` are highlighted for **human review only**. They are not automatically rewritten.

Saving writes:

```text
caption_refiner.edits.json
exported-captions/<image_key>.txt
```

## First experiment

Do **not** run all 87 initially. Pick 5–10 images spanning:

- obvious ordinary standing/frontal pose;
- strong three-quarter body turn;
- near-side-on body with head turning back;
- seated/crouched ambiguity whose lower body lies outside the crop;
- reclining/leaning case;
- meaningful hand-to-head or held-object pose.

The first question is simply:

> Does Qwen turn the SAM3D pose-card into natural language that is more useful than the loose pose wording in the existing caption?

If yes, then we can tune the two-sentence prompt and later make the **normal JSON→caption prompt** reliably put pose/framing into its first two sentences so replacement is mechanically clean.

If no, we have changed only one small experimental component and can discard it without disturbing the existing caption path.
