# Routed Fragment Probe v0.1

This is **Phase 2** of the crop-aware caption-refiner architecture. It consumes the deterministic Phase-1 perception-policy records and routes each image to the smallest visual-semantic task justified by the crop.

It does not compose a final caption, apply head/gaze/laterality experts, or rewrite the existing Fizgig caption.

## Routes

| Phase-1 mode | Phase-2 action |
|---|---|
| `framing_only` | **No pose VLM call.** Emits an empty pose extraction. Framing/head/gaze can dominate later. |
| `configuration` | Image-only Qwen call for directly visible local `BODY RELATIONSHIPS`. Broad posture nouns are forbidden. |
| `pose_allowed` | Image-only Qwen call using the frozen v0.8 `POSE CANDIDATE` + visible-only `BODY RELATIONSHIPS` contract. |
| `pose_guided` | Same image call as `pose_allowed`, with a tiny **structured-text** SAM3D torso hint appended. No pose card or raw reconstruction is shown. |

## SAM3D firewall

SAM3D still cannot promote observability. Phase 1 must already have routed the image to `pose_guided` from observed DWPose crop evidence before Phase 2 will consult SAM3D.

Only these observation-gated torso fields may reach Qwen:

- `torso_camera_orientation`
- `torso_yaw_magnitude_deg`

Head/face geometry, limbs, reconstructed coordinates, support/contact, camera elevation, and the raw SAM3D card/mesh are excluded. The prompt explicitly says these torso values may refine torso-orientation wording only and may not justify posture, limbs, contact, head, gaze, laterality, or hidden anatomy.

## Structural enforcement

The route contract is enforced after generation as well as in the prompt:

- a `POSE CANDIDATE` emitted on the `configuration` route is dropped and recorded as a route violation;
- `framing_only` has no model call at all;
- `POSE CANDIDATE` is stored with `authority: hypothesis`;
- `BODY RELATIONSHIPS` are stored with `authority: visible_candidate`;
- at most five body relationships enter the normalized extraction; the raw response is preserved for audit.

## Calibration run

```bash
cd /workspace/qwen3/qwen3-vl-captioning-validation
git pull --ff-only

RUN_DIR=/workspace/qwen3/qwen3-vl-captioning-validation/runs/SemanticV3-Blind-20260906-160740
IMAGES=/workspace/qwen3/images

KEYS=(
  imageblind-01_00001
  imageblind-01_00014
  imageblind-01_00020
  imageblind-01_00021
  imageblind-01_00023
  imageblind-01_00031
  imageblind-01_00049
  imageblind-01_00051
  imageblind-01_00064
)

QWEN_VLLM_WORKSPACE_ROOT=/workspace/qwen3-vllm \
bash ./run_fragment_probe_routed_v01_workspace.sh "$RUN_DIR" \
  --images-dir "$IMAGES" \
  --only "${KEYS[@]}" \
  --model 32b-fp8 \
  --backend vllm \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-max-model-len 8192 \
  --max-tokens 300 \
  --overwrite
```

`00020`, `00031`, `00051`, and `00064` must already have Phase-1 policy records. If they do not, run Phase 1 for those four first.

The output is:

```text
$RUN_DIR/semantic-v3/fragment-probe-routed-v0.1/
  <key>.routed_fragments.json
  fragment_probe_routed.index.json
```

## Compact review

```bash
OUT="$RUN_DIR/semantic-v3/fragment-probe-routed-v0.1/fragment_probe_routed.index.json"

jq -r '
  .records[] |
  [
    .image_key,
    .policy_mode,
    .status,
    ((.extraction.pose_candidate.text // "-")),
    (([.extraction.body_relationships[]?.text] | join("; ")) // "-"),
    ((.sam3d_guidance.model_facts // {}) | tostring),
    ((.parse.route_violations // []) | join(","))
  ] | @tsv
' "$OUT"
```

## Calibration questions

The first review is deliberately narrow:

- `00021` and `00023` should make **no pose VLM call**.
- `00049` should describe only visible local configuration and must not acquire `seated`, `standing`, hidden legs, or support mechanics.
- `00001` should retain useful broad pose semantics and may use the whitelisted SAM3D torso orientation without gaining hidden-body details.
- `00014` should stay simple: useful broad pose, no SAM3D guidance.
- `00020` and `00064` should preserve the useful gestalt discovered during v0.8 testing while keeping relationship bullets crop-bound.
- `00031` is a contradiction trap for arm configuration.
- `00051` is a tight-crop hallucination trap.

Do not expand to all images until these route-specific fact sheets are defensible. The next layer after this gate is normalization against DWPose/v0.16 framing/laterality/torso evidence and the frozen head/gaze experts, not another free-form caption rewrite.
