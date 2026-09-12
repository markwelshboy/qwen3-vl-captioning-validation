# Caption Fact Sheet Phase 4B.1

Phase 4B.1 is a narrow authority-contract refinement over Phase 4B. It makes no model calls and does not change acquisition routing.

It adds three guardrails:

1. **Crop-governed torso semantic bandwidth.** SAM3D torso geometry may remain available diagnostically, but `framing_only` never exposes it to the composer. Other routes expose torso geometry only when deterministic crop evidence marks the torso strongly observed.
2. **Direction-only head authority stays direction-only.** A head axis with `corroborated_direction` may publish its semantic direction/class, but not angular magnitude. Horizontal `yaw_strength` is also withheld in that case.
3. **Qwen torso-camera-orientation handoff.** Narrow camera-yaw phrases such as `torso angled toward camera` or `torso angled to the side` enter the SAM3D-owned torso-orientation domain. With strongly observed torso geometry, the specialist supersedes the Qwen candidate. Without strong torso observation, neither side is allowed to win automatically; both are retained diagnostically and an explicit review conflict is emitted.

The handoff intentionally does **not** capture in-plane/body-configuration phrases such as `torso bent forward` or `torso oriented horizontally relative to frame`.

Output schema: `caption-fact-sheet-0.2.1`. The output directory remains `semantic-v3/caption-fact-sheet-v0.2` so Phase 4B calibration can be overwritten in place before freezing.
