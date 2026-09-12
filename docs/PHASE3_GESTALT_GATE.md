# Phase 3 gate

Pass if the ordinary visual-semantic branch remains useful while pose/body ownership stays outside this Qwen call.

Review priorities:

- `00021`: useful selfie/scene/appearance semantics without requiring standing/seated.
- `00023`, `00051`: useful tight-crop semantics despite no Phase-2 pose call.
- `00049`: scene/action/social content complementary to configuration facts.
- `00001`, `00064`: gestalt branch should not redundantly recreate detailed body pose.
- `00020`: incomplete interpretation is acceptable; do not tune architecture around this outlier.

Do not compose final captions at this phase. If this gate passes, next build one typed normalized fact sheet per image from independent acquisitions, then inspect it before text-only composition.
