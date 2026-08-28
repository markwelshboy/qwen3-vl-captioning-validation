from __future__ import annotations

from . import compose_fusion_compare as _base
from .caption_projection_143 import build_caption_projection, lint_caption

_BASE_RENDER_FUSION_PROMPT = _base._render_fusion_prompt

_GOVERNANCE_ADDENDUM = r"""

PROJECTION 1.4.3 SEMANTIC-SALIENCE GOVERNANCE
- `framing_camera.framing.normalized_shot_scale` and `normalized_extent_description` are the governing framing description. State that crop/framing once and do not repeat a conflicting source shot label.
- When `pose_orientation.whole_body_posture.allowed` contains `standing`, say that the subject is standing even when the feet are cropped. Do not replace it with vague wording such as "positioned".
- High-confidence interactions represented by `required_claims` are more semantically useful than generic limb geometry. Preserve them in natural language, for example a hand resting on a hip or hands holding an object.
- If `pose_orientation.gesture_semantics` contains a chin-rest gesture, describe the recognizable gesture as the chin resting on the curled/closed hand (or fist only when explicitly supported). Do not serialize it as finger geometry under the chin.
- A generic high-confidence indoor/outdoor scene claim is modest but still useful when no richer setting is qualified. Do not upgrade `indoor` to restaurant, bar, theater, etc. without governed evidence.
- Continue Projection 1.4.2 pose-consistency and contact authority. Prefer one clear semantic pose sentence over a checklist of body-part fields.
"""


def _render_fusion_prompt_143(template: str, caption_evidence: dict, subject_token: str, detail: str) -> str:
    return _BASE_RENDER_FUSION_PROMPT(template, caption_evidence, subject_token, detail) + _GOVERNANCE_ADDENDUM


def main() -> int:
    _base.build_caption_projection = build_caption_projection
    _base.lint_caption = lint_caption
    _base._render_fusion_prompt = _render_fusion_prompt_143
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
