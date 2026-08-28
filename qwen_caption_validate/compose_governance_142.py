from __future__ import annotations

from . import compose_fusion_compare as _base
from .caption_projection_142 import build_caption_projection, lint_caption

_BASE_RENDER_FUSION_PROMPT = _base._render_fusion_prompt

_GOVERNANCE_ADDENDUM = r"""

PROJECTION 1.4.2 POSE-CONSISTENCY GOVERNANCE
- `pose_orientation.upper_torso_depth_relation`, when present, is the governing camera-relative upper-torso orientation. Describe the torso/body as strongly turned in depth or near side-on. Do not reconstruct a frontal/square-on torso from weaker Analyze semantics that Projection has removed.
- `pose_orientation.head_torso_relation`, when present, is RELATIVE pose information: the head is turned substantially toward the camera relative to the depth-turned torso. Say this explicitly. Do not reduce it to ambiguous wording such as "head faces forward".
- Gaze toward the camera and head orientation are separate: it is fine to say the subject turns the head toward the camera and looks toward/at the camera when both are supported.
- Body-to-body contact/support absent from the governed evidence has been deliberately vetoed. Do not reconstruct contact from ordinary body layout or proximity.
- A visible hand does not imply contact with a nearby thigh, torso, table, or other body/object unless governed contact/support explicitly remains.
- Continue Projection 1.4.1 accessory-state preservation and Projection 1.4.0 semantic compression. Use natural pose language, not a geometry checklist.
"""


def _render_fusion_prompt_142(template: str, caption_evidence: dict, subject_token: str, detail: str) -> str:
    return _BASE_RENDER_FUSION_PROMPT(template, caption_evidence, subject_token, detail) + _GOVERNANCE_ADDENDUM


def main() -> int:
    _base.build_caption_projection = build_caption_projection
    _base.lint_caption = lint_caption
    _base._render_fusion_prompt = _render_fusion_prompt_142
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
