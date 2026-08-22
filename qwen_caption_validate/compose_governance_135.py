from __future__ import annotations

from . import compose_fusion_compare as _base
from .caption_projection_135 import build_caption_projection, lint_caption

_BASE_RENDER_FUSION_PROMPT = _base._render_fusion_prompt

_GOVERNANCE_ADDENDUM = r"""

PROJECTION 1.3.5 GOVERNANCE ADDENDUM
- Every item in required_claims is a must-cover visual fact. Express each one naturally in the caption.
- Qualified anatomical laterality licenses only the named body-part side. It does NOT license new signed depth geometry.
- When a semantic body-part record has been corrected from one anatomical side to the other, do not carry side-bound relations such as closer/nearer, forward/retracted, or in-front/behind across that correction unless the relation is independently present in governed evidence.
- For required claim signed_shoulder_nearer_relation, explicitly say that the named anatomical shoulder is closer/nearer to the camera than the opposite shoulder. Do not weaken this to generic shoulder staggering.
- For required claim signed_torso_depth_direction, explicitly describe the torso as angled/turned in depth or not square-on to the camera. Do not call the torso frontal, square-on, or straight-on when this claim is present.
- A named anatomical shoulder being nearer does NOT authorize saying the person turns toward image-left/image-right. Do not invent a frame-facing direction.
- If an interaction actor has unknown anatomical side, never infer the complementary side from a sided target. Say "the other arm", use an unsigned relation, or omit the side.
- Preserve qualified hand contact/support semantics when present. For a support_relation_* required claim, explicitly preserve the support relationship rather than weakening it to mere proximity or contact.
"""


def _render_fusion_prompt_135(template: str, caption_evidence: dict, subject_token: str, detail: str) -> str:
    return _BASE_RENDER_FUSION_PROMPT(template, caption_evidence, subject_token, detail) + _GOVERNANCE_ADDENDUM


def main() -> int:
    _base.build_caption_projection = build_caption_projection
    _base.lint_caption = lint_caption
    _base._render_fusion_prompt = _render_fusion_prompt_135
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
