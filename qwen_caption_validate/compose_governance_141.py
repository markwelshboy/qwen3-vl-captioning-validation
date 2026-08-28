from __future__ import annotations

from . import compose_fusion_compare as _base
from .caption_projection_141 import build_caption_projection, lint_caption

_BASE_RENDER_FUSION_PROMPT = _base._render_fusion_prompt

_GOVERNANCE_ADDENDUM = r"""

PROJECTION 1.4.1 ACCESSORY-STATE GOVERNANCE
- Treat transient accessory state as meaningful image-specific supervision when explicitly present in `transient_appearance.descriptors`.
- Preserve state such as sunglasses perched on the head or a face mask lowered below the chin instead of flattening it to the bare object name.
- Do not invent anatomical left/right for accessories from Analyze summary text.
- Continue to apply Projection 1.4.0 semantic compression: preserve important concepts, merge overlapping evidence, prefer distinctive scene entities over generic surfaces, and avoid checklist prose.
"""


def _render_fusion_prompt_141(template: str, caption_evidence: dict, subject_token: str, detail: str) -> str:
    return _BASE_RENDER_FUSION_PROMPT(template, caption_evidence, subject_token, detail) + _GOVERNANCE_ADDENDUM


def main() -> int:
    _base.build_caption_projection = build_caption_projection
    _base.lint_caption = lint_caption
    _base._render_fusion_prompt = _render_fusion_prompt_141
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
