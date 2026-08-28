from __future__ import annotations

from . import compose_fusion_compare as _base
from .caption_projection_140 import build_caption_projection, lint_caption

_BASE_RENDER_FUSION_PROMPT = _base._render_fusion_prompt

_GOVERNANCE_ADDENDUM = r"""

PROJECTION 1.4.0 SEMANTIC-COMPRESSION GOVERNANCE
- Treat required claims as semantic obligations, NOT as a checklist of clauses. If one natural phrase satisfies several overlapping evidence records, state the concept once and do not restate its supporting chain.
- Prefer the smallest natural description that preserves the important pose. For example, when a visible hand supports the chin/head and the forearm merely mediates that same support chain, describe the hand/chin pose once; do not separately say that the forearm supports the head via the hand.
- Signed depth claims subsume weaker unsigned component measurements when Projection has removed those weaker claims. Do not reconstruct or enumerate removed shoulder/pelvis/torso component claims from ordinary human expectations.
- `environment_lighting.preferred_scene_entities` contains distinctive high-confidence scene objects that should normally outrank generic floor, wall, panel, texture, blur, or lighting-surface inventory when choosing what background detail to verbalize.
- `environment_lighting.important_background_or_nuisance_regions` is OPTIONAL supporting context unless a corresponding item appears in `required_scene_claims`. Do not turn those regions into an exhaustive inventory.
- A broad `required_scene_claims` gestalt such as "park setting" or "kitchen setting" may semantically subsume supporting trees, foliage, cabinetry, appliances, etc. Do not enumerate every supporting observation merely to prove coverage.
- Preserve meaningful non-target objects when they distinguish the image or prevent identity bleed: bags, luggage, carts, boxes, furniture, vehicles, held objects, and similar concrete entities are generally more useful than generic surfaces.
- Use natural sentences. Do not compress unrelated facts into one long comma-separated evidence dump.
- The nominal length ranges are guidance, not targets. Semantic compression outranks word count; a balanced caption around 60-110 words is entirely acceptable when the evidence is already covered.

SAFETY BOUNDARIES RETAINED FROM 1.3.5
- Qualified anatomical laterality licenses only the named body-part side. It does NOT license new signed depth geometry.
- When a semantic body-part record has been corrected from one anatomical side to the other, do not carry side-bound relations such as closer/nearer, forward/retracted, or in-front/behind across that correction unless independently present in governed evidence.
- For required claim `signed_shoulder_nearer_relation`, explicitly preserve which qualified shoulder is closer/nearer to the camera.
- For required claim `signed_torso_depth_direction`, describe the torso as angled/turned in depth or not square-on; do not call it frontal/square-on/straight-on.
- A nearer anatomical shoulder does NOT authorize image-left/image-right turn direction.
- If an interaction actor has unknown anatomical side, never infer the complementary side from a sided target.
- Preserve each remaining required local support relation in meaning, but state it only once at the most direct visible actor.
"""


def _render_fusion_prompt_140(template: str, caption_evidence: dict, subject_token: str, detail: str) -> str:
    return _BASE_RENDER_FUSION_PROMPT(template, caption_evidence, subject_token, detail) + _GOVERNANCE_ADDENDUM


def main() -> int:
    _base.build_caption_projection = build_caption_projection
    _base.lint_caption = lint_caption
    _base._render_fusion_prompt = _render_fusion_prompt_140
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
