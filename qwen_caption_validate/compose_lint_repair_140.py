from __future__ import annotations

from pathlib import Path
from typing import Any

from . import compose_lint_repair_135 as _base
from .caption_projection_140 import lint_caption


REPAIR_PROMPT_140 = """You are repairing an existing text-only identity-training caption against Projection 1.4.0 semantic-compression governance.

Return ONLY the revised caption. Do not explain the repair.

Rules:
- Use only facts present in CAPTION_EVIDENCE_JSON. Do not add or reconstruct visual facts.
- Preserve the trigger token as the grammatical opening.
- Resolve every listed lint violation/warning while preserving supported meaning.
- Make the MINIMUM semantic change needed. A repair must not expand the caption into an evidence checklist.
- Required claims are semantic obligations, not one-clause-per-field quotas. Merge overlapping pose/depth/support facts into one natural statement when possible.
- If a direct hand support relation already expresses a support chain, do not separately restate that a forearm supports the same target via the hand.
- Prefer distinctive `preferred_scene_entities` and meaningful scene gestalt over generic floor/wall/panel/texture inventory.
- If anatomical side is not qualified, rewrite side-neutrally or use safe bilateral wording only when the evidence explicitly supports both sides. Never infer the complementary anatomical side.
- Do not transfer side-bound geometry such as nearer/farther/forward/retracted when laterality is not explicitly qualified for that relation.
- Preserve signed-depth claims and remaining required local support relations exactly in meaning.
- Use natural sentences rather than one long comma-separated list.

CAPTION_EVIDENCE_JSON:
{{CAPTION_EVIDENCE_JSON}}

ORIGINAL_CAPTION:
{{ORIGINAL_CAPTION}}

LINT_FINDINGS_JSON:
{{LINT_FINDINGS_JSON}}
"""


_ORIGINAL_WRITE_JSON = _base._write_json


def _write_json_140(path: Path, value: dict[str, Any]) -> None:
    if path.name == "lint_repair.index.json":
        value = dict(value)
        value["governance_revision"] = "1.4.0"
    _ORIGINAL_WRITE_JSON(path, value)


def main() -> int:
    _base.lint_caption = lint_caption
    _base.REPAIR_PROMPT = REPAIR_PROMPT_140
    _base._write_json = _write_json_140
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
