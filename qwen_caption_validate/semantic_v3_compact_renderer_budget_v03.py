from __future__ import annotations

import re
from typing import Any

from . import semantic_v3_compact_renderer_budget_v01 as budget_v01
from . import semantic_v3_compact_renderer_budget_v02 as budget_v02


ARTIFACT_VERSION = "semantic-v3-compact-renderer-budget-0.3"
RUN_VERSION = "semantic-v3-compact-renderer-budget-0.3-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-budget-v0.3"

_TERMINAL_SENTENCE_RE = re.compile(r"[.!?][\"')\]]*$")


def _renderer_finish_reason(renderer_record: dict[str, Any]) -> str:
    return str((renderer_record.get("performance") or {}).get("finish_reason") or "").strip().lower()


def _caption_ends_complete_sentence(text: str) -> bool:
    return bool(_TERMINAL_SENTENCE_RE.search(text.strip()))


def budget_renderer_record(
    *,
    renderer_record: dict[str, Any],
    pose: dict[str, Any],
) -> dict[str, Any]:
    result = budget_v02.budget_renderer_record(
        renderer_record=renderer_record,
        pose=pose,
    )

    rerender_warnings: list[str] = []
    if _renderer_finish_reason(renderer_record) == "length":
        rerender_warnings.append("source_renderer_truncated")

    budgeted = str(result.get("budgeted_caption") or "").strip()
    if budgeted and not _caption_ends_complete_sentence(budgeted):
        rerender_warnings.append("budgeted_caption_incomplete_sentence")

    result["renderer_finish_reason"] = _renderer_finish_reason(renderer_record) or None
    result["renderer_rerender_warnings"] = rerender_warnings
    result["renderer_rerender_required"] = bool(rerender_warnings)

    if rerender_warnings:
        existing = [str(value) for value in (result.get("remaining_hard_warnings") or [])]
        result["remaining_hard_warnings"] = list(dict.fromkeys(existing + rerender_warnings))
        # A truncated source is not a semantic-repair problem. The source renderer must
        # first be rerun with a larger generation ceiling, then budgeting can be trusted.
        result["semantic_repair_required"] = False
        result["status"] = "renderer_rerender_required"

    return result


def main() -> int:
    # v0.3 preserves v0.2 selection exactly and adds only source-completeness gating.
    budget_v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    budget_v01.RUN_VERSION = RUN_VERSION
    budget_v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    budget_v01._sentence_categories = budget_v02._sentence_categories
    budget_v01._sentence_score = budget_v02._sentence_score
    budget_v01._select_budget_sentences = budget_v02._select_budget_sentences
    budget_v01.budget_renderer_record = budget_renderer_record
    return budget_v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
