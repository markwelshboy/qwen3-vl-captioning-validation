from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_repair as repair_v01
from . import semantic_v3_rich_pose_repair_v05 as repair_v05


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_repair_v06.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-repair-0.6"
RUN_VERSION = "semantic-v3-rich-pose-repair-0.6-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-repair-v0.6"

_BASE_QUALITY_AUDIT = repair_v05.quality_audit
_BASE_FAILURE_LINES = repair_v01._failure_lines

# Blind v0.7 left two semantically-clean but awkward residues after removing
# protected haircut structure: "hair styled in a cut" / "hair styled in a cut featuring".
# This is a local repair-quality issue, not a reason to reopen the universal editor.
_AWKWARD_HAIRCUT_RESIDUE_RE = re.compile(
    r"\bhair\s+styled\s+in\s+a\s+cut(?:\s+featuring)?\b",
    re.IGNORECASE,
)


def _awkward_haircut_residues(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in _AWKWARD_HAIRCUT_RESIDUE_RE.finditer(text):
        value = match.group(0).strip()
        key = value.lower()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _BASE_QUALITY_AUDIT(draft, edited, pose)
    residues = _awkward_haircut_residues(edited)
    warnings = list(base.get("warnings") or [])
    if residues:
        warnings.append("awkward_haircut_residue")

    base["awkward_haircut_residue"] = residues
    base["warnings"] = list(dict.fromkeys(warnings))
    base["passes_basic_gate"] = not base["warnings"]
    return base


def _failure_lines(audit: dict[str, Any]) -> list[str]:
    lines = list(_BASE_FAILURE_LINES(audit))
    residues = [str(value).strip() for value in (audit.get("awkward_haircut_residue") or []) if str(value).strip()]
    if residues:
        lines.append(
            "- Awkward haircut residue after protected-trait removal: "
            + "; ".join(residues)
            + ". Rewrite only this local hair clause using any allowed transient hair state; do not preserve generic haircut wording."
        )
    return lines


def main() -> int:
    repair_v01.DEFAULT_PROMPT = DEFAULT_PROMPT
    repair_v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    repair_v01.RUN_VERSION = RUN_VERSION
    repair_v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    repair_v01.quality_audit = quality_audit
    repair_v01._failure_lines = _failure_lines
    return repair_v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
