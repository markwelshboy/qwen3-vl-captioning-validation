from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_repair as v01


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_repair_v02.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-repair-0.2"
RUN_VERSION = "semantic-v3-rich-pose-repair-0.2-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-repair-v0.2"

_BASE_QUALITY_AUDIT = v01.quality_audit

# Narrow repair-quality guard for the failure observed in calibration: deleting a protected
# hair clause must not cause facial-hair attributes to become grammatically attached to hair.
_HAIR_TO_FACIAL_HAIR_ATTACHMENT_RE = re.compile(
    r"\bhair\b[^.!?]{0,100}\b(?:featuring|including|having|with)\b"
    r"[^.!?]{0,80}\b(?:beard|mustache|moustache|facial\s+hair)\b",
    re.IGNORECASE,
)


def _local_attachment_leaks(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in _HAIR_TO_FACIAL_HAIR_ATTACHMENT_RE.finditer(text):
        value = match.group(0).strip()
        key = value.lower()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _BASE_QUALITY_AUDIT(draft, edited, pose)
    attachment_leaks = _local_attachment_leaks(edited)
    warnings = list(base.get("warnings") or [])
    if attachment_leaks:
        warnings.append("bad_local_attachment")

    base["bad_local_attachment"] = attachment_leaks
    base["warnings"] = list(dict.fromkeys(warnings))
    base["passes_basic_gate"] = not base["warnings"]
    return base


def main() -> int:
    v01.DEFAULT_PROMPT = DEFAULT_PROMPT
    v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    v01.RUN_VERSION = RUN_VERSION
    v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    v01.quality_audit = quality_audit
    return v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
