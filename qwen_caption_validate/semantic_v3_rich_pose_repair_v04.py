from __future__ import annotations

import re
from typing import Any

from . import semantic_v3_rich_pose_editor as editor_v01
from . import semantic_v3_rich_pose_editor_v06 as editor_v06
from . import semantic_v3_rich_pose_repair as repair_v01
from . import semantic_v3_rich_pose_repair_v02 as repair_v02


DEFAULT_PROMPT = repair_v02.DEFAULT_PROMPT
ARTIFACT_VERSION = "semantic-v3-rich-pose-repair-0.4"
RUN_VERSION = "semantic-v3-rich-pose-repair-0.4-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-repair-v0.4"

# Keep the narrow calibration guard, but do not mistake an ordinary sentence whose
# grammatical subject is explicitly "facial hair" for hair->beard attachment corruption.
_HAIR_TO_FACIAL_HAIR_ATTACHMENT_RE = re.compile(
    r"(?<!facial )\bhair\b[^.!?]{0,100}\b(?:featuring|including|having|with)\b"
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
    """Editor v0.6 audit + calibrated repair-only equivalences/grammar gate."""
    base = editor_v06.quality_audit(draft, edited, pose)

    # Preserve the deliberately narrow lossless laterality closure:
    # a governed left/right fist necessarily belongs to that side's hand.
    authorized = repair_v01._expanded_authorized_laterality(pose)
    unauthorized: list[str] = []
    for match in editor_v01._BODY_SIDE_RE.finditer(edited):
        pair = (match.group(1).lower(), match.group(2).lower())
        if pair not in authorized:
            unauthorized.append(match.group(0))

    warnings = [
        value
        for value in (base.get("warnings") or [])
        if value not in {"unsupported_anatomical_laterality", "bad_local_attachment"}
    ]
    if unauthorized:
        warnings.append("unsupported_anatomical_laterality")

    base["unauthorized_anatomical_laterality"] = sorted(set(unauthorized), key=str.lower)
    base["authorized_laterality_expanded"] = sorted(
        [{"side": side, "body_part": part} for side, part in authorized],
        key=lambda item: (item["side"], item["body_part"]),
    )

    attachment_leaks = _local_attachment_leaks(edited)
    if attachment_leaks:
        warnings.append("bad_local_attachment")
    base["bad_local_attachment"] = attachment_leaks

    base["warnings"] = list(dict.fromkeys(warnings))
    base["passes_basic_gate"] = not base["warnings"]
    return base


def main() -> int:
    # Reuse proven fail-only repair mechanics. v0.4 changes provenance and the
    # audit contract used both before and after repair.
    repair_v01.DEFAULT_PROMPT = DEFAULT_PROMPT
    repair_v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    repair_v01.RUN_VERSION = RUN_VERSION
    repair_v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    repair_v01.quality_audit = quality_audit
    return repair_v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
