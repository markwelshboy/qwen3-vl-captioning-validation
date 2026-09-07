from __future__ import annotations

from typing import Any

from . import semantic_v3_rich_pose_editor as editor_v01
from . import semantic_v3_rich_pose_editor_v07 as editor_v07
from . import semantic_v3_rich_pose_repair as repair_v01
from . import semantic_v3_rich_pose_repair_v04 as repair_v04


DEFAULT_PROMPT = editor_v07.PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_repair_v05.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-repair-0.5"
RUN_VERSION = "semantic-v3-rich-pose-repair-0.5-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-repair-v0.5"


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    """Editor v0.7 audit + calibrated repair-only laterality/grammar guards."""
    base = editor_v07.quality_audit(draft, edited, pose)

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

    attachment_leaks = repair_v04._local_attachment_leaks(edited)
    if attachment_leaks:
        warnings.append("bad_local_attachment")
    base["bad_local_attachment"] = attachment_leaks

    base["warnings"] = list(dict.fromkeys(warnings))
    base["passes_basic_gate"] = not base["warnings"]
    return base


def main() -> int:
    repair_v01.DEFAULT_PROMPT = DEFAULT_PROMPT
    repair_v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    repair_v01.RUN_VERSION = RUN_VERSION
    repair_v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    repair_v01.quality_audit = quality_audit
    return repair_v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
