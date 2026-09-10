from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from . import caption_refiner_text_fusion_v14 as v14
from . import caption_refiner_text_fusion_v142 as v142


ARTIFACT_VERSION = "caption-refiner-text-fusion-0.14.3"
RUN_VERSION = "caption-refiner-text-fusion-0.14.3-run"
GATE_POLICY = "sam3d_candidate_up_to_two_body_domains_no_laterality"
_LEFT_RIGHT_RE = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)


def govern_pose_candidate_v143(value: Any) -> dict[str, Any]:
    """Govern a SAM3D-only candidate without forcing an artificial one-domain split.

    v0.12p candidates have no deterministic laterality input, so any left/right term is
    rejected.  A concise relationship spanning two body domains (for example torso +
    upper limbs) is allowed; three-domain pose rewrites remain too broad.
    """
    result = v14.govern_pose_candidate(value)
    raw = v14._extract_delta_text(value)
    if not raw or raw == "NO_CORRECTION":
        return result

    normalized = " ".join(raw.split())
    reasons = [reason for reason in result.get("reasons", []) if reason != "multiple_body_domains"]
    domains = list(result.get("domains") or [])

    if len(domains) > 2:
        reasons.append("too_many_body_domains")
    if _LEFT_RIGHT_RE.search(normalized):
        reasons.append("ungrounded_laterality_language")

    reasons = sorted(set(reasons))
    result["domains"] = domains
    result["reasons"] = reasons
    if reasons:
        result["status"] = "rejected"
        result["accepted_text"] = None
    else:
        result["status"] = "accepted"
        result["accepted_text"] = normalized
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _tag_and_rename_index(output_dir: Path) -> None:
    for path in sorted(output_dir.glob("*.text_refiner.json")):
        payload = _read_json(path)
        if not payload:
            continue
        payload["schema_version"] = ARTIFACT_VERSION
        payload["pose_candidate_gate_policy"] = GATE_POLICY
        _write_json(path, payload)

    old_index = output_dir / "caption_refiner_text_fusion_v142.index.json"
    new_index = output_dir / "caption_refiner_text_fusion_v143.index.json"
    index = _read_json(old_index)
    if not index:
        return
    index["schema_version"] = RUN_VERSION
    index["pose_candidate_gate_policy"] = GATE_POLICY
    records = index.get("records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict):
                record["schema_version"] = ARTIFACT_VERSION
                record["pose_candidate_gate_policy"] = GATE_POLICY
    _write_json(new_index, index)
    if old_index != new_index and old_index.exists():
        old_index.unlink()


def main() -> int:
    argv, output_dir = v142._resolve_output_dir(list(sys.argv))
    sys.argv[:] = argv

    # Patch only this process.  Historical v0.14.2 behavior remains unchanged.
    v14.govern_pose_candidate = govern_pose_candidate_v143
    v142.ARTIFACT_VERSION = ARTIFACT_VERSION
    v142.RUN_VERSION = RUN_VERSION

    print(
        "v0.14.3 pose gate: allow <=2 body domains; reject all SAM3D-only left/right language"
    )
    rc = v142.main()
    if rc == 0:
        _tag_and_rename_index(output_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
