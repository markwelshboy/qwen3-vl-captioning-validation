from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import caption_refiner_delta as impl


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "pose_card_caption_v12p_sam3d_candidate.txt"
ARTIFACT_VERSION = "caption-refiner-delta-0.12p-sam3d-only"
RUN_VERSION = "caption-refiner-delta-0.12p-sam3d-only-run"
CANDIDATE_SCOPE = "sam3d_pose_card_body_geometry_only"


def _disable_dwpose(_run_dir: Path, _supplied: Path | None) -> None:
    """Hard-disable DWPose for the isolated SAM3D candidate experiment."""
    return None


def _no_laterality(*_args, **_kwargs) -> list[dict[str, Any]]:
    """No named laterality facts are allowed into this specialist."""
    return []


def _no_laterality_text(_facts: list[dict[str, Any]]) -> str:
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_output_dir(argv: list[str]) -> tuple[list[str], Path]:
    for i, value in enumerate(argv):
        if value == "--output-dir" and i + 1 < len(argv):
            return argv, Path(argv[i + 1]).expanduser().resolve()
        if value.startswith("--output-dir="):
            return argv, Path(value.split("=", 1)[1]).expanduser().resolve()

    if len(argv) < 2 or argv[1].startswith("-"):
        raise SystemExit("run_dir positional argument is required")
    output = Path(argv[1]).expanduser().resolve() / "caption-refiner-delta-v0.12p-sam3d-only"
    return argv + ["--output-dir", str(output)], output


def _postprocess(output_dir: Path) -> None:
    if not output_dir.is_dir():
        return

    for path in sorted(output_dir.glob("*.delta_refiner.json")):
        payload = _read_json(path)
        if not payload:
            continue
        payload["schema_version"] = ARTIFACT_VERSION
        payload["candidate_scope"] = CANDIDATE_SCOPE
        payload["laterality_facts"] = []
        payload["laterality_source_preference"] = "disabled"
        payload["dwpose_disabled"] = True
        payload["head_gaze_disabled"] = True
        _write_json(path, payload)

    index_path = output_dir / "caption_refiner_delta.index.json"
    index = _read_json(index_path)
    if not index:
        return
    index["schema_version"] = RUN_VERSION
    index["candidate_scope"] = CANDIDATE_SCOPE
    index["dwpose_disabled"] = True
    index["head_gaze_disabled"] = True
    records = index.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            record["schema_version"] = ARTIFACT_VERSION
            record["candidate_scope"] = CANDIDATE_SCOPE
            record["laterality_facts"] = []
            record["laterality_source_preference"] = "disabled"
    _write_json(index_path, index)


def main() -> int:
    argv, output_dir = _ensure_output_dir(list(sys.argv))
    sys.argv[:] = argv

    # Reuse the proven v0.12 pose-card/model machinery, but remove every
    # non-SAM3D specialist input from this process.  The prompt intentionally
    # has no {{LATERALITY_FACTS}} placeholder as a second isolation barrier.
    impl.DEFAULT_PROMPT = DEFAULT_PROMPT
    impl.ARTIFACT_VERSION = ARTIFACT_VERSION
    impl.RUN_VERSION = RUN_VERSION
    impl._resolve_dwpose_dir = _disable_dwpose
    impl._laterality_facts = _no_laterality
    impl._format_laterality_facts = _no_laterality_text

    print("SAM3D pose candidate v0.12p: DWPose/laterality disabled; head/gaze disabled")
    rc = impl.main()
    if rc == 0:
        _postprocess(output_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
