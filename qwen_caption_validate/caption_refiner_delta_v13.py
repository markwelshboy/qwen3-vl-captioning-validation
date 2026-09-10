from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import caption_refiner_delta as impl
from . import pose_atlas_v3 as atlas
from .caption_refiner_specialist_facts import format_head_gaze_facts, render_specialist_prompt
from .dwpose_compat import target_points_from_profile_record


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "pose_card_caption_v13_delta_refiner_specialists.txt"
ARTIFACT_VERSION = "caption-refiner-delta-0.13"
RUN_VERSION = "caption-refiner-delta-0.13-run"

_HEAD_GAZE_DIR: Path | None = None
_CURRENT_KEY: str | None = None
_ORIGINAL_FIND = impl.base._find_for_key


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _head_gaze_path(key: str | None) -> Path | None:
    if _HEAD_GAZE_DIR is None or not key:
        return None
    direct = _HEAD_GAZE_DIR / f"{key}.head_gaze.json"
    if direct.is_file():
        return direct
    matches = sorted(_HEAD_GAZE_DIR.rglob(f"{key}.head_gaze.json"))
    return matches[0] if matches else None


def _head_gaze_record(key: str | None) -> tuple[Path | None, dict[str, Any]]:
    path = _head_gaze_path(key)
    return path, _read_json(path)


def _capturing_find(directory: Path | None, key: str, suffixes, *args, **kwargs):
    global _CURRENT_KEY
    _CURRENT_KEY = key
    return _ORIGINAL_FIND(directory, key, suffixes, *args, **kwargs)


def _render_prompt_v13(template: str, caption: str, laterality_text: str) -> str:
    _, evidence = _head_gaze_record(_CURRENT_KEY)
    specialist_text = format_head_gaze_facts(evidence)
    return render_specialist_prompt(template, caption, laterality_text, specialist_text)


def _extract_custom_args(argv: list[str]) -> tuple[list[str], Path]:
    cleaned = [argv[0]]
    head_gaze: Path | None = None
    i = 1
    while i < len(argv):
        value = argv[i]
        if value == "--head-gaze-dir":
            if i + 1 >= len(argv):
                raise SystemExit("--head-gaze-dir requires a directory")
            head_gaze = Path(argv[i + 1]).expanduser().resolve()
            i += 2
            continue
        if value.startswith("--head-gaze-dir="):
            head_gaze = Path(value.split("=", 1)[1]).expanduser().resolve()
            i += 1
            continue
        cleaned.append(value)
        i += 1

    if head_gaze is None:
        raise SystemExit("caption-refiner delta v0.13 requires --head-gaze-dir DIR")
    if not head_gaze.is_dir():
        raise SystemExit(f"Head/gaze evidence directory not found: {head_gaze}")
    return cleaned, head_gaze


def _ensure_v13_output(argv: list[str]) -> tuple[list[str], Path]:
    for i, value in enumerate(argv):
        if value == "--output-dir" and i + 1 < len(argv):
            return argv, Path(argv[i + 1]).expanduser().resolve()
        if value.startswith("--output-dir="):
            return argv, Path(value.split("=", 1)[1]).expanduser().resolve()

    if len(argv) < 2 or argv[1].startswith("-"):
        raise SystemExit("run_dir positional argument is required")
    output = Path(argv[1]).expanduser().resolve() / "caption-refiner-delta-v0.13"
    return argv + ["--output-dir", str(output)], output


def _enrich_record(payload: dict[str, Any], key: str) -> dict[str, Any]:
    evidence_path, evidence = _head_gaze_record(key)
    facts = format_head_gaze_facts(evidence)
    payload["schema_version"] = ARTIFACT_VERSION
    payload["head_gaze_evidence_source"] = str(evidence_path) if evidence_path else None
    payload["head_gaze_evidence_schema"] = evidence.get("schema_version") if evidence else None
    payload["head_gaze_specialist_facts"] = facts
    payload["head_gaze_evidence"] = evidence if evidence else None
    return payload


def _postprocess(output_dir: Path) -> None:
    if not output_dir.is_dir():
        return

    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("*.delta_refiner.json")):
        payload = _read_json(path)
        key = str(payload.get("image_key") or path.name.removesuffix(".delta_refiner.json"))
        payload = _enrich_record(payload, key)
        _write_json(path, payload)
        summaries[key] = {
            "head_gaze_evidence_source": payload.get("head_gaze_evidence_source"),
            "head_gaze_evidence_schema": payload.get("head_gaze_evidence_schema"),
            "head_gaze_specialist_facts": payload.get("head_gaze_specialist_facts"),
        }

    index_path = output_dir / "caption_refiner_delta.index.json"
    index = _read_json(index_path)
    if not index:
        return
    index["schema_version"] = RUN_VERSION
    index["head_gaze_evidence_dir"] = str(_HEAD_GAZE_DIR) if _HEAD_GAZE_DIR else None
    records = index.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            key = str(record.get("image_key") or "")
            if key in summaries:
                record.update(summaries[key])
    _write_json(index_path, index)


def main() -> int:
    global _HEAD_GAZE_DIR

    cleaned, _HEAD_GAZE_DIR = _extract_custom_args(sys.argv)
    cleaned, output_dir = _ensure_v13_output(cleaned)
    sys.argv[:] = cleaned

    # Keep v0.12 reproducible; install v0.13 behavior only for this process.
    impl.DEFAULT_PROMPT = DEFAULT_PROMPT
    impl.ARTIFACT_VERSION = ARTIFACT_VERSION
    impl.RUN_VERSION = RUN_VERSION
    atlas._dwpose_target_points = target_points_from_profile_record
    impl.base._find_for_key = _capturing_find
    impl._render_prompt = _render_prompt_v13

    rc = impl.main()
    if rc == 0:
        _postprocess(output_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
