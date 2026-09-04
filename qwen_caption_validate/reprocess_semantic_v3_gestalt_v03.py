from __future__ import annotations

import argparse
import fnmatch
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .runner import validate_analysis
from .semantic_v3_gestalt import DEFAULT_SCHEMA, govern_gestalt
from .semantic_v3_gestalt_normalize import normalize_gestalt_representation
from .semantic_v3_gestalt_source_authority import AUTHORITY_VERSION, apply_source_authority


OUTPUT_VERSION = "semantic-v3-gestalt-from-extract-0.3"
OUTPUT_SUBDIR = "gestalt-from-extract-v0.3"
INDEX_VERSION = "semantic-v3-gestalt-from-extract-run-0.3"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _matches(path: Path, only: list[str]) -> bool:
    if not only:
        return True
    key = path.name.removesuffix(".gestalt_v3.json")
    values = (path.name, key)
    return any(
        fnmatch.fnmatch(value, pattern) or pattern == value or pattern in value
        for pattern in only
        for value in values
    )


def reprocess_artifact(
    artifact: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], bool, int]:
    """Apply representation + stable Gestalt governance + source authority with no model."""
    payload = deepcopy(artifact)
    model_output = payload.get("gestalt_model_output")
    evidence = payload.get("evidence_projection")
    if not isinstance(model_output, dict) or not isinstance(evidence, dict):
        payload["schema_version"] = OUTPUT_VERSION
        payload["schema_valid"] = False
        payload["schema_errors"] = ["gestalt_model_output/evidence_projection missing or invalid"]
        payload["gestalt"] = None
        payload["source_authority"] = None
        payload["reprocessed_without_model"] = True
        return payload, False, 0

    normalized, representation = normalize_gestalt_representation(model_output)
    errors = validate_analysis(normalized, schema)
    governed = None
    governance = None
    source_authority = None

    if not errors:
        governed, governance = govern_gestalt(normalized)
        governed, source_authority = apply_source_authority(governed, evidence)
        errors = validate_analysis(governed, schema)
        if errors:
            governed = None

    valid = governed is not None and not errors
    payload["schema_version"] = OUTPUT_VERSION
    payload["gestalt"] = governed
    payload["schema_valid"] = valid
    payload["schema_errors"] = errors
    payload["representation_normalization"] = representation
    payload["governance"] = governance
    payload["source_authority"] = source_authority
    payload["reprocessed_without_model"] = True
    payload["reprocessor_version"] = AUTHORITY_VERSION
    payload["authority_note"] = (
        "Gestalt is derived from the same Extract semantic family; source authority only withholds "
        "claims whose required observation channel is absent."
    )
    return payload, valid, int((source_authority or {}).get("action_count") or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-reprocess-semantic-v3-gestalt-v03",
        description=(
            "Create a version-isolated Gestalt v0.3 tree by applying deterministic source-evidence "
            "authority to preserved v0.2 model outputs. No model or image is loaded."
        ),
    )
    parser.add_argument("source_dir", type=Path, help="Existing Gestalt v0.2 output directory.")
    parser.add_argument("--output-dir", type=Path, help="Destination; defaults to sibling gestalt-from-extract-v0.3/<model-slug>.")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--only", action="append", default=[], help="Image key/basename/pattern; repeatable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source Gestalt directory not found: {source_dir}")

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        # .../semantic-v3/gestalt-from-extract-v0.2/<model-slug>
        semantic_root = source_dir.parent.parent
        output_dir = semantic_root / OUTPUT_SUBDIR / source_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = _read_json(args.schema.expanduser().resolve())
    paths = sorted(path for path in source_dir.glob("*.gestalt_v3.json") if _matches(path, args.only))
    if not paths:
        raise SystemExit("No matching Gestalt artifacts found")

    rows: list[dict[str, Any]] = []
    valid_count = 0
    action_count = 0
    for path in paths:
        artifact = _read_json(path)
        updated, valid, actions = reprocess_artifact(artifact, schema)
        out_path = output_dir / path.name
        _write_json(out_path, updated)
        valid_count += int(valid)
        action_count += actions
        gestalt = updated.get("gestalt") if isinstance(updated.get("gestalt"), dict) else {}
        camera = gestalt.get("camera") if isinstance(gestalt.get("camera"), dict) else {}
        capture = gestalt.get("capture") if isinstance(gestalt.get("capture"), dict) else {}
        orientation = gestalt.get("subject_orientation") if isinstance(gestalt.get("subject_orientation"), dict) else {}
        print(
            f"{updated.get('image_key') or path.name}: {'ok' if valid else 'FAIL'} "
            f"authority_actions={actions} camera={camera.get('elevation')}/{camera.get('pitch')} "
            f"capture={capture.get('mode')} body={orientation.get('body_orientation')} "
            f"foreground={len(gestalt.get('foreground_relations') or [])}"
        )
        rows.append(
            {
                "image_key": updated.get("image_key"),
                "schema_valid": valid,
                "authority_actions": actions,
                "output": str(out_path),
            }
        )

    source_index = _read_json(source_dir / "gestalt_from_extract.index.json")
    index = {
        "schema_version": INDEX_VERSION,
        "source_index": source_index,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "record_count": len(rows),
        "valid_count": valid_count,
        "authority_action_count": action_count,
        "source_authority_version": AUTHORITY_VERSION,
        "GPU/model_load": "none",
        "records": rows,
    }
    _write_json(output_dir / "gestalt_from_extract.index.json", index)
    print(
        f"Gestalt v0.3 authority reprocess: {output_dir}\n"
        f"Reprocessed: {len(rows)}; valid: {valid_count}; authority_actions: {action_count}; GPU/model load: none"
    )
    return 0 if valid_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
