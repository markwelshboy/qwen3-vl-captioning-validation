from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

from .runner import validate_analysis
from .semantic_v3_gestalt import DEFAULT_SCHEMA, govern_gestalt
from .semantic_v3_gestalt_normalize import (
    NORMALIZER_VERSION,
    normalize_gestalt_representation,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
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
    """Rebuild canonical/governed Gestalt from preserved model output only."""
    payload = dict(artifact)
    model_output = payload.get("gestalt_model_output")
    if not isinstance(model_output, dict):
        payload["schema_valid"] = False
        payload["schema_errors"] = ["gestalt_model_output missing or not an object"]
        payload["gestalt"] = None
        payload["governance"] = None
        payload["representation_normalization"] = None
        payload["reprocessed_without_model"] = True
        return payload, False, 0

    normalized, normalization = normalize_gestalt_representation(model_output)
    errors = validate_analysis(normalized, schema)
    governed = None
    governance = None

    if not errors:
        governed, governance = govern_gestalt(normalized)
        governed_errors = validate_analysis(governed, schema)
        if governed_errors:
            errors = governed_errors
            governed = None
        else:
            governance["representation_normalization"] = normalization

    valid = governed is not None and not errors
    payload["gestalt"] = governed
    payload["schema_valid"] = valid
    payload["schema_errors"] = errors
    payload["governance"] = governance
    payload["representation_normalization"] = normalization
    payload["reprocessed_without_model"] = True
    payload["reprocessor_version"] = NORMALIZER_VERSION
    return payload, valid, int(normalization.get("action_count") or 0)


def _refresh_index(output_dir: Path) -> None:
    index_path = output_dir / "gestalt_from_extract.index.json"
    if not index_path.exists():
        return
    index = _read_json(index_path)
    artifact_rows: dict[str, dict[str, Any]] = {}
    for path in output_dir.glob("*.gestalt_v3.json"):
        artifact = _read_json(path)
        key = str(artifact.get("image_key") or path.name.removesuffix(".gestalt_v3.json"))
        artifact_rows[key] = artifact

    records = index.get("records")
    if isinstance(records, list):
        for row in records:
            if not isinstance(row, dict):
                continue
            key = str(row.get("image_key") or "")
            artifact = artifact_rows.get(key)
            if artifact is None:
                continue
            row["schema_valid"] = bool(artifact.get("schema_valid"))
            if artifact.get("reprocessed_without_model"):
                row["status"] = "reprocessed_without_model"

    index["valid_generated"] = sum(1 for item in artifact_rows.values() if item.get("schema_valid") is True)
    index["reprocessed_without_model"] = True
    index["representation_normalizer"] = NORMALIZER_VERSION
    _write_json(index_path, index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-reprocess-semantic-v3-gestalt",
        description="Re-normalize and re-govern preserved Gestalt model outputs without loading a model or image.",
    )
    parser.add_argument("gestalt_dir", type=Path, help="Directory containing *.gestalt_v3.json artifacts.")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--only", action="append", default=[], help="Image key/basename/pattern; repeatable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.gestalt_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise SystemExit(f"Gestalt directory not found: {output_dir}")

    schema_path = args.schema.expanduser().resolve()
    schema = _read_json(schema_path)
    paths = sorted(path for path in output_dir.glob("*.gestalt_v3.json") if _matches(path, args.only))
    if not paths:
        raise SystemExit("No matching Gestalt artifacts found")

    valid_count = 0
    action_count = 0
    for path in paths:
        artifact = _read_json(path)
        updated, valid, actions = reprocess_artifact(artifact, schema)
        _write_json(path, updated)
        valid_count += int(valid)
        action_count += actions
        print(
            f"{updated.get('image_key') or path.name}: "
            f"{'ok' if valid else 'FAIL'} normalization_actions={actions}"
        )
        for error in (updated.get("schema_errors") or [])[:5]:
            print(f"  schema_error: {error}")

    _refresh_index(output_dir)
    print(
        f"Reprocessed: {len(paths)}; valid: {valid_count}; failed: {len(paths) - valid_count}; "
        f"normalization_actions: {action_count}; GPU/model load: none"
    )
    return 0 if valid_count == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
