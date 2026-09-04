from __future__ import annotations

"""Revalidate/re-expand saved x3p2 raw responses without loading a model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import extract_v3
from .extract_v3_contract import audit_extract_contract
from .extract_v3_models import VisualExtractV3
from .extract_v3_models_runtime import ExtractWireV1Runtime
from .extract_v3_wire_contract import expand_extract_wire
from .runner import model_slug, resolve_model_id, validate_analysis


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _schema_hash() -> str:
    schema = ExtractWireV1Runtime.model_json_schema(by_alias=True)
    encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for item in exc.errors(include_url=False, include_context=False, include_input=False):
        loc = ".".join(str(part) for part in item.get("loc") or []) or "$"
        errors.append(f"{loc}: {item.get('msg')} [{item.get('type')}]")
    return errors


def _matches(payload: dict[str, Any], requested: list[str]) -> bool:
    if not requested:
        return True
    image_key = str(payload.get("image_key") or "")
    image_name = Path(str(payload.get("image") or "")).name
    accepted = {image_key, image_name, Path(image_name).stem}
    return any(item in accepted or Path(item).name in accepted or Path(item).stem in accepted for item in requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-reprocess-extract-v3-pydantic",
        description="Revalidate saved x3p2 raw responses and rebuild canonical Extracts without GPU inference.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    output_dir = (
        args.output_dir
        or (run_dir / "extract-v3-pydantic.2" / model_slug(model_id))
    ).expanduser().resolve()
    if not output_dir.is_dir():
        print(f"Pydantic Extract directory not found: {output_dir}", file=sys.stderr)
        return 2

    canonical_schema = _read_json(extract_v3.DEFAULT_SCHEMA)
    wire_schema_hash = _schema_hash()
    paths = sorted(output_dir.glob("*.extract.json"))
    selected = [path for path in paths if _matches(_read_json(path), args.only)]
    if not selected:
        print("No matching saved Extract artifacts found.", file=sys.stderr)
        return 2

    ok = 0
    failed = 0
    for path in selected:
        payload = _read_json(path)
        raw = payload.get("raw_response")
        if not isinstance(raw, str) or not raw.strip():
            print(f"{path.name}: FAIL no raw_response")
            failed += 1
            continue

        try:
            wire = ExtractWireV1Runtime.model_validate_json(raw)
        except ValidationError as exc:
            errors = _validation_errors(exc)
            payload["wire_schema_valid"] = False
            payload["wire_schema_errors"] = errors
            payload["parse_error"] = errors[0] if errors else "runtime validation failed"
            extract_v3._write_json(path, payload)
            print(f"{path.name}: FAIL {payload['parse_error']}")
            failed += 1
            continue

        try:
            canonical_model, expansion = expand_extract_wire(wire)
            warnings = list(expansion.get("warnings") or [])
            warnings.extend(wire.semantic_warnings())
            expansion["warnings"] = warnings
            expansion["runtime_validation_policy"] = "structural_hard_semantic_warning"

            canonical = canonical_model.model_dump(mode="json", by_alias=True)
            VisualExtractV3.model_validate(canonical)
            canonical_errors = validate_analysis(canonical, canonical_schema)
            contract = audit_extract_contract(canonical)
        except Exception as exc:
            payload["expansion_error"] = f"{type(exc).__name__}: {exc}"
            payload["schema_valid"] = False
            extract_v3._write_json(path, payload)
            print(f"{path.name}: FAIL {payload['expansion_error']}")
            failed += 1
            continue

        payload.update(
            {
                "wire_schema_version": "x3p2",
                "wire_contract": "ExtractWireV1Runtime",
                "wire_schema_sha256": wire_schema_hash,
                "wire_extract": wire.model_dump(mode="json", by_alias=True),
                "wire_model_dump": wire.model_dump(mode="json"),
                "wire_schema_valid": True,
                "wire_schema_errors": [],
                "expansion": expansion,
                "expansion_error": None,
                "canonical_contract": "VisualExtractV3",
                "canonical_pydantic_valid": True,
                "extract": canonical,
                "parse_error": None,
                "schema_valid": not canonical_errors,
                "schema_errors": canonical_errors,
                "contract": contract,
            }
        )
        extract_v3._write_json(path, payload)
        status = "ok" if not canonical_errors else "canonical-FAIL"
        print(
            f"{payload.get('image_key')}: {status} "
            f"analyze={'ok' if contract.get('analyze_reconstructable') else 'missing'} "
            f"gestalt={'ok' if contract.get('gestalt_reconstructable') else 'missing'} "
            f"semantic_warnings={len(warnings)}"
        )
        if canonical_errors:
            failed += 1
        else:
            ok += 1

    print(f"Reprocessed: {len(selected)}; valid: {ok}; failed: {failed}; GPU/model load: none")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
