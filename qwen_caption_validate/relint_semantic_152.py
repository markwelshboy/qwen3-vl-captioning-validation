from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .caption_projection_152 import lint_caption


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-relint-semantic-152",
        description=(
            "Re-lint existing Projection 1.5.1 semantic Compose captions with the Projection 1.5.2 linter. "
            "No model is loaded and captions are not regenerated."
        ),
    )
    parser.add_argument("compose_dir", type=Path, help="Existing compose_semantic_151 output directory.")
    parser.add_argument("--only", nargs="+", help="Only relint keys containing one of these strings.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compose_dir = args.compose_dir.expanduser().resolve()
    if not compose_dir.is_dir():
        print(f"Compose directory not found: {compose_dir}", file=sys.stderr)
        return 2

    meta_files = sorted(compose_dir.glob("*.fusion-safe.json"))
    if args.only:
        needles = tuple(args.only)
        meta_files = [p for p in meta_files if any(n in p.name for n in needles)]
    if not meta_files:
        print(f"No .fusion-safe.json metadata found in {compose_dir}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    lines: list[str] = []
    total_violations = 0
    total_warnings = 0
    normalized_aliases = 0
    invalid = 0

    for path in meta_files:
        payload = _read_json(path)
        if payload is None:
            invalid += 1
            continue
        key = path.name.removesuffix(".fusion-safe.json")
        caption = str(payload.get("caption") or "").strip()
        evidence = payload.get("caption_evidence")
        if not caption or not isinstance(evidence, dict):
            invalid += 1
            continue

        lint = lint_caption(caption, evidence)
        violations = lint.get("violations") or []
        warnings = lint.get("warnings") or []
        normalized = lint.get("normalized_findings") or []
        total_violations += len(violations)
        total_warnings += len(warnings)
        normalized_aliases += sum(
            1 for item in normalized if isinstance(item, dict) and item.get("type") == "posture_vocabulary_alias_normalized"
        )

        records.append(
            {
                "result_key": key,
                "violation_count": len(violations),
                "warning_count": len(warnings),
                "violations": violations,
                "warnings": warnings,
                "normalized_findings": normalized,
            }
        )
        lines.extend(
            [
                f"===== {key} =====",
                f"violations={len(violations)} warnings={len(warnings)} normalized_aliases={len(normalized)}",
                f"violations_json: {json.dumps(violations, ensure_ascii=False)}",
                f"warnings_json: {json.dumps(warnings, ensure_ascii=False)}",
                f"normalized_json: {json.dumps(normalized, ensure_ascii=False)}",
                caption,
                "",
            ]
        )

    index = {
        "schema_version": "semantic-compose-relint-1.5.2",
        "source_compose_dir": str(compose_dir),
        "record_count": len(records),
        "invalid_records": invalid,
        "total_violations": total_violations,
        "total_warnings": total_warnings,
        "normalized_reclining_aliases": normalized_aliases,
        "records": records,
    }
    index_path = compose_dir / "relint_152.index.json"
    text_path = compose_dir / "relint_152.txt"
    _write_json(index_path, index)
    text_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Projection 1.5.2 relint: {len(records)} captions; invalid: {invalid}")
    print(f"Violations: {total_violations}; warnings: {total_warnings}; normalized reclining aliases: {normalized_aliases}")
    print(f"JSON: {index_path}")
    print(f"Text: {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
