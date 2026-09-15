from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_SUBDIR = Path("semantic-v3") / "text-composer-v0.9"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "crouch-topology-lexical-probe-v0.1"
SCHEMA_VERSION = "crouch-topology-lexical-probe-0.1"

_CROUCH_OPENING_RE = re.compile(r"^(?P<subject>\S+)\s+crouches\b", re.I)

# This is deliberately a lexical ablation, not another composer pass.  Keeping
# every character after the opening predicate identical isolates whether the
# broad-pose phrase itself encourages a seated/squat-like reconstruction.
_VARIANT_OPENINGS = (
    ("baseline_crouches", "{subject} crouches"),
    ("deep_crouched_stance", "{subject} holds a deep crouched stance"),
    ("deep_standing_crouch", "{subject} holds a deep standing crouch"),
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_variants(caption: str) -> tuple[str, list[dict[str, Any]]]:
    match = _CROUCH_OPENING_RE.search(caption)
    if not match:
        raise ValueError("source caption must begin '<subject> crouches' for a controlled lexical ablation")

    subject = match.group("subject")
    invariant_tail = caption[match.end():]
    tail_sha = _sha256(invariant_tail)
    variants: list[dict[str, Any]] = []
    for variant_id, opening_template in _VARIANT_OPENINGS:
        opening = opening_template.format(subject=subject)
        variant_caption = opening + invariant_tail
        variants.append(
            {
                "variant_id": variant_id,
                "opening": opening,
                "caption": variant_caption,
                "invariant_tail_sha256": tail_sha,
                "only_opening_predicate_changed": True,
            }
        )
    return invariant_tail, variants


def _body_facts(source: dict[str, Any]) -> dict[str, Any]:
    projection = source.get("evidence_projection") if isinstance(source.get("evidence_projection"), dict) else {}
    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    return body


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a controlled three-way lexical ablation for a crouching caption. "
            "No model is loaded and no geometry/evidence is changed."
        )
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--image-key", required=True)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    source_dir = args.source_dir.expanduser().resolve() if args.source_dir else run_dir / DEFAULT_SOURCE_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    source_path = source_dir / f"{args.image_key}.composed.json"
    if not source_path.is_file():
        print(f"Source composed record not found: {source_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.image_key}.lexical_probe.json"
    if out_path.exists() and not args.overwrite:
        print(f"Output exists (use --overwrite): {out_path}", file=sys.stderr)
        return 2

    source = _read_json(source_path)
    caption = str(source.get("caption") or "").strip()
    if not caption:
        print(f"Source caption is empty: {source_path}", file=sys.stderr)
        return 2

    body = _body_facts(source)
    if body.get("broad_pose") != "crouching":
        print(
            f"Expected authoritative broad_pose='crouching', got {body.get('broad_pose')!r}",
            file=sys.stderr,
        )
        return 2

    try:
        invariant_tail, variants = _build_variants(caption)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for variant in variants:
        caption_path = output_dir / f"{args.image_key}.{variant['variant_id']}.caption.txt"
        caption_path.write_text(variant["caption"] + "\n", encoding="utf-8")
        variant["caption_path"] = str(caption_path)

    record = {
        "schema_version": SCHEMA_VERSION,
        "image_key": args.image_key,
        "source_record": str(source_path),
        "source_composer_schema": source.get("schema_version"),
        "source_caption": caption,
        "source_caption_sha256": _sha256(caption),
        "authoritative_body": body,
        "experiment": {
            "question": "Does plain 'crouches' encourage a seated/squat-like reconstruction compared with stance-explicit wording?",
            "controlled_variable": "opening broad-pose lexicalization only",
            "invariant_tail": invariant_tail,
            "invariant_tail_sha256": _sha256(invariant_tail),
            "model_calls": 0,
            "image_access": False,
            "geometry_changed": False,
            "evidence_changed": False,
        },
        "variants": variants,
    }
    _write_json(out_path, record)

    print(f"Controlled crouch lexical probe: {args.image_key}")
    for variant in variants:
        print()
        print(f"[{variant['variant_id']}]")
        print(variant["caption"])
    print(f"\nRecord: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
