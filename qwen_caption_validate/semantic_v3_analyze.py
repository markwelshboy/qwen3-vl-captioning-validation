from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .runner import (
    generate_text,
    load_model,
    model_slug,
    parse_json_response,
    resolve_model_id,
    unload_model,
    validate_analysis,
)
from .semantic_v3_analyze_normalize import normalize_analyze_representation


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_analyze_from_extract_v0_1.txt"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "semantic_analyze_v3.schema.json"
OUTPUT_VERSION = "semantic-v3-analyze-from-extract-0.1"
OUTPUT_SUBDIR = "analyze-from-extract-v0.1"
INDEX_VERSION = "semantic-v3-analyze-from-extract-run-0.1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_json(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _copy_dict(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []


def build_analyze_evidence(extract: dict[str, Any]) -> dict[str, Any]:
    """Project Extract into the semantic/physical evidence surface Analyze may use.

    Visual extraction work stays in Extract. Camera/capture/orientation hypotheses are excluded;
    selected posture/action/support hypotheses are retained as candidates because resolving those
    physical semantics is Analyze's job.
    """
    target = extract.get("target_subject") if isinstance(extract.get("target_subject"), dict) else {}
    hypotheses = extract.get("hypotheses") if isinstance(extract.get("hypotheses"), dict) else {}
    scene = extract.get("scene") if isinstance(extract.get("scene"), dict) else {}

    return {
        "source_schema_version": extract.get("schema_version"),
        "projection_policy": (
            "semantic_physical; selected posture/actions/support hypotheses are candidates; "
            "camera/capture/orientation hypotheses omitted"
        ),
        "framing_context": _copy_dict(extract.get("framing")),
        "subject_evidence": {
            "visible_body_parts": _copy_list(target.get("visible_body_parts")),
            "geometry_landmark_visibility": _copy_dict(target.get("geometry_landmark_visibility")),
            "interactions": _copy_list(target.get("interactions")),
            "gaze": _copy_dict(target.get("gaze")),
        },
        "entities": _copy_list(extract.get("entities")),
        "relations": _copy_list(extract.get("relations")),
        "scene_context": {
            "environment_candidate": deepcopy(scene.get("environment_candidate")),
            "environment_confidence": deepcopy(scene.get("environment_confidence")),
            "environment_cues": _copy_list(scene.get("environment_cues")),
        },
        "candidate_hypotheses": {
            "posture": _copy_dict(hypotheses.get("posture")),
            "actions": _copy_list(hypotheses.get("actions")),
            "support_context": _copy_list(hypotheses.get("support_context")),
        },
        "uncertainties": _copy_list(extract.get("uncertainties")),
    }


def build_prompt(base_prompt: str, evidence: dict[str, Any]) -> str:
    compact = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    return f"{base_prompt.rstrip()}\n\nVISUAL EXTRACT EVIDENCE JSON:\n{compact}\n"


def _matches(path: Path, only: list[str]) -> bool:
    if not only:
        return True
    key = path.name.removesuffix(".extract.json")
    values = (path.name, key)
    return any(
        fnmatch.fnmatch(value, pattern) or pattern == value or pattern in value
        for pattern in only
        for value in values
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-semantic-v3-analyze",
        description=(
            "Interpret canonical Visual Extract v3 records into semantic-analyze-3.0 using a "
            "text-only model call. No image is loaded or sent to the model."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    extract_dir = args.extract_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    if not extract_dir.is_dir():
        raise SystemExit(f"Extract directory not found: {extract_dir}")

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "semantic-v3" / OUTPUT_SUBDIR / slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_paths = sorted(path for path in extract_dir.glob("*.extract.json") if _matches(path, args.only))
    if not extract_paths:
        raise SystemExit("No matching Extract records found")

    base_prompt = args.prompt.expanduser().resolve().read_text(encoding="utf-8")
    schema = _read_json(args.schema.expanduser().resolve())

    pending: list[tuple[Path, dict[str, Any], dict[str, Any], Path]] = []
    records: list[dict[str, Any]] = []
    for source_path in extract_paths:
        wrapper = _read_json(source_path)
        key = str(wrapper.get("image_key") or source_path.name.removesuffix(".extract.json"))
        extract = wrapper.get("extract") if isinstance(wrapper.get("extract"), dict) else None
        if extract is None:
            records.append({"image_key": key, "status": "skipped_missing_canonical_extract"})
            continue
        if wrapper.get("canonical_pydantic_valid") is False or wrapper.get("schema_valid") is False:
            records.append({"image_key": key, "status": "skipped_invalid_source_extract"})
            continue
        out_path = output_dir / f"{key}.analyze_v3.json"
        if out_path.exists() and not args.overwrite:
            existing = _read_json(out_path)
            records.append(
                {
                    "image_key": key,
                    "status": "reused",
                    "schema_valid": existing.get("schema_valid"),
                    "output": str(out_path),
                }
            )
            continue
        pending.append((source_path, wrapper, extract, out_path))

    loaded = None
    if pending:
        print(f"Loading {model_id} for text-only Analyze-from-Extract ...")
        loaded = load_model(
            model_id,
            backend=args.backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(f"Loaded in {loaded.load_seconds:.2f}s. Processing {len(pending)} Extract record(s); image input disabled by design.")

    generated = 0
    valid = 0
    try:
        for source_path, wrapper, extract, out_path in pending:
            key = str(wrapper.get("image_key") or source_path.name.removesuffix(".extract.json"))
            evidence = build_analyze_evidence(extract)
            prompt = build_prompt(base_prompt, evidence)
            raw, seconds = generate_text(loaded, prompt, max_new_tokens=args.max_tokens)
            parsed, parse_error = parse_json_response(raw)

            normalized = None
            normalization = None
            schema_errors: list[str] = []
            if isinstance(parsed, dict):
                normalized, normalization = normalize_analyze_representation(parsed)
                schema_errors = validate_analysis(normalized, schema)
            elif parse_error:
                schema_errors = [parse_error]
            else:
                schema_errors = ["model output did not parse to a JSON object"]

            is_valid = normalized is not None and not schema_errors
            payload = {
                "schema_version": OUTPUT_VERSION,
                "image_key": key,
                "source_extract_path": str(source_path),
                "source_extract_sha256": _sha256_json(extract),
                "source_extract_schema_version": extract.get("schema_version"),
                "source_wire_schema_version": wrapper.get("wire_schema_version"),
                "source_extract_normalization": ((wrapper.get("expansion") or {}).get("normalization")),
                "model": model_id,
                "backend": loaded.backend if loaded is not None else None,
                "inference_seconds": seconds,
                "evidence_projection": evidence,
                "analyze_model_output": parsed,
                "representation_normalization": normalization,
                "analyze": normalized if is_valid else None,
                "raw_response": raw,
                "parse_error": parse_error,
                "schema_valid": is_valid,
                "schema_errors": schema_errors,
                "evidence_family": "semantic_visual",
                "authority_note": (
                    "Analyze interprets the same immutable Extract evidence as Gestalt; it is not "
                    "an independent visual vote. Pose/Fusion remain authoritative for governed geometry."
                ),
            }
            _write_json(out_path, payload)
            generated += 1
            valid += int(is_valid)

            result = normalized or parsed or {}
            posture = result.get("posture") if isinstance(result.get("posture"), dict) else {}
            print(
                f"{key}: schema={'ok' if is_valid else 'FAIL'} "
                f"posture={posture.get('value')}/{posture.get('assessment')} "
                f"actions={len(result.get('actions') or [])} "
                f"interactions={len(result.get('interactions') or [])} "
                f"support={len(result.get('support_context') or [])} "
                f"{seconds:.3f}s"
            )
            if parse_error:
                print(f"  parse_error: {parse_error}")
            for error in schema_errors[:5]:
                print(f"  schema_error: {error}")

            records.append(
                {
                    "image_key": key,
                    "status": "written" if is_valid else "written_invalid",
                    "schema_valid": is_valid,
                    "output": str(out_path),
                    "posture": posture,
                    "action_count": len(result.get("actions") or []),
                    "interaction_count": len(result.get("interactions") or []),
                    "support_count": len(result.get("support_context") or []),
                }
            )
    finally:
        if loaded is not None:
            unload_model(loaded)

    index = {
        "schema_version": INDEX_VERSION,
        "run_dir": str(run_dir),
        "extract_dir": str(extract_dir),
        "model": model_id,
        "prompt": str(args.prompt.expanduser().resolve()),
        "schema": str(args.schema.expanduser().resolve()),
        "output_dir": str(output_dir),
        "generated": generated,
        "valid_generated": valid,
        "record_count": len(records),
        "records": sorted(records, key=lambda item: str(item.get("image_key") or "")),
        "evidence_policy": (
            "one image-conditioned Extract pass; Analyze is text-only; selected posture/actions/support "
            "hypotheses are candidates; camera/capture/orientation hypotheses omitted"
        ),
    }
    _write_json(output_dir / "analyze_from_extract.index.json", index)
    print(f"Analyze-from-Extract: {output_dir}")
    print(f"Generated: {generated}; valid: {valid}/{generated if generated else 0}; total records: {len(records)}")
    return 0 if valid == generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
