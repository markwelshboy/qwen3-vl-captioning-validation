from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
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


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_gestalt_from_extract_v0_1.txt"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "composition_gestalt_v1_4.schema.json"
OUTPUT_VERSION = "semantic-v3-gestalt-from-extract-0.1"

_BODY_SIDE_RE = re.compile(
    r"\b(?:left|right)\s+(?=(?:hand|fist|forearm|arm|elbow|wrist|shoulder|leg|knee|foot|hip|thigh)\b)",
    re.I,
)


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


def build_gestalt_evidence(extract: dict[str, Any]) -> dict[str, Any]:
    """Project canonical Visual Extract into the evidence surface Gestalt is allowed to see.

    The projection deliberately excludes image_overview, transient appearance, raw VLM text,
    and wire/governance internals. Gestalt gets composition-relevant observation fields plus
    explicit Extract hypotheses as candidates, never a second image interpretation.
    """
    target = extract.get("target_subject") if isinstance(extract.get("target_subject"), dict) else {}
    hypotheses = extract.get("hypotheses") if isinstance(extract.get("hypotheses"), dict) else {}
    scene = extract.get("scene") if isinstance(extract.get("scene"), dict) else {}

    return {
        "source_schema_version": extract.get("schema_version"),
        "framing": _copy_dict(extract.get("framing")),
        "subject_evidence": {
            "visible_body_parts": _copy_list(target.get("visible_body_parts")),
            "geometry_landmark_visibility": _copy_dict(target.get("geometry_landmark_visibility")),
            "orientation_cues": _copy_list(target.get("orientation_cues")),
            "gaze": _copy_dict(target.get("gaze")),
            "interactions": _copy_list(target.get("interactions")),
        },
        "entities": _copy_list(extract.get("entities")),
        "relations": _copy_list(extract.get("relations")),
        "scene": {
            "environment_candidate": deepcopy(scene.get("environment_candidate")),
            "background_regions": _copy_list(scene.get("background_regions")),
        },
        "composition_observations": _copy_list(extract.get("composition_observations")),
        "candidate_hypotheses": {
            "posture": _copy_dict(hypotheses.get("posture")),
            "torso_orientation": _copy_dict(hypotheses.get("torso_orientation")),
            "head_orientation": _copy_dict(hypotheses.get("head_orientation")),
            "head_body_relation": _copy_dict(hypotheses.get("head_body_relation")),
            "camera": _copy_dict(hypotheses.get("camera")),
            "capture": _copy_dict(hypotheses.get("capture")),
            "support_context": _copy_list(hypotheses.get("support_context")),
        },
        "uncertainties": _copy_list(extract.get("uncertainties")),
    }


def _scrub_body_laterality(text: Any) -> tuple[Any, bool]:
    if not isinstance(text, str):
        return text, False
    clean, count = _BODY_SIDE_RE.subn("", text)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean, bool(count)


def govern_gestalt(gestalt: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply only representation/authority governance inherited from Gestalt v1.4.

    This intentionally does not try to repair semantic camera/orientation/support mistakes.
    Those remain visible during calibration and are later inputs to Fusion policy.
    """
    governed = deepcopy(gestalt)
    scrubbed: list[dict[str, Any]] = []

    for index, item in enumerate(governed.get("salient_body_configuration") or []):
        if not isinstance(item, dict):
            continue
        original = item.get("description")
        clean, changed = _scrub_body_laterality(original)
        if changed:
            item["description"] = clean
            scrubbed.append(
                {
                    "field": f"salient_body_configuration[{index}].description",
                    "original": original,
                    "governed": clean,
                }
            )

    original_summary = governed.get("composition_summary")
    clean_summary, changed = _scrub_body_laterality(original_summary)
    if changed:
        governed["composition_summary"] = clean_summary
        scrubbed.append(
            {
                "field": "composition_summary",
                "original": original_summary,
                "governed": clean_summary,
            }
        )

    support_audit: list[dict[str, Any]] = []
    for index, item in enumerate(governed.get("support_context") or []):
        if not isinstance(item, dict):
            continue
        ownership = item.get("target_ownership")
        evidence_status = item.get("evidence_status")
        eligible = ownership == "external_scene" and evidence_status in {"observed", "contextual"}
        support_audit.append(
            {
                "index": index,
                "subject_relation": item.get("subject_relation"),
                "target": item.get("target"),
                "target_description": item.get("target_description"),
                "target_ownership": ownership,
                "evidence_status": evidence_status,
                "external_support_candidate": eligible,
            }
        )

    orientation = governed.get("subject_orientation") if isinstance(governed.get("subject_orientation"), dict) else {}
    audit = {
        "schema_version": "semantic-v3-gestalt-governance-0.1",
        "policy": {
            "anatomical_laterality": "side-neutral prose; frame direction remains allowed in body_faces_frame/background frame_location",
            "support_target": "external support candidate only when target_ownership=external_scene and evidence_status observed/contextual",
            "semantic_repair": "none; model semantic errors remain visible for calibration/Fusion",
        },
        "body_laterality_scrubbed": scrubbed,
        "support_context_audit": support_audit,
        "subject_orientation_audit": {
            "body_orientation": orientation.get("body_orientation"),
            "body_faces_frame": orientation.get("body_faces_frame"),
            "body_confidence": orientation.get("body_confidence"),
            "torso_evidence_quality": orientation.get("torso_evidence_quality"),
            "head_relative_body": orientation.get("head_relative_body"),
            "posture_independent": True,
            "frame_direction_not_anatomical_laterality": True,
        },
    }
    return governed, audit


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
        prog="qwen-semantic-v3-gestalt",
        description=(
            "Interpret canonical Visual Extract v3 records into composition-gestalt-1.4 "
            "using a text-only model call. No image is loaded or sent to the model."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Validation run directory used only for default output placement.")
    parser.add_argument("--extract-dir", type=Path, required=True, help="Directory containing <image>.extract.json canonical Extract wrappers.")
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", action="append", default=[], help="Image key/basename/pattern; repeatable.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=1600)
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
        else run_dir / "semantic-v3" / "gestalt-from-extract-v0.1" / slug
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
            records.append({"image_key": key, "status": "skipped_missing_canonical_extract", "source_extract_path": str(source_path)})
            continue
        if wrapper.get("canonical_pydantic_valid") is False or wrapper.get("schema_valid") is False:
            records.append({"image_key": key, "status": "skipped_invalid_source_extract", "source_extract_path": str(source_path)})
            continue

        out_path = output_dir / f"{key}.gestalt_v3.json"
        if out_path.exists() and not args.overwrite:
            existing = _read_json(out_path)
            records.append(
                {
                    "image_key": key,
                    "status": "reused",
                    "source_extract_sha256": existing.get("source_extract_sha256"),
                    "schema_valid": existing.get("schema_valid"),
                    "output": str(out_path),
                }
            )
            continue
        pending.append((source_path, wrapper, extract, out_path))

    loaded = None
    if pending:
        print(f"Loading {model_id} for text-only Gestalt-from-Extract ...")
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
            source_sha = _sha256_json(extract)
            evidence = build_gestalt_evidence(extract)
            prompt = build_prompt(base_prompt, evidence)
            raw, seconds = generate_text(loaded, prompt, max_new_tokens=args.max_tokens)
            parsed, parse_error = parse_json_response(raw)
            schema_errors = validate_analysis(parsed, schema) if isinstance(parsed, dict) else []

            governed = None
            governance = None
            if isinstance(parsed, dict) and not schema_errors:
                governed, governance = govern_gestalt(parsed)
                governed_errors = validate_analysis(governed, schema)
                if governed_errors:
                    schema_errors = governed_errors
                    governed = None

            is_valid = governed is not None and not schema_errors
            payload = {
                "schema_version": OUTPUT_VERSION,
                "image_key": key,
                "source_extract_path": str(source_path),
                "source_extract_sha256": source_sha,
                "source_extract_schema_version": extract.get("schema_version"),
                "source_wire_schema_version": wrapper.get("wire_schema_version"),
                "source_extract_normalization": ((wrapper.get("expansion") or {}).get("normalization")),
                "model": model_id,
                "backend": loaded.backend if loaded is not None else None,
                "inference_seconds": seconds,
                "evidence_projection": evidence,
                "gestalt_model_output": parsed,
                "gestalt": governed,
                "raw_response": raw,
                "parse_error": parse_error,
                "schema_valid": is_valid,
                "schema_errors": schema_errors,
                "governance": governance,
                "evidence_family": "semantic_visual",
                "authority_note": "Gestalt is an interpretation of the same immutable Extract evidence as Analyze; it is not an independent vote.",
            }
            _write_json(out_path, payload)
            generated += 1
            valid += int(is_valid)
            gestalt = governed or parsed or {}
            camera = gestalt.get("camera") if isinstance(gestalt.get("camera"), dict) else {}
            capture = gestalt.get("capture") if isinstance(gestalt.get("capture"), dict) else {}
            orientation = gestalt.get("subject_orientation") if isinstance(gestalt.get("subject_orientation"), dict) else {}
            print(
                f"{key}: schema={'ok' if is_valid else 'FAIL'} "
                f"camera={camera.get('elevation')}/{camera.get('pitch')} "
                f"capture={capture.get('mode')} "
                f"body={orientation.get('body_orientation')}/{orientation.get('body_faces_frame')} "
                f"support={len(gestalt.get('support_context') or [])} "
                f"{seconds:.3f}s"
            )
            if parse_error:
                print(f"  parse_error: {parse_error}")
            if schema_errors:
                for error in schema_errors[:5]:
                    print(f"  schema_error: {error}")

            records.append(
                {
                    "image_key": key,
                    "status": "written" if is_valid else "written_invalid",
                    "source_extract_sha256": source_sha,
                    "schema_valid": is_valid,
                    "output": str(out_path),
                    "camera": camera,
                    "capture": capture,
                    "subject_orientation": orientation,
                    "framing": gestalt.get("framing"),
                    "support_context": gestalt.get("support_context") or [],
                    "foreground_relations": gestalt.get("foreground_relations") or [],
                    "composition_summary": gestalt.get("composition_summary"),
                }
            )
    finally:
        if loaded is not None:
            unload_model(loaded)

    index = {
        "schema_version": "semantic-v3-gestalt-from-extract-run-0.1",
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
        "evidence_policy": "one image-conditioned Extract pass; Gestalt uses canonical Extract JSON only",
    }
    _write_json(output_dir / "gestalt_from_extract.index.json", index)
    print(f"Gestalt-from-Extract: {output_dir}")
    print(f"Generated: {generated}; valid: {valid}/{generated if generated else 0}; total records: {len(records)}")
    return 0 if valid == generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
