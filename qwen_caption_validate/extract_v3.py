from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

from .extract_v3_contract import audit_extract_contract
from .runner import (
    generate,
    load_model,
    model_slug,
    parse_json_response,
    resolve_backend,
    resolve_model_id,
    unload_model,
    validate_analysis,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "extract_v3.txt"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "extract_v3.schema.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _install_image_only_vllm() -> None:
    try:
        import vllm
    except ImportError:
        return

    original_llm = vllm.LLM
    if getattr(original_llm, "_extract_v3_image_only", False):
        return

    def image_only_llm(*args, **kwargs):
        requested = kwargs.get("limit_mm_per_prompt")
        if requested is not None and requested != {"image": 1, "video": 0}:
            raise RuntimeError(
                "Extract v3 requires image-only vLLM limits {'image': 1, 'video': 0}; "
                f"refusing conflicting limits {requested!r}"
            )
        kwargs["limit_mm_per_prompt"] = {"image": 1, "video": 0}
        return original_llm(*args, **kwargs)

    image_only_llm._extract_v3_image_only = True  # type: ignore[attr-defined]
    vllm.LLM = image_only_llm


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _discover_images(run_dir: Path, images_dir: Path | None) -> list[Path]:
    root = images_dir or (run_dir / "images")
    if not root.is_dir():
        raise FileNotFoundError(f"Extract image directory not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _matches(path: Path, only: list[str]) -> bool:
    if not only:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) or pattern in path.name for pattern in only)


def _result_key(path: Path) -> str:
    return path.stem


def _record_summary(payload: dict[str, Any]) -> dict[str, Any]:
    extract = payload.get("extract") if isinstance(payload.get("extract"), dict) else {}
    hypotheses = extract.get("hypotheses") if isinstance(extract.get("hypotheses"), dict) else {}
    posture = hypotheses.get("posture") if isinstance(hypotheses.get("posture"), dict) else {}
    orientation = hypotheses.get("torso_orientation") if isinstance(hypotheses.get("torso_orientation"), dict) else {}
    return {
        "image_key": payload.get("image_key"),
        "schema_valid": bool(payload.get("schema_valid")),
        "contract": payload.get("contract") or {},
        "entity_count": len(extract.get("entities") or []) if isinstance(extract, dict) else 0,
        "relation_count": len(extract.get("relations") or []) if isinstance(extract, dict) else 0,
        "posture_candidate": posture.get("value"),
        "torso_orientation_candidate": orientation.get("orientation_band"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-visual-extract-v3",
        description=(
            "Run the single image-conditioned V3 visual Extract over cached run images. "
            "Later Analyze/Gestalt stages should reason from these records without re-reading pixels."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Validation run containing images/, used as the persistent output root.")
    parser.add_argument("--images-dir", type=Path, help="Optional image directory override; outputs still live under run_dir.")
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4200)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    try:
        images = [path for path in _discover_images(run_dir, images_dir) if _matches(path, args.only)]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not images:
        print("No matching images found for Extract v3.", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    output_dir = (args.output_dir or (run_dir / "extract-v3.0" / slug)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    schema_path = args.schema.expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = _read_json(schema_path)

    pending: list[tuple[Path, str, Path]] = []
    reused: list[dict[str, Any]] = []
    for image in images:
        key = _result_key(image)
        out_path = output_dir / f"{key}.extract.json"
        if args.overwrite or not out_path.exists():
            pending.append((image, key, out_path))
        else:
            reused.append(_record_summary(_read_json(out_path)))

    if pending and resolve_backend(model_id, args.backend) == "vllm":
        _install_image_only_vllm()

    generated: list[dict[str, Any]] = []
    loaded = None
    if pending:
        print(f"Loading {model_id} for V3 visual Extract ...")
        loaded = load_model(
            model_id,
            backend=args.backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(f"Loaded in {loaded.load_seconds:.2f}s. Extracting {len(pending)} image(s).")

    try:
        for image, key, out_path in pending:
            raw, seconds = generate(loaded, image, prompt, max_new_tokens=args.max_tokens)
            parsed, parse_error = parse_json_response(raw)
            schema_errors = validate_analysis(parsed, schema) if isinstance(parsed, dict) else []
            contract = audit_extract_contract(parsed) if isinstance(parsed, dict) else {
                "schema_version": "visual-extract-contract-audit-0.1",
                "analyze_reconstructable": False,
                "gestalt_reconstructable": False,
                "analyze_missing_paths": ["unparsed_extract"],
                "gestalt_missing_paths": ["unparsed_extract"],
            }
            payload = {
                "schema_version": "visual-extract-artifact-3.0",
                "image_key": key,
                "image": str(image),
                "model": model_id,
                "backend": loaded.backend if loaded is not None else None,
                "inference_seconds": seconds,
                "extract": parsed,
                "raw_response": raw,
                "parse_error": parse_error,
                "schema_valid": isinstance(parsed, dict) and not schema_errors,
                "schema_errors": schema_errors,
                "contract": contract,
            }
            _write_json(out_path, payload)
            record = _record_summary(payload)
            generated.append(record)
            print(
                f"{key}: schema={'ok' if record['schema_valid'] else 'FAIL'} "
                f"analyze={'ok' if record['contract'].get('analyze_reconstructable') else 'missing'} "
                f"gestalt={'ok' if record['contract'].get('gestalt_reconstructable') else 'missing'} "
                f"entities={record['entity_count']} relations={record['relation_count']} "
                f"posture={record['posture_candidate']} torso={record['torso_orientation_candidate']}"
            )
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(reused + generated, key=lambda item: str(item.get("image_key") or ""))
    index = {
        "schema_version": "visual-extract-run-3.0",
        "run_dir": str(run_dir),
        "images_dir": str(images_dir or (run_dir / "images")),
        "model": model_id,
        "prompt": str(prompt_path),
        "schema": str(schema_path),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "generated": len(generated),
        "reused": len(reused),
        "schema_valid_count": sum(1 for record in records if record.get("schema_valid")),
        "analyze_reconstructable_count": sum(1 for record in records if (record.get("contract") or {}).get("analyze_reconstructable")),
        "gestalt_reconstructable_count": sum(1 for record in records if (record.get("contract") or {}).get("gestalt_reconstructable")),
        "records": records,
    }
    _write_json(output_dir / "extract.index.json", index)
    print(f"Extract v3: {output_dir}")
    print(
        f"Generated: {len(generated)}; reused: {len(reused)}; "
        f"schema valid: {index['schema_valid_count']}/{len(records)}; "
        f"Analyze contract: {index['analyze_reconstructable_count']}/{len(records)}; "
        f"Gestalt contract: {index['gestalt_reconstructable_count']}/{len(records)}"
    )
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
