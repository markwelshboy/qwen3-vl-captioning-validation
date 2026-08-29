from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

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
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "composition_gestalt_v1.txt"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "composition_gestalt_v1.schema.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _install_image_only_vllm() -> None:
    try:
        import vllm
    except ImportError:
        return

    original_llm = vllm.LLM
    if getattr(original_llm, "_composition_gestalt_image_only", False):
        return

    def image_only_llm(*args, **kwargs):
        kwargs["limit_mm_per_prompt"] = {"image": 1, "video": 0}
        return original_llm(*args, **kwargs)

    image_only_llm._composition_gestalt_image_only = True  # type: ignore[attr-defined]
    vllm.LLM = image_only_llm


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _discover_images(run_dir: Path) -> list[Path]:
    images_dir = run_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"run images directory not found: {images_dir}")
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _result_key(path: Path) -> str:
    return path.stem


def _matches(path: Path, only: list[str]) -> bool:
    if not only:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) or pattern in path.name for pattern in only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-composition-gestalt-probe",
        description=(
            "Run a top-down camera/composition probe over images already cached in a validation run. "
            "This is report-only and does not modify Analyze, Fusion, or Pose Semantics."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    images = [path for path in _discover_images(run_dir) if _matches(path, args.only)]
    if not images:
        print("No matching cached run images found.", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    output_dir = (args.output_dir or (run_dir / "composition-gestalt-v1" / slug)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt.expanduser().resolve().read_text(encoding="utf-8")
    schema = _read_json(args.schema.expanduser().resolve())

    pending: list[tuple[Path, str, Path]] = []
    existing_records: list[dict[str, Any]] = []
    for image in images:
        key = _result_key(image)
        out_path = output_dir / f"{key}.composition_gestalt.json"
        if args.overwrite or not out_path.exists():
            pending.append((image, key, out_path))
        else:
            payload = _read_json(out_path)
            gestalt = payload.get("gestalt") if isinstance(payload.get("gestalt"), dict) else {}
            existing_records.append(
                {
                    "image_key": key,
                    "schema_valid": bool(payload.get("schema_valid")),
                    "camera_elevation": (gestalt.get("camera") or {}).get("elevation"),
                    "camera_pitch": (gestalt.get("camera") or {}).get("pitch"),
                    "capture_mode": (gestalt.get("capture") or {}).get("mode"),
                    "selfie_holding_hand": (gestalt.get("capture") or {}).get("selfie_holding_hand"),
                    "composition_summary": gestalt.get("composition_summary"),
                }
            )

    if pending and resolve_backend(model_id, args.backend) == "vllm":
        _install_image_only_vllm()

    generated_records: list[dict[str, Any]] = []
    loaded = None
    if pending:
        print(f"Loading {model_id} for top-down composition gestalt probe ...")
        loaded = load_model(
            model_id,
            backend=args.backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(f"Loaded in {loaded.load_seconds:.2f}s. Processing {len(pending)} image(s).")

    try:
        for image, key, out_path in pending:
            raw, seconds = generate(loaded, image, prompt, max_new_tokens=args.max_tokens)
            parsed, parse_error = parse_json_response(raw)
            schema_errors = validate_analysis(parsed, schema) if isinstance(parsed, dict) else []
            payload = {
                "image_key": key,
                "image": str(image),
                "model": model_id,
                "backend": loaded.backend if loaded is not None else None,
                "inference_seconds": seconds,
                "gestalt": parsed,
                "raw_response": raw,
                "parse_error": parse_error,
                "schema_valid": isinstance(parsed, dict) and not schema_errors,
                "schema_errors": schema_errors,
            }
            _write_json(out_path, payload)

            gestalt = parsed if isinstance(parsed, dict) else {}
            camera = gestalt.get("camera") if isinstance(gestalt.get("camera"), dict) else {}
            capture = gestalt.get("capture") if isinstance(gestalt.get("capture"), dict) else {}
            record = {
                "image_key": key,
                "schema_valid": payload["schema_valid"],
                "camera_elevation": camera.get("elevation"),
                "camera_pitch": camera.get("pitch"),
                "camera_confidence": camera.get("confidence"),
                "capture_mode": capture.get("mode"),
                "capture_confidence": capture.get("confidence"),
                "selfie_holding_hand": capture.get("selfie_holding_hand"),
                "holding_hand_confidence": capture.get("holding_hand_confidence"),
                "framing": gestalt.get("framing"),
                "support_context": gestalt.get("support_context") or [],
                "foreground_relations": gestalt.get("foreground_relations") or [],
                "salient_body_configuration": gestalt.get("salient_body_configuration") or [],
                "composition_summary": gestalt.get("composition_summary"),
            }
            generated_records.append(record)
            print(
                f"{key}: camera={record['camera_elevation']}/{record['camera_pitch']} "
                f"capture={record['capture_mode']} hand={record['selfie_holding_hand']} :: "
                f"{record['composition_summary']}"
            )
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(existing_records + generated_records, key=lambda item: str(item.get("image_key") or ""))
    index = {
        "schema_version": "composition-gestalt-probe-run-1.0",
        "run_dir": str(run_dir),
        "model": model_id,
        "prompt": str(args.prompt),
        "schema": str(args.schema),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "generated": len(generated_records),
        "reused": len(existing_records),
        "records": records,
    }
    _write_json(output_dir / "composition_gestalt.index.json", index)
    print(f"Composition gestalt: {output_dir}")
    print(f"Generated: {len(generated_records)}; reused: {len(existing_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
