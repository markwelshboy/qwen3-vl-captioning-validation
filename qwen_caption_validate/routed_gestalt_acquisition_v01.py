from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.1"
DEFAULT_BODY_SUBDIR = Path("semantic-v3") / "fragment-probe-routed-v0.1"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "routed-gestalt-v0.1"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "routed_gestalt_acquisition_v01.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SCHEMA_VERSION = "routed-gestalt-0.1"
ALLOWED_KEYS = {
    "schema_version",
    "appearance",
    "expression_action",
    "objects",
    "scene",
    "secondary_people",
    "gestalt",
    "uncertainties",
}
LIST_FIELDS = (
    "appearance",
    "expression_action",
    "objects",
    "scene",
    "secondary_people",
    "uncertainties",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(raw) if ch == "{"):
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a JSON object")


def _clean_list(value: Any, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.strip().split())
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    unexpected = sorted(str(key) for key in payload if key not in ALLOWED_KEYS)
    normalized: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for field in LIST_FIELDS:
        normalized[field] = _clean_list(payload.get(field))
    gestalt = payload.get("gestalt")
    if isinstance(gestalt, str):
        gestalt = " ".join(gestalt.strip().split()) or None
    else:
        gestalt = None
    normalized["gestalt"] = gestalt
    return normalized, unexpected


def _resolve_image(policy: dict[str, Any], images_dir: Path | None) -> Path | None:
    source = policy.get("image")
    if source:
        candidate = Path(str(source)).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    key = str(policy.get("image_key") or "")
    if not key or images_dir is None or not images_dir.is_dir():
        return None
    for ext in IMAGE_EXTENSIONS:
        direct = images_dir / f"{key}{ext}"
        if direct.is_file():
            return direct.resolve()
    matches = sorted(
        p for p in images_dir.rglob(f"{key}.*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return matches[0].resolve() if matches else None


def _policy_files(policy_dir: Path, only: set[str]) -> list[Path]:
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if only:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in only]
    return paths


def _body_reference(body_dir: Path, key: str) -> dict[str, Any] | None:
    path = body_dir / f"{key}.routed_fragments.json"
    if not path.is_file():
        return None
    record = _read_json(path)
    return {
        "path": str(path),
        "status": record.get("status"),
        "policy_mode": record.get("policy_mode"),
        "model_call": record.get("model_call"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-3 route-independent Qwen acquisition for appearance, action, objects, scene, and gestalt."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--body-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    body_dir = args.body_dir.expanduser().resolve() if args.body_dir else run_dir / DEFAULT_BODY_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    prompt_path = args.prompt.expanduser().resolve()

    if not policy_dir.is_dir():
        print(f"Policy directory not found: {policy_dir}", file=sys.stderr)
        return 2
    if not prompt_path.is_file():
        print(f"Prompt not found: {prompt_path}", file=sys.stderr)
        return 2

    requested = set(args.only)
    policy_paths = _policy_files(policy_dir, requested)
    if not policy_paths:
        print(f"No matching perception-policy records found in {policy_dir}", file=sys.stderr)
        return 2
    found = {p.name.removesuffix(".perception_policy.json") for p in policy_paths}
    missing_requested = sorted(requested - found)
    if missing_requested:
        print("WARNING: requested keys without policy records: " + ", ".join(missing_requested), file=sys.stderr)

    prompt = prompt_path.read_text(encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []

    for policy_path in policy_paths:
        policy = _read_json(policy_path)
        key = str(policy.get("image_key") or policy_path.name.removesuffix(".perception_policy.json"))
        mode = str((policy.get("policy") or {}).get("mode") or "unknown")
        out_path = output_dir / f"{key}.gestalt.json"
        if out_path.is_file() and not args.overwrite:
            records.append(_read_json(out_path))
            continue
        image = _resolve_image(policy, images_dir)
        if image is None:
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "policy_mode": mode,
                "policy_source": str(policy_path),
                "error": "source_image_not_found",
            }
            _write_json(out_path, record)
            records.append(record)
            continue
        jobs.append({
            "key": key,
            "mode": mode,
            "image": image,
            "policy_path": policy_path,
            "out_path": out_path,
            "body_reference": _body_reference(body_dir, key),
        })

    if jobs:
        from .extract_v3_wire import _install_image_only_vllm
        from .runner import generate, load_model, resolve_backend, resolve_model_id, unload_model

        model_id = resolve_model_id(args.model)
        backend = resolve_backend(model_id, args.backend)
        if backend == "vllm":
            _install_image_only_vllm()
        print(f"Loading {model_id} once for {len(jobs)} gestalt acquisition call(s) ...")
        loaded = load_model(
            model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(f"Loaded in {loaded.load_seconds:.2f}s")
        try:
            for index, job in enumerate(jobs, start=1):
                print(f"\n[{index}/{len(jobs)}] {job['key']}: {job['mode']}")
                base = {
                    "schema_version": SCHEMA_VERSION,
                    "image_key": job["key"],
                    "policy_mode": job["mode"],
                    "policy_source": str(job["policy_path"]),
                    "body_acquisition": job["body_reference"],
                    "guardrails": {
                        "pose_owned_elsewhere": True,
                        "configuration_owned_elsewhere": True,
                        "framing_owned_elsewhere": True,
                        "head_pose_owned_elsewhere": True,
                        "gaze_owned_elsewhere": True,
                        "anatomical_laterality_owned_elsewhere": True,
                    },
                }
                try:
                    raw, seconds = generate(loaded, job["image"], prompt, max_new_tokens=args.max_tokens)
                    raw = raw.strip()
                    parsed = _extract_json_object(raw)
                    normalized, unexpected = _normalize_payload(parsed)
                    record = {
                        **base,
                        "status": "ok",
                        "image": str(job["image"]),
                        "model": model_id,
                        "backend": backend,
                        "inference_seconds": seconds,
                        "prompt_source": str(prompt_path),
                        "prompt_sha256": _sha256(prompt),
                        "acquisition": normalized,
                        "parse": {"unexpected_keys": unexpected},
                        "raw_response": raw,
                    }
                    _write_json(job["out_path"], record)
                    records.append(record)
                    print(f"  gestalt={normalized.get('gestalt') or '-'}")
                    if unexpected:
                        print("  unexpected keys: " + ", ".join(unexpected))
                except Exception as exc:
                    record = {
                        **base,
                        "status": "error",
                        "image": str(job["image"]),
                        "model": model_id,
                        "backend": backend,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _write_json(job["out_path"], record)
                    records.append(record)
                    print(f"ERROR: {record['error']}", file=sys.stderr)
        finally:
            unload_model(loaded)

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    mode_counts = Counter(str(r.get("policy_mode") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "body_dir": str(body_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "invariants": {
            "same_gestalt_prompt_for_all_policy_modes": True,
            "gestalt_prompt_has_no_pose_geometry_input": True,
            "body_fragments_are_referenced_but_not_injected_into_prompt": True,
            "pose_configuration_framing_head_gaze_laterality_owned_elsewhere": True,
        },
        "records": records,
    }
    index_path = output_dir / "routed_gestalt.index.json"
    _write_json(index_path, index)
    print(f"\nIndex: {index_path}")
    return 1 if status_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
