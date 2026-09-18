from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "selfie-composition-observer-v0.1"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "selfie_composition_observer_v01.txt"
SCHEMA_VERSION = "selfie-composition-observer-0.1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

CAPTURE_LABELS = {"selfie_style", "portrait_like", "unclear"}
BODY_REGIONS = {"arm", "forearm", "hand", "other"}
EXTENSIONS = {"outstretched", "extended", "bent", "unclear"}
FRAME_REGIONS = {
    "lower_frame_left",
    "lower_frame_right",
    "frame_left",
    "frame_right",
    "lower_center",
    "center",
    "other",
}
SALIENCE = {"large", "medium", "small"}


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


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _clean_list(value: Any, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _enum(value: Any, allowed: set[str], default: str | None = None) -> str | None:
    text = _clean_text(value)
    return text if text in allowed else default


def _normalize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    unexpected: list[str] = []

    capture = payload.get("capture_style") if isinstance(payload.get("capture_style"), dict) else {}
    capture_unexpected = sorted(str(k) for k in capture if k not in {"label", "basis"})
    unexpected.extend(f"capture_style.{k}" for k in capture_unexpected)
    capture_label = _enum(capture.get("label"), CAPTURE_LABELS, "unclear")
    capture_norm = {
        "label": capture_label,
        "basis": _clean_list(capture.get("basis"), limit=3),
    }

    elements_in = payload.get("foreground_body_elements")
    elements: list[dict[str, Any]] = []
    if isinstance(elements_in, list):
        for idx, raw in enumerate(elements_in[:3]):
            if not isinstance(raw, dict):
                continue
            bad = sorted(
                str(k)
                for k in raw
                if k not in {"body_region", "extension", "frame_region", "salience", "composer_text"}
            )
            unexpected.extend(f"foreground_body_elements[{idx}].{k}" for k in bad)
            region = _enum(raw.get("body_region"), BODY_REGIONS)
            frame_region = _enum(raw.get("frame_region"), FRAME_REGIONS)
            salience = _enum(raw.get("salience"), SALIENCE)
            composer_text = _clean_text(raw.get("composer_text"))
            if not region or not frame_region or not salience or not composer_text:
                continue
            elements.append(
                {
                    "body_region": region,
                    "extension": _enum(raw.get("extension"), EXTENSIONS, "unclear"),
                    "frame_region": frame_region,
                    "salience": salience,
                    "composer_text": composer_text,
                    "anatomical_side": None,
                    "authority": "direct_visual_composition_candidate",
                }
            )

    top_unexpected = sorted(
        str(k)
        for k in payload
        if k not in {"schema_version", "capture_style", "foreground_body_elements"}
    )
    unexpected.extend(top_unexpected)

    return {
        "schema_version": SCHEMA_VERSION,
        "capture_style": capture_norm,
        "foreground_body_elements": elements,
    }, sorted(set(unexpected))


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
        p
        for p in images_dir.rglob(f"{key}.*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return matches[0].resolve() if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shadow Qwen observer for selfie-style capture cues and salient foreground body composition."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--images-dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--model", default="32b-fp8")
    p.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--max-tokens", type=int, default=350)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--vllm-max-model-len", type=int, default=8192)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    prompt_path = args.prompt.expanduser().resolve()

    if not run_dir.is_dir() or not policy_dir.is_dir() or not prompt_path.is_file():
        print(
            f"Required input missing: run={run_dir} policy={policy_dir} prompt={prompt_path}",
            file=sys.stderr,
        )
        return 2

    requested = set(args.only)
    policy_paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        policy_paths = [
            p for p in policy_paths
            if p.name.removesuffix(".perception_policy.json") in requested
        ]
    if not policy_paths:
        print("No matching production v0.2 perception-policy records found.", file=sys.stderr)
        return 2

    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha = _sha256(prompt)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for policy_path in policy_paths:
        policy = _read_json(policy_path)
        key = str(policy.get("image_key") or policy_path.name.removesuffix(".perception_policy.json"))
        out_path = output_dir / f"{key}.selfie_composition.json"
        if out_path.is_file() and not args.overwrite:
            records.append(_read_json(out_path))
            continue

        image = _resolve_image(policy, images_dir)
        if image is None:
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "policy_source": str(policy_path),
                "error": "source_image_not_found",
            }
            _write_json(out_path, record)
            records.append(record)
            continue

        jobs.append(
            {
                "key": key,
                "image": image,
                "policy_path": policy_path,
                "out_path": out_path,
            }
        )

    if jobs:
        from .extract_v3_wire import _install_image_only_vllm
        from .runner import generate, load_model, resolve_backend, resolve_model_id, unload_model

        model_id = resolve_model_id(args.model)
        backend = resolve_backend(model_id, args.backend)
        if backend == "vllm":
            _install_image_only_vllm()
        print(f"Loading {model_id} once for {len(jobs)} selfie-composition observer call(s) ...")
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
                print(f"\n[{index}/{len(jobs)}] {job['key']}")
                try:
                    raw, seconds = generate(
                        loaded,
                        job["image"],
                        prompt,
                        max_new_tokens=args.max_tokens,
                    )
                    raw = raw.strip()
                    parsed = _extract_json_object(raw)
                    normalized, unexpected = _normalize_payload(parsed)
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "status": "ok",
                        "image_key": job["key"],
                        "image": str(job["image"]),
                        "policy_source": str(job["policy_path"]),
                        "model": model_id,
                        "backend": backend,
                        "inference_seconds": seconds,
                        "prompt_source": str(prompt_path),
                        "prompt_sha256": prompt_sha,
                        "observation": normalized,
                        "parse": {"unexpected_keys": unexpected},
                        "raw_response": raw,
                        "invariants": {
                            "camera_height_pitch_owned_by_geometry": True,
                            "foreground_body_frame_region_is_image_relative": True,
                            "foreground_body_anatomical_side_is_never_inferred": True,
                            "camera_holding_is_never_inferred": True,
                            "complete_arm_chain_not_required_for_compositional_salience": True,
                        },
                    }
                    _write_json(job["out_path"], record)
                    records.append(record)
                    capture = normalized["capture_style"]["label"]
                    elements = normalized["foreground_body_elements"]
                    summary = "; ".join(e["composer_text"] for e in elements) or "-"
                    print(f"  capture={capture} | foreground={summary}")
                except Exception as exc:
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "status": "error",
                        "image_key": job["key"],
                        "image": str(job["image"]),
                        "policy_source": str(job["policy_path"]),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _write_json(job["out_path"], record)
                    records.append(record)
                    print(f"ERROR: {record['error']}", file=sys.stderr)
        finally:
            unload_model(loaded)

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    capture_counts = Counter(
        str((((r.get("observation") or {}).get("capture_style") or {}).get("label")) or "unknown")
        for r in records
        if r.get("status") == "ok"
    )
    salient_count = sum(
        bool(((r.get("observation") or {}).get("foreground_body_elements") or []))
        for r in records
        if r.get("status") == "ok"
    )
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "capture_style_counts": dict(sorted(capture_counts.items())),
        "foreground_body_present_count": salient_count,
        "records": records,
    }
    _write_json(output_dir / "selfie_composition_observer.index.json", index)
    print(f"\nIndex: {output_dir / 'selfie_composition_observer.index.json'}")
    return 1 if status_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
