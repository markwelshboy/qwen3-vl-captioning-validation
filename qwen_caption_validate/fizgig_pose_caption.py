from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from .extract_v3_wire import _install_image_only_vllm
from .runner import (
    discover_images,
    generate,
    load_model,
    resolve_backend,
    resolve_model_id,
    unload_model,
)
from .semantic_v3_rich_caption import _generate_vllm_batch


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fizgig_custom1_pose_aware_v01.txt"
ARTIFACT_VERSION = "fizgig-custom1-pose-aware-caption-0.1"
RUN_VERSION = "fizgig-custom1-pose-aware-caption-0.1-run"
METHODOLOGY_ID = "custom1"
METHODOLOGY_NAME = "Portrait Identity — Pose Aware"
DEFAULT_MAX_TOKENS = 220
DEFAULT_MAX_ATTEMPTS = 3

GRAMMAR_PROFILES: dict[str, dict[str, str]] = {
    "feminine": {
        "gender_grammar": "feminine",
        "subject_pronoun": "she",
        "object_pronoun": "her",
        "possessive_pronoun": "her",
        "reflexive_pronoun": "herself",
    },
    "masculine": {
        "gender_grammar": "masculine",
        "subject_pronoun": "he",
        "object_pronoun": "him",
        "possessive_pronoun": "his",
        "reflexive_pronoun": "himself",
    },
    "neutral": {
        "gender_grammar": "gender-neutral",
        "subject_pronoun": "they",
        "object_pronoun": "them",
        "possessive_pronoun": "their",
        "reflexive_pronoun": "themself",
    },
}

STRICT_BINDING = {
    "require_exact_trigger": True,
    "require_trigger_first": True,
    "require_single_trigger": True,
    "reject_detached_trailing_trigger": True,
    "reject_generic_subject_after_trigger": True,
    "retry_on_failure": True,
    "max_attempts": DEFAULT_MAX_ATTEMPTS,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _caption_stats(text: str) -> dict[str, int]:
    words = re.findall(r"\S+", text)
    sentences = [piece for piece in re.split(r"(?<=[.!?])\s+", text.strip()) if piece.strip()]
    return {"characters": len(text), "words": len(words), "sentences": len(sentences)}


def _render_prompt(template: str, *, trigger: str, grammar: str, protected_traits: list[str]) -> tuple[str, dict[str, str]]:
    profile = GRAMMAR_PROFILES[grammar]
    variables = {
        "TRIGGER": trigger,
        "GENDER_GRAMMAR": profile["gender_grammar"],
        "SUBJECT_PRONOUN": profile["subject_pronoun"],
        "OBJECT_PRONOUN": profile["object_pronoun"],
        "POSSESSIVE_PRONOUN": profile["possessive_pronoun"],
        "REFLEXIVE_PRONOUN": profile["reflexive_pronoun"],
        "PROTECTED_TRAITS": ", ".join(protected_traits) if protected_traits else "none configured",
    }
    rendered = template
    for name, value in variables.items():
        rendered = rendered.replace(f"[{name}]", value)
    unresolved = sorted(set(re.findall(r"\[([A-Z][A-Z0-9_]*)\]", rendered)))
    if unresolved:
        raise ValueError(f"Unresolved prompt variable(s): {', '.join(unresolved)}")
    return rendered, variables


def validate_caption(caption: str, trigger: str) -> dict[str, Any]:
    """Mirror fizgig-web Custom 1 STRICT_BINDING validation."""
    text = " ".join(str(caption or "").strip().split())
    trigger = trigger.strip()
    errors: list[str] = []
    warnings: list[str] = []
    if not text:
        return {"valid": False, "errors": ["Caption is empty"], "warnings": warnings}
    if not trigger:
        return {"valid": False, "errors": ["Trigger word is not configured"], "warnings": warnings}

    exact_count = text.count(trigger)
    ci_count = text.lower().count(trigger.lower())
    if exact_count == 0:
        errors.append(f'Exact trigger "{trigger}" is missing or changed case')
    if not text.startswith(trigger):
        errors.append(f'Caption must begin directly with "{trigger}"')
    elif len(text) > len(trigger) and text[len(trigger)].isalnum():
        errors.append("Trigger is not a standalone first token")
    if ci_count != 1:
        errors.append(f"Trigger must occur exactly once (found {ci_count})")
    if re.search(rf"[,;:]\s*{re.escape(trigger)}[.!?]?\s*$", text, re.IGNORECASE):
        errors.append("Detached trailing trigger is not allowed")
    generic = re.match(
        rf"^{re.escape(trigger)}\s*[,;:\-]\s*(?:a|the)\s+(?:woman|man|person|subject)\b",
        text,
        re.IGNORECASE,
    )
    if generic:
        errors.append("Trigger may not be followed by a generic replacement subject")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _retry_prompt(base_prompt: str, validation: dict[str, Any]) -> str:
    reason = "; ".join(validation.get("errors", [])) or "output contract failed"
    return (
        base_prompt
        + "\n\nRETRY CORRECTION: The previous candidate was rejected because "
        + reason
        + ". Follow the requested output structure exactly. Output only the caption."
    )


def _matches_only(path: Path, only: list[str]) -> bool:
    if not only:
        return True
    values = {path.name.lower(), path.stem.lower()}
    wanted = {str(value).lower() for value in only}
    return bool(values & wanted)


def _attempt_record(text: str, validation: dict[str, Any], perf: dict[str, Any]) -> dict[str, Any]:
    return {
        "caption": text,
        "validation": validation,
        "finish_reason": perf.get("finish_reason"),
        "prompt_tokens": perf.get("prompt_tokens"),
        "output_tokens": perf.get("output_tokens"),
        "prepare_seconds": perf.get("prepare_seconds"),
        "ttft_seconds": perf.get("ttft_seconds"),
        "decode_seconds": perf.get("decode_seconds"),
        "decode_tokens_per_second": perf.get("decode_tokens_per_second"),
        "inference_seconds": perf.get("inference_seconds"),
    }


def _payload(
    *,
    image: Path,
    key: str,
    caption: str,
    raw_response: str,
    validation: dict[str, Any],
    history: list[dict[str, Any]],
    model_id: str,
    backend: str,
    prompt_path: Path,
    rendered_prompt: str,
    variables: dict[str, str],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_VERSION,
        "image_key": key,
        "image": str(image),
        "caption": caption,
        "raw_response": raw_response,
        "caption_stats": _caption_stats(caption) if caption else _caption_stats(raw_response),
        "valid": bool(validation.get("valid")),
        "validation": validation,
        "attempts": len(history),
        "attempt_history": history,
        "model": model_id,
        "backend": backend,
        "methodology": {
            "methodology_id": METHODOLOGY_ID,
            "methodology_name": METHODOLOGY_NAME,
            "source": "fizgig-web/backend/app/caption_methodologies.py DEFAULT_POSE_PROMPT",
            "prompt": str(prompt_path),
            "rendered_instruction": rendered_prompt,
            "variables": variables,
            "max_tokens": max_tokens,
            "validation": STRICT_BINDING,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone clone of fizgig-web Custom 1 / Portrait Identity — Pose Aware caption generation."
    )
    parser.add_argument("images_dir", type=Path, help="Directory containing images to caption.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <images_dir parent>/fizgig-pose-captions.")
    parser.add_argument("--trigger", required=True, help="Exact training trigger token, e.g. sH1VX.")
    parser.add_argument("--grammar", choices=sorted(GRAMMAR_PROFILES), default="feminine")
    parser.add_argument("--protected-trait", action="append", default=[], help="Additional project-protected identity phrase; repeatable.")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.expanduser().resolve()
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 2
    if not args.trigger.strip():
        print("--trigger must not be empty", file=sys.stderr)
        return 2
    if args.batch_size < 1 or args.max_tokens < 1 or not (1 <= args.max_attempts <= 5):
        print("--batch-size/--max-tokens must be positive; --max-attempts must be 1..5", file=sys.stderr)
        return 2

    output_dir = (args.output_dir or images_dir.parent / "fizgig-pose-captions").expanduser().resolve()
    text_dir = output_dir / "texts"
    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    template = prompt_path.read_text(encoding="utf-8")
    rendered_prompt, variables = _render_prompt(
        template,
        trigger=args.trigger.strip(),
        grammar=args.grammar,
        protected_traits=[str(value).strip() for value in args.protected_trait if str(value).strip()],
    )

    images = [path for path in discover_images(images_dir, recursive=args.recursive) if _matches_only(path, args.only)]
    if not images:
        print("No matching images found.", file=sys.stderr)
        return 2

    model_id = resolve_model_id(args.model)
    backend = resolve_backend(model_id, args.backend)
    reused: list[dict[str, Any]] = []
    pending: list[tuple[Path, str, Path]] = []
    for image in images:
        key = image.stem
        out_path = output_dir / f"{key}.caption.json"
        if out_path.exists() and not args.overwrite:
            reused.append(_read_json(out_path))
        else:
            pending.append((image, key, out_path))

    loaded = None
    model_load_seconds = 0.0
    generated: list[dict[str, Any]] = []
    failures: list[str] = []
    if pending:
        if backend == "vllm":
            _install_image_only_vllm()
        print(f"Loading {model_id} for Fizgig Custom 1 captions ...")
        loaded = load_model(
            model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        model_load_seconds = float(loaded.load_seconds)
        print(
            f"Loaded in {model_load_seconds:.2f}s. Captioning {len(pending)} image(s); "
            f"batch_size={args.batch_size} max_tokens={args.max_tokens}."
        )

    try:
        if loaded is not None and loaded.backend == "vllm":
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset:offset + args.batch_size]
                items, batch_perf = _generate_vllm_batch(
                    loaded,
                    [row[0] for row in batch],
                    rendered_prompt,
                    max_new_tokens=args.max_tokens,
                )
                for (image, key, out_path), item in zip(batch, items):
                    history: list[dict[str, Any]] = []
                    text = str(item.get("text") or "").strip()
                    validation = validate_caption(text, args.trigger)
                    history.append(_attempt_record(text, validation, item))

                    attempt = 1
                    while not validation["valid"] and attempt < args.max_attempts:
                        attempt += 1
                        retry_prompt = _retry_prompt(rendered_prompt, validation)
                        retry_items, _ = _generate_vllm_batch(
                            loaded,
                            [image],
                            retry_prompt,
                            max_new_tokens=args.max_tokens,
                        )
                        retry_item = retry_items[0]
                        text = str(retry_item.get("text") or "").strip()
                        validation = validate_caption(text, args.trigger)
                        history.append(_attempt_record(text, validation, retry_item))

                    caption = text if validation["valid"] else ""
                    payload = _payload(
                        image=image,
                        key=key,
                        caption=caption,
                        raw_response=text,
                        validation=validation,
                        history=history,
                        model_id=model_id,
                        backend=loaded.backend,
                        prompt_path=prompt_path,
                        rendered_prompt=rendered_prompt,
                        variables=variables,
                        max_tokens=args.max_tokens,
                    )
                    payload["performance"] = {"initial_batch_generation_seconds": batch_perf.get("generation_seconds")}
                    _write_json(out_path, payload)
                    if caption:
                        (text_dir / f"{key}.txt").write_text(caption + "\n", encoding="utf-8")
                        print(f"{key}: PASS attempt={len(history)} — {caption}")
                    else:
                        failures.append(key)
                        print(f"{key}: FAIL after {len(history)} attempts — {validation['errors']}", file=sys.stderr)
                    generated.append(payload)

        elif loaded is not None:
            for image, key, out_path in pending:
                history: list[dict[str, Any]] = []
                validation: dict[str, Any] = {"valid": False, "errors": ["No generation attempt completed"], "warnings": []}
                text = ""
                prompt = rendered_prompt
                for attempt in range(1, args.max_attempts + 1):
                    started = time.perf_counter()
                    text, inference_seconds = generate(loaded, image, prompt, max_new_tokens=args.max_tokens)
                    text = text.strip()
                    validation = validate_caption(text, args.trigger)
                    history.append(_attempt_record(text, validation, {"inference_seconds": inference_seconds or (time.perf_counter()-started)}))
                    if validation["valid"]:
                        break
                    prompt = _retry_prompt(rendered_prompt, validation)
                caption = text if validation["valid"] else ""
                payload = _payload(
                    image=image,
                    key=key,
                    caption=caption,
                    raw_response=text,
                    validation=validation,
                    history=history,
                    model_id=model_id,
                    backend=loaded.backend,
                    prompt_path=prompt_path,
                    rendered_prompt=rendered_prompt,
                    variables=variables,
                    max_tokens=args.max_tokens,
                )
                _write_json(out_path, payload)
                if caption:
                    (text_dir / f"{key}.txt").write_text(caption + "\n", encoding="utf-8")
                    print(f"{key}: PASS attempt={len(history)} — {caption}")
                else:
                    failures.append(key)
                    print(f"{key}: FAIL after {len(history)} attempts — {validation['errors']}", file=sys.stderr)
                generated.append(payload)
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(reused + generated, key=lambda row: str(row.get("image_key") or ""))
    valid_count = sum(1 for row in records if row.get("caption") and (row.get("validation") or {}).get("valid"))
    index = {
        "schema_version": RUN_VERSION,
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "methodology_id": METHODOLOGY_ID,
        "methodology_name": METHODOLOGY_NAME,
        "prompt": str(prompt_path),
        "variables": variables,
        "model": model_id,
        "backend": backend,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "model_load_seconds": model_load_seconds,
        "record_count": len(records),
        "valid_count": valid_count,
        "failed_count": len(records) - valid_count,
        "generated": len(generated),
        "reused": len(reused),
        "records": [
            {
                "image_key": row.get("image_key"),
                "valid": bool((row.get("validation") or {}).get("valid")),
                "attempts": row.get("attempts"),
                "words": (row.get("caption_stats") or {}).get("words"),
                "sentences": (row.get("caption_stats") or {}).get("sentences"),
            }
            for row in records
        ],
    }
    _write_json(output_dir / "fizgig_pose_caption.index.json", index)
    print(f"Fizgig pose-aware captions: {output_dir}")
    print(f"Records: {len(records)}; valid: {valid_count}; failed: {len(records)-valid_count}; generated: {len(generated)}; reused: {len(reused)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
