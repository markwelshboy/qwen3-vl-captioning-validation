from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import extract_v3
from .runner import load_model, model_slug, resolve_backend, resolve_model_id, unload_model
from .semantic_v3_text_only_bootstrap import install_text_only_vllm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_editor_v01.txt"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 700
ARTIFACT_VERSION = "semantic-v3-rich-pose-editor-0.1"
RUN_VERSION = "semantic-v3-rich-pose-editor-0.1-run"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _text_stats(text: str) -> dict[str, int]:
    words = re.findall(r"\S+", text)
    sentences = [piece for piece in re.split(r"(?<=[.!?])\s+", text.strip()) if piece.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
    }


def _pose_corrections(pose: dict[str, Any]) -> list[str]:
    # Deliberately exclude conditional_hints. A withheld reconstruction is not editor evidence.
    return [str(value).strip() for value in (pose.get("caption_ready_phrases") or []) if str(value).strip()]


def _authorized_laterality(pose: dict[str, Any]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    components = pose.get("components") if isinstance(pose.get("components"), dict) else {}
    for relation in components.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        side = str(relation.get("side") or "").lower()
        phrase = str(relation.get("phrase") or "").lower()
        if side not in {"left", "right"}:
            continue
        for body_part in ("hand", "fist", "wrist", "arm", "elbow", "shoulder", "hip", "knee", "leg", "foot"):
            if body_part in phrase:
                result.add((side, body_part))
    return result


def build_editor_input(rich: dict[str, Any], pose: dict[str, Any], prompt_template: str) -> dict[str, Any]:
    draft = str(rich.get("caption") or "").strip()
    if not draft:
        raise ValueError("Rich caption artifact has no caption text")
    corrections = _pose_corrections(pose)
    correction_text = "\n".join(f"- {value}" for value in corrections) if corrections else "- None. Preserve the draft geometry except for removing unsupported anatomical laterality."
    prompt = (
        prompt_template.replace("{{RICH_DRAFT}}", draft)
        .replace("{{POSE_CORRECTIONS}}", correction_text)
    )
    return {
        "rich_draft": draft,
        "pose_corrections": corrections,
        "pose_conditional_hints_withheld": pose.get("conditional_hints") or [],
        "pose_injection_priority": pose.get("injection_priority"),
        "authorized_laterality": sorted([{"side": side, "body_part": part} for side, part in _authorized_laterality(pose)], key=lambda item: (item["side"], item["body_part"])),
        "editor_prompt": prompt,
    }


_HAIR_COLOR_RE = re.compile(
    r"\b(?:blond(?:e)?|brunette|auburn|ginger|red|brown|black|gray|grey|silver|dark|light)(?:[- ]\w+){0,2}\s+hair\b",
    re.IGNORECASE,
)
_HAIR_LENGTH_RE = re.compile(
    r"\b(?:very\s+)?(?:long|short|shoulder-length|chin-length|medium-length|waist-length)(?:[- ]\w+){0,2}\s+hair\b",
    re.IGNORECASE,
)
_EYE_COLOR_RE = re.compile(
    r"\b(?:blue|green|brown|hazel|gray|grey|dark|light)(?:[- ]\w+){0,1}\s+eyes\b",
    re.IGNORECASE,
)
_SKIN_RE = re.compile(
    r"\b(?:fair|pale|light|dark|brown|olive|tan|tanned)(?:[- ]\w+){0,2}\s+skin\b",
    re.IGNORECASE,
)
_AGE_RE = re.compile(
    r"\b(?:young|middle-aged|elderly|older|teenage|teenaged)\s+(?:woman|man|person|subject)\b",
    re.IGNORECASE,
)
_BODY_SIDE_RE = re.compile(
    r"\b(left|right)\s+(hand|fist|wrist|arm|elbow|shoulder|hip|knee|leg|foot)\b",
    re.IGNORECASE,
)


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    draft_stats = _text_stats(draft)
    edited_stats = _text_stats(edited)
    draft_words = draft_stats["words"]
    edited_words = edited_stats["words"]
    ratio = edited_words / draft_words if draft_words else 0.0

    identity_leaks: list[str] = []
    for pattern in (_HAIR_COLOR_RE, _HAIR_LENGTH_RE, _EYE_COLOR_RE, _SKIN_RE, _AGE_RE):
        identity_leaks.extend(match.group(0) for match in pattern.finditer(edited))

    authorized = _authorized_laterality(pose)
    unauthorized_laterality: list[str] = []
    for match in _BODY_SIDE_RE.finditer(edited):
        pair = (match.group(1).lower(), match.group(2).lower())
        if pair not in authorized:
            unauthorized_laterality.append(match.group(0))

    warnings: list[str] = []
    if ratio < 0.72:
        warnings.append("edited_caption_overcompressed")
    elif ratio > 1.35:
        warnings.append("edited_caption_expanded_substantially")
    if identity_leaks:
        warnings.append("intrinsic_identity_leakage")
    if unauthorized_laterality:
        warnings.append("unsupported_anatomical_laterality")

    return {
        "draft_stats": draft_stats,
        "edited_stats": edited_stats,
        "word_ratio_vs_draft": round(ratio, 4),
        "identity_leaks": sorted(set(identity_leaks), key=str.lower),
        "unauthorized_anatomical_laterality": sorted(set(unauthorized_laterality), key=str.lower),
        "warnings": warnings,
        "passes_basic_gate": not warnings,
    }


def _generate_text_batch(loaded, prompts: list[str], *, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import SamplingParams

    prepared: list[str] = []
    prepare_by_prompt: list[float] = []
    prepare_started = time.perf_counter()
    for prompt in prompts:
        item_started = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        prepared.append(
            loaded.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        prepare_by_prompt.append(time.perf_counter() - item_started)
    prepare_seconds = time.perf_counter() - prepare_started

    generation_started = time.perf_counter()
    outputs = loaded.model.generate(
        prepared,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
        use_tqdm=False,
    )
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != len(prompts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} editor prompts")

    items: list[dict[str, Any]] = []
    for prep_seconds, output in zip(prepare_by_prompt, outputs):
        perf = extract_v3._request_perf_fields(output, max_new_tokens)
        perf["prepare_seconds"] = prep_seconds
        items.append(perf)

    output_tokens = sum(int(item.get("output_tokens") or 0) for item in items)
    prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in items)
    return items, {
        "prepare_seconds": prepare_seconds,
        "generation_seconds": generation_seconds,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "aggregate_output_tokens_per_second": output_tokens / generation_seconds if generation_seconds > 0 else 0.0,
    }


def _fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservatively edit rich VLM captions using governed Pose language. Text-only: the editor never sees the image."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rich-dir", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="32b-fp8", help="Text-only editor checkpoint")
    parser.add_argument("--rich-model", default="32b-fp8", help="Model slug used to locate default rich-caption artifacts")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build editor input artifacts without loading a model")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    if args.batch_size < 1 or args.max_tokens < 1:
        print("--batch-size and --max-tokens must be >= 1", file=sys.stderr)
        return 2

    editor_model_id = resolve_model_id(args.model)
    editor_slug = model_slug(editor_model_id)
    rich_slug = model_slug(resolve_model_id(args.rich_model))
    rich_dir = (args.rich_dir or (run_dir / "semantic-v3" / "rich-caption-v0.1" / rich_slug)).expanduser().resolve()
    pose_dir = (args.pose_dir or (run_dir / "semantic-v3" / "pose-language-v0.1")).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "semantic-v3" / "rich-pose-editor-v0.1" / editor_slug)).expanduser().resolve()
    if not rich_dir.is_dir():
        print(f"Rich caption directory not found: {rich_dir}", file=sys.stderr)
        return 2
    if not pose_dir.is_dir():
        print(f"Pose language directory not found: {pose_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")

    rich_paths = sorted(rich_dir.glob("*.rich_caption.json"))
    selected = set(args.only)
    if selected:
        rich_paths = [path for path in rich_paths if path.name.removesuffix(".rich_caption.json") in selected]
    if not rich_paths:
        print("No matching rich-caption artifacts found.", file=sys.stderr)
        return 2

    prepared_records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    missing_pose: list[str] = []
    for rich_path in rich_paths:
        key = rich_path.name.removesuffix(".rich_caption.json")
        pose_path = pose_dir / f"{key}.pose_language.json"
        if not pose_path.is_file():
            missing_pose.append(key)
            continue
        rich = _read_json(rich_path)
        pose = _read_json(pose_path)
        editor_input = build_editor_input(rich, pose, prompt_template)
        input_path = output_dir / f"{key}.editor_input.json"
        _write_json(input_path, {
            "schema_version": ARTIFACT_VERSION + "-input",
            "image_key": key,
            "rich_source": str(rich_path),
            "pose_source": str(pose_path),
            "prompt_template": str(prompt_path),
            **editor_input,
        })
        prepared_records.append({"image_key": key, "editor_input": editor_input, "rich": rich, "pose": pose})

        edited_path = output_dir / f"{key}.edited_caption.json"
        if not args.dry_run and (args.overwrite or not edited_path.exists()):
            pending.append({
                "image_key": key,
                "edited_path": edited_path,
                "editor_input": editor_input,
                "rich": rich,
                "pose": pose,
            })
        elif edited_path.exists():
            reused.append(_read_json(edited_path))

    if missing_pose:
        print(f"Missing Pose language for {len(missing_pose)} image(s): {', '.join(missing_pose)}", file=sys.stderr)
        return 2

    if args.dry_run:
        index = {
            "schema_version": RUN_VERSION,
            "dry_run": True,
            "run_dir": str(run_dir),
            "rich_dir": str(rich_dir),
            "pose_dir": str(pose_dir),
            "output_dir": str(output_dir),
            "record_count": len(prepared_records),
            "records": [
                {
                    "image_key": item["image_key"],
                    "pose_corrections": item["editor_input"]["pose_corrections"],
                    "withheld_conditional_hint_count": len(item["editor_input"]["pose_conditional_hints_withheld"]),
                }
                for item in prepared_records
            ],
        }
        _write_json(output_dir / "editor.index.json", index)
        print(f"Editor inputs: {output_dir}")
        print(f"Records: {len(prepared_records)}; dry_run=yes; model not loaded")
        return 0

    backend = resolve_backend(editor_model_id, args.backend)
    if pending and backend == "vllm":
        install_text_only_vllm()

    loaded = None
    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    if pending:
        print(f"Loading {editor_model_id} for text-only rich/Pose editing ...")
        loaded = load_model(
            editor_model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(
            f"Loaded in {loaded.load_seconds:.2f}s. Editing {len(pending)} caption(s). "
            f"batch_size={args.batch_size} max_tokens={args.max_tokens}"
        )

    try:
        if loaded is not None and loaded.backend == "vllm":
            total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
            for batch_index, offset in enumerate(range(0, len(pending), args.batch_size), start=1):
                batch_started = time.perf_counter()
                batch = pending[offset : offset + args.batch_size]
                items, batch_perf = _generate_text_batch(
                    loaded,
                    [item["editor_input"]["editor_prompt"] for item in batch],
                    max_new_tokens=args.max_tokens,
                )
                for source, item in zip(batch, items):
                    edited = str(item.get("text") or "").strip()
                    audit = quality_audit(source["editor_input"]["rich_draft"], edited, source["pose"])
                    payload = {
                        "schema_version": ARTIFACT_VERSION,
                        "image_key": source["image_key"],
                        "model": editor_model_id,
                        "backend": loaded.backend,
                        "rich_source_caption": source["editor_input"]["rich_draft"],
                        "pose_corrections": source["editor_input"]["pose_corrections"],
                        "pose_conditional_hints_withheld": source["editor_input"]["pose_conditional_hints_withheld"],
                        "edited_caption": edited,
                        "quality_audit": audit,
                        "performance": {
                            "batch_index": batch_index,
                            "batch_size": len(batch),
                            "prepare_seconds": item.get("prepare_seconds"),
                            "shared_batch_generation_seconds": batch_perf["generation_seconds"],
                            "prompt_tokens": item.get("prompt_tokens"),
                            "output_tokens": item.get("output_tokens"),
                            "max_new_tokens": args.max_tokens,
                            "finish_reason": item.get("finish_reason"),
                            "ttft_seconds": item.get("ttft_seconds"),
                            "queue_seconds": item.get("queue_seconds"),
                            "decode_seconds": item.get("decode_seconds"),
                            "decode_tokens_per_second": item.get("decode_tokens_per_second"),
                            "engine_e2e_seconds": item.get("engine_e2e_seconds"),
                        },
                    }
                    _write_json(source["edited_path"], payload)
                    generated.append(payload)
                    print(
                        "RICH_POSE_EDITOR_PERF "
                        f"image={source['image_key']} batch={batch_index}/{total_batches} "
                        f"prepare={_fmt_seconds(item.get('prepare_seconds'))} "
                        f"batch_generate={batch_perf['generation_seconds']:.3f}s "
                        f"prompt_tokens={item.get('prompt_tokens')} output_tokens={item.get('output_tokens')}/{args.max_tokens} "
                        f"decode_tok_s={item.get('decode_tokens_per_second') or 'n/a'} "
                        f"word_ratio={audit['word_ratio_vs_draft']:.3f} "
                        f"warnings={','.join(audit['warnings']) or 'none'} finish={item.get('finish_reason')}"
                    )

                batch_wall = time.perf_counter() - batch_started
                amortized = batch_wall / len(batch) if batch else 0.0
                record = {
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "images": [item["image_key"] for item in batch],
                    "prepare_seconds": batch_perf["prepare_seconds"],
                    "generation_seconds": batch_perf["generation_seconds"],
                    "wall_seconds": batch_wall,
                    "amortized_seconds_per_image": amortized,
                    "prompt_tokens": batch_perf["prompt_tokens"],
                    "output_tokens": batch_perf["output_tokens"],
                    "aggregate_output_tokens_per_second": batch_perf["aggregate_output_tokens_per_second"],
                }
                batch_runtime.append(record)
                print(
                    "RICH_POSE_EDITOR_BATCH_PERF "
                    f"batch={batch_index}/{total_batches} size={len(batch)} "
                    f"prepare={batch_perf['prepare_seconds']:.3f}s generate={batch_perf['generation_seconds']:.3f}s "
                    f"wall={batch_wall:.3f}s amortized={amortized:.3f}s/image "
                    f"prompt_tokens={batch_perf['prompt_tokens']} output_tokens={batch_perf['output_tokens']} "
                    f"aggregate_output_tok_s={batch_perf['aggregate_output_tokens_per_second']:.2f}"
                )
        elif loaded is not None:
            from .runner import generate_text
            for source in pending:
                edited, inference_seconds = generate_text(
                    loaded,
                    source["editor_input"]["editor_prompt"],
                    max_new_tokens=args.max_tokens,
                )
                audit = quality_audit(source["editor_input"]["rich_draft"], edited, source["pose"])
                payload = {
                    "schema_version": ARTIFACT_VERSION,
                    "image_key": source["image_key"],
                    "model": editor_model_id,
                    "backend": loaded.backend,
                    "rich_source_caption": source["editor_input"]["rich_draft"],
                    "pose_corrections": source["editor_input"]["pose_corrections"],
                    "pose_conditional_hints_withheld": source["editor_input"]["pose_conditional_hints_withheld"],
                    "edited_caption": edited.strip(),
                    "quality_audit": audit,
                    "performance": {"inference_seconds": inference_seconds, "max_new_tokens": args.max_tokens},
                }
                _write_json(source["edited_path"], payload)
                generated.append(payload)
    finally:
        if loaded is not None:
            unload_model(loaded)

    run_wall = time.perf_counter() - run_started
    records = sorted(reused + generated, key=lambda item: str(item.get("image_key") or ""))
    index = {
        "schema_version": RUN_VERSION,
        "dry_run": False,
        "run_dir": str(run_dir),
        "rich_dir": str(rich_dir),
        "pose_dir": str(pose_dir),
        "output_dir": str(output_dir),
        "model": editor_model_id,
        "backend": backend,
        "prompt": str(prompt_path),
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "record_count": len(records),
        "generated": len(generated),
        "reused": len(reused),
        "batch_runtime": batch_runtime,
        "run_wall_seconds": run_wall,
        "basic_gate_pass_count": sum(1 for item in records if (item.get("quality_audit") or {}).get("passes_basic_gate")),
        "records": [
            {
                "image_key": item.get("image_key"),
                "pose_corrections": item.get("pose_corrections") or [],
                "quality_audit": item.get("quality_audit") or {},
                "performance": item.get("performance") or {},
            }
            for item in records
        ],
    }
    _write_json(output_dir / "editor.index.json", index)
    print(f"Rich/Pose edited captions: {output_dir}")
    print(
        f"Records: {len(records)}; generated: {len(generated)}; reused: {len(reused)}; "
        f"basic_gate_pass={index['basic_gate_pass_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
