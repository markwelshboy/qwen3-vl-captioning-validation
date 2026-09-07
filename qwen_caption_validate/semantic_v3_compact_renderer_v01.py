from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor as editor_v01
from . import semantic_v3_rich_pose_repair_v07 as repair_v07
from .runner import load_model, model_slug, resolve_backend, resolve_model_id, unload_model
from .semantic_v3_text_only_bootstrap import install_text_only_vllm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_compact_renderer_v01.txt"
ARTIFACT_VERSION = "semantic-v3-compact-renderer-0.1"
RUN_VERSION = "semantic-v3-compact-renderer-0.1-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-v0.1"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 320

PROFILE_BUDGETS: dict[str, tuple[int, int]] = {
    "compact": (60, 90),
    "medium": (100, 140),
}

GRAMMAR_PROFILES: dict[str, dict[str, str]] = {
    "feminine": {
        "label": "feminine",
        "subject_pronoun": "she",
        "object_pronoun": "her",
        "possessive_pronoun": "her",
        "reflexive_pronoun": "herself",
    },
    "masculine": {
        "label": "masculine",
        "subject_pronoun": "he",
        "object_pronoun": "him",
        "possessive_pronoun": "his",
        "reflexive_pronoun": "himself",
    },
    "neutral": {
        "label": "gender-neutral",
        "subject_pronoun": "they",
        "object_pronoun": "them",
        "possessive_pronoun": "their",
        "reflexive_pronoun": "themself",
    },
}

_EXPECTED_COMPRESSION_WARNINGS = {
    "edited_caption_overcompressed",
    "edited_caption_expanded_substantially",
}

_SECONDARY_PERSON_RE = re.compile(
    r"\b(?:another|other|second)\s+(?:person|woman|man|child|girl|boy|passenger|figure)\b"
    r"|\b(?:person|woman|man|child|girl|boy|passenger|figure)\s+(?:in|at)\s+the\s+background\b"
    r"|\b(?:portrait|photograph|photo|poster|screen|tattoo|artwork)\b[^.!?]{0,90}\b(?:person|woman|man|face|figure)\b",
    re.IGNORECASE,
)

_BARE_BODY_START_RE = re.compile(
    r"^(?:the|a|one)\s+"
    r"(?P<body>hand|hands|fist|fists|wrist|wrists|arm|arms|elbow|elbows|shoulder|shoulders|"
    r"torso|body|hip|hips|leg|legs|knee|knees|foot|feet)\b",
    re.IGNORECASE,
)

_GENERIC_PRIMARY_START_RE = re.compile(
    r"^(?:a|the)\s+(?:woman|man|person|subject)\b",
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _sentences(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text.strip()) if piece.strip()]


def _pose_corrections(pose: dict[str, Any]) -> list[str]:
    return [str(value).strip() for value in (pose.get("caption_ready_phrases") or []) if str(value).strip()]


def _semantic_caption_for_key(
    key: str,
    *,
    edited_path: Path,
    repair_dir: Path,
) -> tuple[str, Path, str]:
    repaired_path = repair_dir / f"{key}.repaired_caption.json"
    if repaired_path.is_file():
        repaired = _read_json(repaired_path)
        caption = str(repaired.get("repaired_caption") or "").strip()
        if not caption:
            raise ValueError(f"Repair artifact has no repaired_caption: {repaired_path}")
        return caption, repaired_path, "repair-v0.7"

    edited = _read_json(edited_path)
    caption = str(edited.get("edited_caption") or "").strip()
    if not caption:
        raise ValueError(f"Editor artifact has no edited_caption: {edited_path}")
    return caption, edited_path, "editor-v0.7"


def build_renderer_input(
    *,
    semantic_caption: str,
    pose: dict[str, Any],
    trigger: str,
    grammar_profile: str,
    profile: str,
    prompt_template: str,
) -> dict[str, Any]:
    if profile not in PROFILE_BUDGETS:
        raise ValueError(f"Unknown renderer profile: {profile}")
    if grammar_profile not in GRAMMAR_PROFILES:
        raise ValueError(f"Unknown grammar profile: {grammar_profile}")

    min_words, max_words = PROFILE_BUDGETS[profile]
    grammar = GRAMMAR_PROFILES[grammar_profile]
    corrections = _pose_corrections(pose)
    pose_text = "\n".join(f"- {value}" for value in corrections) if corrections else "- None"

    prompt = (
        prompt_template.replace("{{MIN_WORDS}}", str(min_words))
        .replace("{{MAX_WORDS}}", str(max_words))
        .replace("{{TRIGGER}}", trigger)
        .replace("{{GRAMMAR_PROFILE}}", grammar["label"])
        .replace("{{SUBJECT_PRONOUN}}", grammar["subject_pronoun"])
        .replace("{{OBJECT_PRONOUN}}", grammar["object_pronoun"])
        .replace("{{POSSESSIVE_PRONOUN}}", grammar["possessive_pronoun"])
        .replace("{{REFLEXIVE_PRONOUN}}", grammar["reflexive_pronoun"])
        .replace("{{POSE_CORRECTIONS}}", pose_text)
        .replace("{{SEMANTIC_CAPTION}}", semantic_caption)
    )

    return {
        "profile": profile,
        "target_word_range": {"min": min_words, "max": max_words},
        "trigger": trigger,
        "grammar_profile": grammar_profile,
        "grammar": grammar,
        "pose_corrections": corrections,
        "semantic_caption": semantic_caption,
        "renderer_prompt": prompt,
    }


def _trigger_audit(text: str, trigger: str) -> dict[str, Any]:
    escaped = re.escape(trigger)
    exact_matches = list(re.finditer(rf"(?<!\w){escaped}(?!\w)", text))
    ci_matches = list(re.finditer(rf"(?<!\w){escaped}(?!\w)", text, re.IGNORECASE))

    starts_exact = bool(re.match(rf"^{escaped}(?=\s)", text))
    detached_trailing = bool(re.search(rf"[,;:]\s*{escaped}[.!?]?\s*$", text, re.IGNORECASE))
    generic_after_trigger = bool(
        re.match(
            rf"^{escaped}\s*[,;:\-]\s*(?:a|the)\s+(?:woman|man|person|subject)\b",
            text,
            re.IGNORECASE,
        )
        or re.match(
            rf"^{escaped}\s+(?:is|appears|looks)\s+(?:like\s+)?(?:a|the)\s+(?:woman|man|person|subject)\b",
            text,
            re.IGNORECASE,
        )
    )

    warnings: list[str] = []
    if not exact_matches:
        warnings.append("trigger_missing")
    if len(ci_matches) != len(exact_matches):
        warnings.append("trigger_case_changed")
    if not starts_exact:
        warnings.append("trigger_not_direct_first_subject")
    if detached_trailing:
        warnings.append("detached_trailing_trigger")
    if generic_after_trigger:
        warnings.append("generic_primary_subject_after_trigger")

    return {
        "exact_trigger_count": len(exact_matches),
        "case_insensitive_trigger_count": len(ci_matches),
        "starts_with_exact_trigger_as_direct_token": starts_exact,
        "detached_trailing_trigger": detached_trailing,
        "generic_primary_subject_after_trigger": generic_after_trigger,
        "warnings": warnings,
    }


def _ownership_review(text: str, trigger_count: int) -> dict[str, Any]:
    sentences = _sentences(text)
    secondary_mentions = [match.group(0) for match in _SECONDARY_PERSON_RE.finditer(text)]
    bare_body_starts: list[str] = []
    generic_primary_starts: list[str] = []
    for sentence in sentences[1:]:
        if _BARE_BODY_START_RE.match(sentence):
            bare_body_starts.append(sentence)
        if _GENERIC_PRIMARY_START_RE.match(sentence):
            generic_primary_starts.append(sentence)

    review_warnings: list[str] = []
    if secondary_mentions and trigger_count <= 1:
        review_warnings.append("single_trigger_despite_secondary_person_context")
    if bare_body_starts and (secondary_mentions or len(sentences) >= 4):
        review_warnings.append("possible_detached_body_ownership")
    if generic_primary_starts:
        review_warnings.append("possible_generic_primary_reintroduction")

    return {
        "sentence_count": len(sentences),
        "secondary_person_mentions": secondary_mentions,
        "bare_body_part_sentence_starts": bare_body_starts,
        "generic_person_sentence_starts_after_first": generic_primary_starts,
        "review_warnings": review_warnings,
    }


def quality_audit(
    *,
    semantic_caption: str,
    rendered_caption: str,
    pose: dict[str, Any],
    trigger: str,
    profile: str,
) -> dict[str, Any]:
    min_words, max_words = PROFILE_BUDGETS[profile]
    word_count = _word_count(rendered_caption)
    source_words = _word_count(semantic_caption)
    word_ratio = word_count / source_words if source_words else 0.0

    trigger_audit = _trigger_audit(rendered_caption, trigger)
    ownership = _ownership_review(rendered_caption, trigger_audit["exact_trigger_count"])

    semantic = repair_v07.quality_audit(semantic_caption, rendered_caption, pose)
    semantic_policy_warnings = [
        str(value)
        for value in (semantic.get("warnings") or [])
        if str(value) not in _EXPECTED_COMPRESSION_WARNINGS
    ]

    warnings: list[str] = []
    if word_count < min_words:
        warnings.append("rendered_caption_below_target_words")
    elif word_count > max_words:
        warnings.append("rendered_caption_above_target_words")
    warnings.extend(trigger_audit["warnings"])
    warnings.extend(semantic_policy_warnings)
    warnings = list(dict.fromkeys(warnings))

    return {
        "profile": profile,
        "target_word_range": {"min": min_words, "max": max_words},
        "source_word_count": source_words,
        "rendered_word_count": word_count,
        "word_ratio_vs_semantic": round(word_ratio, 4),
        "trigger": trigger_audit,
        "semantic_policy_audit": semantic,
        "semantic_policy_warnings_after_expected_compression": semantic_policy_warnings,
        "ownership_review": ownership,
        "warnings": warnings,
        "passes_basic_gate": not warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render frozen Semantic V3 captions into compact/medium trigger-bound training captions. "
            "Text-only: no image re-analysis occurs."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--editor-dir", type=Path)
    parser.add_argument("--repair-dir", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="32b-fp8", help="Text-only renderer checkpoint")
    parser.add_argument("--source-model", default="32b-fp8", help="Model slug used to locate frozen editor/repair artifacts")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILE_BUDGETS), default=["compact"])
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--grammar-profile", choices=sorted(GRAMMAR_PROFILES), required=True)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build renderer inputs without loading a model")
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

    trigger = str(args.trigger).strip()
    if not trigger or re.search(r"\s", trigger):
        print("--trigger must be one non-empty token with no whitespace", file=sys.stderr)
        return 2

    renderer_model_id = resolve_model_id(args.model)
    renderer_slug = model_slug(renderer_model_id)
    source_slug = model_slug(resolve_model_id(args.source_model))
    editor_dir = (
        args.editor_dir
        or (run_dir / "semantic-v3" / "rich-pose-editor-v0.7" / source_slug)
    ).expanduser().resolve()
    repair_dir = (
        args.repair_dir
        or (run_dir / "semantic-v3" / "rich-pose-repair-v0.7" / source_slug)
    ).expanduser().resolve()
    pose_dir = (
        args.pose_dir
        or (run_dir / "semantic-v3" / "pose-language-v0.1")
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or (run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR / renderer_slug)
    ).expanduser().resolve()

    for label, path in (("editor", editor_dir), ("repair", repair_dir), ("pose", pose_dir)):
        if not path.is_dir():
            print(f"{label.capitalize()} directory not found: {path}", file=sys.stderr)
            return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    if not prompt_path.is_file():
        print(f"Renderer prompt not found: {prompt_path}", file=sys.stderr)
        return 2
    prompt_template = prompt_path.read_text(encoding="utf-8")

    selected = set(args.only)
    edited_paths = sorted(editor_dir.glob("*.edited_caption.json"))
    if selected:
        edited_paths = [path for path in edited_paths if path.name.removesuffix(".edited_caption.json") in selected]
    if not edited_paths:
        print("No matching frozen editor artifacts found.", file=sys.stderr)
        return 2

    prepared: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []

    for edited_path in edited_paths:
        key = edited_path.name.removesuffix(".edited_caption.json")
        pose_path = pose_dir / f"{key}.pose_language.json"
        if not pose_path.is_file():
            print(f"Missing Pose language for {key}: {pose_path}", file=sys.stderr)
            return 2
        pose = _read_json(pose_path)
        semantic_caption, semantic_path, semantic_stage = _semantic_caption_for_key(
            key,
            edited_path=edited_path,
            repair_dir=repair_dir,
        )

        for profile in args.profiles:
            renderer_input = build_renderer_input(
                semantic_caption=semantic_caption,
                pose=pose,
                trigger=trigger,
                grammar_profile=args.grammar_profile,
                profile=profile,
                prompt_template=prompt_template,
            )
            input_path = output_dir / f"{key}.{profile}.renderer_input.json"
            _write_json(
                input_path,
                {
                    "schema_version": ARTIFACT_VERSION + "-input",
                    "image_key": key,
                    "profile": profile,
                    "semantic_source": str(semantic_path),
                    "semantic_source_stage": semantic_stage,
                    "pose_source": str(pose_path),
                    "prompt_template": str(prompt_path),
                    **renderer_input,
                },
            )

            output_path = output_dir / f"{key}.{profile}.rendered_caption.json"
            record = {
                "image_key": key,
                "profile": profile,
                "semantic_caption": semantic_caption,
                "semantic_source": str(semantic_path),
                "semantic_source_stage": semantic_stage,
                "pose": pose,
                "renderer_input": renderer_input,
                "output_path": output_path,
            }
            prepared.append(record)
            if not args.dry_run and (args.overwrite or not output_path.exists()):
                pending.append(record)
            elif output_path.exists():
                reused.append(_read_json(output_path))

    if args.dry_run:
        index = {
            "schema_version": RUN_VERSION,
            "dry_run": True,
            "run_dir": str(run_dir),
            "editor_dir": str(editor_dir),
            "repair_dir": str(repair_dir),
            "pose_dir": str(pose_dir),
            "output_dir": str(output_dir),
            "trigger": trigger,
            "grammar_profile": args.grammar_profile,
            "profiles": args.profiles,
            "record_count": len(prepared),
            "image_count": len(edited_paths),
            "records": [
                {
                    "image_key": item["image_key"],
                    "profile": item["profile"],
                    "semantic_source_stage": item["semantic_source_stage"],
                    "semantic_word_count": _word_count(item["semantic_caption"]),
                    "pose_corrections": item["renderer_input"]["pose_corrections"],
                }
                for item in prepared
            ],
        }
        _write_json(output_dir / "renderer.index.json", index)
        print(f"Compact renderer inputs: {output_dir}")
        print(
            f"Images: {len(edited_paths)}; render_records={len(prepared)}; "
            f"profiles={','.join(args.profiles)}; dry_run=yes"
        )
        return 0

    backend = resolve_backend(renderer_model_id, args.backend)
    if pending and backend == "vllm":
        install_text_only_vllm()

    loaded = None
    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    if pending:
        print(f"Loading {renderer_model_id} for trigger-bound training-caption rendering ...")
        loaded_started = time.perf_counter()
        loaded = load_model(
            renderer_model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(
            f"Loaded in {time.perf_counter() - loaded_started:.2f}s. "
            f"Rendering {len(pending)} caption(s). batch_size={args.batch_size} max_tokens={args.max_tokens}"
        )

    try:
        if loaded is not None and loaded.backend == "vllm":
            total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
            for batch_index, offset in enumerate(range(0, len(pending), args.batch_size), start=1):
                batch_started = time.perf_counter()
                batch = pending[offset : offset + args.batch_size]
                perf_items, batch_perf = editor_v01._generate_text_batch(
                    loaded,
                    [item["renderer_input"]["renderer_prompt"] for item in batch],
                    max_new_tokens=args.max_tokens,
                )
                for source, perf in zip(batch, perf_items):
                    rendered = str(perf.get("text") or "").strip()
                    if not rendered:
                        raise RuntimeError(f"Empty renderer output for {source['image_key']} {source['profile']}")
                    audit = quality_audit(
                        semantic_caption=source["semantic_caption"],
                        rendered_caption=rendered,
                        pose=source["pose"],
                        trigger=trigger,
                        profile=source["profile"],
                    )
                    payload = {
                        "schema_version": ARTIFACT_VERSION,
                        "image_key": source["image_key"],
                        "profile": source["profile"],
                        "model": renderer_model_id,
                        "backend": loaded.backend,
                        "trigger": trigger,
                        "grammar_profile": args.grammar_profile,
                        "semantic_source": source["semantic_source"],
                        "semantic_source_stage": source["semantic_source_stage"],
                        "semantic_caption": source["semantic_caption"],
                        "pose_corrections": source["renderer_input"]["pose_corrections"],
                        "rendered_caption": rendered,
                        "quality_audit": audit,
                        "performance": {
                            "batch_index": batch_index,
                            "batch_size": len(batch),
                            "prepare_seconds": perf.get("prepare_seconds"),
                            "shared_batch_generation_seconds": batch_perf["generation_seconds"],
                            "prompt_tokens": perf.get("prompt_tokens"),
                            "output_tokens": perf.get("output_tokens"),
                            "max_new_tokens": args.max_tokens,
                            "finish_reason": perf.get("finish_reason"),
                            "ttft_seconds": perf.get("ttft_seconds"),
                            "queue_seconds": perf.get("queue_seconds"),
                            "decode_seconds": perf.get("decode_seconds"),
                            "decode_tokens_per_second": perf.get("decode_tokens_per_second"),
                            "engine_e2e_seconds": perf.get("engine_e2e_seconds"),
                        },
                    }
                    _write_json(source["output_path"], payload)
                    generated.append(payload)
                    print(
                        "SEMANTIC_V3_RENDERER_PERF "
                        f"image={source['image_key']} profile={source['profile']} "
                        f"words={audit['rendered_word_count']} trigger_count={audit['trigger']['exact_trigger_count']} "
                        f"gate={'pass' if audit['passes_basic_gate'] else 'FAIL'} "
                        f"review={','.join(audit['ownership_review']['review_warnings']) or 'none'} "
                        f"finish={perf.get('finish_reason')}"
                    )

                batch_wall = time.perf_counter() - batch_started
                batch_runtime.append(
                    {
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "items": [f"{item['image_key']}:{item['profile']}" for item in batch],
                        "prepare_seconds": batch_perf["prepare_seconds"],
                        "generation_seconds": batch_perf["generation_seconds"],
                        "wall_seconds": batch_wall,
                        "amortized_seconds_per_render": batch_wall / len(batch) if batch else 0.0,
                        "prompt_tokens": batch_perf["prompt_tokens"],
                        "output_tokens": batch_perf["output_tokens"],
                        "aggregate_output_tokens_per_second": batch_perf["aggregate_output_tokens_per_second"],
                    }
                )
        elif loaded is not None:
            from .runner import generate_text

            for source in pending:
                rendered, inference_seconds = generate_text(
                    loaded,
                    source["renderer_input"]["renderer_prompt"],
                    max_new_tokens=args.max_tokens,
                )
                rendered = rendered.strip()
                audit = quality_audit(
                    semantic_caption=source["semantic_caption"],
                    rendered_caption=rendered,
                    pose=source["pose"],
                    trigger=trigger,
                    profile=source["profile"],
                )
                payload = {
                    "schema_version": ARTIFACT_VERSION,
                    "image_key": source["image_key"],
                    "profile": source["profile"],
                    "model": renderer_model_id,
                    "backend": loaded.backend,
                    "trigger": trigger,
                    "grammar_profile": args.grammar_profile,
                    "semantic_source": source["semantic_source"],
                    "semantic_source_stage": source["semantic_source_stage"],
                    "semantic_caption": source["semantic_caption"],
                    "pose_corrections": source["renderer_input"]["pose_corrections"],
                    "rendered_caption": rendered,
                    "quality_audit": audit,
                    "performance": {"inference_seconds": inference_seconds, "max_new_tokens": args.max_tokens},
                }
                _write_json(source["output_path"], payload)
                generated.append(payload)
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(reused + generated, key=lambda item: (str(item.get("image_key") or ""), str(item.get("profile") or "")))
    by_profile: dict[str, dict[str, Any]] = {}
    for profile in args.profiles:
        items = [item for item in records if item.get("profile") == profile]
        counts = [int((item.get("quality_audit") or {}).get("rendered_word_count") or 0) for item in items]
        by_profile[profile] = {
            "record_count": len(items),
            "gate_pass_count": sum(1 for item in items if (item.get("quality_audit") or {}).get("passes_basic_gate")),
            "review_warning_count": sum(
                1
                for item in items
                if ((item.get("quality_audit") or {}).get("ownership_review") or {}).get("review_warnings")
            ),
            "word_count_min": min(counts) if counts else None,
            "word_count_median": statistics.median(counts) if counts else None,
            "word_count_max": max(counts) if counts else None,
            "trigger_count_distribution": {
                str(count): sum(
                    1
                    for item in items
                    if int((((item.get("quality_audit") or {}).get("trigger") or {}).get("exact_trigger_count") or 0)) == count
                )
                for count in sorted(
                    {
                        int((((item.get("quality_audit") or {}).get("trigger") or {}).get("exact_trigger_count") or 0))
                        for item in items
                    }
                )
            },
        }

    index = {
        "schema_version": RUN_VERSION,
        "dry_run": False,
        "run_dir": str(run_dir),
        "editor_dir": str(editor_dir),
        "repair_dir": str(repair_dir),
        "pose_dir": str(pose_dir),
        "output_dir": str(output_dir),
        "model": renderer_model_id,
        "backend": backend,
        "trigger": trigger,
        "grammar_profile": args.grammar_profile,
        "profiles": args.profiles,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "record_count": len(records),
        "generated": len(generated),
        "reused": len(reused),
        "batch_runtime": batch_runtime,
        "run_wall_seconds": time.perf_counter() - run_started,
        "profile_summary": by_profile,
        "records": [
            {
                "image_key": item.get("image_key"),
                "profile": item.get("profile"),
                "semantic_source_stage": item.get("semantic_source_stage"),
                "quality_audit": item.get("quality_audit") or {},
                "performance": item.get("performance") or {},
            }
            for item in records
        ],
    }
    _write_json(output_dir / "renderer.index.json", index)
    print(f"Trigger-bound training captions: {output_dir}")
    print(
        f"Records: {len(records)}; generated={len(generated)}; reused={len(reused)}; "
        + "; ".join(
            f"{profile}:pass={summary['gate_pass_count']}/{summary['record_count']} "
            f"words={summary['word_count_min']}/{summary['word_count_median']}/{summary['word_count_max']}"
            for profile, summary in by_profile.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
