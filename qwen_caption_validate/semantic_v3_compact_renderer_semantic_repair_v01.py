from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from . import semantic_v3_compact_renderer_finish_v01 as finish_v01
from . import semantic_v3_compact_renderer_v01 as renderer_v01
from . import semantic_v3_rich_pose_editor as editor_v01
from .runner import load_model, model_slug, resolve_backend, resolve_model_id, unload_model
from .semantic_v3_text_only_bootstrap import install_text_only_vllm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_compact_renderer_semantic_repair_v01.txt"
ARTIFACT_VERSION = "semantic-v3-compact-renderer-semantic-repair-0.1"
RUN_VERSION = "semantic-v3-compact-renderer-semantic-repair-0.1-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-semantic-repair-v0.1"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 170

_LENGTH_WARNINGS = {
    "rendered_caption_above_target_words",
    "rendered_caption_below_target_words",
}

_read_json = finish_v01._read_json
_write_json = finish_v01._write_json
quality_audit = finish_v01.quality_audit


def _semantic_warnings(audit: dict[str, Any]) -> list[str]:
    return [
        str(value)
        for value in (audit.get("warnings") or [])
        if str(value) not in _LENGTH_WARNINGS
    ]


def _semantic_failure_lines(audit: dict[str, Any]) -> list[str]:
    lines = []
    for line in finish_v01._failure_lines(audit):
        if line.startswith("- Overlength:") or line.startswith("- Underlength:"):
            continue
        lines.append(line)
    if not lines and _semantic_warnings(audit):
        lines.append(
            "- Clear the listed semantic validation warning(s) with the smallest local edit possible."
        )
    return lines


def build_semantic_repair_input(
    *,
    budget_record: dict[str, Any],
    pose: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    profile = str(budget_record.get("profile") or "")
    if profile not in renderer_v01.PROFILE_BUDGETS:
        raise ValueError(f"Unknown renderer profile: {profile}")

    trigger = str(budget_record.get("trigger") or "").strip()
    grammar_profile = str(budget_record.get("grammar_profile") or "").strip()
    if grammar_profile not in renderer_v01.GRAMMAR_PROFILES:
        raise ValueError(f"Unknown grammar profile: {grammar_profile}")

    semantic_caption = str(budget_record.get("semantic_caption") or "").strip()
    current_caption = str(budget_record.get("budgeted_caption") or "").strip()
    if not semantic_caption or not current_caption or not trigger:
        raise ValueError(
            "Budget artifact missing semantic_caption, budgeted_caption, or trigger"
        )

    audit = quality_audit(
        semantic_caption=semantic_caption,
        rendered_caption=current_caption,
        pose=pose,
        trigger=trigger,
        profile=profile,
    )
    semantic_warnings = _semantic_warnings(audit)
    failure_lines = _semantic_failure_lines(audit)

    min_words, max_words = renderer_v01.PROFILE_BUDGETS[profile]
    grammar = renderer_v01.GRAMMAR_PROFILES[grammar_profile]
    corrections = [
        str(value).strip()
        for value in (pose.get("caption_ready_phrases") or [])
        if str(value).strip()
    ]
    pose_text = "\n".join(f"- {value}" for value in corrections) if corrections else "- None"
    failure_text = "\n".join(failure_lines) if failure_lines else "- None"

    prompt = (
        prompt_template.replace("{{PROFILE}}", profile)
        .replace("{{MIN_WORDS}}", str(min_words))
        .replace("{{MAX_WORDS}}", str(max_words))
        .replace("{{TRIGGER}}", trigger)
        .replace("{{SUBJECT_PRONOUN}}", grammar["subject_pronoun"])
        .replace("{{OBJECT_PRONOUN}}", grammar["object_pronoun"])
        .replace("{{POSSESSIVE_PRONOUN}}", grammar["possessive_pronoun"])
        .replace("{{REFLEXIVE_PRONOUN}}", grammar["reflexive_pronoun"])
        .replace("{{POSE_CORRECTIONS}}", pose_text)
        .replace("{{FAILURES}}", failure_text)
        .replace("{{CURRENT_CAPTION}}", current_caption)
    )

    return {
        "profile": profile,
        "trigger": trigger,
        "grammar_profile": grammar_profile,
        "semantic_caption": semantic_caption,
        "current_caption": current_caption,
        "pose_corrections": corrections,
        "initial_quality_audit": audit,
        "semantic_warnings": semantic_warnings,
        "semantic_failure_lines": failure_lines,
        "repair_prompt": prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conditionally repair only genuine semantic failures remaining after "
            "Semantic V3 deterministic compact/medium budgeting. Length alone never "
            "triggers model inference. No image analysis occurs."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--budget-dir", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--source-model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(renderer_v01.PROFILE_BUDGETS),
        default=["compact", "medium"],
    )
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["auto", "transformers", "vllm"],
        default="vllm",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
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

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    source_slug = model_slug(resolve_model_id(args.source_model))
    budget_dir = (
        args.budget_dir
        or (
            run_dir
            / "semantic-v3"
            / "compact-renderer-budget-v0.2"
            / source_slug
        )
    ).expanduser().resolve()
    pose_dir = (
        args.pose_dir or (run_dir / "semantic-v3" / "pose-language-v0.1")
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or (run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR / slug)
    ).expanduser().resolve()

    for label, path in (("budget", budget_dir), ("pose", pose_dir)):
        if not path.is_dir():
            print(f"{label.capitalize()} directory not found: {path}", file=sys.stderr)
            return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    if not prompt_path.is_file():
        print(f"Semantic repair prompt not found: {prompt_path}", file=sys.stderr)
        return 2
    prompt_template = prompt_path.read_text(encoding="utf-8")

    selected_keys = set(args.only)
    selected_profiles = set(args.profiles)
    source_paths = sorted(budget_dir.glob("*.budgeted_caption.json"))

    prepared: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    passthrough: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []

    for source_path in source_paths:
        source = _read_json(source_path)
        profile = str(source.get("profile") or "")
        key = str(source.get("image_key") or "")
        if profile not in selected_profiles:
            continue
        if selected_keys and key not in selected_keys:
            continue

        pose_path = pose_dir / f"{key}.pose_language.json"
        if not pose_path.is_file():
            print(f"Missing Pose language: {pose_path}", file=sys.stderr)
            return 2
        pose = _read_json(pose_path)
        repair_input = build_semantic_repair_input(
            budget_record=source,
            pose=pose,
            prompt_template=prompt_template,
        )
        input_path = output_dir / f"{key}.{profile}.semantic_repair_input.json"
        _write_json(
            input_path,
            {
                "schema_version": ARTIFACT_VERSION + "-input",
                "image_key": key,
                "profile": profile,
                "budget_source": str(source_path),
                "pose_source": str(pose_path),
                "prompt_template": str(prompt_path),
                **repair_input,
            },
        )

        output_path = output_dir / f"{key}.{profile}.semantic_repaired_caption.json"
        record = {
            "image_key": key,
            "profile": profile,
            "source_path": source_path,
            "source": source,
            "pose": pose,
            "repair_input": repair_input,
            "output_path": output_path,
        }
        prepared.append(record)

        needs_repair = bool(repair_input["semantic_warnings"])
        if args.dry_run:
            continue

        if not needs_repair:
            current = str(source.get("budgeted_caption") or "")
            audit = repair_input["initial_quality_audit"]
            payload = {
                "schema_version": ARTIFACT_VERSION,
                "image_key": key,
                "profile": profile,
                "model": source.get("renderer_model") or source.get("model"),
                "backend": source.get("renderer_backend") or source.get("backend"),
                "trigger": source.get("trigger"),
                "grammar_profile": source.get("grammar_profile"),
                "budget_source": str(source_path),
                "pose_source": str(pose_path),
                "initial_caption": current,
                "final_caption": current,
                "repair_action": "passthrough",
                "initial_semantic_warnings": [],
                "final_semantic_warnings": [],
                "initial_quality_audit": audit,
                "quality_audit": audit,
                "performance": {"generated": False},
            }
            _write_json(output_path, payload)
            passthrough.append(payload)
        elif args.overwrite or not output_path.exists():
            pending.append(record)
        else:
            reused.append(_read_json(output_path))

    if not prepared:
        print("No matching budget artifacts found.", file=sys.stderr)
        return 2

    if args.dry_run:
        candidates = [
            item
            for item in prepared
            if item["repair_input"]["semantic_warnings"]
        ]
        index = {
            "schema_version": RUN_VERSION,
            "dry_run": True,
            "run_dir": str(run_dir),
            "budget_dir": str(budget_dir),
            "pose_dir": str(pose_dir),
            "output_dir": str(output_dir),
            "profiles": args.profiles,
            "record_count": len(prepared),
            "semantic_repair_candidate_count": len(candidates),
            "records": [
                {
                    "image_key": item["image_key"],
                    "profile": item["profile"],
                    "semantic_warnings": item["repair_input"]["semantic_warnings"],
                    "semantic_failure_lines": item["repair_input"]["semantic_failure_lines"],
                    "initial_quality_audit": item["repair_input"]["initial_quality_audit"],
                }
                for item in prepared
            ],
        }
        _write_json(output_dir / "semantic_repair.index.json", index)
        print(f"Post-budget semantic repair inputs: {output_dir}")
        print(
            f"Records: {len(prepared)}; semantic_repair_candidates={len(candidates)}; dry_run=yes"
        )
        return 0

    backend = resolve_backend(model_id, args.backend)
    if pending and backend == "vllm":
        install_text_only_vllm()

    loaded = None
    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    if pending:
        print(f"Loading {model_id} for post-budget semantic repair ...")
        load_started = time.perf_counter()
        loaded = load_model(
            model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(
            f"Loaded in {time.perf_counter() - load_started:.2f}s. "
            f"Repairing {len(pending)} semantic failure(s). "
            f"batch_size={args.batch_size} max_tokens={args.max_tokens}"
        )

    try:
        if loaded is not None and loaded.backend == "vllm":
            for batch_index, offset in enumerate(
                range(0, len(pending), args.batch_size), start=1
            ):
                batch_started = time.perf_counter()
                batch = pending[offset : offset + args.batch_size]
                perf_items, batch_perf = editor_v01._generate_text_batch(
                    loaded,
                    [item["repair_input"]["repair_prompt"] for item in batch],
                    max_new_tokens=args.max_tokens,
                )
                for source, perf in zip(batch, perf_items):
                    repaired = str(perf.get("text") or "").strip()
                    if not repaired:
                        raise RuntimeError(
                            f"Empty semantic repair output for {source['image_key']} {source['profile']}"
                        )
                    src = source["source"]
                    audit = quality_audit(
                        semantic_caption=str(src.get("semantic_caption") or ""),
                        rendered_caption=repaired,
                        pose=source["pose"],
                        trigger=str(src.get("trigger") or ""),
                        profile=source["profile"],
                    )
                    final_semantic = _semantic_warnings(audit)
                    payload = {
                        "schema_version": ARTIFACT_VERSION,
                        "image_key": source["image_key"],
                        "profile": source["profile"],
                        "model": model_id,
                        "backend": loaded.backend,
                        "trigger": src.get("trigger"),
                        "grammar_profile": src.get("grammar_profile"),
                        "budget_source": str(source["source_path"]),
                        "pose_source": str(
                            pose_dir / f"{source['image_key']}.pose_language.json"
                        ),
                        "initial_caption": src.get("budgeted_caption"),
                        "final_caption": repaired,
                        "repair_action": "generated",
                        "initial_semantic_warnings": source["repair_input"]["semantic_warnings"],
                        "final_semantic_warnings": final_semantic,
                        "initial_quality_audit": source["repair_input"]["initial_quality_audit"],
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
                            "decode_seconds": perf.get("decode_seconds"),
                            "decode_tokens_per_second": perf.get("decode_tokens_per_second"),
                            "engine_e2e_seconds": perf.get("engine_e2e_seconds"),
                        },
                    }
                    _write_json(source["output_path"], payload)
                    generated.append(payload)
                    print(
                        "SEMANTIC_V3_POST_BUDGET_REPAIR_PERF "
                        f"image={source['image_key']} profile={source['profile']} "
                        f"words={audit['rendered_word_count']} "
                        f"semantic_gate={'pass' if not final_semantic else 'FAIL'} "
                        f"full_gate={'pass' if audit['passes_basic_gate'] else 'FAIL'} "
                        f"finish={perf.get('finish_reason')}"
                    )

                batch_wall = time.perf_counter() - batch_started
                batch_runtime.append(
                    {
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "items": [
                            f"{item['image_key']}:{item['profile']}" for item in batch
                        ],
                        "generation_seconds": batch_perf["generation_seconds"],
                        "wall_seconds": batch_wall,
                        "amortized_seconds_per_repair": (
                            batch_wall / len(batch) if batch else 0.0
                        ),
                        "prompt_tokens": batch_perf["prompt_tokens"],
                        "output_tokens": batch_perf["output_tokens"],
                        "aggregate_output_tokens_per_second": batch_perf[
                            "aggregate_output_tokens_per_second"
                        ],
                    }
                )
        elif loaded is not None:
            from .runner import generate_text

            for source in pending:
                repaired, inference_seconds = generate_text(
                    loaded,
                    source["repair_input"]["repair_prompt"],
                    max_new_tokens=args.max_tokens,
                )
                repaired = repaired.strip()
                src = source["source"]
                audit = quality_audit(
                    semantic_caption=str(src.get("semantic_caption") or ""),
                    rendered_caption=repaired,
                    pose=source["pose"],
                    trigger=str(src.get("trigger") or ""),
                    profile=source["profile"],
                )
                final_semantic = _semantic_warnings(audit)
                payload = {
                    "schema_version": ARTIFACT_VERSION,
                    "image_key": source["image_key"],
                    "profile": source["profile"],
                    "model": model_id,
                    "backend": loaded.backend,
                    "trigger": src.get("trigger"),
                    "grammar_profile": src.get("grammar_profile"),
                    "budget_source": str(source["source_path"]),
                    "initial_caption": src.get("budgeted_caption"),
                    "final_caption": repaired,
                    "repair_action": "generated",
                    "initial_semantic_warnings": source["repair_input"]["semantic_warnings"],
                    "final_semantic_warnings": final_semantic,
                    "initial_quality_audit": source["repair_input"]["initial_quality_audit"],
                    "quality_audit": audit,
                    "performance": {
                        "inference_seconds": inference_seconds,
                        "max_new_tokens": args.max_tokens,
                    },
                }
                _write_json(source["output_path"], payload)
                generated.append(payload)
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(
        passthrough + reused + generated,
        key=lambda item: (
            str(item.get("image_key") or ""),
            str(item.get("profile") or ""),
        ),
    )
    by_profile: dict[str, dict[str, Any]] = {}
    for profile in args.profiles:
        items = [item for item in records if item.get("profile") == profile]
        counts = [
            int((item.get("quality_audit") or {}).get("rendered_word_count") or 0)
            for item in items
        ]
        by_profile[profile] = {
            "record_count": len(items),
            "generated_count": sum(
                1 for item in items if item.get("repair_action") == "generated"
            ),
            "passthrough_count": sum(
                1 for item in items if item.get("repair_action") == "passthrough"
            ),
            "semantic_pass_count": sum(
                1 for item in items if not (item.get("final_semantic_warnings") or [])
            ),
            "full_gate_pass_count": sum(
                1
                for item in items
                if (item.get("quality_audit") or {}).get("passes_basic_gate")
            ),
            "word_count_min": min(counts) if counts else None,
            "word_count_median": statistics.median(counts) if counts else None,
            "word_count_max": max(counts) if counts else None,
        }

    index = {
        "schema_version": RUN_VERSION,
        "dry_run": False,
        "run_dir": str(run_dir),
        "budget_dir": str(budget_dir),
        "pose_dir": str(pose_dir),
        "output_dir": str(output_dir),
        "model": model_id,
        "backend": backend,
        "profiles": args.profiles,
        "record_count": len(records),
        "generated": len(generated),
        "passthrough": len(passthrough),
        "reused": len(reused),
        "batch_runtime": batch_runtime,
        "run_wall_seconds": time.perf_counter() - run_started,
        "profile_summary": by_profile,
        "records": [
            {
                "image_key": item.get("image_key"),
                "profile": item.get("profile"),
                "repair_action": item.get("repair_action"),
                "initial_semantic_warnings": item.get("initial_semantic_warnings") or [],
                "final_semantic_warnings": item.get("final_semantic_warnings") or [],
                "quality_audit": item.get("quality_audit") or {},
                "performance": item.get("performance") or {},
            }
            for item in records
        ],
    }
    _write_json(output_dir / "semantic_repair.index.json", index)

    print(f"Post-budget semantic repair captions: {output_dir}")
    print(
        f"Records: {len(records)}; generated={len(generated)}; "
        f"passthrough={len(passthrough)}; reused={len(reused)}; "
        + "; ".join(
            f"{profile}:semantic_pass={summary['semantic_pass_count']}/{summary['record_count']} "
            f"full_pass={summary['full_gate_pass_count']}/{summary['record_count']} "
            f"words={summary['word_count_min']}/{summary['word_count_median']}/{summary['word_count_max']}"
            for profile, summary in by_profile.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
