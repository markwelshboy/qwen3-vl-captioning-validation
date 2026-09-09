from __future__ import annotations

import argparse
import copy
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from . import semantic_v3_compact_renderer_semantic_repair_v01 as v01
from . import semantic_v3_compact_renderer_v01 as renderer_v01
from . import semantic_v3_rich_pose_editor as editor_v01
from .runner import load_model, model_slug, resolve_backend, resolve_model_id, unload_model
from .semantic_v3_text_only_bootstrap import install_text_only_vllm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_compact_renderer_semantic_repair_v02.txt"
ARTIFACT_VERSION = "semantic-v3-compact-renderer-semantic-repair-0.2"
RUN_VERSION = "semantic-v3-compact-renderer-semantic-repair-0.2-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-semantic-repair-v0.2"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 256

_read_json = v01._read_json
_write_json = v01._write_json
quality_audit = v01.quality_audit
_semantic_warnings = v01._semantic_warnings

_SIDE_RE = re.compile(r"^(?:left|right)\s+", re.IGNORECASE)
_HAIR_LENGTH_LOCATION_RE = re.compile(
    r"^(?:falls?|hangs?|reaches?|extends?)\b.*\b(?:shoulders?|neck|chest|waist)\b",
    re.IGNORECASE,
)


def _replace_exact_phrase(text: str, phrase: str, replacement: str) -> tuple[str, bool]:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    updated, count = pattern.subn(replacement, text, count=1)
    return updated, bool(count)


def _cleanup_after_local_deletion(text: str) -> str:
    # The protected hair-length phrase often sits between the noun "hair" and a
    # valid transient participial clause: "hair [LEAK], partially framing ...".
    # Preserve that transient state without leaving a dangling participle.
    text = re.sub(
        r"\bhair\s*,\s*partially\s+framing\b",
        "hair, with strands partially framing",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bhair\s*,\s*framing\b",
        "hair, with strands framing",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def deterministic_local_repair(
    *,
    caption: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Apply only exact, semantically safe local edits identified by the audit.

    This stage deliberately refuses broad rewriting. If a warning cannot be repaired
    by an exact safe edit, it remains for the text-model fallback.
    """
    current = caption
    edits: list[dict[str, str]] = []
    semantic = audit.get("semantic_policy_audit") or {}

    # Safe rule 1: neutralize only the exact side word in an unauthorized anatomical
    # phrase, e.g. "right hand" -> "hand". The audit already decides which phrase is
    # unauthorized, so authorized "left fist" elsewhere is untouched.
    for phrase in semantic.get("unauthorized_anatomical_laterality") or []:
        phrase = str(phrase).strip()
        neutral = _SIDE_RE.sub("", phrase).strip()
        if not phrase or neutral == phrase or not neutral:
            continue
        updated, changed = _replace_exact_phrase(current, phrase, neutral)
        if changed:
            edits.append(
                {
                    "kind": "neutralize_unsupported_laterality",
                    "before": phrase,
                    "after": neutral,
                }
            )
            current = updated

    # Safe rule 2: remove only a verb phrase whose audited identity leak is a hair
    # length/location claim (shoulders/neck/chest/waist). Transient state such as
    # tousled/damp/face-framing remains. The local grammar cleanup is intentionally
    # tiny and only handles the participial shape produced by deleting such a clause.
    for phrase in semantic.get("identity_leaks") or []:
        phrase = str(phrase).strip()
        if not phrase or not _HAIR_LENGTH_LOCATION_RE.search(phrase):
            continue
        updated, changed = _replace_exact_phrase(current, phrase, "")
        if changed:
            updated = _cleanup_after_local_deletion(updated)
            edits.append(
                {
                    "kind": "remove_hair_length_location",
                    "before": phrase,
                    "after": "",
                }
            )
            current = updated

    return {
        "caption": current,
        "edits": edits,
        "changed": current != caption,
    }


def build_model_repair_input(
    *,
    budget_record: dict[str, Any],
    pose: dict[str, Any],
    current_caption: str,
    prompt_template: str,
) -> dict[str, Any]:
    modified = copy.deepcopy(budget_record)
    modified["budgeted_caption"] = current_caption
    return v01.build_semantic_repair_input(
        budget_record=modified,
        pose=pose,
        prompt_template=prompt_template,
    )


def _payload_base(
    *,
    source: dict[str, Any],
    source_path: Path,
    pose_path: Path,
    initial_caption: str,
    final_caption: str,
    action: str,
    initial_audit: dict[str, Any],
    final_audit: dict[str, Any],
    deterministic_edits: list[dict[str, str]],
    deterministic_caption: str,
    model: str | None,
    backend: str | None,
    performance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_VERSION,
        "image_key": source.get("image_key"),
        "profile": source.get("profile"),
        "model": model,
        "backend": backend,
        "trigger": source.get("trigger"),
        "grammar_profile": source.get("grammar_profile"),
        "budget_source": str(source_path),
        "pose_source": str(pose_path),
        "initial_caption": initial_caption,
        "deterministic_caption": deterministic_caption,
        "deterministic_edits": deterministic_edits,
        "final_caption": final_caption,
        "repair_action": action,
        "initial_semantic_warnings": _semantic_warnings(initial_audit),
        "final_semantic_warnings": _semantic_warnings(final_audit),
        "initial_quality_audit": initial_audit,
        "quality_audit": final_audit,
        "performance": performance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic-first post-budget semantic repair. Exact safe local edits "
            "are applied and re-audited before any model is loaded; only unresolved "
            "semantic failures reach the text-model fallback. Length never triggers repair."
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

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    source_slug = model_slug(resolve_model_id(args.source_model))
    budget_dir = (
        args.budget_dir
        or run_dir / "semantic-v3" / "compact-renderer-budget-v0.2" / source_slug
    ).expanduser().resolve()
    pose_dir = (
        args.pose_dir or run_dir / "semantic-v3" / "pose-language-v0.1"
    ).expanduser().resolve()
    output_dir = (
        args.output_dir or run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR / slug
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
    finished: list[dict[str, Any]] = []
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
        initial_caption = str(source.get("budgeted_caption") or "").strip()
        semantic_caption = str(source.get("semantic_caption") or "").strip()
        trigger = str(source.get("trigger") or "").strip()
        if not initial_caption or not semantic_caption or not trigger:
            print(f"Incomplete budget artifact: {source_path}", file=sys.stderr)
            return 2

        initial_audit = quality_audit(
            semantic_caption=semantic_caption,
            rendered_caption=initial_caption,
            pose=pose,
            trigger=trigger,
            profile=profile,
        )
        initial_semantic = _semantic_warnings(initial_audit)
        deterministic = deterministic_local_repair(
            caption=initial_caption,
            audit=initial_audit,
        )
        deterministic_caption = str(deterministic["caption"])
        deterministic_audit = quality_audit(
            semantic_caption=semantic_caption,
            rendered_caption=deterministic_caption,
            pose=pose,
            trigger=trigger,
            profile=profile,
        )
        remaining_semantic = _semantic_warnings(deterministic_audit)
        model_input = build_model_repair_input(
            budget_record=source,
            pose=pose,
            current_caption=deterministic_caption,
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
                "initial_caption": initial_caption,
                "initial_semantic_warnings": initial_semantic,
                "deterministic_caption": deterministic_caption,
                "deterministic_edits": deterministic["edits"],
                "deterministic_semantic_warnings": remaining_semantic,
                "model_repair_required": bool(remaining_semantic),
                "model_repair_prompt": model_input["repair_prompt"] if remaining_semantic else None,
            },
        )

        output_path = output_dir / f"{key}.{profile}.semantic_repaired_caption.json"
        record = {
            "image_key": key,
            "profile": profile,
            "source": source,
            "source_path": source_path,
            "pose": pose,
            "pose_path": pose_path,
            "initial_caption": initial_caption,
            "initial_audit": initial_audit,
            "deterministic_caption": deterministic_caption,
            "deterministic_edits": deterministic["edits"],
            "deterministic_audit": deterministic_audit,
            "remaining_semantic": remaining_semantic,
            "model_input": model_input,
            "output_path": output_path,
        }
        prepared.append(record)

        if args.dry_run:
            continue
        if output_path.exists() and not args.overwrite:
            reused.append(_read_json(output_path))
            continue

        if not initial_semantic:
            payload = _payload_base(
                source=source,
                source_path=source_path,
                pose_path=pose_path,
                initial_caption=initial_caption,
                final_caption=initial_caption,
                action="passthrough",
                initial_audit=initial_audit,
                final_audit=initial_audit,
                deterministic_edits=[],
                deterministic_caption=initial_caption,
                model=source.get("renderer_model") or source.get("model"),
                backend=source.get("renderer_backend") or source.get("backend"),
                performance={"generated": False},
            )
            _write_json(output_path, payload)
            finished.append(payload)
        elif not remaining_semantic:
            payload = _payload_base(
                source=source,
                source_path=source_path,
                pose_path=pose_path,
                initial_caption=initial_caption,
                final_caption=deterministic_caption,
                action="deterministic",
                initial_audit=initial_audit,
                final_audit=deterministic_audit,
                deterministic_edits=deterministic["edits"],
                deterministic_caption=deterministic_caption,
                model=None,
                backend="deterministic",
                performance={"generated": False},
            )
            _write_json(output_path, payload)
            finished.append(payload)
        else:
            pending.append(record)

    if not prepared:
        print("No matching budget artifacts found.", file=sys.stderr)
        return 2

    if args.dry_run:
        initial_candidates = [item for item in prepared if _semantic_warnings(item["initial_audit"])]
        deterministic_resolved = [
            item
            for item in initial_candidates
            if not item["remaining_semantic"]
        ]
        model_candidates = [item for item in initial_candidates if item["remaining_semantic"]]
        index = {
            "schema_version": RUN_VERSION,
            "dry_run": True,
            "record_count": len(prepared),
            "initial_semantic_candidate_count": len(initial_candidates),
            "deterministic_resolved_count": len(deterministic_resolved),
            "model_repair_candidate_count": len(model_candidates),
            "records": [
                {
                    "image_key": item["image_key"],
                    "profile": item["profile"],
                    "initial_semantic_warnings": _semantic_warnings(item["initial_audit"]),
                    "deterministic_edits": item["deterministic_edits"],
                    "remaining_semantic_warnings": item["remaining_semantic"],
                }
                for item in prepared
            ],
        }
        _write_json(output_dir / "semantic_repair.index.json", index)
        print(f"Deterministic-first semantic repair inputs: {output_dir}")
        print(
            f"Records: {len(prepared)}; initial_candidates={len(initial_candidates)}; "
            f"deterministic_resolved={len(deterministic_resolved)}; "
            f"model_candidates={len(model_candidates)}; dry_run=yes"
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
        print(f"Loading {model_id} for unresolved post-budget semantic repair ...")
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
            f"Repairing {len(pending)} unresolved caption(s). "
            f"batch_size={args.batch_size} max_tokens={args.max_tokens}"
        )

    try:
        if loaded is not None and loaded.backend == "vllm":
            for batch_index, offset in enumerate(range(0, len(pending), args.batch_size), start=1):
                batch_started = time.perf_counter()
                batch = pending[offset : offset + args.batch_size]
                perf_items, batch_perf = editor_v01._generate_text_batch(
                    loaded,
                    [item["model_input"]["repair_prompt"] for item in batch],
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
                    payload = _payload_base(
                        source=src,
                        source_path=source["source_path"],
                        pose_path=source["pose_path"],
                        initial_caption=source["initial_caption"],
                        final_caption=repaired,
                        action="generated_after_deterministic",
                        initial_audit=source["initial_audit"],
                        final_audit=audit,
                        deterministic_edits=source["deterministic_edits"],
                        deterministic_caption=source["deterministic_caption"],
                        model=model_id,
                        backend=loaded.backend,
                        performance={
                            "batch_index": batch_index,
                            "batch_size": len(batch),
                            "prompt_tokens": perf.get("prompt_tokens"),
                            "output_tokens": perf.get("output_tokens"),
                            "max_new_tokens": args.max_tokens,
                            "finish_reason": perf.get("finish_reason"),
                            "shared_batch_generation_seconds": batch_perf["generation_seconds"],
                        },
                    )
                    _write_json(source["output_path"], payload)
                    generated.append(payload)
                    print(
                        "SEMANTIC_V3_POST_BUDGET_REPAIR_V02 "
                        f"image={source['image_key']} profile={source['profile']} "
                        f"semantic_gate={'pass' if not _semantic_warnings(audit) else 'FAIL'} "
                        f"full_gate={'pass' if audit['passes_basic_gate'] else 'FAIL'} "
                        f"finish={perf.get('finish_reason')}"
                    )
                batch_runtime.append(
                    {
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "generation_seconds": batch_perf["generation_seconds"],
                        "wall_seconds": time.perf_counter() - batch_started,
                    }
                )
        elif loaded is not None:
            from .runner import generate_text
            for source in pending:
                repaired, inference_seconds = generate_text(
                    loaded,
                    source["model_input"]["repair_prompt"],
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
                payload = _payload_base(
                    source=src,
                    source_path=source["source_path"],
                    pose_path=source["pose_path"],
                    initial_caption=source["initial_caption"],
                    final_caption=repaired,
                    action="generated_after_deterministic",
                    initial_audit=source["initial_audit"],
                    final_audit=audit,
                    deterministic_edits=source["deterministic_edits"],
                    deterministic_caption=source["deterministic_caption"],
                    model=model_id,
                    backend=loaded.backend,
                    performance={"inference_seconds": inference_seconds, "max_new_tokens": args.max_tokens},
                )
                _write_json(source["output_path"], payload)
                generated.append(payload)
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(
        finished + reused + generated,
        key=lambda item: (str(item.get("image_key") or ""), str(item.get("profile") or "")),
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
            "passthrough_count": sum(item.get("repair_action") == "passthrough" for item in items),
            "deterministic_count": sum(item.get("repair_action") == "deterministic" for item in items),
            "generated_count": sum(item.get("repair_action") == "generated_after_deterministic" for item in items),
            "semantic_pass_count": sum(not (item.get("final_semantic_warnings") or []) for item in items),
            "full_gate_pass_count": sum(bool((item.get("quality_audit") or {}).get("passes_basic_gate")) for item in items),
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
                "deterministic_edits": item.get("deterministic_edits") or [],
                "final_semantic_warnings": item.get("final_semantic_warnings") or [],
                "quality_audit": item.get("quality_audit") or {},
                "performance": item.get("performance") or {},
            }
            for item in records
        ],
    }
    _write_json(output_dir / "semantic_repair.index.json", index)

    print(f"Deterministic-first post-budget semantic repair captions: {output_dir}")
    print(
        f"Records: {len(records)}; generated={len(generated)}; reused={len(reused)}; "
        + "; ".join(
            f"{profile}:semantic_pass={summary['semantic_pass_count']}/{summary['record_count']} "
            f"full_pass={summary['full_gate_pass_count']}/{summary['record_count']} "
            f"passthrough={summary['passthrough_count']} deterministic={summary['deterministic_count']} "
            f"generated={summary['generated_count']} words={summary['word_count_min']}/{summary['word_count_median']}/{summary['word_count_max']}"
            for profile, summary in by_profile.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
