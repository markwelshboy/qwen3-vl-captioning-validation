from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from . import semantic_v3_compact_renderer_v01 as renderer_v01
from . import semantic_v3_compact_renderer_v02 as renderer_v02
from . import semantic_v3_rich_pose_editor as editor_v01
from .runner import load_model, model_slug, resolve_backend, resolve_model_id, unload_model
from .semantic_v3_text_only_bootstrap import install_text_only_vllm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_compact_renderer_finish_v01.txt"
ARTIFACT_VERSION = "semantic-v3-compact-renderer-finish-0.1"
RUN_VERSION = "semantic-v3-compact-renderer-finish-0.1-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-finish-v0.1"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 190

PROFILE_PREFERRED_RANGES: dict[str, tuple[int, int]] = {
    "compact": (70, 82),
    "medium": (112, 128),
}

_BARE_HAIR_COLOR = {
    "blond", "blonde", "brunette", "auburn", "ginger", "red", "brown",
    "black", "gray", "grey", "silver", "white", "dark", "light",
}


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


def _explicit_hair_color(text: str, color: str) -> bool:
    escaped = re.escape(color)
    patterns = (
        rf"\b{escaped}\s+hair\b",
        rf"\bhair\s+(?:is|appears|looks)\s+{escaped}\b",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _filtered_semantic_audit(
    semantic_caption: str,
    rendered_caption: str,
    pose: dict[str, Any],
) -> dict[str, Any]:
    """Reuse frozen semantic policy but suppress bare hair-color false positives
    unless the color is explicitly applied to hair.

    The editor audit intentionally used generous local hair context while calibrating
    long captions. During rendering, valid nouns such as "black headband" can occur
    in the same short sentence as "damp hair", which can make a bare color token look
    like hair identity. A bare color is a renderer leak only when the prose actually
    applies that color to hair.
    """
    base = renderer_v01.repair_v07.quality_audit(semantic_caption, rendered_caption, pose)

    identity = [str(v) for v in (base.get("identity_leaks") or [])]
    filtered_identity: list[str] = []
    for value in identity:
        clean = value.strip()
        if clean.lower() in _BARE_HAIR_COLOR and not _explicit_hair_color(rendered_caption, clean):
            continue
        filtered_identity.append(clean)

    warnings = [
        str(v)
        for v in (base.get("warnings") or [])
        if str(v) != "intrinsic_identity_leakage"
    ]
    generic = [str(v) for v in (base.get("generic_identity_paraphrase_leaks") or [])]
    dye = [str(v) for v in (base.get("hair_dye_detail_leaks") or [])]
    if filtered_identity or generic or dye:
        warnings.append("intrinsic_identity_leakage")

    base["identity_leaks"] = list(dict.fromkeys(filtered_identity))
    base["warnings"] = list(dict.fromkeys(warnings))
    base["passes_basic_gate"] = not base["warnings"]
    return base


def quality_audit(
    *,
    semantic_caption: str,
    rendered_caption: str,
    pose: dict[str, Any],
    trigger: str,
    profile: str,
) -> dict[str, Any]:
    min_words, max_words = renderer_v01.PROFILE_BUDGETS[profile]
    word_count = _word_count(rendered_caption)
    source_words = _word_count(semantic_caption)
    word_ratio = word_count / source_words if source_words else 0.0

    trigger_audit = renderer_v02._trigger_audit(rendered_caption, trigger)
    ownership = renderer_v01._ownership_review(rendered_caption, trigger_audit["exact_trigger_count"])
    semantic = _filtered_semantic_audit(semantic_caption, rendered_caption, pose)
    semantic_policy_warnings = [
        str(value)
        for value in (semantic.get("warnings") or [])
        if str(value) not in renderer_v01._EXPECTED_COMPRESSION_WARNINGS
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
        "preferred_word_range": {
            "min": PROFILE_PREFERRED_RANGES[profile][0],
            "max": PROFILE_PREFERRED_RANGES[profile][1],
        },
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


def _failure_lines(audit: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    warnings = set(str(v) for v in (audit.get("warnings") or []))

    target = audit.get("target_word_range") or {}
    preferred = audit.get("preferred_word_range") or {}
    if "rendered_caption_above_target_words" in warnings:
        lines.append(
            f"- Overlength: current caption is {audit.get('rendered_word_count')} words; "
            f"final allowed maximum is {target.get('max')}. Delete lower-priority facts "
            f"until it lands around {preferred.get('min')}-{preferred.get('max')} words."
        )
    if "rendered_caption_below_target_words" in warnings:
        lines.append(
            f"- Underlength: current caption is {audit.get('rendered_word_count')} words; "
            f"final minimum is {target.get('min')}. Add only high-value facts already present "
            f"in the validated semantic source, aiming around {preferred.get('min')}-{preferred.get('max')} words."
        )

    trigger = audit.get("trigger") or {}
    for value in trigger.get("warnings") or []:
        lines.append(f"- Trigger binding failure: {value}.")

    semantic = audit.get("semantic_policy_audit") or {}
    identity = [str(v) for v in (semantic.get("identity_leaks") or []) if str(v).strip()]
    generic = [str(v) for v in (semantic.get("generic_identity_paraphrase_leaks") or []) if str(v).strip()]
    dye = [str(v) for v in (semantic.get("hair_dye_detail_leaks") or []) if str(v).strip()]
    age = [str(v) for v in (semantic.get("age_proxy_identity_leaks") or []) if str(v).strip()]
    laterality = [str(v) for v in (semantic.get("unauthorized_anatomical_laterality") or []) if str(v).strip()]
    conflicts = [str(v) for v in (semantic.get("conflicting_pose_wording") or []) if str(v).strip()]
    awkward = [str(v) for v in (semantic.get("awkward_haircut_residue") or []) if str(v).strip()]
    bad_attach = [str(v) for v in (semantic.get("bad_local_attachment") or []) if str(v).strip()]

    if identity or generic or dye:
        values = list(dict.fromkeys(identity + generic + dye))
        lines.append("- Protected primary-subject identity wording: " + "; ".join(values) + ". Remove only that identity meaning.")
    if age:
        lines.append("- Apparent-age proxy wording: " + "; ".join(age) + ". Delete these age cues.")
    if laterality:
        lines.append(
            "- Unsupported anatomical laterality: "
            + "; ".join(laterality)
            + ". Remove only the unsupported side word; keep the action/body-part relation in neutral wording."
        )
    if conflicts:
        lines.append("- Pose wording conflicts with governed Pose: " + "; ".join(conflicts) + ". Keep the governed Pose formulation.")
    if awkward:
        lines.append("- Awkward haircut residue: " + "; ".join(awkward) + ". Repair only the local hair grammar.")
    if bad_attach:
        lines.append("- Bad local grammatical attachment: " + "; ".join(bad_attach) + ". Repair only the local attachment.")

    if not lines:
        lines.append("- Re-render conservatively to clear the deterministic gate without adding detail.")
    return lines


def build_finish_input(
    *,
    renderer_record: dict[str, Any],
    pose: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    profile = str(renderer_record.get("profile") or "")
    if profile not in renderer_v01.PROFILE_BUDGETS:
        raise ValueError(f"Unknown renderer profile: {profile}")
    trigger = str(renderer_record.get("trigger") or "").strip()
    grammar_profile = str(renderer_record.get("grammar_profile") or "").strip()
    if grammar_profile not in renderer_v01.GRAMMAR_PROFILES:
        raise ValueError(f"Unknown grammar profile: {grammar_profile}")
    semantic_caption = str(renderer_record.get("semantic_caption") or "").strip()
    current_caption = str(renderer_record.get("rendered_caption") or "").strip()
    if not semantic_caption or not current_caption:
        raise ValueError("Renderer artifact is missing semantic_caption or rendered_caption")

    audit = quality_audit(
        semantic_caption=semantic_caption,
        rendered_caption=current_caption,
        pose=pose,
        trigger=trigger,
        profile=profile,
    )
    min_words, max_words = renderer_v01.PROFILE_BUDGETS[profile]
    preferred_min, preferred_max = PROFILE_PREFERRED_RANGES[profile]
    grammar = renderer_v01.GRAMMAR_PROFILES[grammar_profile]
    corrections = [
        str(v).strip()
        for v in (pose.get("caption_ready_phrases") or [])
        if str(v).strip()
    ]
    pose_text = "\n".join(f"- {v}" for v in corrections) if corrections else "- None"
    failure_text = "\n".join(_failure_lines(audit))

    prompt = (
        prompt_template.replace("{{PROFILE}}", profile)
        .replace("{{MIN_WORDS}}", str(min_words))
        .replace("{{MAX_WORDS}}", str(max_words))
        .replace("{{PREFERRED_MIN_WORDS}}", str(preferred_min))
        .replace("{{PREFERRED_MAX_WORDS}}", str(preferred_max))
        .replace("{{TRIGGER}}", trigger)
        .replace("{{SUBJECT_PRONOUN}}", grammar["subject_pronoun"])
        .replace("{{OBJECT_PRONOUN}}", grammar["object_pronoun"])
        .replace("{{POSSESSIVE_PRONOUN}}", grammar["possessive_pronoun"])
        .replace("{{REFLEXIVE_PRONOUN}}", grammar["reflexive_pronoun"])
        .replace("{{POSE_CORRECTIONS}}", pose_text)
        .replace("{{FAILURES}}", failure_text)
        .replace("{{CURRENT_CAPTION}}", current_caption)
        .replace("{{SEMANTIC_CAPTION}}", semantic_caption)
    )
    return {
        "profile": profile,
        "trigger": trigger,
        "grammar_profile": grammar_profile,
        "semantic_caption": semantic_caption,
        "current_caption": current_caption,
        "pose_corrections": corrections,
        "initial_quality_audit": audit,
        "hard_failures": _failure_lines(audit),
        "finish_prompt": prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conditionally finish Semantic V3 compact-renderer outputs that miss the hard "
            "word/trigger/semantic gate. Text-only; no image analysis occurs."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--renderer-dir", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--source-model", default="32b-fp8")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--profiles", nargs="+", choices=sorted(renderer_v01.PROFILE_BUDGETS), default=["compact", "medium"])
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
    renderer_dir = (
        args.renderer_dir
        or (run_dir / "semantic-v3" / "compact-renderer-v0.2" / source_slug)
    ).expanduser().resolve()
    pose_dir = (
        args.pose_dir
        or (run_dir / "semantic-v3" / "pose-language-v0.1")
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or (run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR / slug)
    ).expanduser().resolve()

    for label, path in (("renderer", renderer_dir), ("pose", pose_dir)):
        if not path.is_dir():
            print(f"{label.capitalize()} directory not found: {path}", file=sys.stderr)
            return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    if not prompt_path.is_file():
        print(f"Finisher prompt not found: {prompt_path}", file=sys.stderr)
        return 2
    prompt_template = prompt_path.read_text(encoding="utf-8")

    selected_keys = set(args.only)
    selected_profiles = set(args.profiles)
    source_paths = sorted(renderer_dir.glob("*.rendered_caption.json"))
    if selected_keys:
        source_paths = [
            path for path in source_paths
            if path.name.split(".", 1)[0] in selected_keys
        ]
    prepared: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    passthrough: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []

    for source_path in source_paths:
        source = _read_json(source_path)
        profile = str(source.get("profile") or "")
        if profile not in selected_profiles:
            continue
        key = str(source.get("image_key") or "")
        pose_path = pose_dir / f"{key}.pose_language.json"
        if not pose_path.is_file():
            print(f"Missing Pose language: {pose_path}", file=sys.stderr)
            return 2
        pose = _read_json(pose_path)
        finish_input = build_finish_input(
            renderer_record=source,
            pose=pose,
            prompt_template=prompt_template,
        )
        input_path = output_dir / f"{key}.{profile}.finish_input.json"
        _write_json(input_path, {
            "schema_version": ARTIFACT_VERSION + "-input",
            "image_key": key,
            "profile": profile,
            "renderer_source": str(source_path),
            "pose_source": str(pose_path),
            "prompt_template": str(prompt_path),
            **finish_input,
        })

        output_path = output_dir / f"{key}.{profile}.finished_caption.json"
        record = {
            "image_key": key,
            "profile": profile,
            "source_path": source_path,
            "source": source,
            "pose": pose,
            "finish_input": finish_input,
            "output_path": output_path,
        }
        prepared.append(record)

        needs_finish = not bool(finish_input["initial_quality_audit"]["passes_basic_gate"])
        if args.dry_run:
            continue
        if not needs_finish:
            payload = {
                "schema_version": ARTIFACT_VERSION,
                "image_key": key,
                "profile": profile,
                "model": source.get("model"),
                "backend": source.get("backend"),
                "trigger": source.get("trigger"),
                "grammar_profile": source.get("grammar_profile"),
                "renderer_source": str(source_path),
                "semantic_caption": source.get("semantic_caption"),
                "pose_corrections": source.get("pose_corrections") or [],
                "initial_caption": source.get("rendered_caption"),
                "finished_caption": source.get("rendered_caption"),
                "finish_action": "passthrough",
                "initial_quality_audit": finish_input["initial_quality_audit"],
                "quality_audit": finish_input["initial_quality_audit"],
                "performance": {"generated": False},
            }
            _write_json(output_path, payload)
            passthrough.append(payload)
        elif args.overwrite or not output_path.exists():
            pending.append(record)
        else:
            reused.append(_read_json(output_path))

    if not prepared:
        print("No matching renderer artifacts found.", file=sys.stderr)
        return 2

    if args.dry_run:
        candidates = [
            item for item in prepared
            if not item["finish_input"]["initial_quality_audit"]["passes_basic_gate"]
        ]
        index = {
            "schema_version": RUN_VERSION,
            "dry_run": True,
            "run_dir": str(run_dir),
            "renderer_dir": str(renderer_dir),
            "pose_dir": str(pose_dir),
            "output_dir": str(output_dir),
            "profiles": args.profiles,
            "record_count": len(prepared),
            "finish_candidate_count": len(candidates),
            "records": [
                {
                    "image_key": item["image_key"],
                    "profile": item["profile"],
                    "initial_quality_audit": item["finish_input"]["initial_quality_audit"],
                }
                for item in prepared
            ],
        }
        _write_json(output_dir / "finish.index.json", index)
        print(f"Renderer finisher inputs: {output_dir}")
        print(f"Records: {len(prepared)}; finish_candidates={len(candidates)}; dry_run=yes")
        return 0

    backend = resolve_backend(model_id, args.backend)
    if pending and backend == "vllm":
        install_text_only_vllm()

    loaded = None
    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    if pending:
        print(f"Loading {model_id} for conditional training-caption finishing ...")
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
            f"Finishing {len(pending)} caption(s). batch_size={args.batch_size} max_tokens={args.max_tokens}"
        )

    try:
        if loaded is not None and loaded.backend == "vllm":
            total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
            for batch_index, offset in enumerate(range(0, len(pending), args.batch_size), start=1):
                batch_started = time.perf_counter()
                batch = pending[offset: offset + args.batch_size]
                perf_items, batch_perf = editor_v01._generate_text_batch(
                    loaded,
                    [item["finish_input"]["finish_prompt"] for item in batch],
                    max_new_tokens=args.max_tokens,
                )
                for source, perf in zip(batch, perf_items):
                    finished = str(perf.get("text") or "").strip()
                    if not finished:
                        raise RuntimeError(
                            f"Empty finisher output for {source['image_key']} {source['profile']}"
                        )
                    src = source["source"]
                    audit = quality_audit(
                        semantic_caption=str(src.get("semantic_caption") or ""),
                        rendered_caption=finished,
                        pose=source["pose"],
                        trigger=str(src.get("trigger") or ""),
                        profile=source["profile"],
                    )
                    payload = {
                        "schema_version": ARTIFACT_VERSION,
                        "image_key": source["image_key"],
                        "profile": source["profile"],
                        "model": model_id,
                        "backend": loaded.backend,
                        "trigger": src.get("trigger"),
                        "grammar_profile": src.get("grammar_profile"),
                        "renderer_source": str(source["source_path"]),
                        "semantic_caption": src.get("semantic_caption"),
                        "pose_corrections": src.get("pose_corrections") or [],
                        "initial_caption": src.get("rendered_caption"),
                        "finished_caption": finished,
                        "finish_action": "generated",
                        "initial_quality_audit": source["finish_input"]["initial_quality_audit"],
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
                        "SEMANTIC_V3_RENDERER_FINISH_PERF "
                        f"image={source['image_key']} profile={source['profile']} "
                        f"words={audit['rendered_word_count']} trigger_count={audit['trigger']['exact_trigger_count']} "
                        f"gate={'pass' if audit['passes_basic_gate'] else 'FAIL'} "
                        f"review={','.join(audit['ownership_review']['review_warnings']) or 'none'} "
                        f"finish={perf.get('finish_reason')}"
                    )

                batch_wall = time.perf_counter() - batch_started
                batch_runtime.append({
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "items": [f"{item['image_key']}:{item['profile']}" for item in batch],
                    "generation_seconds": batch_perf["generation_seconds"],
                    "wall_seconds": batch_wall,
                    "amortized_seconds_per_render": batch_wall / len(batch) if batch else 0.0,
                    "prompt_tokens": batch_perf["prompt_tokens"],
                    "output_tokens": batch_perf["output_tokens"],
                    "aggregate_output_tokens_per_second": batch_perf["aggregate_output_tokens_per_second"],
                })
        elif loaded is not None:
            from .runner import generate_text
            for source in pending:
                finished, inference_seconds = generate_text(
                    loaded,
                    source["finish_input"]["finish_prompt"],
                    max_new_tokens=args.max_tokens,
                )
                finished = finished.strip()
                src = source["source"]
                audit = quality_audit(
                    semantic_caption=str(src.get("semantic_caption") or ""),
                    rendered_caption=finished,
                    pose=source["pose"],
                    trigger=str(src.get("trigger") or ""),
                    profile=source["profile"],
                )
                payload = {
                    "schema_version": ARTIFACT_VERSION,
                    "image_key": source["image_key"],
                    "profile": source["profile"],
                    "model": model_id,
                    "backend": loaded.backend,
                    "trigger": src.get("trigger"),
                    "grammar_profile": src.get("grammar_profile"),
                    "renderer_source": str(source["source_path"]),
                    "semantic_caption": src.get("semantic_caption"),
                    "pose_corrections": src.get("pose_corrections") or [],
                    "initial_caption": src.get("rendered_caption"),
                    "finished_caption": finished,
                    "finish_action": "generated",
                    "initial_quality_audit": source["finish_input"]["initial_quality_audit"],
                    "quality_audit": audit,
                    "performance": {"inference_seconds": inference_seconds, "max_new_tokens": args.max_tokens},
                }
                _write_json(source["output_path"], payload)
                generated.append(payload)
    finally:
        if loaded is not None:
            unload_model(loaded)

    records = sorted(
        passthrough + reused + generated,
        key=lambda item: (str(item.get("image_key") or ""), str(item.get("profile") or "")),
    )
    by_profile: dict[str, dict[str, Any]] = {}
    for profile in args.profiles:
        items = [item for item in records if item.get("profile") == profile]
        counts = [int((item.get("quality_audit") or {}).get("rendered_word_count") or 0) for item in items]
        by_profile[profile] = {
            "record_count": len(items),
            "generated_count": sum(1 for item in items if item.get("finish_action") == "generated"),
            "passthrough_count": sum(1 for item in items if item.get("finish_action") == "passthrough"),
            "gate_pass_count": sum(1 for item in items if (item.get("quality_audit") or {}).get("passes_basic_gate")),
            "review_warning_count": sum(
                1 for item in items
                if ((item.get("quality_audit") or {}).get("ownership_review") or {}).get("review_warnings")
            ),
            "word_count_min": min(counts) if counts else None,
            "word_count_median": statistics.median(counts) if counts else None,
            "word_count_max": max(counts) if counts else None,
        }

    index = {
        "schema_version": RUN_VERSION,
        "dry_run": False,
        "run_dir": str(run_dir),
        "renderer_dir": str(renderer_dir),
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
                "finish_action": item.get("finish_action"),
                "initial_quality_audit": item.get("initial_quality_audit") or {},
                "quality_audit": item.get("quality_audit") or {},
                "performance": item.get("performance") or {},
            }
            for item in records
        ],
    }
    _write_json(output_dir / "finish.index.json", index)
    print(f"Finished trigger-bound training captions: {output_dir}")
    print(
        f"Records: {len(records)}; generated={len(generated)}; passthrough={len(passthrough)}; reused={len(reused)}; "
        + "; ".join(
            f"{profile}:pass={summary['gate_pass_count']}/{summary['record_count']} "
            f"words={summary['word_count_min']}/{summary['word_count_median']}/{summary['word_count_max']}"
            for profile, summary in by_profile.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
