from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor as v01
from . import semantic_v3_rich_pose_editor_v04 as v04
from .runner import load_model, model_slug, resolve_backend, resolve_model_id, unload_model
from .semantic_v3_text_only_bootstrap import install_text_only_vllm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_repair_v01.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-repair-0.1"
RUN_VERSION = "semantic-v3-rich-pose-repair-0.1-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-repair-v0.1"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_TOKENS = 700


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _expanded_authorized_laterality(pose: dict[str, Any]) -> set[tuple[str, str]]:
    """Apply only semantically lossless laterality closure.

    A governed left/right fist is necessarily that side's hand, so wording such as
    'left hand ... with the fist under the jaw' is not an unsupported side guess.
    Do not infer broader limb-chain laterality from wrist/arm/etc.
    """
    authorized = set(v01._authorized_laterality(pose))
    for side in ("left", "right"):
        if (side, "fist") in authorized:
            authorized.add((side, "hand"))
    return authorized


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    """v0.4 semantic audit plus lossless fist->hand laterality equivalence."""
    base = v04.quality_audit(draft, edited, pose)
    authorized = _expanded_authorized_laterality(pose)

    unauthorized: list[str] = []
    for match in v01._BODY_SIDE_RE.finditer(edited):
        pair = (match.group(1).lower(), match.group(2).lower())
        if pair not in authorized:
            unauthorized.append(match.group(0))

    warnings = [
        value
        for value in (base.get("warnings") or [])
        if value != "unsupported_anatomical_laterality"
    ]
    if unauthorized:
        warnings.append("unsupported_anatomical_laterality")

    base["unauthorized_anatomical_laterality"] = sorted(set(unauthorized), key=str.lower)
    base["authorized_laterality_expanded"] = sorted(
        [{"side": side, "body_part": part} for side, part in authorized],
        key=lambda item: (item["side"], item["body_part"]),
    )
    base["warnings"] = list(dict.fromkeys(warnings))
    base["passes_basic_gate"] = not base["warnings"]
    return base


def _failure_lines(audit: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    identity = audit.get("identity_leaks") or []
    generic = audit.get("generic_identity_paraphrase_leaks") or []
    dye = audit.get("hair_dye_detail_leaks") or []
    meta = audit.get("meta_redaction_language") or []
    laterality = audit.get("unauthorized_anatomical_laterality") or []
    conflicts = audit.get("conflicting_pose_wording") or []

    if identity:
        lines.append("- Protected identity information: " + "; ".join(map(str, identity)))
    if generic:
        lines.append("- Generic identity paraphrase/euphemism: " + "; ".join(map(str, generic)))
    if dye:
        lines.append("- Hair dye/treatment residue: " + "; ".join(map(str, dye)))
    if meta:
        lines.append("- Meta-redaction language: " + "; ".join(map(str, meta)))
    if laterality:
        lines.append("- Unsupported anatomical laterality: " + "; ".join(map(str, laterality)))
    if conflicts:
        lines.append("- Conflicting pose wording: " + "; ".join(map(str, conflicts)))

    # Length warnings are not safely repairable without reintroducing source material.
    for warning in audit.get("warnings") or []:
        if warning in {
            "edited_caption_overcompressed",
            "edited_caption_expanded_substantially",
        }:
            lines.append(f"- Non-repairable length gate: {warning}")
    return lines


def _repairable(audit: dict[str, Any]) -> bool:
    warnings = set(audit.get("warnings") or [])
    if not warnings:
        return False
    nonrepairable = {
        "edited_caption_overcompressed",
        "edited_caption_expanded_substantially",
    }
    return not bool(warnings & nonrepairable)


def build_repair_input(
    *,
    edited_caption: str,
    pose: dict[str, Any],
    editor_input: dict[str, Any],
    audit: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    corrections = [str(v).strip() for v in (pose.get("caption_ready_phrases") or []) if str(v).strip()]
    pose_text = "\n".join(f"- {value}" for value in corrections) if corrections else "- None"
    redactions = editor_input.get("mandatory_redactions") if isinstance(editor_input.get("mandatory_redactions"), dict) else {}
    transient = [str(v).strip() for v in (redactions.get("transient_hair_mentions_to_preserve") or []) if str(v).strip()]
    transient_text = "\n".join(f"- {value}" for value in transient) if transient else "- None"
    failure_lines = _failure_lines(audit)
    failure_text = "\n".join(failure_lines) if failure_lines else "- None"

    prompt = (
        prompt_template.replace("{{CURRENT_CAPTION}}", edited_caption)
        .replace("{{POSE_CORRECTIONS}}", pose_text)
        .replace("{{TRANSIENT_HAIR}}", transient_text)
        .replace("{{FAILURES}}", failure_text)
    )
    return {
        "current_caption": edited_caption,
        "pose_corrections": corrections,
        "transient_hair_mentions_to_preserve": transient,
        "initial_quality_audit": audit,
        "repairable": _repairable(audit),
        "repair_prompt": prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-audit rich/Pose edited captions and repair only genuine deterministic validation failures."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--editor-dir", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--editor-model", default="32b-fp8", help="Model slug used to locate v0.4 editor artifacts")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Re-audit and write repair inputs without loading a model")
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

    repair_model_id = resolve_model_id(args.model)
    repair_slug = model_slug(repair_model_id)
    editor_slug = model_slug(resolve_model_id(args.editor_model))
    editor_dir = (args.editor_dir or (run_dir / "semantic-v3" / "rich-pose-editor-v0.4" / editor_slug)).expanduser().resolve()
    pose_dir = (args.pose_dir or (run_dir / "semantic-v3" / "pose-language-v0.1")).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR / repair_slug)).expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()

    if not editor_dir.is_dir():
        print(f"Editor directory not found: {editor_dir}", file=sys.stderr)
        return 2
    if not pose_dir.is_dir():
        print(f"Pose directory not found: {pose_dir}", file=sys.stderr)
        return 2
    if not prompt_path.is_file():
        print(f"Repair prompt not found: {prompt_path}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = prompt_path.read_text(encoding="utf-8")

    selected = set(args.only)
    edited_paths = sorted(editor_dir.glob("*.edited_caption.json"))
    if selected:
        edited_paths = [p for p in edited_paths if p.name.removesuffix(".edited_caption.json") in selected]
    if not edited_paths:
        print("No matching v0.4 edited-caption artifacts found.", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    for edited_path in edited_paths:
        key = edited_path.name.removesuffix(".edited_caption.json")
        input_path = editor_dir / f"{key}.editor_input.json"
        pose_path = pose_dir / f"{key}.pose_language.json"
        if not input_path.is_file() or not pose_path.is_file():
            print(f"Missing editor input or Pose artifact for {key}", file=sys.stderr)
            return 2

        edited = _read_json(edited_path)
        editor_input = _read_json(input_path)
        pose = _read_json(pose_path)
        caption = str(edited.get("edited_caption") or "").strip()
        draft = str(edited.get("rich_source_caption") or editor_input.get("rich_draft") or "").strip()
        if not caption or not draft:
            print(f"Missing caption text for {key}", file=sys.stderr)
            return 2

        audit = quality_audit(draft, caption, pose)
        repair_input = build_repair_input(
            edited_caption=caption,
            pose=pose,
            editor_input=editor_input,
            audit=audit,
            prompt_template=prompt_template,
        )
        repair_input_path = output_dir / f"{key}.repair_input.json"
        _write_json(
            repair_input_path,
            {
                "schema_version": ARTIFACT_VERSION + "-input",
                "image_key": key,
                "editor_source": str(edited_path),
                "editor_input_source": str(input_path),
                "pose_source": str(pose_path),
                "prompt_template": str(prompt_path),
                **repair_input,
            },
        )

        repaired_path = output_dir / f"{key}.repaired_caption.json"
        record = {
            "image_key": key,
            "draft": draft,
            "edited_caption": caption,
            "pose": pose,
            "repair_input": repair_input,
            "repaired_path": repaired_path,
            "initial_pass": bool(audit.get("passes_basic_gate")),
        }
        records.append(record)

        if audit.get("passes_basic_gate"):
            continue
        if not repair_input["repairable"]:
            continue
        if not args.dry_run and (args.overwrite or not repaired_path.exists()):
            pending.append(record)
        elif repaired_path.exists():
            reused.append(_read_json(repaired_path))

    if args.dry_run:
        index = {
            "schema_version": RUN_VERSION,
            "dry_run": True,
            "run_dir": str(run_dir),
            "editor_dir": str(editor_dir),
            "pose_dir": str(pose_dir),
            "output_dir": str(output_dir),
            "record_count": len(records),
            "initial_pass_count": sum(1 for r in records if r["initial_pass"]),
            "repair_candidate_count": sum(
                1 for r in records
                if (not r["initial_pass"] and r["repair_input"]["repairable"])
            ),
            "records": [
                {
                    "image_key": r["image_key"],
                    "initial_pass": r["initial_pass"],
                    "warnings": r["repair_input"]["initial_quality_audit"].get("warnings") or [],
                    "repairable": r["repair_input"]["repairable"],
                }
                for r in records
            ],
        }
        _write_json(output_dir / "repair.index.json", index)
        print(f"Repair inputs: {output_dir}")
        print(
            f"Records: {len(records)}; initial_pass={index['initial_pass_count']}; "
            f"repair_candidates={index['repair_candidate_count']}; dry_run=yes"
        )
        return 0

    backend = resolve_backend(repair_model_id, args.backend)
    if pending and backend == "vllm":
        install_text_only_vllm()

    generated: list[dict[str, Any]] = []
    batch_runtime: list[dict[str, Any]] = []
    loaded = None
    run_started = time.perf_counter()
    if pending:
        print(f"Loading {repair_model_id} for fail-only rich/Pose repair ...")
        loaded_started = time.perf_counter()
        loaded = load_model(
            repair_model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        print(
            f"Loaded in {time.perf_counter() - loaded_started:.2f}s. "
            f"Repairing {len(pending)} failed caption(s). batch_size={args.batch_size} max_tokens={args.max_tokens}"
        )

        total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
        for batch_index in range(total_batches):
            batch = pending[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
            prompts = [item["repair_input"]["repair_prompt"] for item in batch]
            perf_items, batch_perf = v01._generate_text_batch(
                loaded,
                prompts,
                max_new_tokens=args.max_tokens,
            )
            batch_runtime.append(batch_perf)
            for item, perf in zip(batch, perf_items):
                repaired = str(perf.get("text") or "").strip()
                if not repaired:
                    raise RuntimeError(f"Empty repair output for {item['image_key']}")
                final_audit = quality_audit(item["draft"], repaired, item["pose"])
                artifact = {
                    "schema_version": ARTIFACT_VERSION,
                    "image_key": item["image_key"],
                    "model": repair_model_id,
                    "backend": backend,
                    "source_edited_caption": item["edited_caption"],
                    "repaired_caption": repaired,
                    "pose_corrections": item["repair_input"]["pose_corrections"],
                    "initial_quality_audit": item["repair_input"]["initial_quality_audit"],
                    "final_quality_audit": final_audit,
                    "performance": perf,
                }
                _write_json(item["repaired_path"], artifact)
                generated.append(artifact)
                print(
                    f"RICH_POSE_REPAIR_PERF image={item['image_key']} "
                    f"warnings={','.join(final_audit.get('warnings') or []) or 'none'} "
                    f"finish={perf.get('finish_reason') or 'n/a'}"
                )

    if loaded is not None:
        unload_model(loaded)

    # Build final status across all records: initial pass, repaired pass, reused repair, or unresolved.
    repaired_by_key = {item["image_key"]: item for item in generated}
    repaired_by_key.update({str(item.get("image_key")): item for item in reused if item.get("image_key")})
    final_records: list[dict[str, Any]] = []
    final_pass_count = 0
    for record in records:
        key = record["image_key"]
        initial_audit = record["repair_input"]["initial_quality_audit"]
        if record["initial_pass"]:
            final_pass = True
            status = "initial_pass"
            final_audit = initial_audit
        elif key in repaired_by_key:
            final_audit = repaired_by_key[key].get("final_quality_audit") or {}
            final_pass = bool(final_audit.get("passes_basic_gate"))
            status = "repaired_pass" if final_pass else "repair_failed"
        else:
            final_audit = initial_audit
            final_pass = False
            status = "unresolved"
        if final_pass:
            final_pass_count += 1
        final_records.append(
            {
                "image_key": key,
                "status": status,
                "initial_warnings": initial_audit.get("warnings") or [],
                "final_warnings": final_audit.get("warnings") or [],
                "passes_final_gate": final_pass,
            }
        )

    index = {
        "schema_version": RUN_VERSION,
        "dry_run": False,
        "run_dir": str(run_dir),
        "editor_dir": str(editor_dir),
        "pose_dir": str(pose_dir),
        "output_dir": str(output_dir),
        "model": repair_model_id,
        "backend": backend,
        "prompt": str(prompt_path),
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "record_count": len(records),
        "initial_pass_count": sum(1 for r in records if r["initial_pass"]),
        "repair_candidate_count": sum(
            1 for r in records
            if (not r["initial_pass"] and r["repair_input"]["repairable"])
        ),
        "generated_repairs": len(generated),
        "reused_repairs": len(reused),
        "final_pass_count": final_pass_count,
        "run_wall_seconds": time.perf_counter() - run_started,
        "batch_runtime": batch_runtime,
        "records": final_records,
    }
    _write_json(output_dir / "repair.index.json", index)
    print(f"Rich/Pose repaired captions: {output_dir}")
    print(
        f"Records: {len(records)}; initial_pass={index['initial_pass_count']}; "
        f"repairs_generated={len(generated)}; final_pass={final_pass_count}"
    )
    return 0 if final_pass_count == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
