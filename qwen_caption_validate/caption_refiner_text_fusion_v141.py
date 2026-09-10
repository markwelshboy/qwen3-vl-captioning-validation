from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import caption_refiner_text_fusion_v14 as v14
from . import runner
from .caption_refiner_specialist_facts import format_head_gaze_facts


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "caption_refiner_v141_text_fusion_structured.txt"
ARTIFACT_VERSION = "caption-refiner-text-fusion-0.14.1"
RUN_VERSION = "caption-refiner-text-fusion-0.14.1-run"

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["NO_CORRECTION", "CORRECTION"],
        },
        "correction": {
            "anyOf": [
                {"type": "string", "minLength": 1, "maxLength": 280},
                {"type": "null"},
            ]
        },
    },
    "required": ["decision", "correction"],
}

_META_OUTPUT_RE = re.compile(
    r"\b(?:current\s+caption|caption\s+(?:states?|says?|describes?|asserts?)|"
    r"specialist(?:\s+evidence)?|deterministic\s+laterality|the\s+evidence|"
    r"evidence\s+(?:confirms?|indicates?|shows?|provides?|contradicts?)|"
    r"confirms?|contradicts?|therefore|however)\b",
    re.IGNORECASE,
)
_TERMINAL_RE = re.compile(r"[.!?][\"'”’]?$", re.UNICODE)


def validate_correction_text(text: str) -> tuple[bool, str | None]:
    value = " ".join(text.strip().split())
    if not value:
        return False, "empty_correction"
    if len(value.split()) > 35:
        return False, "correction_too_long"
    if not _TERMINAL_RE.search(value):
        return False, "correction_not_terminal_sentence"
    if v14._sentence_count(value) != 1:
        return False, "correction_not_exactly_one_sentence"
    if _META_OUTPUT_RE.search(value):
        return False, "meta_or_rationale_language"
    return True, None


def interpret_structured_decision(value: Any, raw_response: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "decision": None,
        "text": None,
        "contract_ok": False,
        "contract_reason": None,
        "raw_response": raw_response,
        "structured": value if isinstance(value, dict) else None,
    }
    if not isinstance(value, dict):
        result["contract_reason"] = "structured_output_not_object"
        return result

    decision = value.get("decision")
    correction = value.get("correction")
    if decision not in {"NO_CORRECTION", "CORRECTION"}:
        result["contract_reason"] = "invalid_decision"
        return result

    result["decision"] = decision
    if decision == "NO_CORRECTION":
        if correction not in (None, ""):
            result["text"] = "NO_CORRECTION"
            result["contract_reason"] = "no_correction_with_nonempty_correction"
            return result
        result["text"] = "NO_CORRECTION"
        result["contract_ok"] = True
        return result

    if not isinstance(correction, str):
        result["contract_reason"] = "correction_missing_or_not_string"
        return result

    normalized = " ".join(correction.strip().split())
    result["text"] = normalized
    ok, reason = validate_correction_text(normalized)
    result["contract_ok"] = ok
    result["contract_reason"] = reason
    return result


def _parse_structured(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value, _ = runner.parse_json_response(raw)
    return value if isinstance(value, dict) else None


def _generate_vllm_batch(loaded, prompts: list[str], max_tokens: int) -> tuple[list[dict[str, Any]], float]:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    text_prompts = [runner._chat_text(loaded, prompt) for prompt in prompts]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=DECISION_SCHEMA),
    )
    started = time.perf_counter()
    outputs = loaded.model.generate(text_prompts, sampling_params=sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started
    if len(outputs) != len(prompts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} text-refiner requests")

    decisions: list[dict[str, Any]] = []
    for output in outputs:
        raw = output.outputs[0].text.strip()
        decisions.append(interpret_structured_decision(_parse_structured(raw), raw))
    return decisions, elapsed


def _generate_transformers_one(loaded, prompt: str, max_tokens: int) -> tuple[dict[str, Any], float]:
    raw, elapsed = runner.generate_text(loaded, prompt, max_new_tokens=max_tokens)
    return interpret_structured_decision(_parse_structured(raw), raw), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured text-only governed evidence fusion refiner v0.14.1")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--base-evidence-dir",
        type=Path,
        required=True,
        help="Delta-refiner output containing canonical existing caption and laterality facts; v0.13 is suitable.",
    )
    parser.add_argument(
        "--pose-candidate-dir",
        type=Path,
        help="Optional pose-only delta-refiner output (normally v0.12); candidate is gated before prompt exposure.",
    )
    parser.add_argument("--head-gaze-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    base_dir = args.base_evidence_dir.expanduser().resolve()
    pose_candidate_dir = args.pose_candidate_dir.expanduser().resolve() if args.pose_candidate_dir else None
    head_gaze_dir = args.head_gaze_dir.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "caption-refiner-text-fusion-v0.14.1").expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()

    checks = [(run_dir, "Run"), (base_dir, "Base evidence"), (head_gaze_dir, "Head/gaze evidence")]
    if pose_candidate_dir is not None:
        checks.append((pose_candidate_dir, "Pose candidate"))
    for path, label in checks:
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2
    if not prompt_path.is_file():
        print(f"Prompt not found: {prompt_path}", file=sys.stderr)
        return 2
    if args.batch_size < 1 or args.max_tokens < 1:
        print("--batch-size and --max-tokens must be >= 1", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    template = prompt_path.read_text(encoding="utf-8")
    base_records = v14._index_by_key(base_dir)
    pose_candidates = v14._index_by_key(pose_candidate_dir)

    records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    missing: list[dict[str, str]] = []

    for key in sorted(base_records):
        if not v14._matches_only(key, args.only):
            continue
        base = base_records[key]
        caption = str(base.get("existing_caption") or "").strip()
        if not caption:
            missing.append({"image_key": key, "reason": "missing_existing_caption"})
            continue

        candidate_source = pose_candidates.get(key, {}) if pose_candidate_dir else {}
        gate = v14.govern_pose_candidate(candidate_source.get("pose_delta") if candidate_source else None)
        accepted_candidate = gate.get("accepted_text") if gate.get("status") == "accepted" else None
        laterality = v14.caption_safe_laterality_facts(
            base.get("laterality_facts"),
            current_caption=caption,
            accepted_pose_candidate=accepted_candidate,
        )
        laterality_text = v14.format_caption_safe_laterality(laterality)

        hg_path = v14._head_gaze_path(head_gaze_dir, key)
        hg = v14._read_json(hg_path)
        head_gaze_text = format_head_gaze_facts(hg)
        pose_prompt_text = v14.pose_candidate_prompt_text(gate)
        prompt = v14.render_prompt(
            template,
            current_caption=caption,
            pose_candidate_text=pose_prompt_text,
            laterality_text=laterality_text,
            head_gaze_text=head_gaze_text,
        )

        out_path = output_dir / f"{key}.text_refiner.json"
        existing = v14._read_json(out_path) if out_path.exists() and not args.overwrite else {}
        base_source_path = v14._delta_record_path(base_dir, key)
        pose_source_path = v14._delta_record_path(pose_candidate_dir, key) if pose_candidate_dir else None
        record = {
            "schema_version": ARTIFACT_VERSION,
            "image_key": key,
            "existing_caption": caption,
            "base_evidence_source": str(base_source_path) if base_source_path else None,
            "pose_candidate_source": str(pose_source_path) if pose_source_path else None,
            "pose_candidate_gate": gate,
            "laterality_facts_supplied": laterality,
            "laterality_text_supplied": laterality_text,
            "head_gaze_evidence_source": str(hg_path) if hg_path else None,
            "head_gaze_evidence_schema": hg.get("schema_version") if hg else None,
            "head_gaze_specialist_facts": head_gaze_text,
            "text_only": True,
            "structured_decision": True,
            "correction_delta": existing.get("correction_delta") if existing else None,
        }
        records.append(record)

        if args.dry_run:
            v14._write_json(out_path, record)
        elif args.overwrite or not record.get("correction_delta"):
            record["_prompt"] = prompt
            pending.append((record, out_path))
        else:
            v14._write_json(out_path, record)

    model_id = runner.resolve_model_id(args.model)
    backend = runner.resolve_backend(model_id, args.backend)
    loaded = None
    load_seconds = 0.0
    if pending:
        print(f"Loading {model_id} for {len(pending)} structured text-only refinement(s) ...")
        loaded = runner.load_model(
            model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        load_seconds = float(loaded.load_seconds)
        print(f"Loaded in {load_seconds:.2f}s")

    try:
        if loaded is not None and loaded.backend == "vllm":
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset:offset + args.batch_size]
                prompts = [record["_prompt"] for record, _ in batch]
                decisions, elapsed = _generate_vllm_batch(loaded, prompts, args.max_tokens)
                per_item = elapsed / max(1, len(batch))
                for (record, out_path), decision in zip(batch, decisions):
                    record.pop("_prompt", None)
                    decision["generation_seconds_approx"] = per_item
                    record["correction_delta"] = decision
                    v14._write_json(out_path, record)
                    print(
                        f"{record['image_key']}: {decision.get('text')} "
                        f"[decision={decision.get('decision')} ok={decision.get('contract_ok')}]"
                    )
        elif loaded is not None:
            for record, out_path in pending:
                prompt = record.pop("_prompt")
                decision, elapsed = _generate_transformers_one(loaded, prompt, args.max_tokens)
                decision["generation_seconds"] = elapsed
                record["correction_delta"] = decision
                v14._write_json(out_path, record)
                print(
                    f"{record['image_key']}: {decision.get('text')} "
                    f"[decision={decision.get('decision')} ok={decision.get('contract_ok')}]"
                )
    finally:
        if loaded is not None:
            runner.unload_model(loaded)

    for record in records:
        record.pop("_prompt", None)

    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "base_evidence_dir": str(base_dir),
        "pose_candidate_dir": str(pose_candidate_dir) if pose_candidate_dir else None,
        "head_gaze_evidence_dir": str(head_gaze_dir),
        "prompt": str(prompt_path),
        "model_id": model_id,
        "backend": backend,
        "model_load_seconds": load_seconds,
        "record_count": len(records),
        "missing": missing,
        "records": records,
    }
    v14._write_json(output_dir / "caption_refiner_text_fusion_v141.index.json", index)
    print(f"Structured text-only caption refinement v0.14.1: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
