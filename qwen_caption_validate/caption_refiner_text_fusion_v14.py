from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import runner
from .caption_refiner_specialist_facts import format_head_gaze_facts


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "caption_refiner_v14_text_fusion.txt"
ARTIFACT_VERSION = "caption-refiner-text-fusion-0.14"
RUN_VERSION = "caption-refiner-text-fusion-0.14-run"

_POSTURE_RE = re.compile(
    r"\b(?:standing|stand(?:ing)?|seated|sitting|sit(?:ting)?|reclining|reclined|lying|laying|"
    r"crouching|crouched|kneeling|kneels?|squatting|squats?)\b",
    re.IGNORECASE,
)
_HEAD_GAZE_RE = re.compile(
    r"\b(?:head|face|chin|gaze|eyes?|looking|looks?|eye[- ]?contact)\b",
    re.IGNORECASE,
)
_SUPPORT_SCENE_RE = re.compile(
    r"\b(?:chair|sofa|couch|bed|floor|wall|table|furniture|supported?|supporting|"
    r"leaning\s+against|resting\s+on)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(?:reaching|gesturing|holding|walking|running|dancing|moving|interacting|grabbing|"
    r"touching|carrying)\b",
    re.IGNORECASE,
)
_HIDDEN_RE = re.compile(
    r"\b(?:hidden|occluded|cropped\s+out|outside\s+(?:the\s+)?(?:photograph|frame)|"
    r"out\s+of\s+(?:the\s+)?frame)\b",
    re.IGNORECASE,
)

_DOMAIN_PATTERNS = {
    "trunk": re.compile(r"\b(?:torso|trunk|shoulders?|hips?|upper\s+body|body\s+orientation)\b", re.IGNORECASE),
    "upper_limb": re.compile(r"\b(?:arms?|elbows?|wrists?|hands?)\b", re.IGNORECASE),
    "lower_limb": re.compile(r"\b(?:legs?|knees?|ankles?|feet|foot)\b", re.IGNORECASE),
}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _extract_delta_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text")
        return str(text).strip() if text is not None else ""
    return ""


def _sentence_count(text: str) -> int:
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return len(pieces)


def govern_pose_candidate(value: Any) -> dict[str, Any]:
    """Gate a visual pose-refiner delta before it can reach the text-only writer.

    Rejected raw text is retained in the artifact for audit but is never inserted into
    the model prompt.
    """
    raw = _extract_delta_text(value)
    result: dict[str, Any] = {
        "status": "unavailable",
        "accepted_text": None,
        "raw_text": raw or None,
        "reasons": [],
    }
    if not raw:
        result["reasons"] = ["missing_candidate"]
        return result
    if raw == "NO_CORRECTION":
        result["status"] = "no_correction"
        result["reasons"] = ["pose_specialist_found_no_correction"]
        return result

    normalized = " ".join(raw.split())
    reasons: list[str] = []
    words = normalized.split()
    if len(words) > 45:
        reasons.append("too_long")
    if _sentence_count(normalized) != 1:
        reasons.append("not_exactly_one_sentence")
    if _POSTURE_RE.search(normalized):
        reasons.append("semantic_posture_language")
    if _HEAD_GAZE_RE.search(normalized):
        reasons.append("head_or_gaze_language")
    if _SUPPORT_SCENE_RE.search(normalized):
        reasons.append("support_or_scene_language")
    if _ACTION_RE.search(normalized):
        reasons.append("action_language")
    if _HIDDEN_RE.search(normalized):
        reasons.append("hidden_or_out_of_frame_language")

    domains = [name for name, pattern in _DOMAIN_PATTERNS.items() if pattern.search(normalized)]
    if len(domains) > 1:
        reasons.append("multiple_body_domains")
    if not domains:
        reasons.append("no_supported_body_geometry_domain")

    if reasons:
        result["status"] = "rejected"
        result["reasons"] = reasons
        result["domains"] = domains
        return result

    result["status"] = "accepted"
    result["accepted_text"] = normalized
    result["domains"] = domains
    return result


def pose_candidate_prompt_text(gate: dict[str, Any]) -> str:
    if gate.get("status") == "accepted" and gate.get("accepted_text"):
        return f"- Accepted pose geometry candidate: {gate['accepted_text']}"
    return "- No pose geometry candidate is available; do not introduce body geometry from pose-specialist evidence."


def _laterality_fact_relevant(fact: dict[str, Any], reference_text: str) -> bool:
    joint = str(fact.get("joint") or "").lower()
    text = reference_text.lower()
    if joint in {"shoulder", "elbow", "wrist"}:
        return any(term in text for term in (joint, "arm", "hand"))
    if joint in {"hip", "knee", "ankle"}:
        return any(term in text for term in (joint, "leg", "foot", "feet"))
    return joint != "" and joint in text


def caption_safe_laterality_facts(
    facts: Any,
    *,
    current_caption: str,
    accepted_pose_candidate: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(facts, list):
        return []
    reference = current_caption
    if accepted_pose_candidate:
        reference += " " + accepted_pose_candidate
    safe: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("inside_frame") is not True:
            continue
        if not _laterality_fact_relevant(fact, reference):
            continue
        anatomical = str(fact.get("anatomical_side") or "")
        frame = str(fact.get("frame_side") or "")
        joint = str(fact.get("joint") or "")
        if anatomical not in {"subject-left", "subject-right"}:
            continue
        if not joint:
            continue
        safe.append({
            "joint": joint,
            "anatomical_side": anatomical,
            "frame_side": frame,
            "source": fact.get("source"),
        })
    return safe


def format_caption_safe_laterality(facts: list[dict[str, Any]]) -> str:
    if not facts:
        return "- No caption-relevant deterministic laterality facts are available; do not introduce a new left/right limb claim."
    return "\n".join(
        f"- {fact['anatomical_side']} {fact['joint']}: {fact.get('frame_side') or 'frame location unavailable'}."
        for fact in facts
    )


def render_prompt(
    template: str,
    *,
    current_caption: str,
    pose_candidate_text: str,
    laterality_text: str,
    head_gaze_text: str,
) -> str:
    return (
        template
        .replace("{{CURRENT_CAPTION}}", current_caption.strip())
        .replace("{{POSE_CANDIDATE}}", pose_candidate_text.strip())
        .replace("{{LATERALITY_FACTS}}", laterality_text.strip())
        .replace("{{HEAD_GAZE_FACTS}}", head_gaze_text.strip())
    )


def validate_output_contract(text: str) -> tuple[bool, str | None]:
    value = text.strip()
    if value == "NO_CORRECTION":
        return True, None
    if not value:
        return False, "empty_output"
    if "\n" in value:
        return False, "multiline_output"
    if _sentence_count(value) != 1:
        return False, "not_exactly_one_sentence"
    if len(value.split()) > 65:
        return False, "output_too_long"
    return True, None


def _head_gaze_path(directory: Path, key: str) -> Path | None:
    direct = directory / f"{key}.head_gaze.json"
    if direct.is_file():
        return direct
    matches = sorted(directory.rglob(f"{key}.head_gaze.json"))
    return matches[0] if matches else None


def _pose_record_path(directory: Path, key: str) -> Path | None:
    direct = directory / f"{key}.delta_refiner.json"
    if direct.is_file():
        return direct
    matches = sorted(directory.rglob(f"{key}.delta_refiner.json"))
    return matches[0] if matches else None


def _load_pose_records(directory: Path) -> list[dict[str, Any]]:
    index = _read_json(directory / "caption_refiner_delta.index.json")
    records = index.get("records") if isinstance(index, dict) else None
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]

    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.delta_refiner.json")):
        value = _read_json(path)
        if value:
            out.append(value)
    return out


def _matches_only(key: str, only: list[str]) -> bool:
    if not only:
        return True
    low = key.lower()
    return any(token.lower() == low or token.lower() in low for token in only)


def _generate_vllm_batch(loaded, prompts: list[str], max_tokens: int) -> tuple[list[str], float]:
    from vllm import SamplingParams

    text_prompts = [runner._chat_text(loaded, prompt) for prompt in prompts]
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    started = time.perf_counter()
    outputs = loaded.model.generate(text_prompts, sampling_params=sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started
    if len(outputs) != len(prompts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} text-refiner requests")
    return [output.outputs[0].text.strip() for output in outputs], elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-only governed evidence fusion refiner v0.14")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pose-evidence-dir", type=Path, required=True,
                        help="v0.12 pose-only delta-refiner output; supplies current caption, laterality, and gated pose candidate")
    parser.add_argument("--head-gaze-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    pose_dir = args.pose_evidence_dir.expanduser().resolve()
    head_gaze_dir = args.head_gaze_dir.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "caption-refiner-text-fusion-v0.14").expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()

    for path, label in ((run_dir, "Run"), (pose_dir, "Pose evidence"), (head_gaze_dir, "Head/gaze evidence")):
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
    pose_records = [record for record in _load_pose_records(pose_dir)
                    if _matches_only(str(record.get("image_key") or ""), args.only)]

    records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    missing: list[dict[str, str]] = []

    for summary in pose_records:
        key = str(summary.get("image_key") or "")
        if not key:
            continue
        source_path = _pose_record_path(pose_dir, key)
        source = _read_json(source_path) if source_path else summary
        if not source:
            source = summary

        caption = str(source.get("existing_caption") or summary.get("existing_caption") or "").strip()
        if not caption:
            missing.append({"image_key": key, "reason": "missing_existing_caption"})
            continue

        gate = govern_pose_candidate(source.get("pose_delta", summary.get("pose_delta")))
        accepted_candidate = gate.get("accepted_text") if gate.get("status") == "accepted" else None
        laterality = caption_safe_laterality_facts(
            source.get("laterality_facts", summary.get("laterality_facts")),
            current_caption=caption,
            accepted_pose_candidate=accepted_candidate,
        )
        laterality_text = format_caption_safe_laterality(laterality)

        hg_path = _head_gaze_path(head_gaze_dir, key)
        hg = _read_json(hg_path)
        head_gaze_text = format_head_gaze_facts(hg)
        pose_prompt_text = pose_candidate_prompt_text(gate)
        prompt = render_prompt(
            template,
            current_caption=caption,
            pose_candidate_text=pose_prompt_text,
            laterality_text=laterality_text,
            head_gaze_text=head_gaze_text,
        )

        out_path = output_dir / f"{key}.text_refiner.json"
        existing = _read_json(out_path) if out_path.exists() and not args.overwrite else {}
        record = {
            "schema_version": ARTIFACT_VERSION,
            "image_key": key,
            "existing_caption": caption,
            "pose_evidence_source": str(source_path) if source_path else None,
            "pose_candidate_gate": gate,
            "laterality_facts_supplied": laterality,
            "laterality_text_supplied": laterality_text,
            "head_gaze_evidence_source": str(hg_path) if hg_path else None,
            "head_gaze_evidence_schema": hg.get("schema_version") if hg else None,
            "head_gaze_specialist_facts": head_gaze_text,
            "text_only": True,
            "correction_delta": existing.get("correction_delta") if existing else None,
        }
        records.append(record)

        if args.dry_run:
            _write_json(out_path, record)
        elif args.overwrite or not record.get("correction_delta"):
            record["_prompt"] = prompt
            pending.append((record, out_path))
        else:
            _write_json(out_path, record)

    model_id = runner.resolve_model_id(args.model)
    backend = runner.resolve_backend(model_id, args.backend)
    loaded = None
    load_seconds = 0.0
    if pending:
        print(f"Loading {model_id} for {len(pending)} text-only refinement(s) ...")
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
                texts, elapsed = _generate_vllm_batch(loaded, prompts, args.max_tokens)
                per_item = elapsed / max(1, len(batch))
                for (record, out_path), text in zip(batch, texts):
                    ok, reason = validate_output_contract(text)
                    record.pop("_prompt", None)
                    record["correction_delta"] = {
                        "text": text,
                        "contract_ok": ok,
                        "contract_reason": reason,
                        "generation_seconds_approx": per_item,
                    }
                    _write_json(out_path, record)
                    print(f"{record['image_key']}: {text}")
        elif loaded is not None:
            for record, out_path in pending:
                prompt = record.pop("_prompt")
                text, elapsed = runner.generate_text(loaded, prompt, max_new_tokens=args.max_tokens)
                ok, reason = validate_output_contract(text)
                record["correction_delta"] = {
                    "text": text,
                    "contract_ok": ok,
                    "contract_reason": reason,
                    "generation_seconds": elapsed,
                }
                _write_json(out_path, record)
                print(f"{record['image_key']}: {text}")
    finally:
        if loaded is not None:
            runner.unload_model(loaded)

    # Strip any transient prompt left by an interrupted/non-generating path.
    for record in records:
        record.pop("_prompt", None)

    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "pose_evidence_dir": str(pose_dir),
        "head_gaze_evidence_dir": str(head_gaze_dir),
        "prompt": str(prompt_path),
        "model_id": model_id,
        "backend": backend,
        "model_load_seconds": load_seconds,
        "record_count": len(records),
        "missing": missing,
        "records": records,
    }
    _write_json(output_dir / "caption_refiner_text_fusion.index.json", index)
    print(f"Text-only caption refinement v0.14: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
