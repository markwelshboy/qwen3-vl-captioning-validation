from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v01 as engine
from . import fact_sheet_text_composer_v06 as phase55

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v06.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.7"
SCHEMA_VERSION = "fact-sheet-text-composer-0.7"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"

_TORSO_SIGNED_FRAME_RE = re.compile(
    r"\b(?:torso|upper\s+torso)\b[^.!?]{0,140}\b(?:toward|to)\s+frame\s+(left|right)\b",
    re.I,
)
_UPPER_MORE_TURNED_RE = re.compile(
    r"\bupper\s+torso\b[^.!?]{0,100}\b(?:more|further)\b[^.!?]{0,70}\b(?:turned|angled|toward|to)\b",
    re.I,
)
_UPPER_LESS_TURNED_RE = re.compile(
    r"\bupper\s+torso\b[^.!?]{0,120}\b(?:less\s+turned|closer\s+to\s+frontal|more\s+frontal)\b",
    re.I,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _orientation_magnitude(value: dict[str, Any]) -> float | None:
    for key in ("approx_yaw_deg", "yaw_magnitude_deg"):
        raw = value.get(key)
        if isinstance(raw, (int, float)):
            return float(raw)
    return None


def _articulated_relative_orientation(torso: dict[str, Any]) -> dict[str, Any] | None:
    if torso.get("mode") != "articulated":
        return None
    body = torso.get("body_orientation") if isinstance(torso.get("body_orientation"), dict) else {}
    upper = torso.get("upper_torso_orientation") if isinstance(torso.get("upper_torso_orientation"), dict) else {}
    body_direction = _clean(body.get("turn_direction"))
    upper_direction = _clean(upper.get("turn_direction"))
    body_mag = _orientation_magnitude(body)
    upper_mag = _orientation_magnitude(upper)
    relative = torso.get("relative_twist_magnitude_deg")
    if not isinstance(relative, (int, float)) and body_mag is not None and upper_mag is not None:
        relative = abs(body_mag - upper_mag)
    if not isinstance(relative, (int, float)):
        relative = None

    if body_mag is None or upper_mag is None:
        return None

    relationship: str
    if body_direction and upper_direction and body_direction != upper_direction:
        relationship = "counter_rotated"
    elif upper_mag + 1.0 < body_mag:
        relationship = "upper_torso_closer_to_frontal"
    elif upper_mag > body_mag + 1.0:
        relationship = "upper_torso_more_turned"
    else:
        relationship = "approximately_aligned"

    result: dict[str, Any] = {
        "relationship": relationship,
        "body_yaw_deg": round(body_mag, 1),
        "upper_torso_yaw_deg": round(upper_mag, 1),
    }
    if body_direction:
        result["body_turn_direction"] = body_direction
    if upper_direction:
        result["upper_torso_turn_direction"] = upper_direction
    if relative is not None:
        result["relative_twist_magnitude_deg"] = round(float(relative), 1)

    frame_phrase = (upper_direction or body_direction or "").replace("_", " ")
    amount = round(float(relative)) if relative is not None else None
    if relationship == "upper_torso_closer_to_frontal":
        result["composer_relation"] = (
            f"upper torso is less turned toward {frame_phrase} and closer to frontal than the body"
            + (f" by about {amount} degrees" if amount is not None else "")
        )
    elif relationship == "upper_torso_more_turned":
        result["composer_relation"] = (
            f"upper torso is more turned toward {frame_phrase} than the body"
            + (f" by about {amount} degrees" if amount is not None else "")
        )
    elif relationship == "counter_rotated":
        result["composer_relation"] = (
            "upper torso and body turn in opposite frame directions"
            + (f" with about {amount} degrees of relative twist" if amount is not None else "")
        )
    return result


def _projection(
    sheet: dict[str, Any],
    *,
    trigger_token: str | None = None,
    subject_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    projection, audit = phase55._projection(
        sheet,
        trigger_token=trigger_token,
        subject_class=subject_class,
    )
    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    torso = body.get("torso_orientation") if isinstance(body.get("torso_orientation"), dict) else {}
    relationship = _articulated_relative_orientation(torso)
    if relationship:
        torso["articulated_relative_orientation"] = relationship
        body["torso_orientation"] = torso
        authoritative["body"] = body
        projection["authoritative_facts"] = authoritative
    audit["articulated_relative_orientation_projected"] = relationship is not None
    audit["articulated_relative_orientation"] = relationship
    return projection, audit


def _torso_authorized_frame_directions(projection: dict[str, Any]) -> set[str]:
    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    torso = body.get("torso_orientation") if isinstance(body.get("torso_orientation"), dict) else {}
    directions: set[str] = set()
    direct = _clean(torso.get("turn_direction"))
    if direct:
        directions.add(direct)
    for key in ("body_orientation", "upper_torso_orientation"):
        part = torso.get(key) if isinstance(torso.get(key), dict) else {}
        direction = _clean(part.get("turn_direction"))
        if direction:
            directions.add(direction)
    return directions


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = phase55._caption_audit(caption, projection)
    violations = list(audit.get("violations") or [])

    allowed_directions = _torso_authorized_frame_directions(projection)
    used_directions = {f"frame_{m.group(1).lower()}" for m in _TORSO_SIGNED_FRAME_RE.finditer(caption)}
    unauthorized = sorted(used_directions - allowed_directions)
    if unauthorized:
        violations.append("unauthorized_torso_turn_direction:" + ",".join(unauthorized))

    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    torso = body.get("torso_orientation") if isinstance(body.get("torso_orientation"), dict) else {}
    relation = torso.get("articulated_relative_orientation") if isinstance(torso.get("articulated_relative_orientation"), dict) else {}
    relationship = relation.get("relationship")
    if relationship == "upper_torso_closer_to_frontal" and _UPPER_MORE_TURNED_RE.search(caption):
        violations.append("articulated_torso_relationship_reversed")
    if relationship == "upper_torso_more_turned" and _UPPER_LESS_TURNED_RE.search(caption):
        violations.append("articulated_torso_relationship_reversed")

    audit["violations"] = sorted(set(violations))
    audit["torso_authorized_frame_directions"] = sorted(allowed_directions)
    audit["torso_used_frame_directions"] = sorted(used_directions)
    audit["articulated_relationship"] = relationship
    return audit


def _normalize_caption(text: str) -> str:
    return engine._normalize_caption(text)


def _read_json(path: Path) -> dict[str, Any]:
    return engine._read_json(path)


def _write_json(path: Path, value: Any) -> None:
    engine._write_json(path, value)


def _input_files(input_dir: Path, only: set[str]) -> list[Path]:
    return engine._input_files(input_dir, only)


def _retry_prompt(original_prompt: str, caption: str, violations: list[str]) -> str:
    rules: list[str] = []
    for violation in violations:
        if violation == "gaze_language_without_publishable_gaze":
            rules.append("Remove every statement about gaze, looking, or unknown gaze. Do not replace it with a disclaimer.")
        elif violation.startswith("unauthorized_torso_turn_direction"):
            rules.append("Remove any frame-left/frame-right torso direction not explicitly supplied by the torso evidence. Preserve unsigned torso magnitude/orientation.")
        elif violation == "articulated_torso_relationship_reversed":
            rules.append("Correct the articulated torso comparison using authoritative_facts.body.torso_orientation.articulated_relative_orientation; do not invert more/less turned.")
        elif violation == "unauthorized_anatomical_laterality":
            rules.append("Remove anatomical left/right claims that are not explicitly present in authoritative body configuration while preserving authorized sided relations.")
        else:
            rules.append(f"Repair the audit violation `{violation}` using only the supplied authoritative evidence.")
    joined = "\n- ".join(rules)
    return (
        original_prompt
        + "\n\nAUDIT-GUIDED REVISION\n"
        + "The previous caption failed a deterministic factual audit. Rewrite the whole caption once, preserving all valid detail and changing only what is required to satisfy the audit. Do not discuss the audit.\n"
        + f"Previous caption:\n{caption}\n\nRequired corrections:\n- {joined}\n"
        + "Return only the revised caption as plain text."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-5.6 text-only caption composition with deterministic torso semantics and one audit-guided retry.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--length", choices=["concise", "balanced", "dense"], default="dense")
    p.add_argument("--trigger-token")
    p.add_argument("--subject-class")
    p.add_argument("--model", default="32b-fp8")
    p.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--vllm-max-model-len", type=int, default=8192)
    p.add_argument("--audit-retries", type=int, choices=[0, 1], default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else run_dir / DEFAULT_INPUT_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    prompt_path = args.prompt.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Fact-sheet directory not found: {input_dir}", file=sys.stderr)
        return 2
    if not prompt_path.is_file():
        print(f"Prompt not found: {prompt_path}", file=sys.stderr)
        return 2

    input_paths = _input_files(input_dir, set(args.only))
    if not input_paths:
        print(f"No matching fact sheets found in {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    template = prompt_path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []

    for input_path in input_paths:
        key = input_path.name.removesuffix(".fact_sheet.json")
        out_path = output_dir / f"{key}.composed.json"
        caption_path = output_dir / f"{key}.caption.txt"
        if out_path.is_file() and not args.overwrite:
            records.append(_read_json(out_path))
            continue
        sheet = _read_json(input_path)
        if sheet.get("schema_version") != EXPECTED_FACT_SHEET_SCHEMA:
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "input_fact_sheet": str(input_path),
                "error": f"expected {EXPECTED_FACT_SHEET_SCHEMA}, got {sheet.get('schema_version')}",
            }
            _write_json(out_path, record)
            records.append(record)
            continue
        projection, projection_audit = _projection(sheet, trigger_token=args.trigger_token, subject_class=args.subject_class)
        evidence_json = json.dumps(projection, indent=2, ensure_ascii=False)
        effective_prompt = template.replace("{{LENGTH_PROFILE}}", args.length).replace("{{COMPOSER_EVIDENCE_JSON}}", evidence_json)
        jobs.append({
            "key": key,
            "input_path": input_path,
            "out_path": out_path,
            "caption_path": caption_path,
            "sheet": sheet,
            "projection": projection,
            "projection_audit": projection_audit,
            "prompt": effective_prompt,
        })

    if jobs:
        from .runner import generate_text, load_model, resolve_backend, resolve_model_id, unload_model

        model_id = resolve_model_id(args.model)
        backend = resolve_backend(model_id, args.backend)
        print(f"Loading {model_id} once for {len(jobs)} text-only composer call(s) ...")
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
                print(f"\n[{index}/{len(jobs)}] {job['key']}: text-only compose")
                try:
                    raw, seconds = generate_text(loaded, job["prompt"], max_new_tokens=args.max_tokens)
                    caption = _normalize_caption(raw)
                    if not caption:
                        raise ValueError("empty composer response")
                    initial_caption = caption
                    initial_raw = raw
                    initial_audit = _caption_audit(caption, job["projection"])
                    final_audit = initial_audit
                    retry_attempts: list[dict[str, Any]] = []
                    total_seconds = float(seconds)

                    if initial_audit["violations"] and args.audit_retries:
                        retry_prompt = _retry_prompt(job["prompt"], caption, list(initial_audit["violations"]))
                        retry_raw, retry_seconds = generate_text(loaded, retry_prompt, max_new_tokens=args.max_tokens)
                        retry_caption = _normalize_caption(retry_raw)
                        if retry_caption:
                            retry_audit = _caption_audit(retry_caption, job["projection"])
                            retry_attempts.append({
                                "attempt": 1,
                                "violations_requested": list(initial_audit["violations"]),
                                "caption": retry_caption,
                                "caption_audit": retry_audit,
                                "raw_response": retry_raw,
                                "inference_seconds": retry_seconds,
                            })
                            caption = retry_caption
                            raw = retry_raw
                            final_audit = retry_audit
                            total_seconds += float(retry_seconds)

                    status = "ok" if not final_audit["violations"] else "needs_review"
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "status": status,
                        "image_key": job["key"],
                        "policy_mode": ((job["sheet"].get("policy") or {}).get("mode")),
                        "input_fact_sheet": str(job["input_path"]),
                        "model": model_id,
                        "backend": backend,
                        "inference_seconds": total_seconds,
                        "length_profile": args.length,
                        "prompt_source": str(prompt_path),
                        "prompt_sha256": _sha256(template),
                        "composer_has_image_access": False,
                        "evidence_projection": job["projection"],
                        "projection_audit": job["projection_audit"],
                        "caption": caption,
                        "caption_audit": final_audit,
                        "raw_response": raw,
                        "initial_caption": initial_caption,
                        "initial_caption_audit": initial_audit,
                        "initial_raw_response": initial_raw,
                        "audit_retry_count": len(retry_attempts),
                        "audit_retry_attempts": retry_attempts,
                    }
                    _write_json(job["out_path"], record)
                    job["caption_path"].write_text(caption + "\n", encoding="utf-8")
                    records.append(record)
                    print(f"  {status} | {final_audit['word_count']} words | retries={len(retry_attempts)}")
                    print(f"  {caption}")
                    if final_audit["violations"]:
                        print("  VIOLATIONS: " + ", ".join(final_audit["violations"]))
                except Exception as exc:
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "status": "error",
                        "image_key": job["key"],
                        "input_fact_sheet": str(job["input_path"]),
                        "model": model_id,
                        "backend": backend,
                        "composer_has_image_access": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _write_json(job["out_path"], record)
                    records.append(record)
                    print(f"ERROR: {record['error']}", file=sys.stderr)
        finally:
            unload_model(loaded)

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    retry_count = sum(int(r.get("audit_retry_count") or 0) for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "length_profile": args.length,
        "trigger_token": args.trigger_token,
        "subject_class": args.subject_class,
        "audit_retry_count": retry_count,
        "invariants": {
            "text_only": True,
            "composer_has_image_access": False,
            "only_composer_eligible_specialist_facts_are_projected": True,
            "held_and_diagnostic_only_facts_are_not_projected": True,
            "review_conflict_domains_are_omitted": True,
            "holistic_context_is_sanitized_and_non_authoritative": True,
            "articulated_torso_relationship_is_deterministically_projected": True,
            "unsigned_torso_direction_is_audited": True,
            "at_most_one_audit_guided_retry": True,
            "retry_does_not_receive_image_access": True,
        },
        "records": records,
    }
    index_path = output_dir / "text_composer.index.json"
    _write_json(index_path, index)
    print(f"\nIndex: {index_path}")
    return 1 if status_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
