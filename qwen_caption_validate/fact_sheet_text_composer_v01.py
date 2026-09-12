from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.1"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v01.txt"
SCHEMA_VERSION = "fact-sheet-text-composer-0.1"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.2.1"

SUBJECT_HINT_RE = re.compile(r"\b(woman|man|person)\b", re.I)
FRAMING_WORD_RE = re.compile(
    r"\b(close[- ]?up|medium close[- ]?up|full[- ]?body|full length|waist[- ]?up|upper[- ]?body)\b",
    re.I,
)
POSTURE_WORD_RE = re.compile(
    r"\b(stands?|standing|sits?|sitting|seated|lies?|lying|reclin(?:e|es|ed|ing)|crouch(?:es|ed|ing)?|kneel(?:s|ed|ing)?)\b",
    re.I,
)
LOOKING_CLAUSE_RE = re.compile(r"(?:,?\s*(?:while\s+)?)?looking\b[^,.;]*", re.I)
RESTRICTED_CONTEXT_RE = re.compile(
    r"\b(looking|gaze|standing|stand|stands|seated|sitting|sit|lying|lies|reclined|reclining|"
    r"crouch(?:ed|ing)?|kneel(?:ed|ing)?|bent forward|lean(?:ing|ed)?|head|neck|torso|shoulder|"
    r"arm|forearm|hand|hands|hip|knee|leg|foot|feet|frame_left|frame_right|frame left|frame right|"
    r"close[- ]?up|medium close[- ]?up|full[- ]?body|waist[- ]?up|camera angle|high angle|low angle)\b",
    re.I,
)
ANATOMICAL_LATERALITY_RE = re.compile(
    r"\b(left|right)\s+(hand|arm|forearm|wrist|elbow|shoulder|hip|knee|leg|ankle|foot|eye|ear)\b",
    re.I,
)
POLICY_LANGUAGE_RE = re.compile(
    r"\b(SAM3D|DWPose|Qwen|reconstruction|authority|routing|policy|fact sheet|diagnostic|confidence score|hidden anatomy)\b",
    re.I,
)
HEAD_ORIENTATION_RE = re.compile(
    r"\b(head|face|neck)\b.{0,40}\b(turned|turning|tilted|tilting|angled|yaw|pitch|facing)\b"
    r"|\b(turned|turning|tilted|tilting|angled)\b.{0,40}\b(head|face|neck)\b",
    re.I,
)
TORSO_ORIENTATION_RE = re.compile(
    r"\b(torso|body)\b.{0,50}\b(frontal|three[- ]quarter|side[- ]on|profile|angled|turned|facing|toward camera|away from camera)\b",
    re.I,
)
GAZE_LANGUAGE_RE = re.compile(r"\b(gaze|looking|looks|looked)\b", re.I)
TOWARD_CAMERA_RE = re.compile(r"\b(looking|gaze|eyes?)\b.{0,35}\b(at|into|toward)\s+(the\s+)?camera\b|\beye contact\b", re.I)
OFF_CAMERA_RE = re.compile(r"\b(looking|gaze|eyes?)\b.{0,35}\b(away from|off[- ]camera)\b", re.I)

POSE_GROUPS: dict[str, re.Pattern[str]] = {
    "standing": re.compile(r"\b(stand|stands|standing)\b", re.I),
    "seated": re.compile(r"\b(seated|sit|sits|sitting)\b", re.I),
    "lying": re.compile(r"\b(lie|lies|lying|reclined|reclining)\b", re.I),
    "crouching": re.compile(r"\b(crouch|crouches|crouched|crouching)\b", re.I),
    "kneeling": re.compile(r"\b(kneel|kneels|kneeled|kneeling)\b", re.I),
}

FRAMING_PHRASES = {
    "full_length": "full-body framing",
    "three_quarter_or_long": "long or three-quarter framing",
    "waist_or_upper_body": "upper-body or waist-up framing",
    "close_or_medium_close": "close framing",
    "face_or_partial_body": "tight face or partial-body framing",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _clean(value)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _composer_values(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _clean(item.get("composer_text"))
        if text:
            out.append(text)
    return _dedupe(out)


def _reference_hint(context: dict[str, Any]) -> str | None:
    gestalt = context.get("gestalt") if isinstance(context.get("gestalt"), dict) else {}
    text = _clean(gestalt.get("text"))
    if not text:
        return None
    match = SUBJECT_HINT_RE.search(text)
    return match.group(1).lower() if match else None


def _sanitize_holistic_context(context: dict[str, Any]) -> str | None:
    gestalt = context.get("gestalt") if isinstance(context.get("gestalt"), dict) else {}
    text = _clean(gestalt.get("text"))
    if not text:
        return None
    text = LOOKING_CLAUSE_RE.sub("", text)
    text = FRAMING_WORD_RE.sub("", text)
    text = POSTURE_WORD_RE.sub("", text)
    text = re.sub(r"\b(?:bent|leaning|leaned)\s+(?:forward|backward|back)\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:toward|to)\s+frame\s+(?:left|right)\b", "", text, flags=re.I)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;")
    return text or None


def _safe_context_values(context: dict[str, Any]) -> list[str]:
    values = context.get("expression_action")
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        text = _clean(item.get("text"))
        if not text or RESTRICTED_CONTEXT_RE.search(text):
            continue
        out.append(text)
    return _dedupe(out)


def _broad_pose(body: dict[str, Any]) -> str | None:
    pose = body.get("pose_candidate") if isinstance(body.get("pose_candidate"), dict) else {}
    text = _clean(pose.get("text"))
    if not text:
        return None
    if pose.get("promotion_status") not in {"candidate", "accepted", "resolved"}:
        return None
    return text


def _torso_fact(body: dict[str, Any]) -> dict[str, Any] | None:
    torso = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}
    if not torso.get("available") or not torso.get("composer_eligible"):
        return None
    orientation = _clean(torso.get("torso_camera_orientation"))
    if not orientation:
        return None
    return {"camera_orientation": orientation}


def _head_fact(facts: dict[str, Any]) -> dict[str, Any] | None:
    raw = facts.get("head_pose") if isinstance(facts.get("head_pose"), dict) else {}
    if not raw.get("available"):
        return None
    out: dict[str, Any] = {}
    horizontal = raw.get("horizontal") if isinstance(raw.get("horizontal"), dict) else {}
    vertical = raw.get("vertical") if isinstance(raw.get("vertical"), dict) else {}
    if horizontal.get("publishable") and _clean(horizontal.get("value")):
        out["horizontal"] = _clean(horizontal.get("value"))
    if vertical.get("publishable") and _clean(vertical.get("value")):
        out["vertical"] = _clean(vertical.get("value"))
    strength = _clean(raw.get("yaw_strength"))
    if strength:
        out["yaw_strength"] = strength
    return out or None


def _gaze_fact(facts: dict[str, Any]) -> dict[str, Any] | None:
    raw = facts.get("gaze") if isinstance(facts.get("gaze"), dict) else {}
    if not raw.get("publishable"):
        return None
    out: dict[str, Any] = {}
    for key in ("horizontal", "vertical"):
        value = _clean(raw.get(key))
        if value:
            out[key] = value
    relationship = _clean(raw.get("camera_relationship"))
    if relationship and relationship != "uncertain":
        out["camera_relationship"] = relationship
    return out or None


def _visual_fact_values(visual: dict[str, Any], field: str) -> list[str]:
    return _composer_values(visual.get(field))


def _projection(
    sheet: dict[str, Any],
    *,
    trigger_token: str | None = None,
    subject_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    visual = facts.get("visual") if isinstance(facts.get("visual"), dict) else {}
    framing = facts.get("framing") if isinstance(facts.get("framing"), dict) else {}
    context = sheet.get("context_only") if isinstance(sheet.get("context_only"), dict) else {}
    audit = sheet.get("audit") if isinstance(sheet.get("audit"), dict) else {}

    extent = _clean(framing.get("extent"))
    framing_text = FRAMING_PHRASES.get(extent or "", extent)
    conflicts = audit.get("review_conflicts") if isinstance(audit.get("review_conflicts"), list) else []
    conflict_domains = sorted({
        str(item.get("domain"))
        for item in conflicts
        if isinstance(item, dict) and item.get("domain")
    })

    authoritative = {
        "framing": {"extent": framing_text} if framing_text else {},
        "body": {
            "broad_pose": _broad_pose(body),
            "configuration": _composer_values(body.get("configuration")),
            "torso_orientation": _torso_fact(body),
        },
        "head": _head_fact(facts),
        "gaze": _gaze_fact(facts),
        "appearance": _visual_fact_values(visual, "appearance"),
        "objects": _visual_fact_values(visual, "objects"),
        "scene": _visual_fact_values(visual, "scene"),
        "secondary_people": _visual_fact_values(visual, "secondary_people"),
        "safe_expression_action": _safe_context_values(context),
    }

    # Remove empty/null leaves so absence itself remains meaningful to the model.
    body_out = authoritative["body"]
    body_out = {k: v for k, v in body_out.items() if v not in (None, [], {})}
    authoritative["body"] = body_out
    authoritative = {k: v for k, v in authoritative.items() if v not in (None, [], {})}

    projection = {
        "schema_version": SCHEMA_VERSION + "-evidence",
        "subject": {
            "trigger_token": _clean(trigger_token) or "",
            "subject_class": _clean(subject_class) or "",
            "reference_hint": _reference_hint(context),
        },
        "authoritative_facts": authoritative,
        "holistic_context_non_authoritative": _sanitize_holistic_context(context),
        "omitted_review_conflict_domains": conflict_domains,
    }
    projection_audit = {
        "source_fact_sheet_schema": sheet.get("schema_version"),
        "source_phase": audit.get("phase"),
        "raw_images_available_to_composer": False,
        "diagnostic_only_torso_removed": not bool(_torso_fact(body)) and bool((body.get("torso_geometry") or {}).get("available")),
        "held_configuration_removed": any(
            isinstance(item, dict) and not _clean(item.get("composer_text"))
            for item in (body.get("configuration") or [])
        ),
        "review_conflict_domains_omitted": conflict_domains,
        "context_expression_action_filtered": True,
        "holistic_context_sanitized": True,
    }
    return projection, projection_audit


def _allowed_pose_groups(pose_text: str | None) -> set[str]:
    if not pose_text:
        return set()
    allowed: set[str] = set()
    for name, pattern in POSE_GROUPS.items():
        if pattern.search(pose_text):
            allowed.add(name)
    return allowed


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    warnings: list[str] = []
    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    head = authoritative.get("head") if isinstance(authoritative.get("head"), dict) else None
    gaze = authoritative.get("gaze") if isinstance(authoritative.get("gaze"), dict) else None

    allowed_groups = _allowed_pose_groups(_clean(body.get("broad_pose")))
    used_groups = {name for name, pattern in POSE_GROUPS.items() if pattern.search(caption)}
    unauthorized_groups = sorted(used_groups - allowed_groups)
    if unauthorized_groups:
        violations.append("unauthorized_broad_pose:" + ",".join(unauthorized_groups))

    if ANATOMICAL_LATERALITY_RE.search(caption):
        violations.append("unauthorized_anatomical_laterality")
    if POLICY_LANGUAGE_RE.search(caption):
        violations.append("policy_or_model_language_leak")

    if not gaze and GAZE_LANGUAGE_RE.search(caption):
        violations.append("gaze_language_without_publishable_gaze")
    if gaze:
        relationship = _clean(gaze.get("camera_relationship"))
        if relationship != "toward_camera" and TOWARD_CAMERA_RE.search(caption):
            violations.append("toward_camera_claim_without_authority")
        if relationship != "off_camera" and OFF_CAMERA_RE.search(caption):
            violations.append("off_camera_claim_without_authority")

    if not head and HEAD_ORIENTATION_RE.search(caption):
        violations.append("head_orientation_without_publishable_head")

    conflict_domains = set(projection.get("omitted_review_conflict_domains") or [])
    if "torso_camera_orientation" in conflict_domains and TORSO_ORIENTATION_RE.search(caption):
        violations.append("torso_orientation_reintroduced_despite_review_conflict")

    if len(caption.split()) < 25:
        warnings.append("very_short_caption")

    return {
        "violations": sorted(set(violations)),
        "warnings": sorted(set(warnings)),
        "word_count": len(caption.split()),
    }


def _normalize_caption(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:text)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return " ".join(value.split())


def _input_files(input_dir: Path, only: set[str]) -> list[Path]:
    paths = sorted(input_dir.glob("*.fact_sheet.json"))
    if only:
        paths = [p for p in paths if p.name.removesuffix(".fact_sheet.json") in only]
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-5 text-only caption composition from the frozen Phase-4B.1 fact sheet.")
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
        projection, projection_audit = _projection(
            sheet,
            trigger_token=args.trigger_token,
            subject_class=args.subject_class,
        )
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
                    caption_audit = _caption_audit(caption, job["projection"])
                    status = "ok" if not caption_audit["violations"] else "needs_review"
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "status": status,
                        "image_key": job["key"],
                        "policy_mode": ((job["sheet"].get("policy") or {}).get("mode")),
                        "input_fact_sheet": str(job["input_path"]),
                        "model": model_id,
                        "backend": backend,
                        "inference_seconds": seconds,
                        "length_profile": args.length,
                        "prompt_source": str(prompt_path),
                        "prompt_sha256": _sha256(template),
                        "composer_has_image_access": False,
                        "evidence_projection": job["projection"],
                        "projection_audit": job["projection_audit"],
                        "caption": caption,
                        "caption_audit": caption_audit,
                        "raw_response": raw,
                    }
                    _write_json(job["out_path"], record)
                    job["caption_path"].write_text(caption + "\n", encoding="utf-8")
                    records.append(record)
                    print(f"  {status} | {caption_audit['word_count']} words")
                    print(f"  {caption}")
                    if caption_audit["violations"]:
                        print("  VIOLATIONS: " + ", ".join(caption_audit["violations"]))
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
        "invariants": {
            "text_only": True,
            "composer_has_image_access": False,
            "only_composer_eligible_specialist_facts_are_projected": True,
            "held_and_diagnostic_only_facts_are_not_projected": True,
            "review_conflict_domains_are_omitted": True,
            "holistic_context_is_sanitized_and_non_authoritative": True,
            "no_automatic_repair_after_composition": True,
        },
        "records": records,
    }
    index_path = output_dir / "text_composer.index.json"
    _write_json(index_path, index)
    print(f"\nIndex: {index_path}")
    return 1 if status_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
