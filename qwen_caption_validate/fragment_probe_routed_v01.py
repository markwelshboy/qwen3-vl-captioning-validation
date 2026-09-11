from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.1"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "fragment-probe-routed-v0.1"
DEFAULT_POSE_PROMPT = PACKAGE_ROOT / "prompts" / "fragment_probe_v08_pose_candidate_relationships.txt"
DEFAULT_CONFIGURATION_PROMPT = PACKAGE_ROOT / "prompts" / "fragment_probe_route_configuration_v01.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SCHEMA_VERSION = "fragment-probe-routed-0.1"
VALID_MODES = {"framing_only", "configuration", "pose_allowed", "pose_guided"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _route_spec(mode: str) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported policy mode: {mode}")
    if mode == "framing_only":
        return {
            "prompt_route": "no_pose_call",
            "model_call": False,
            "pose_candidate_permitted": False,
            "body_relationships_permitted": False,
            "sam3d_guidance_permitted": False,
        }
    if mode == "configuration":
        return {
            "prompt_route": "visible_configuration_only",
            "model_call": True,
            "pose_candidate_permitted": False,
            "body_relationships_permitted": True,
            "sam3d_guidance_permitted": False,
        }
    if mode == "pose_allowed":
        return {
            "prompt_route": "image_pose_candidate_and_relationships",
            "model_call": True,
            "pose_candidate_permitted": True,
            "body_relationships_permitted": True,
            "sam3d_guidance_permitted": False,
        }
    return {
        "prompt_route": "image_pose_candidate_relationships_plus_sam3d_torso",
        "model_call": True,
        "pose_candidate_permitted": True,
        "body_relationships_permitted": True,
        "sam3d_guidance_permitted": True,
    }


def _parse_fact_sheet(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {
            "pose_candidates": [],
            "body_relationships": [],
            "unparsed_lines": [],
            "no_reliable_pose_facts": False,
            "parse_status": "empty",
        }
    if raw.upper() == "NO RELIABLE POSE FACTS":
        return {
            "pose_candidates": [],
            "body_relationships": [],
            "unparsed_lines": [],
            "no_reliable_pose_facts": True,
            "parse_status": "ok",
        }

    section: str | None = None
    pose: list[str] = []
    relationships: list[str] = []
    unparsed: list[str] = []
    for source_line in raw.splitlines():
        line = source_line.strip()
        if not line:
            continue
        heading = line.rstrip(":").strip().upper()
        if heading == "POSE CANDIDATE":
            section = "pose"
            continue
        if heading == "BODY RELATIONSHIPS":
            section = "relationships"
            continue
        if line.startswith("- "):
            value = line[2:].strip()
            if not value:
                continue
            if section == "pose":
                pose.append(value)
            elif section == "relationships":
                relationships.append(value)
            else:
                unparsed.append(line)
        else:
            unparsed.append(line)

    status = "ok" if not unparsed else "partial"
    return {
        "pose_candidates": pose,
        "body_relationships": relationships,
        "unparsed_lines": unparsed,
        "no_reliable_pose_facts": False,
        "parse_status": status,
    }


def _normalize_extraction(parsed: dict[str, Any], mode: str) -> tuple[dict[str, Any], list[str]]:
    spec = _route_spec(mode)
    violations: list[str] = []
    pose_values = [str(v).strip() for v in parsed.get("pose_candidates") or [] if str(v).strip()]
    rel_values = [str(v).strip() for v in parsed.get("body_relationships") or [] if str(v).strip()]

    pose_candidate = None
    if pose_values:
        if spec["pose_candidate_permitted"]:
            pose_candidate = {"text": pose_values[0], "authority": "hypothesis"}
            if len(pose_values) > 1:
                violations.append("multiple_pose_candidates")
        else:
            violations.append("pose_candidate_not_permitted_for_route")

    if not spec["body_relationships_permitted"] and rel_values:
        violations.append("body_relationships_not_permitted_for_route")
        rel_values = []

    if len(rel_values) > 5:
        violations.append("more_than_five_body_relationships")
        rel_values = rel_values[:5]

    relationships = [
        {"text": value, "authority": "visible_candidate"}
        for value in rel_values
    ]
    return {
        "pose_candidate": pose_candidate,
        "body_relationships": relationships,
    }, violations


def _sam3d_model_facts(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Return only the torso-level SAM3D fields permitted to reach Qwen.

    Deliberately excludes face/head, limbs, contacts, camera elevation, and raw
    reconstructed joint coordinates. The Phase-1 crop gate must already have
    selected pose_guided before these facts are considered.
    """
    body = diagnostic.get("body_camera_relation") if isinstance(diagnostic, dict) else None
    gate = diagnostic.get("dwpose_visibility_gate") if isinstance(diagnostic, dict) else None
    if not isinstance(body, dict) or not isinstance(gate, dict):
        return {}
    if not bool(gate.get("body_yaw_observation_gate")):
        return {}

    facts: dict[str, Any] = {}
    band = body.get("orientation_band")
    if band in {"frontal", "slightly_angled", "three_quarter", "side_on", "rear_three_quarter", "rear"}:
        facts["torso_camera_orientation"] = str(band)
    yaw = body.get("yaw_deg")
    if isinstance(yaw, (int, float)):
        facts["torso_yaw_magnitude_deg"] = round(abs(float(yaw)), 1)
    return facts


def _format_sam3d_guidance(facts: dict[str, Any]) -> str:
    if not facts:
        return ""
    lines = [
        "",
        "EXPERT TORSO GUIDANCE",
        "The crop has already passed the observed broad-pose gate. The structured values below are corroborating torso geometry only.",
        "They may refine torso-orientation wording, but MUST NOT be used to infer broad posture, limbs, support/contact, head, gaze, laterality, or hidden anatomy.",
    ]
    if "torso_camera_orientation" in facts:
        lines.append(f"- torso camera orientation: {str(facts['torso_camera_orientation']).replace('_', ' ')}")
    if "torso_yaw_magnitude_deg" in facts:
        lines.append(f"- torso yaw magnitude: {facts['torso_yaw_magnitude_deg']:.1f} degrees")
    lines.extend([
        "Treat this as corroborating guidance, not as proof of anything outside the visible crop.",
        "Return the fact sheet using the original output contract above.",
    ])
    return "\n".join(lines)


def _effective_prompt(mode: str, pose_prompt: str, configuration_prompt: str, sam3d_facts: dict[str, Any] | None = None) -> str:
    if mode == "configuration":
        return configuration_prompt
    if mode == "pose_allowed":
        return pose_prompt
    if mode == "pose_guided":
        return pose_prompt.rstrip() + "\n" + _format_sam3d_guidance(sam3d_facts or {}) + "\n"
    return ""


def _resolve_image(policy: dict[str, Any], images_dir: Path | None) -> Path | None:
    source = policy.get("image")
    if source:
        candidate = Path(str(source)).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    key = str(policy.get("image_key") or "")
    if not key or images_dir is None or not images_dir.is_dir():
        return None
    for ext in IMAGE_EXTENSIONS:
        direct = images_dir / f"{key}{ext}"
        if direct.is_file():
            return direct.resolve()
    matches = sorted(
        p for p in images_dir.rglob(f"{key}.*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return matches[0].resolve() if matches else None


def _load_sam3d_guidance(policy: dict[str, Any]) -> dict[str, Any]:
    sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
    sam_path_text = sources.get("sam3d_arrays")
    dwpose_path_text = sources.get("dwpose")
    result = {
        "used": False,
        "model_facts": {},
        "source": str(sam_path_text) if sam_path_text else None,
        "dwpose_source": str(dwpose_path_text) if dwpose_path_text else None,
        "reason": None,
        "raw_sam3d_image_shown": False,
    }
    if not sam_path_text:
        result["reason"] = "missing_sam3d_arrays"
        return result
    sam_path = Path(str(sam_path_text)).expanduser()
    if not sam_path.is_file():
        result["reason"] = "sam3d_arrays_not_found"
        return result

    try:
        import numpy as np
        from .sam3d_subject_geometry_diagnostic_02 import build_subject_geometry

        with np.load(sam_path, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        dwpose = None
        if dwpose_path_text:
            dwpose_path = Path(str(dwpose_path_text)).expanduser()
            if dwpose_path.is_file():
                dwpose = _read_json(dwpose_path)
        diagnostic = build_subject_geometry(arrays, dwpose)
        facts = _sam3d_model_facts(diagnostic)
        result["model_facts"] = facts
        result["used"] = bool(facts)
        result["reason"] = None if facts else "no_whitelisted_observation_gated_torso_facts"
        return result
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result


def _policy_files(policy_dir: Path, only: set[str]) -> list[Path]:
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if only:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in only]
    return paths


def _guardrails(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "broad_pose_permitted": bool(spec["pose_candidate_permitted"]),
        "body_relationships_permitted": bool(spec["body_relationships_permitted"]),
        "sam3d_guidance_permitted": bool(spec["sam3d_guidance_permitted"]),
        "raw_sam3d_image_shown": False,
        "hidden_anatomy_may_not_be_used_as_visible_evidence": True,
        "head_gaze_framing_camera_laterality_owned_elsewhere": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 route-aware pose fragment extraction driven by Phase-1 perception policy.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pose-prompt", type=Path, default=DEFAULT_POSE_PROMPT)
    parser.add_argument("--configuration-prompt", type=Path, default=DEFAULT_CONFIGURATION_PROMPT)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    pose_prompt_path = args.pose_prompt.expanduser().resolve()
    configuration_prompt_path = args.configuration_prompt.expanduser().resolve()
    for path in (policy_dir, pose_prompt_path, configuration_prompt_path):
        if not path.exists():
            print(f"Required path not found: {path}", file=sys.stderr)
            return 2

    requested = set(args.only)
    policy_paths = _policy_files(policy_dir, requested)
    if not policy_paths:
        print(f"No matching perception-policy records found in {policy_dir}", file=sys.stderr)
        return 2
    found = {p.name.removesuffix(".perception_policy.json") for p in policy_paths}
    missing_requested = sorted(requested - found)
    if missing_requested:
        print("WARNING: requested keys without policy records: " + ", ".join(missing_requested), file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    pose_prompt = pose_prompt_path.read_text(encoding="utf-8")
    configuration_prompt = configuration_prompt_path.read_text(encoding="utf-8")

    records: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for policy_path in policy_paths:
        policy = _read_json(policy_path)
        key = str(policy.get("image_key") or policy_path.name.removesuffix(".perception_policy.json"))
        mode = str((policy.get("policy") or {}).get("mode") or "")
        out_path = output_dir / f"{key}.routed_fragments.json"
        if out_path.is_file() and not args.overwrite:
            records.append(_read_json(out_path))
            continue
        try:
            spec = _route_spec(mode)
        except ValueError as exc:
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "policy_source": str(policy_path),
                "error": str(exc),
            }
            _write_json(out_path, record)
            records.append(record)
            continue

        base = {
            "schema_version": SCHEMA_VERSION,
            "image_key": key,
            "policy_source": str(policy_path),
            "policy_mode": mode,
            "pose_relevance": policy.get("pose_relevance"),
            "prompt_route": spec["prompt_route"],
            "guardrails": _guardrails(spec),
        }
        if not spec["model_call"]:
            record = {
                **base,
                "status": "skipped_by_policy",
                "model_call": False,
                "sam3d_guidance": {"used": False, "model_facts": {}, "raw_sam3d_image_shown": False, "reason": "route_does_not_permit_pose_call"},
                "extraction": {"pose_candidate": None, "body_relationships": []},
                "parse": {"parse_status": "not_applicable", "route_violations": []},
                "raw_response": None,
            }
            _write_json(out_path, record)
            records.append(record)
            print(f"{key}: {mode} -> skipped pose VLM")
            continue

        image = _resolve_image(policy, images_dir)
        if image is None:
            record = {**base, "status": "error", "model_call": False, "error": "source_image_not_found"}
            _write_json(out_path, record)
            records.append(record)
            continue

        guidance = _load_sam3d_guidance(policy) if spec["sam3d_guidance_permitted"] else {
            "used": False,
            "model_facts": {},
            "source": None,
            "dwpose_source": None,
            "reason": "route_does_not_permit_sam3d_guidance",
            "raw_sam3d_image_shown": False,
        }
        prompt = _effective_prompt(mode, pose_prompt, configuration_prompt, guidance.get("model_facts") or {})
        jobs.append({
            "key": key,
            "policy": policy,
            "base": base,
            "spec": spec,
            "image": image,
            "out_path": out_path,
            "guidance": guidance,
            "prompt": prompt,
        })

    if jobs:
        from .extract_v3_wire import _install_image_only_vllm
        from .runner import generate, load_model, resolve_backend, resolve_model_id, unload_model

        model_id = resolve_model_id(args.model)
        backend = resolve_backend(model_id, args.backend)
        if backend == "vllm":
            _install_image_only_vllm()
        print(f"Loading {model_id} once for {len(jobs)} routed pose-fragment call(s) ...")
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
                key = job["key"]
                print(f"\n[{index}/{len(jobs)}] {key}: {job['base']['policy_mode']}")
                try:
                    text, seconds = generate(loaded, job["image"], job["prompt"], max_new_tokens=args.max_tokens)
                    text = text.strip()
                    parsed = _parse_fact_sheet(text)
                    extraction, violations = _normalize_extraction(parsed, job["base"]["policy_mode"])
                    record = {
                        **job["base"],
                        "status": "ok",
                        "model_call": True,
                        "image": str(job["image"]),
                        "model": model_id,
                        "backend": backend,
                        "inference_seconds": seconds,
                        "prompt_sha256": _sha256(job["prompt"]),
                        "prompt_sources": {
                            "pose": str(pose_prompt_path),
                            "configuration": str(configuration_prompt_path),
                        },
                        "sam3d_guidance": job["guidance"],
                        "extraction": extraction,
                        "parse": {
                            "parse_status": parsed["parse_status"],
                            "no_reliable_pose_facts": parsed["no_reliable_pose_facts"],
                            "unparsed_lines": parsed["unparsed_lines"],
                            "route_violations": violations,
                        },
                        "raw_response": text,
                    }
                    _write_json(job["out_path"], record)
                    records.append(record)
                    candidate = (extraction.get("pose_candidate") or {}).get("text")
                    rels = [item["text"] for item in extraction.get("body_relationships") or []]
                    print(f"  pose={candidate or '-'}")
                    print(f"  relationships={'; '.join(rels) if rels else '-'}")
                    if violations:
                        print("  route violations: " + ", ".join(violations))
                except Exception as exc:
                    record = {
                        **job["base"],
                        "status": "error",
                        "model_call": True,
                        "image": str(job["image"]),
                        "model": model_id,
                        "backend": backend,
                        "sam3d_guidance": job["guidance"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _write_json(job["out_path"], record)
                    records.append(record)
                    print(f"ERROR: {record['error']}", file=sys.stderr)
        finally:
            unload_model(loaded)

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    mode_counts = Counter(str(r.get("policy_mode") or "unknown") for r in records)
    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "mode_counts": dict(sorted(mode_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "invariants": {
            "framing_only_has_no_pose_model_call": True,
            "configuration_forbids_broad_pose_candidate": True,
            "sam3d_guidance_only_in_pose_guided": True,
            "sam3d_guidance_is_structured_text_not_raw_visual": True,
            "sam3d_guidance_whitelist": ["torso_camera_orientation", "torso_yaw_magnitude_deg"],
            "hidden_anatomy_may_not_be_used_as_visible_evidence": True,
        },
        "records": records,
    }
    _write_json(output_dir / "fragment_probe_routed.index.json", index)
    print(f"\nIndex: {output_dir / 'fragment_probe_routed.index.json'}")
    return 1 if status_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
