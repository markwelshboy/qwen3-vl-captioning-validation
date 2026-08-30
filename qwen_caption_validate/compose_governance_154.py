from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import compose_fusion_compare as _base
from .caption_projection_154 import lint_caption
from .runner import generate_text, load_model, model_slug, resolve_model_id, unload_model


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FUSION_PROMPT = PACKAGE_ROOT / "prompts" / "compose_identity_fusion_safe_v1.txt"

_GOVERNANCE_ADDENDUM = r"""

PROJECTION 1.5.4 SUBJECT-GEOMETRY SEMANTIC GOVERNANCE
- The supplied caption evidence has already passed Projection 1.5.4. Treat its FACT semantic pose/orientation and required claims as the caption authority; do not reconstruct discarded low-level anatomy from audit history or scene context.
- `pose_orientation.semantic_pose.posture`, when non-null, is a qualified Pose Semantics v0.10 FACT. State it naturally once. If it is null, do not infer standing, seated, squatting, reclining, kneeling, or lying from furniture, crop, or context.
- `pose_orientation.semantic_pose.gestures` contains caption-preferred recognizable action/support primitives. Express them naturally and economically.
- `pose_orientation.subject_geometry_orientation.body_orientation` is the governed body-to-camera orientation. Express `slightly_angled`, `three_quarter`, `side_on`, rear-three-quarter, or rear in ordinary photographic language; never quote reconstructed degrees.
- If `faces_frame` is left/right, it means the BODY is oriented toward that side of the image frame. Do not convert this into anatomical left/right.
- `face_orientation` describes face/head orientation relative to the camera, not eye gaze. Do not turn `toward_camera` into "looking at the camera" unless independent gaze evidence explicitly supports that statement.
- A `head_body_relation` of `turned_toward_camera` is a compound human pose fact. Express body orientation/frame direction plus the head turn as ONE compact relation rather than separate torso, shoulder-depth, and head clauses.
- Do not resurrect `signed_shoulder_nearer_relation`, `signed_torso_depth_direction`, raw shoulder depth, root yaw degrees, or other component geometry removed by Projection 1.5.4.
- Subject-relative camera-center geometry remains audit-only at this stage. Do not infer high-angle/low-angle photography, selfie capture mode, camera hand, or world camera elevation from it.
- Prefer natural human phrasing such as "body nearly side-on with the head turned toward the camera" over scientific or measurement-like language.
- Continue the identity firewall, qualified laterality, framing, scene, clothing/accessory, and lighting rules already present in the base prompt/evidence.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-compose-governance-154",
        description="Generate text-only captions directly from cached Projection 1.5.4 caption evidence.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis-model", default="32b-fp8")
    parser.add_argument("--compose-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--projection-dir", type=Path)
    parser.add_argument("--fusion-prompt", type=Path, default=DEFAULT_FUSION_PROMPT)
    parser.add_argument("--subject-token")
    parser.add_argument("--detail", choices=["concise", "balanced", "detailed"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--run-label")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-caption-tokens", type=int, default=450)
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    return _base._read_json(path)


def _render_prompt(template: str, evidence: dict[str, Any], subject_token: str, detail: str) -> str:
    return _base._render_fusion_prompt(template, evidence, subject_token, detail) + _GOVERNANCE_ADDENDUM


def _projection_ok(evidence: dict[str, Any] | None) -> bool:
    return isinstance(evidence, dict) and evidence.get("projection_revision") == "1.5.4"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        print(f"Run manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    analysis_model_id = resolve_model_id(args.analysis_model)
    analysis_slug = model_slug(analysis_model_id)
    projection_dir = (
        args.projection_dir.expanduser().resolve()
        if args.projection_dir
        else run_dir / "projection-v1.5.4" / analysis_slug
    )
    if not projection_dir.is_dir():
        print(f"Projection 1.5.4 directory not found: {projection_dir}", file=sys.stderr)
        return 2

    compose_model_id = resolve_model_id(args.compose_model)
    compose_slug = model_slug(compose_model_id)
    run_label = _base._safe_label(args.run_label or f"semantic154-{args.backend}-{args.quantization}-{args.dtype}")
    subject_token = args.subject_token or manifest.get("subject_token") or "sH1Vx"
    detail = args.detail or manifest.get("detail") or "balanced"
    fusion_template = args.fusion_prompt.read_text(encoding="utf-8")

    image_records = list(manifest.get("images") or [])
    if args.only:
        needles = tuple(args.only)
        image_records = [
            record
            for record in image_records
            if any(needle in str(record.get("result_key") or "") for needle in needles)
        ]
    if args.limit is not None:
        image_records = image_records[: args.limit]

    output_dir = run_dir / "compose_semantic_154" / f"{analysis_slug}__compose__{compose_slug}__{run_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    missing = invalid_projection = pending = 0
    for record in image_records:
        key = str(record.get("result_key") or "")
        evidence_path = projection_dir / f"{key}.caption_evidence.json"
        audit_path = projection_dir / f"{key}.projection_audit.json"
        evidence = _read_json(evidence_path)
        audit = _read_json(audit_path)
        if evidence is None or audit is None:
            missing += 1
            print(f"WARNING: missing Projection 1.5.4 evidence/audit for {key}; skipping", file=sys.stderr)
            continue
        if not _projection_ok(evidence):
            invalid_projection += 1
            print(f"WARNING: {key} caption evidence is not Projection 1.5.4; skipping", file=sys.stderr)
            continue

        txt = output_dir / f"{key}.fusion-safe.txt"
        meta = output_dir / f"{key}.fusion-safe.json"
        need = args.overwrite or not txt.exists() or not meta.exists()
        pending += int(need)
        prepared.append(
            {
                **record,
                "result_key": key,
                "caption_evidence": evidence,
                "projection_audit": audit,
                "projection_evidence_path": str(evidence_path),
                "projection_audit_path": str(audit_path),
                "txt": txt,
                "meta": meta,
                "need": need,
            }
        )

    if not prepared:
        print("No complete Projection 1.5.4 records found.", file=sys.stderr)
        return 2

    loaded = None
    load_seconds: float | None = None
    if pending:
        print(
            f"Loading {compose_model_id} for Projection 1.5.4 text-only Compose "
            f"({pending} generation(s); run={run_label}) ..."
        )
        loaded = load_model(
            compose_model_id,
            backend=args.backend,
            dtype=args.dtype,
            quantization=args.quantization,
            attn_implementation=args.attn,
            cache_dir=args.cache_dir,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        load_seconds = loaded.load_seconds
        print(f"Loaded in {loaded.load_seconds:.1f}s. Compose remains text-only.")

    try:
        for item in tqdm(prepared, desc="Semantic governed Compose 1.5.4"):
            if not item["need"]:
                continue
            prompt = _render_prompt(fusion_template, item["caption_evidence"], subject_token, detail)
            caption, seconds = generate_text(loaded, prompt, max_new_tokens=args.max_caption_tokens)
            metadata = {
                "schema_version": "compose-semantic-1.5.4",
                "image": item.get("relative_path"),
                "analysis_model": analysis_model_id,
                "compose_model": compose_model_id,
                "compose_seconds": seconds,
                "subject_token": subject_token,
                "detail": detail,
                "image_conditioned": False,
                "run_label": run_label,
                "backend_requested": args.backend,
                "dtype_requested": args.dtype,
                "quantization_requested": args.quantization,
                "projection_revision": "1.5.4",
                "projection_evidence_path": item["projection_evidence_path"],
                "projection_audit_path": item["projection_audit_path"],
                "caption_evidence": item["caption_evidence"],
                "projection_audit": item["projection_audit"],
                "caption_lint": lint_caption(caption, item["caption_evidence"]),
            }
            _base._write_caption(item["txt"], item["meta"], caption, metadata)
    finally:
        if loaded is not None:
            print(f"Unloading {compose_model_id} ...")
            unload_model(loaded)

    records: list[dict[str, Any]] = []
    aggregate_lines: list[str] = []
    for item in prepared:
        caption = _base._read_text(item["txt"])
        meta = _read_json(item["meta"])
        pose = item["caption_evidence"].get("pose_orientation") or {}
        semantic_pose = pose.get("semantic_pose") or {}
        subject_orientation = pose.get("subject_geometry_orientation") or {}
        lint = (meta or {}).get("caption_lint") or {}
        records.append(
            {
                "result_key": item["result_key"],
                "relative_path": item.get("relative_path"),
                "semantic_pose": semantic_pose,
                "subject_geometry_orientation": subject_orientation,
                "caption": caption,
                "word_count": (meta or {}).get("word_count"),
                "compose_seconds": (meta or {}).get("compose_seconds"),
                "caption_lint": lint,
            }
        )
        aggregate_lines.extend(
            [
                f"===== {item['result_key']} =====",
                f"semantic_pose: {json.dumps(semantic_pose, ensure_ascii=False)}",
                f"subject_geometry_orientation: {json.dumps(subject_orientation, ensure_ascii=False)}",
                f"lint: violations={lint.get('violation_count', 0)} warnings={lint.get('warning_count', 0)}",
                caption or "<missing>",
                "",
            ]
        )

    summary = {
        "schema_version": "compose-semantic-1.5.4-run",
        "projection_revision": "1.5.4",
        "analysis_model": analysis_model_id,
        "compose_model": compose_model_id,
        "compose_model_load_seconds": load_seconds,
        "projection_source": str(projection_dir),
        "output_dir": str(output_dir),
        "run_label": run_label,
        "record_count": len(records),
        "missing_projection_records": missing,
        "invalid_projection_records": invalid_projection,
        "records": records,
    }
    summary_path = output_dir / "compose_semantic_154.index.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    aggregate_path = output_dir / "compose_semantic_154.txt"
    aggregate_path.write_text("\n".join(aggregate_lines), encoding="utf-8")

    print(f"Done. Captions: {output_dir}")
    print(f"      JSON:     {summary_path}")
    print(f"      Text:     {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
