from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .analysis_v2_normalize import normalize_analysis_v2
from .caption_lint import lint_caption
from .caption_projection import build_caption_projection
from .pose_evidence import build_pose_evidence
from .runner import (
    generate_text,
    load_model,
    model_slug,
    render_compose_prompt,
    resolve_model_id,
    unload_model,
)


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS_PROMPT = PACKAGE_ROOT / "prompts" / "compose_identity_pose_v1.txt"
DEFAULT_DWPOSE_PROMPT = PACKAGE_ROOT / "prompts" / "compose_identity_pose_dwpose_v1.txt"
DEFAULT_FUSION_PROMPT = PACKAGE_ROOT / "prompts" / "compose_identity_fusion_safe_v1.txt"
VARIANTS = ("analysis", "dwpose", "fusion-safe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-compose-fusion-compare",
        description=(
            "Generate text-only Analyze, DWPose-assisted, and governed Fusion-v2.3 captions "
            "from cached evidence. No image-conditioned inference is performed."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing Analyze-v2.1 run directory.")
    parser.add_argument(
        "--analysis-model",
        default="32b-fp8",
        help="Model that produced the cached Analyze run (default: 32b-fp8).",
    )
    parser.add_argument(
        "--compose-model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Text-only model used to write all comparison captions (default: Qwen3-VL-4B-Instruct).",
    )
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose directory (default: <run_dir>/dwpose).")
    parser.add_argument(
        "--fusion-dir",
        type=Path,
        help="Fusion-v2.3 directory (default: <run_dir>/fusion-v2.3/<analysis-model-slug>).",
    )
    parser.add_argument("--analysis-prompt", type=Path, default=DEFAULT_ANALYSIS_PROMPT)
    parser.add_argument("--dwpose-prompt", type=Path, default=DEFAULT_DWPOSE_PROMPT)
    parser.add_argument("--fusion-prompt", type=Path, default=DEFAULT_FUSION_PROMPT)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Caption variants to generate (default: analysis dwpose fusion-safe).",
    )
    parser.add_argument("--subject-token", help="Override trigger token; otherwise inherit run.json.")
    parser.add_argument("--gender-grammar", help="Optional grammar label carried into the caption policy.")
    parser.add_argument("--subject-pronoun", help="Optional subject pronoun, e.g. she/he/they.")
    parser.add_argument("--object-pronoun", help="Optional object pronoun, e.g. her/him/them.")
    parser.add_argument("--possessive-pronoun", help="Optional possessive pronoun, e.g. her/his/their.")
    parser.add_argument(
        "--protected-trait",
        action="append",
        default=[],
        help="Additional project-protected identity trait. May be repeated.",
    )
    parser.add_argument(
        "--detail",
        choices=["concise", "balanced", "detailed"],
        help="Override detail profile; otherwise inherit run.json.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N matching images.")
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only process result keys containing any supplied string, e.g. 00015 00012.",
    )
    parser.add_argument(
        "--run-label",
        help=(
            "Label used to isolate outputs for precision/quantization comparisons. "
            "Default: <backend>-<quantization>-<dtype>."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-caption-tokens", type=int, default=450)
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    return parser.parse_args()


def _safe_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return text.strip("-._") or "default"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _render_pose_prompt(
    template: str,
    analysis: dict[str, Any],
    pose_evidence: dict[str, Any],
    subject_token: str,
    detail: str,
) -> str:
    return (
        template.replace("{{SUBJECT_TOKEN}}", subject_token)
        .replace("{{DETAIL_PROFILE}}", detail)
        .replace("{{ANALYSIS_JSON}}", json.dumps(analysis, indent=2, ensure_ascii=False))
        .replace("{{POSE_EVIDENCE_JSON}}", json.dumps(pose_evidence, indent=2, ensure_ascii=False))
    )


def _render_fusion_prompt(
    template: str,
    caption_evidence: dict[str, Any],
    subject_token: str,
    detail: str,
) -> str:
    return (
        template.replace("{{SUBJECT_TOKEN}}", subject_token)
        .replace("{{DETAIL_PROFILE}}", detail)
        .replace("{{CAPTION_EVIDENCE_JSON}}", json.dumps(caption_evidence, indent=2, ensure_ascii=False))
    )


def _caption_policy(args: argparse.Namespace, manifest: dict[str, Any], subject_token: str) -> dict[str, Any]:
    stored = manifest.get("caption_policy") or {}
    protected = [str(value) for value in (stored.get("protected_traits") or []) if str(value).strip()]
    protected.extend(str(value) for value in args.protected_trait if str(value).strip())
    return {
        "trigger_token": subject_token,
        "gender_grammar": args.gender_grammar or stored.get("gender_grammar"),
        "subject_pronoun": args.subject_pronoun or stored.get("subject_pronoun"),
        "object_pronoun": args.object_pronoun or stored.get("object_pronoun"),
        "possessive_pronoun": args.possessive_pronoun or stored.get("possessive_pronoun"),
        "protected_traits": protected,
    }


def _write_caption(
    path: Path,
    meta_path: Path,
    caption: str,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption.strip() + "\n", encoding="utf-8")
    payload = dict(metadata)
    payload["caption"] = caption.strip()
    payload["word_count"] = len(caption.strip().split())
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _caption_cell(title: str, caption: str | None, metadata: dict[str, Any] | None) -> str:
    text = html.escape(caption or "")
    words = (metadata or {}).get("word_count")
    seconds = (metadata or {}).get("compose_seconds")
    lint = (metadata or {}).get("caption_lint") or {}
    stats: list[str] = []
    if words is not None:
        stats.append(f"{words} words")
    if seconds is not None:
        try:
            stats.append(f"{float(seconds):.2f}s")
        except (TypeError, ValueError):
            pass
    badge = ""
    if lint:
        if lint.get("violation_count"):
            badge = '<span class="badge fail">LINT FAIL</span>'
        elif lint.get("warning_count"):
            badge = '<span class="badge warn">LINT WARN</span>'
        else:
            badge = '<span class="badge pass">LINT PASS</span>'
    stat_text = " · ".join(stats)
    return (
        f'<div class="captionbox"><h3>{html.escape(title)} {badge}</h3>'
        f'<p>{text or "<em>missing</em>"}</p>'
        f'<div class="stats">{html.escape(stat_text)}</div></div>'
    )


def _build_html(
    run_dir: Path,
    analysis_slug: str,
    compose_slug: str,
    run_label: str,
    records: list[dict[str, Any]],
) -> Path:
    out = run_dir / f"compose_fusion_compare_{analysis_slug}__compose__{compose_slug}__{run_label}.html"
    cards: list[str] = []
    for record in records:
        image_src = html.escape(record.get("report_image") or "")
        key = html.escape(record["result_key"])
        existing = html.escape(record.get("existing_caption") or "")
        analysis_caption = record.get("captions", {}).get("analysis")
        dwpose_caption = record.get("captions", {}).get("dwpose")
        fusion_caption = record.get("captions", {}).get("fusion-safe")
        metadata = record.get("metadata") or {}
        safe_json = html.escape(json.dumps(record.get("caption_evidence") or {}, indent=2, ensure_ascii=False))
        audit_json = html.escape(json.dumps(record.get("firewall_audit") or {}, indent=2, ensure_ascii=False))
        lint_json = html.escape(json.dumps((metadata.get("fusion-safe") or {}).get("caption_lint") or {}, indent=2, ensure_ascii=False))
        normalization = html.escape(json.dumps(record.get("normalization_actions") or [], indent=2, ensure_ascii=False))
        raw_summary = html.escape(record.get("image_summary") or "")
        cards.append(
            f"""
<section class="card">
  <h2>{key}</h2>
  <div class="topgrid">
    <div><img src="{image_src}" loading="lazy"></div>
    <div>
      <h3>Raw Analyze summary <span class="badge blocked">withheld from governed Compose</span></h3>
      <p class="summary">{raw_summary or '<em>none</em>'}</p>
      <h3>Existing caption</h3><p>{existing or '<em>none</em>'}</p>
      <details><summary>Normalization actions</summary><pre>{normalization}</pre></details>
    </div>
  </div>
  <div class="captiongrid">
    {_caption_cell('A · Analyze-only', analysis_caption, metadata.get('analysis'))}
    {_caption_cell('B · Analyze + DWPose', dwpose_caption, metadata.get('dwpose'))}
    {_caption_cell('C · Governed task-shaped evidence', fusion_caption, metadata.get('fusion-safe'))}
  </div>
  <details><summary>Caption authority lint for C</summary><pre>{lint_json}</pre></details>
  <details><summary>Caption Evidence 1.3 passed to C</summary><pre>{safe_json}</pre></details>
  <details><summary>Firewall + projection audit</summary><pre>{audit_json}</pre></details>
</section>
"""
        )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Governed caption comparison</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;line-height:1.45;background:#f5f5f5;color:#171717}}
h1{{margin-bottom:6px}} .lede{{color:#555;max-width:1100px}}
.card{{background:white;border:1px solid #ddd;border-radius:10px;padding:18px;margin:20px 0}}
.topgrid{{display:grid;grid-template-columns:minmax(260px,32%) 1fr;gap:22px;align-items:start}}
.captiongrid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0}}
.captionbox{{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}}
.captionbox h3{{margin-top:0}} .stats{{font-size:12px;color:#666}}
img{{width:100%;height:auto;max-height:620px;object-fit:contain;background:#111;border-radius:6px}}
.summary{{font-weight:500}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:12px;border-radius:6px;font-size:12px}}
.badge{{font-size:11px;border-radius:999px;padding:3px 7px;font-weight:700;vertical-align:middle}}
.blocked{{background:#f3d6d6;color:#762020}} .pass{{background:#d9f0df;color:#1d6030}} .warn{{background:#fff0c9;color:#704d00}} .fail{{background:#f3d6d6;color:#762020}}
details{{margin-top:10px}}
@media(max-width:1050px){{.captiongrid{{grid-template-columns:1fr}}}}
@media(max-width:850px){{.topgrid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Analyze vs DWPose vs governed task-shaped Compose</h1>
<p class="lede">All captions are text-only generations from cached evidence; the image is shown only for human review. Variant C receives caption-evidence-1.3: quarantined transient appearance, side-neutral pose/orientation, directly qualified whole-body posture, framing/camera, environment/lighting, required claims, and hard constraints. Raw reconstruction, raw summary prose, free-form uncertainty prose, horizontal frame-side hints, intrinsic-identity subparts, unsupported posture, and unsupported distal anatomy are withheld.</p>
<p class="lede"><strong>Analyze source:</strong> {html.escape(analysis_slug)} &nbsp; <strong>Compose model:</strong> {html.escape(compose_slug)} &nbsp; <strong>Run:</strong> {html.escape(run_label)}</p>
{''.join(cards)}
</body></html>"""
    out.write_text(document, encoding="utf-8")
    return out


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run.json"
    if not manifest_path.exists():
        print(f"Run manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    analysis_model_id = resolve_model_id(args.analysis_model)
    analysis_slug = model_slug(analysis_model_id)
    analysis_dir = run_dir / analysis_slug
    if not analysis_dir.is_dir():
        print(f"Analysis directory not found for {analysis_model_id}: {analysis_dir}", file=sys.stderr)
        return 2

    compose_model_id = resolve_model_id(args.compose_model)
    compose_slug = model_slug(compose_model_id)
    default_label = f"{args.backend}-{args.quantization}-{args.dtype}"
    run_label = _safe_label(args.run_label or default_label)
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    fusion_dir = (
        args.fusion_dir.expanduser().resolve()
        if args.fusion_dir
        else run_dir / "fusion-v2.3" / analysis_slug
    )

    if "dwpose" in args.variants and not dwpose_dir.is_dir():
        print(f"DWPose directory not found: {dwpose_dir}", file=sys.stderr)
        return 2
    if "fusion-safe" in args.variants and not fusion_dir.is_dir():
        print(f"Fusion-v2.3 directory not found: {fusion_dir}", file=sys.stderr)
        return 2

    analysis_template = args.analysis_prompt.read_text(encoding="utf-8")
    dwpose_template = args.dwpose_prompt.read_text(encoding="utf-8")
    fusion_template = args.fusion_prompt.read_text(encoding="utf-8")
    subject_token = args.subject_token or manifest.get("subject_token") or "sH1Vx"
    detail = args.detail or manifest.get("detail") or "balanced"
    caption_policy = _caption_policy(args, manifest, subject_token)

    image_records = list(manifest.get("images") or [])
    if args.only:
        needles = tuple(args.only)
        image_records = [
            record
            for record in image_records
            if any(needle in str(record.get("result_key", "")) for needle in needles)
        ]
    if args.limit is not None:
        image_records = image_records[: args.limit]

    compare_dir = run_dir / "compose_fusion_compare" / f"{analysis_slug}__compose__{compose_slug}__{run_label}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    pending_count = 0
    for record in image_records:
        key = str(record.get("result_key") or "")
        analysis_record = _read_json(analysis_dir / f"{key}.analysis.json")
        raw_analysis = (analysis_record or {}).get("analysis")
        if not isinstance(raw_analysis, dict):
            print(f"WARNING: missing/invalid analysis for {key}; skipping", file=sys.stderr)
            continue
        analysis, normalization_actions = normalize_analysis_v2(raw_analysis)

        pose_evidence: dict[str, Any] | None = None
        if "dwpose" in args.variants:
            dwpose_record = _read_json(dwpose_dir / f"{key}.dwpose.json")
            if dwpose_record is None:
                print(f"WARNING: missing DWPose record for {key}; DWPose variant unavailable", file=sys.stderr)
            else:
                pose_evidence = build_pose_evidence(dwpose_record)

        fusion_payload: dict[str, Any] | None = None
        caption_evidence: dict[str, Any] | None = None
        firewall_audit: dict[str, Any] | None = None
        if "fusion-safe" in args.variants:
            fusion_payload = _read_json(fusion_dir / f"{key}.fused_v2_3.json")
            if fusion_payload is None:
                print(f"WARNING: missing Fusion-v2.3 record for {key}; safe variant unavailable", file=sys.stderr)
            else:
                caption_evidence, firewall_audit = build_caption_projection(
                    fusion_payload,
                    analysis,
                    caption_policy=caption_policy,
                )

        variant_files: dict[str, dict[str, Any]] = {}
        for variant in args.variants:
            available = (
                variant == "analysis"
                or (variant == "dwpose" and pose_evidence is not None)
                or (variant == "fusion-safe" and caption_evidence is not None)
            )
            if not available:
                continue
            txt = compare_dir / f"{key}.{variant}.txt"
            meta = compare_dir / f"{key}.{variant}.json"
            need = args.overwrite or not txt.exists() or not meta.exists()
            pending_count += int(need)
            variant_files[variant] = {"txt": txt, "meta": meta, "need": need}

        prepared.append(
            {
                **record,
                "analysis": analysis,
                "normalization_actions": normalization_actions,
                "pose_evidence": pose_evidence,
                "fusion_payload": fusion_payload,
                "caption_evidence": caption_evidence,
                "firewall_audit": firewall_audit,
                "variant_files": variant_files,
            }
        )

    if not prepared:
        print("No matching cached analysis records found.", file=sys.stderr)
        return 2

    loaded = None
    model_load_seconds: float | None = None
    if pending_count:
        print(
            f"Loading {compose_model_id} for text-only governed Compose comparison "
            f"({pending_count} generation(s); run={run_label}) ..."
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
        model_load_seconds = loaded.load_seconds
        print(f"Loaded in {loaded.load_seconds:.1f}s. No image-conditioned inference will be run.")

    try:
        for item in tqdm(prepared, desc="Governed Compose A/B/C"):
            key = item["result_key"]
            analysis = item["analysis"]
            for variant, files in item["variant_files"].items():
                if not files["need"]:
                    continue
                if variant == "analysis":
                    prompt = render_compose_prompt(analysis_template, analysis, subject_token, detail)
                elif variant == "dwpose":
                    prompt = _render_pose_prompt(
                        dwpose_template,
                        analysis,
                        item["pose_evidence"],
                        subject_token,
                        detail,
                    )
                else:
                    prompt = _render_fusion_prompt(
                        fusion_template,
                        item["caption_evidence"],
                        subject_token,
                        detail,
                    )

                caption, seconds = generate_text(loaded, prompt, max_new_tokens=args.max_caption_tokens)
                metadata: dict[str, Any] = {
                    "image": item.get("relative_path"),
                    "analysis_model": analysis_model_id,
                    "compose_model": compose_model_id,
                    "variant": variant,
                    "compose_seconds": seconds,
                    "subject_token": subject_token,
                    "detail": detail,
                    "image_conditioned": False,
                    "run_label": run_label,
                    "backend_requested": args.backend,
                    "dtype_requested": args.dtype,
                    "quantization_requested": args.quantization,
                    "caption_policy": caption_policy,
                    "normalization_actions": item.get("normalization_actions") or [],
                }
                if variant == "dwpose":
                    metadata["pose_evidence"] = item["pose_evidence"]
                if variant == "fusion-safe":
                    metadata["caption_evidence"] = item["caption_evidence"]
                    metadata["firewall_audit"] = item["firewall_audit"]
                    metadata["caption_lint"] = lint_caption(caption, item["caption_evidence"] or {})
                _write_caption(files["txt"], files["meta"], caption, metadata)
    finally:
        if loaded is not None:
            print(f"Unloading {compose_model_id} ...")
            unload_model(loaded)

    summary_records: list[dict[str, Any]] = []
    for item in prepared:
        captions: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        for variant, files in item["variant_files"].items():
            captions[variant] = _read_text(files["txt"])
            metadata[variant] = _read_json(files["meta"])
        summary_records.append(
            {
                "relative_path": item.get("relative_path"),
                "result_key": item["result_key"],
                "report_image": item.get("report_image"),
                "existing_caption": item.get("existing_caption"),
                "image_summary": item["analysis"].get("image_summary"),
                "normalization_actions": item.get("normalization_actions") or [],
                "captions": captions,
                "metadata": metadata,
                "caption_evidence": item.get("caption_evidence"),
                "firewall_audit": item.get("firewall_audit"),
            }
        )

    summary = {
        "schema_version": "compose-fusion-compare-1.3",
        "caption_evidence_schema": "caption-evidence-1.3",
        "analysis_model": analysis_model_id,
        "compose_model": compose_model_id,
        "compose_model_load_seconds": model_load_seconds,
        "run_label": run_label,
        "backend_requested": args.backend,
        "dtype_requested": args.dtype,
        "quantization_requested": args.quantization,
        "analysis_source": str(analysis_dir),
        "dwpose_source": str(dwpose_dir) if dwpose_dir.is_dir() else None,
        "fusion_source": str(fusion_dir) if fusion_dir.is_dir() else None,
        "subject_token": subject_token,
        "caption_policy": caption_policy,
        "detail": detail,
        "variants": args.variants,
        "image_conditioned_compose": False,
        "records": summary_records,
    }
    stem = f"compose_fusion_compare_{analysis_slug}__compose__{compose_slug}__{run_label}"
    summary_path = run_dir / f"{stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path = _build_html(run_dir, analysis_slug, compose_slug, run_label, summary_records)
    print(f"Done. JSON:   {summary_path}")
    print(f"      Report: {html_path}")
    print(f"      Captions: {compare_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
