from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
DEFAULT_BASE_PROMPT = PACKAGE_ROOT / "prompts" / "compose_identity_pose_v1.txt"
DEFAULT_POSE_PROMPT = PACKAGE_ROOT / "prompts" / "compose_identity_pose_dwpose_v1.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-compose-compare",
        description=(
            "Generate text-only baseline and DWPose-assisted captions from an existing "
            "Qwen analysis run. No image-conditioned inference is performed."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="Existing validation run directory, e.g. runs/analysis-v1-nf4")
    parser.add_argument("--model", default="32b", help="Analysis/compose model alias or Hugging Face model ID (default: 32b).")
    parser.add_argument("--dwpose-dir", type=Path, help="DWPose output directory (default: <run_dir>/dwpose).")
    parser.add_argument("--baseline-prompt", type=Path, default=DEFAULT_BASE_PROMPT)
    parser.add_argument("--pose-prompt", type=Path, default=DEFAULT_POSE_PROMPT)
    parser.add_argument("--subject-token", help="Override subject token; otherwise inherit run.json.")
    parser.add_argument("--detail", choices=["concise", "balanced", "detailed"], help="Override detail profile; otherwise inherit run.json.")
    parser.add_argument("--limit", type=int, help="Process only the first N matching images.")
    parser.add_argument("--only", nargs="+", help="Only process result keys containing any of these strings, e.g. 00001 00004 00014.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-caption-tokens", type=int, default=450)
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], help="Override backend; otherwise inherit run.json.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], help="Override dtype; otherwise inherit run.json.")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], help="Override quantization; otherwise inherit run.json.")
    parser.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"], help="Override attention implementation; otherwise inherit run.json.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float)
    parser.add_argument("--vllm-max-model-len", type=int)
    return parser.parse_args()


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


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _write_caption(path: Path, meta_path: Path, caption: str, metadata: dict[str, Any]) -> None:
    path.write_text(caption.strip() + "\n", encoding="utf-8")
    payload = dict(metadata)
    payload["caption"] = caption.strip()
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_html(run_dir: Path, slug: str, records: list[dict[str, Any]]) -> Path:
    out = run_dir / f"compose_compare_{slug}.html"
    cards: list[str] = []
    for record in records:
        image_src = html.escape(record.get("report_image") or "")
        key = html.escape(record["result_key"])
        existing = html.escape(record.get("existing_caption") or "")
        baseline = html.escape(record.get("baseline_caption") or "")
        pose = html.escape(record.get("pose_caption") or "")
        summary = html.escape(record.get("image_summary") or "")
        pose_json = html.escape(json.dumps(record.get("pose_evidence") or {}, indent=2, ensure_ascii=False))
        cards.append(
            f"""
<section class="card">
  <h2>{key}</h2>
  <div class="grid">
    <div><img src="{image_src}" loading="lazy"></div>
    <div>
      <p class="summary">{summary}</p>
      <h3>Existing caption</h3><p>{existing or '<em>none</em>'}</p>
      <h3>Baseline Compose</h3><p>{baseline or '<em>missing</em>'}</p>
      <h3>DWPose-assisted Compose</h3><p>{pose or '<em>missing</em>'}</p>
      <details><summary>Compact DWPose evidence</summary><pre>{pose_json}</pre></details>
    </div>
  </div>
</section>
"""
        )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Compose comparison — {html.escape(slug)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;line-height:1.45;background:#f6f6f6;color:#171717}}
h1{{margin-bottom:6px}} .lede{{color:#555}}
.card{{background:white;border:1px solid #ddd;border-radius:10px;padding:18px;margin:18px 0}}
.grid{{display:grid;grid-template-columns:minmax(260px,34%) 1fr;gap:22px;align-items:start}}
img{{width:100%;height:auto;max-height:620px;object-fit:contain;background:#111;border-radius:6px}}
h2{{margin-top:0}} h3{{margin-bottom:4px}} p{{margin-top:4px}} .summary{{font-weight:600}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:12px;border-radius:6px;font-size:12px}}
@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Baseline vs DWPose-assisted Compose</h1>
<p class="lede">Both captions are text-only generations from the same cached visual analysis. The second receives only the additional compact DWPose evidence block; no image is re-read.</p>
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

    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    model_dir = run_dir / slug
    if not model_dir.is_dir():
        print(f"Analysis directory not found for {model_id}: {model_dir}", file=sys.stderr)
        return 2

    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    if not dwpose_dir.is_dir():
        print(f"DWPose directory not found: {dwpose_dir}", file=sys.stderr)
        return 2

    baseline_template = args.baseline_prompt.read_text(encoding="utf-8")
    pose_template = args.pose_prompt.read_text(encoding="utf-8")
    subject_token = args.subject_token or manifest.get("subject_token") or "sH1Vx"
    detail = args.detail or manifest.get("detail") or "balanced"

    image_records = list(manifest.get("images") or [])
    if args.only:
        needles = tuple(args.only)
        image_records = [r for r in image_records if any(n in str(r.get("result_key", "")) for n in needles)]
    if args.limit is not None:
        image_records = image_records[: args.limit]

    compare_dir = model_dir / "compose_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    pending_count = 0
    for record in image_records:
        key = record["result_key"]
        analysis_record = _read_json(model_dir / f"{key}.analysis.json")
        analysis = (analysis_record or {}).get("analysis")
        if not isinstance(analysis, dict):
            print(f"WARNING: missing/invalid analysis for {key}; skipping", file=sys.stderr)
            continue
        dwpose_record = _read_json(dwpose_dir / f"{key}.dwpose.json")
        if dwpose_record is None:
            print(f"WARNING: missing DWPose record for {key}; skipping", file=sys.stderr)
            continue

        baseline_path = compare_dir / f"{key}.baseline.txt"
        baseline_meta = compare_dir / f"{key}.baseline.json"
        pose_path = compare_dir / f"{key}.dwpose.txt"
        pose_meta = compare_dir / f"{key}.dwpose.json"
        need_baseline = args.overwrite or not baseline_path.exists() or not baseline_meta.exists()
        need_pose = args.overwrite or not pose_path.exists() or not pose_meta.exists()
        pending_count += int(need_baseline) + int(need_pose)
        prepared.append(
            {
                **record,
                "analysis": analysis,
                "pose_evidence": build_pose_evidence(dwpose_record),
                "baseline_path": baseline_path,
                "baseline_meta": baseline_meta,
                "pose_path": pose_path,
                "pose_meta": pose_meta,
                "need_baseline": need_baseline,
                "need_pose": need_pose,
            }
        )

    if not prepared:
        print("No matching analysis + DWPose records found.", file=sys.stderr)
        return 2

    backend = args.backend or manifest.get("backend") or "transformers"
    dtype = args.dtype or manifest.get("dtype") or "bfloat16"
    quantization = args.quantization or manifest.get("quantization") or "4bit"
    attn = args.attn if args.attn is not None else manifest.get("attention")
    gpu_mem = args.vllm_gpu_memory_utilization or manifest.get("vllm_gpu_memory_utilization") or 0.92
    max_model_len = args.vllm_max_model_len or manifest.get("vllm_max_model_len") or 8192

    loaded = None
    if pending_count:
        print(f"Loading {model_id} for text-only Compose comparison ({pending_count} generation(s)) ...")
        loaded = load_model(
            model_id,
            backend=backend,
            dtype=dtype,
            quantization=quantization,
            attn_implementation=attn,
            cache_dir=args.cache_dir,
            vllm_gpu_memory_utilization=gpu_mem,
            vllm_max_model_len=max_model_len,
        )
        print(f"Loaded in {loaded.load_seconds:.1f}s. No image-conditioned inference will be run.")

    try:
        for item in tqdm(prepared, desc="Compose A/B"):
            key = item["result_key"]
            analysis = item["analysis"]
            pose_evidence = item["pose_evidence"]

            if item["need_baseline"]:
                prompt = render_compose_prompt(baseline_template, analysis, subject_token, detail)
                caption, seconds = generate_text(loaded, prompt, max_new_tokens=args.max_caption_tokens)
                _write_caption(
                    item["baseline_path"],
                    item["baseline_meta"],
                    caption,
                    {
                        "image": item.get("relative_path"),
                        "model": model_id,
                        "variant": "baseline",
                        "compose_seconds": seconds,
                        "subject_token": subject_token,
                        "detail": detail,
                        "image_conditioned": False,
                    },
                )

            if item["need_pose"]:
                prompt = _render_pose_prompt(pose_template, analysis, pose_evidence, subject_token, detail)
                caption, seconds = generate_text(loaded, prompt, max_new_tokens=args.max_caption_tokens)
                _write_caption(
                    item["pose_path"],
                    item["pose_meta"],
                    caption,
                    {
                        "image": item.get("relative_path"),
                        "model": model_id,
                        "variant": "dwpose_assisted",
                        "compose_seconds": seconds,
                        "subject_token": subject_token,
                        "detail": detail,
                        "image_conditioned": False,
                        "pose_evidence": pose_evidence,
                    },
                )
    finally:
        if loaded is not None:
            print(f"Unloading {model_id} ...")
            unload_model(loaded)

    summary_records: list[dict[str, Any]] = []
    for item in prepared:
        summary_records.append(
            {
                "relative_path": item.get("relative_path"),
                "result_key": item["result_key"],
                "report_image": item.get("report_image"),
                "existing_caption": item.get("existing_caption"),
                "image_summary": item["analysis"].get("image_summary"),
                "baseline_caption": _read_text(item["baseline_path"]),
                "pose_caption": _read_text(item["pose_path"]),
                "pose_evidence": item["pose_evidence"],
            }
        )

    summary_path = run_dir / f"compose_compare_{slug}.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "compose-compare-1.0",
                "model": model_id,
                "analysis_source": str(model_dir),
                "dwpose_source": str(dwpose_dir),
                "subject_token": subject_token,
                "detail": detail,
                "records": summary_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    html_path = _build_html(run_dir, slug, summary_records)
    print(f"Done. JSON:   {summary_path}")
    print(f"      Report: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
