from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pre(value: Any) -> str:
    if value is None:
        return '<div class="missing">Not available</div>'
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False)
    return f"<pre>{html.escape(text)}</pre>"


def _fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "—"


def build_report(run_dir: Path, model_slugs: list[str]) -> Path:
    manifest = _read_json(run_dir / "run.json") or {}
    items = manifest.get("images", [])
    model_map = manifest.get("models", {})
    model_runtime = manifest.get("model_runtime", {})

    summaries: list[str] = []
    for slug in model_slugs:
        analysis_times: list[float] = []
        compose_times: list[float] = []
        for item in items:
            key = item["result_key"]
            result = _read_json(run_dir / slug / f"{key}.analysis.json")
            caption_meta = _read_json(run_dir / slug / f"{key}.caption.json")
            if result:
                value = result.get("analysis_seconds", result.get("inference_seconds"))
                if isinstance(value, (int, float)):
                    analysis_times.append(float(value))
            if caption_meta:
                value = caption_meta.get("compose_seconds")
                if isinstance(value, (int, float)):
                    compose_times.append(float(value))

        runtime = model_runtime.get(slug, {})
        load_seconds = runtime.get("load_seconds")
        avg_analysis = sum(analysis_times) / len(analysis_times) if analysis_times else None
        avg_compose = sum(compose_times) / len(compose_times) if compose_times else None
        total_generation = sum(analysis_times) + sum(compose_times)
        total_with_load = total_generation + (float(load_seconds) if isinstance(load_seconds, (int, float)) else 0.0)
        model_label = model_map.get(slug, slug)
        summaries.append(
            f"""
            <section class="summary-card">
              <strong>{html.escape(model_label)}</strong>
              <div class="summary-grid">
                <span>Load <b>{_fmt_seconds(load_seconds)}</b></span>
                <span>Avg analysis <b>{_fmt_seconds(avg_analysis)}</b></span>
                <span>Avg compose <b>{_fmt_seconds(avg_compose)}</b></span>
                <span>Generation <b>{_fmt_seconds(total_generation)}</b></span>
                <span>Total incl. load <b>{_fmt_seconds(total_with_load)}</b></span>
              </div>
            </section>
            """
        )

    cards: list[str] = []
    for item in items:
        rel_image = item["report_image"]
        stem_key = item["result_key"]
        existing = item.get("existing_caption")

        columns: list[str] = []
        for slug in model_slugs:
            result = _read_json(run_dir / slug / f"{stem_key}.analysis.json")
            caption_meta = _read_json(run_dir / slug / f"{stem_key}.caption.json")
            caption_path = run_dir / slug / f"{stem_key}.caption.txt"
            caption = None
            if caption_meta:
                caption = caption_meta.get("caption")
            elif caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8")
            model_label = model_map.get(slug, slug)

            analysis_seconds = None
            compose_seconds = None
            if result:
                analysis = result.get("analysis")
                raw = result.get("raw_response") if not analysis else None
                status = "valid" if result.get("schema_valid") else "warning"
                validation = result.get("schema_errors") or []
                analysis_seconds = result.get("analysis_seconds", result.get("inference_seconds"))
            else:
                analysis = raw = None
                status = "missing"
                validation = []
            if caption_meta:
                compose_seconds = caption_meta.get("compose_seconds")

            parts = []
            if analysis_seconds is not None:
                parts.append(f"A {_fmt_seconds(analysis_seconds)}")
            if compose_seconds is not None:
                parts.append(f"C {_fmt_seconds(compose_seconds)}")
            if analysis_seconds is not None or compose_seconds is not None:
                total = float(analysis_seconds or 0) + float(compose_seconds or 0)
                parts.append(f"Σ {_fmt_seconds(total)}")
            metric = " · ".join(parts) if parts else status

            validation_html = ""
            if validation:
                validation_html = "<details><summary>Schema warnings</summary>" + _pre(validation) + "</details>"
            raw_html = ""
            if raw:
                raw_html = "<details><summary>Raw model output</summary>" + _pre(raw) + "</details>"
            caption_html = ""
            if caption:
                caption_html = "<h4>Pre-cached composed caption</h4>" + _pre(caption)

            columns.append(
                f"""
                <section class="model-col">
                  <div class="model-head"><strong>{html.escape(model_label)}</strong><span class="pill {status}">{html.escape(metric)}</span></div>
                  {_pre(analysis)}
                  {raw_html}
                  {validation_html}
                  {caption_html}
                </section>
                """
            )

        cards.append(
            f"""
            <article class="card">
              <div class="image-col">
                <h2>{html.escape(item['relative_path'])}</h2>
                <img src="{html.escape(rel_image)}" loading="lazy" />
                <h4>Existing sidecar caption</h4>
                {_pre(existing)}
              </div>
              <div class="results-grid">{''.join(columns)}</div>
            </article>
            """
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-VL Captioning Validation</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin:0; background:#07111f; color:#e7edf6; }}
header {{ position:sticky; top:0; z-index:2; padding:18px 24px; background:#0b1727ee; border-bottom:1px solid #24344a; backdrop-filter:blur(10px); }}
header h1 {{ margin:0 0 5px; font-size:20px; }}
header p {{ margin:0; color:#9fb0c8; }}
.runtime-summary {{ padding:14px 18px 0; display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }}
.summary-card {{ background:#0d1a2b; border:1px solid #24344a; border-radius:12px; padding:12px 14px; }}
.summary-card strong {{ display:block; margin-bottom:8px; }}
.summary-grid {{ display:flex; flex-wrap:wrap; gap:7px 14px; color:#9fb0c8; font-size:12px; }}
.summary-grid b {{ color:#e7edf6; }}
main {{ padding:18px; display:grid; gap:18px; }}
.card {{ display:grid; grid-template-columns:minmax(280px, 34vw) 1fr; gap:18px; padding:16px; background:#0d1a2b; border:1px solid #24344a; border-radius:14px; }}
.image-col img {{ width:100%; max-height:72vh; object-fit:contain; background:#050b13; border-radius:10px; }}
.image-col h2 {{ margin-top:0; font-size:15px; word-break:break-all; }}
.results-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; align-items:start; }}
.model-col {{ min-width:0; }}
.model-head {{ display:flex; justify-content:space-between; gap:10px; align-items:center; margin-bottom:8px; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#081321; border:1px solid #203249; border-radius:9px; padding:12px; font-size:12px; line-height:1.45; max-height:65vh; overflow:auto; }}
.pill {{ font-size:11px; padding:3px 8px; border-radius:999px; background:#23344c; }}
.pill.valid {{ background:#153b2c; }} .pill.warning {{ background:#563d12; }} .pill.missing {{ background:#3b2530; }}
.missing {{ color:#72849d; font-style:italic; }}
details {{ margin:8px 0; }} summary {{ cursor:pointer; color:#d8b75f; }}
h4 {{ margin:14px 0 6px; color:#b9c8dc; }}
@media (max-width: 1000px) {{ .card {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header><h1>Qwen3-VL Captioning Validation</h1><p>{html.escape(manifest.get('run_name',''))} · {len(items)} images · A = analysis · C = compose</p></header>
<div class="runtime-summary">{''.join(summaries)}</div>
<main>{''.join(cards)}</main>
</body></html>"""
    output = run_dir / "report.html"
    output.write_text(doc, encoding="utf-8")
    return output
