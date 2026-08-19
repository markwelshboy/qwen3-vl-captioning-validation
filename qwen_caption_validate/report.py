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


def build_report(run_dir: Path, model_slugs: list[str]) -> Path:
    manifest = _read_json(run_dir / "run.json") or {}
    items = manifest.get("images", [])
    model_map = manifest.get("models", {})

    cards: list[str] = []
    for item in items:
        rel_image = item["report_image"]
        stem_key = item["result_key"]
        existing = item.get("existing_caption")

        columns: list[str] = []
        for slug in model_slugs:
            result = _read_json(run_dir / slug / f"{stem_key}.analysis.json")
            caption_path = run_dir / slug / f"{stem_key}.caption.txt"
            caption = caption_path.read_text(encoding="utf-8") if caption_path.exists() else None
            model_label = model_map.get(slug, slug)
            if result:
                analysis = result.get("analysis")
                raw = result.get("raw_response") if not analysis else None
                status = "valid" if result.get("schema_valid") else "warning"
                validation = result.get("schema_errors") or []
                metric = f"{result.get('inference_seconds', 0):.1f}s"
            else:
                analysis = raw = None
                status = "missing"
                validation = []
                metric = ""

            validation_html = ""
            if validation:
                validation_html = "<details><summary>Schema warnings</summary>" + _pre(validation) + "</details>"
            raw_html = ""
            if raw:
                raw_html = "<details><summary>Raw model output</summary>" + _pre(raw) + "</details>"
            caption_html = ""
            if caption:
                caption_html = "<h4>Composed caption</h4>" + _pre(caption)

            columns.append(
                f"""
                <section class="model-col">
                  <div class="model-head"><strong>{html.escape(model_label)}</strong><span class="pill {status}">{html.escape(metric or status)}</span></div>
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
<header><h1>Qwen3-VL Captioning Validation</h1><p>{html.escape(manifest.get('run_name',''))} · {len(items)} images</p></header>
<main>{''.join(cards)}</main>
</body></html>"""
    output = run_dir / "report.html"
    output.write_text(doc, encoding="utf-8")
    return output
