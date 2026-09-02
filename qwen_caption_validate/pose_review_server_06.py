from __future__ import annotations

"""v0.6 local Pose Review UI for v0.11 plus selected-card packet export.

Each card gets a packet checkbox. Selected cards can be exported as a fully
self-contained HTML file (images embedded as data URIs, text searchable) plus a
JSON companion containing the compact record, annotation and raw profile JSON.
The HTML can be opened directly or printed to PDF from the browser.
"""

import base64
import html
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import pose_review_server as base
from . import pose_review_server_05 as _v05  # noqa: F401  (applies prior UI extensions)


EXTRA_UI = r'''
<style>
.packet-tools{display:flex;gap:8px;align-items:center}.packet-select{display:flex;align-items:center;gap:5px;font-size:12px;color:#c8d1da;white-space:nowrap}.packet-select input{width:17px;height:17px;accent-color:#77c7ff}.card.packet-selected{box-shadow:0 0 0 2px #77c7ff inset}
</style>
<script>
(function(){
  const selected=new Set();
  const controls=document.querySelector('.controls');
  const exportBtn=document.createElement('button');
  exportBtn.id='exportSelectedPacket';
  exportBtn.textContent='Export selected packet (0)';
  exportBtn.disabled=true;
  const selectVisible=document.createElement('button');
  selectVisible.textContent='Select visible';
  const clearSelected=document.createElement('button');
  clearSelected.textContent='Clear selected';
  controls.insertBefore(selectVisible, document.getElementById('export'));
  controls.insertBefore(clearSelected, document.getElementById('export'));
  controls.insertBefore(exportBtn, document.getElementById('export'));

  function updatePacketButton(){
    exportBtn.textContent=`Export selected packet (${selected.size})`;
    exportBtn.disabled=selected.size===0;
  }
  window.togglePacketSelection=function(key,checked){
    if(checked) selected.add(key); else selected.delete(key);
    const cardEl=document.querySelector(`article.card[data-key="${CSS.escape(key)}"]`);
    if(cardEl) cardEl.classList.toggle('packet-selected',checked);
    updatePacketButton();
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const d=r.directional_recline_diagnostic||{};
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#a8d7ff"><b>v0.11 direction-aware recline</b></td></tr>
      <tr><td>Upper inclination (raw)</td><td>${d.raw_upper_body_inclination_score_percent??0}%</td></tr>
      <tr><td>Support-relative direction</td><td>${esc(d.direction||'-')}</td></tr>
      <tr><td>Shoulder shift toward feet</td><td>${d.shoulder_shift_toward_feet_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Retreat from support</td><td>${d.retreat_from_support_score_percent??0}%</td></tr>
      <tr><td>Advance toward support</td><td>${d.advance_toward_support_score_percent??0}%</td></tr>
      <tr><td>Directional upper recline</td><td>${d.directional_upper_recline_score_percent??0}%</td></tr>
      <tr><td>Forward-bend recline veto</td><td>${d.hard_forward_bend_recline_rejection?'YES — HARD×':'no'}</td></tr>`;
    text=text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
    const checked=selected.has(r.image_key)?'checked':'';
    text=text.replace(
      `<div class="head"><h2>${esc(r.image_key)}</h2>`,
      `<div class="head"><div class="packet-tools"><label class="packet-select"><input type="checkbox" ${checked} onchange="togglePacketSelection('${esc(r.image_key)}',this.checked)">packet</label><h2>${esc(r.image_key)}</h2></div>`
    );
    if(checked) text=text.replace('<article class="card ', '<article class="card packet-selected ');
    return text;
  };

  selectVisible.onclick=()=>{filtered().forEach(r=>selected.add(r.image_key));render();updatePacketButton();};
  clearSelected.onclick=()=>{selected.clear();render();updatePacketButton();};
  exportBtn.onclick=async()=>{
    if(!selected.size)return;
    const response=await fetch('/api/export-selected-packet',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({keys:[...selected]})
    });
    const result=await response.json();
    if(!response.ok){alert(result.error||'Selected packet export failed');return;}
    window.open(result.url,'_blank');
    alert(`Selected packet exported:\n${result.html}\n${result.json}\n\nThe HTML is self-contained and can be printed to PDF.`);
  };

  updatePacketButton();
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _score_line(record: dict[str, Any]) -> str:
    scores = record.get("posture_score_percent") or {}
    return " · ".join(
        f"{name} {scores.get(name, 0)}%"
        for name in ("standing", "crouching", "squatting", "sitting", "reclined")
    )


def _rejection_text(record: dict[str, Any]) -> str:
    per_pose = ((record.get("physical_governance") or {}).get("per_pose") or {})
    parts: list[str] = []
    for name, row in per_pose.items():
        if not (row or {}).get("hard_rejected"):
            continue
        reasons = ", ".join((row or {}).get("hard_rejection_reasons") or [])
        parts.append(f"{name}: {reasons}")
    return " ; ".join(parts) or "-"


def _packet_html(
    bundle: Path,
    records: list[dict[str, Any]],
    annotations: dict[str, Any],
    raw_by_key: dict[str, Any],
) -> str:
    cards: list[str] = []
    anno_rows = annotations.get("records") or {}
    for record in records:
        key = str(record.get("image_key") or "")
        anno = anno_rows.get(key) or {}
        original = base._safe_media(bundle, str(record.get("original") or ""))
        overlay = base._safe_media(bundle, str(record.get("overlay") or ""))
        original_uri = _data_uri(original) if original is not None and original.is_file() else ""
        overlay_uri = _data_uri(overlay) if overlay is not None and overlay.is_file() else ""
        gov = record.get("physical_governance") or {}
        auth = gov.get("authority") or {}
        direction = record.get("directional_recline_diagnostic") or {}
        support = record.get("independent_support_diagnostic") or {}
        leg = record.get("leg_state_diagnostic") or {}
        upper = record.get("upper_body_recline_diagnostic") or {}
        notes = html.escape(str(anno.get("notes") or ""))
        modifiers = html.escape(", ".join(anno.get("modifiers") or []))
        raw_json = html.escape(json.dumps(raw_by_key.get(key) or {}, indent=2, ensure_ascii=False))
        cards.append(f'''<article class="card">
<h2>{html.escape(key)}</h2>
<div class="images"><figure><figcaption>Original</figcaption><img src="{original_uri}"></figure><figure><figcaption>DWPose + SAM3D overlay</figcaption><img src="{overlay_uri}"></figure></div>
<div class="cols"><table>
<tr><th>Projected</th><td>{html.escape(str(record.get("pose") or "-"))}</td></tr>
<tr><th>Best candidate</th><td>{html.escape(str(record.get("best_candidate_pose") or "-"))}</td></tr>
<tr><th>Scores</th><td>{html.escape(_score_line(record))}</td></tr>
<tr><th>Reconstruction</th><td>{record.get("reconstruction_match_percent",0)}%</td></tr>
<tr><th>Crop authority</th><td>{auth.get("crop_support_percent",record.get("crop_support_percent",0))}% · {html.escape(str(auth.get("authority_path") or record.get("support_class") or "-"))}</td></tr>
<tr><th>Winner margin</th><td>{record.get("winner_margin_percent",0)}%</td></tr>
<tr><th>Leg state</th><td>{html.escape(str(leg.get("state") or "-"))} · flex {leg.get("bilateral_flexion_score_percent",0)}% · straight {leg.get("bilateral_straight_score_percent",0)}%</td></tr>
<tr><th>Foot feasibility</th><td>{support.get("support_feasibility_score_percent",0)}% · external support {support.get("external_support_requirement_percent",0)}%</td></tr>
<tr><th>Upper inclination</th><td>{direction.get("raw_upper_body_inclination_score_percent",0)}%</td></tr>
<tr><th>Direction</th><td>{html.escape(str(direction.get("direction") or "-"))} · shift {direction.get("shoulder_shift_toward_feet_shoulder_widths","-")} sw</td></tr>
<tr><th>Directional recline</th><td>{direction.get("directional_upper_recline_score_percent",0)}% · forward-veto {"YES" if direction.get("hard_forward_bend_recline_rejection") else "no"}</td></tr>
<tr><th>Upper-body path</th><td>{upper.get("score_percent",0)}% raw · authority {upper.get("path_authority_percent",0)}%</td></tr>
<tr><th>Hard rejections</th><td>{html.escape(_rejection_text(record))}</td></tr>
</table><section class="annotation"><h3>Human review</h3>
<p><b>Verdict:</b> {html.escape(str(anno.get("verdict") or "-"))}</p>
<p><b>Human pose:</b> {html.escape(str(anno.get("human_pose") or "-"))}</p>
<p><b>Modifiers:</b> {modifiers or "-"}</p>
<p><b>Notes:</b><br>{notes.replace(chr(10),'<br>') or '-'}</p></section></div>
<details><summary>Raw profile JSON</summary><pre>{raw_json}</pre></details>
</article>''')

    created = datetime.now(timezone.utc).isoformat()
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Pose Review Selected Packet</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:20px;background:#0b0d10;color:#edf1f5}}h1{{margin-bottom:4px}}.meta{{color:#aab3bd;margin-bottom:20px}}.card{{page-break-after:always;border:1px solid #39424c;border-radius:10px;padding:14px;margin:0 0 22px;background:#14181d}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}figure{{margin:0}}figcaption{{color:#aab3bd;font-size:12px;margin-bottom:4px}}img{{max-width:100%;max-height:650px;object-fit:contain;background:#080a0d}}.cols{{display:grid;grid-template-columns:1.1fr .9fr;gap:16px;margin-top:12px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:5px;border-bottom:1px solid #2b323a;text-align:left;vertical-align:top}}th{{width:180px;color:#aab3bd}}.annotation{{border:1px solid #39424c;border-radius:8px;padding:10px}}pre{{white-space:pre-wrap;font-size:11px;background:#090b0e;padding:10px}}summary{{cursor:pointer}}@media print{{body{{background:white;color:black;margin:8mm}}.card{{background:white;border-color:#bbb}}th{{color:#444}}.annotation{{border-color:#bbb}}details{{display:none}}}}
</style></head><body><h1>Pose Review — Selected Packet</h1><div class="meta">Generated {html.escape(created)} · {len(records)} selected record(s)</div>{''.join(cards)}</body></html>'''


def _export_selected_packet(bundle: Path, keys: list[str]) -> tuple[Path, Path]:
    index = base._load_json(bundle / "pose_review.index.json", {"records": []})
    annotations = base._load_json(bundle / "pose_review_annotations.json", {"records": {}})
    by_key = {str(r.get("image_key")): r for r in index.get("records") or []}
    records = [by_key[k] for k in keys if k in by_key]
    if not records:
        raise ValueError("no_valid_selected_keys")

    raw_by_key: dict[str, Any] = {}
    for record in records:
        key = str(record.get("image_key"))
        raw_path = base._safe_media(bundle, str(record.get("raw_json") or ""))
        raw_by_key[key] = base._load_json(raw_path, {}) if raw_path is not None and raw_path.is_file() else {}

    payload = {
        "schema_version": "pose-review-selected-packet-0.1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_schema_version": index.get("schema_version"),
        "selected_keys": [str(r.get("image_key")) for r in records],
        "records": [
            {
                "record": r,
                "annotation": (annotations.get("records") or {}).get(str(r.get("image_key"))) or {},
                "raw_profile": raw_by_key.get(str(r.get("image_key"))) or {},
            }
            for r in records
        ],
    }
    json_path = bundle / "pose_review_selected_packet.json"
    html_path = bundle / "pose_review_selected_packet.html"
    base._atomic_json(json_path, payload)
    html_path.write_text(_packet_html(bundle, records, annotations, raw_by_key), encoding="utf-8")
    return html_path, json_path


_original_make_handler = base.make_handler


def _make_handler_v06(bundle: Path):
    Parent = _original_make_handler(bundle)

    class Handler(Parent):
        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/export-selected-packet":
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._json({"error": "invalid_json"}, 400)
                    return
                keys = body.get("keys") if isinstance(body, dict) else None
                if not isinstance(keys, list):
                    self._json({"error": "keys_must_be_array"}, 400)
                    return
                clean = [str(k) for k in keys if str(k).strip()]
                try:
                    html_path, json_path = _export_selected_packet(bundle, clean)
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                self._json({
                    "ok": True,
                    "html": str(html_path),
                    "json": str(json_path),
                    "url": "/media/" + html_path.name,
                })
                return
            super().do_POST()

    return Handler


base.make_handler = _make_handler_v06


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
