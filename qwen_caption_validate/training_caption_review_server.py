from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import mimetypes
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ANNOTATION_SCHEMA = "training-caption-review-annotations-0.1"

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic V3 Training Caption Review</title>
<style>
:root{color-scheme:dark;--bg:#0b0d10;--panel:#14181d;--panel2:#1b2026;--text:#edf1f5;--muted:#aab3bd;--line:#303841;--cyan:#36d7ff;--amber:#ffbf3f;--good:#6ee7a8;--warn:#ffcf70;--bad:#ff7474;--blue:#77c7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,sans-serif}.toolbar{position:sticky;top:0;z-index:10;background:rgba(11,13,16,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:11px 15px}.toolbar h1{font-size:19px;margin:0 0 8px}.controls{display:flex;flex-wrap:wrap;gap:7px;align-items:center}.controls input,.controls select,.controls button,.card button,.card select,.card textarea{background:#151a20;color:var(--text);border:1px solid #404a55;border-radius:7px;padding:7px}.controls button,.card button{cursor:pointer}.legend{margin-left:auto;color:var(--muted);font-size:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 9px}.cyan{background:var(--cyan)}.amber{background:var(--amber)}.summary{padding:8px 15px;color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(980px,1fr));gap:14px;padding:0 14px 30px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.card.reviewed{border-color:#476957}.card.packet-selected{box-shadow:0 0 0 2px var(--blue) inset}.head{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:10px 12px;background:var(--panel2);border-bottom:1px solid var(--line)}.headleft{display:flex;align-items:center;gap:9px}.head h2{font-size:16px;margin:0}.packet-select{font-size:12px;color:var(--muted);display:flex;gap:4px;align-items:center}.chips{display:flex;flex-wrap:wrap;gap:5px}.chip{font-size:11px;background:#252c34;border-radius:999px;padding:3px 7px}.chip.good{background:#17392b}.chip.warn{background:#4b3d1e}.visuals{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:9px}.visuals figure{margin:0;min-width:0}.visuals figcaption{font-size:12px;color:var(--muted);padding:3px}.visuals img{display:block;width:100%;max-height:520px;object-fit:contain;background:#080a0d;border-radius:7px}.pose-ref{margin:0 10px 10px;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:#101419;font-size:13px}.pose-ref h3{font-size:13px;margin:0 0 5px;color:#bcd8ee}.pose-lines{margin:4px 0 0;padding-left:20px}.captions{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 10px 10px}.profile{border:1px solid var(--line);border-radius:9px;background:#11151a;overflow:hidden}.profile-head{display:flex;justify-content:space-between;align-items:center;padding:8px 9px;background:#181e24}.profile h3{margin:0;font-size:14px;text-transform:capitalize}.caption{font-size:14px;line-height:1.5;padding:10px;white-space:pre-wrap;border-bottom:1px solid var(--line)}.grade{padding:8px 9px}.row{display:grid;grid-template-columns:145px 1fr;gap:7px;align-items:center;margin:6px 0}.row label{font-size:12px;color:var(--muted)}.row select{width:100%}.issues{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 8px;margin:7px 0}.issues label{font-size:12px;color:#d6dde4;display:flex;gap:5px;align-items:flex-start}.issues input{margin-top:2px}.notes{width:100%;min-height:65px;resize:vertical}.overall{margin:0 10px 11px;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:#101419}.overall-grid{display:grid;grid-template-columns:170px 190px 1fr;gap:8px;align-items:center}.overall textarea{min-height:58px;resize:vertical;width:100%}.actions{display:flex;gap:7px;align-items:center;margin-top:8px}.saved{font-size:12px;color:var(--good)}.raw{display:none;margin:0 10px 10px;max-height:420px;overflow:auto;background:#080a0d;border:1px solid var(--line);padding:9px;border-radius:7px;font-size:11px;white-space:pre-wrap}.raw.open{display:block}@media(max-width:1050px){.grid{grid-template-columns:1fr}.captions,.visuals{grid-template-columns:1fr}.overall-grid{grid-template-columns:1fr}.legend{width:100%;margin:3px 0 0}}
</style></head><body>
<div class="toolbar"><h1>Semantic V3 — Training Caption Fit-for-Purpose Review</h1><div class="controls">
<input id="search" placeholder="image key"><select id="pose"><option value="">all poses</option></select>
<select id="filter"><option value="all">all records</option><option value="unreviewed">unreviewed only</option><option value="major">major issue</option><option value="pipeline">pipeline fix expected</option><option value="posebad">pose rigid / wrong</option><option value="compactbad">compact weak / bad</option><option value="mediumbad">medium weak / bad</option></select>
<select id="sort"><option value="key">sort: key</option><option value="pose">sort: pose</option><option value="unreviewed">sort: unreviewed first</option></select>
<button id="selectVisible">Select visible</button><button id="clearSelected">Clear selected</button><button id="packet">Export selected packet (0)</button><button id="export">Export JSON + CSV</button>
<span class="legend"><span class="dot cyan"></span>DWPose <span class="dot amber"></span>SAM3D</span>
</div></div><div id="summary" class="summary"></div><main id="grid" class="grid"></main>
<script>
let INDEX=null,ANNO={records:{}},SELECTED=new Set();
const issues=['major_factual_error','pose_mismatch','missing_high_value_detail','too_much_low_value_detail','ownership_or_coreference','identity_policy_problem','grammar_or_fluency'];
const issueLabels={major_factual_error:'major factual error',pose_mismatch:'pose mismatch',missing_high_value_detail:'missing high-value detail',too_much_low_value_detail:'too much low-value detail',ownership_or_coreference:'ownership / coreference',identity_policy_problem:'identity-policy problem',grammar_or_fluency:'grammar / fluency'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const el=id=>document.getElementById(id), searchEl=el('search'),poseEl=el('pose'),filterEl=el('filter'),sortEl=el('sort'),gridEl=el('grid'),summaryEl=el('summary'),packetEl=el('packet');
function slug(k){return k.replace(/[^A-Za-z0-9_-]/g,'_')}
function annotation(key){return ANNO.records?.[key]||{}}
function profileAnno(key,p){return annotation(key)?.[p]||{}}
function isReviewed(key){return annotation(key)?.reviewed===true}
function hasMajor(key){return ['compact','medium'].some(p=>(profileAnno(key,p).issues||[]).includes('major_factual_error'))}
function wantsFix(key){return ['compact','medium'].some(p=>profileAnno(key,p).pipeline_fix==='yes')}
function poseBad(key){return ['compact','medium'].some(p=>['anatomical_rigid','wrong'].includes(profileAnno(key,p).pose_language))}
function fitBad(key,p){return ['usable_with_issue','bad'].includes(profileAnno(key,p).fit)}
async function load(){INDEX=await (await fetch('/api/index')).json();ANNO=await (await fetch('/api/annotations')).json();const poses=[...new Set(INDEX.records.map(r=>r.pose).filter(Boolean))].sort();poseEl.innerHTML='<option value="">all poses</option>'+poses.map(p=>`<option>${esc(p)}</option>`).join('');render()}
function filtered(){let rows=[...INDEX.records];const q=searchEl.value.trim().toLowerCase();if(q)rows=rows.filter(r=>r.image_key.toLowerCase().includes(q));if(poseEl.value)rows=rows.filter(r=>r.pose===poseEl.value);const f=filterEl.value;if(f==='unreviewed')rows=rows.filter(r=>!isReviewed(r.image_key));if(f==='major')rows=rows.filter(r=>hasMajor(r.image_key));if(f==='pipeline')rows=rows.filter(r=>wantsFix(r.image_key));if(f==='posebad')rows=rows.filter(r=>poseBad(r.image_key));if(f==='compactbad')rows=rows.filter(r=>fitBad(r.image_key,'compact'));if(f==='mediumbad')rows=rows.filter(r=>fitBad(r.image_key,'medium'));rows.sort((a,b)=>{if(sortEl.value==='pose')return String(a.pose||'').localeCompare(String(b.pose||''))||a.image_key.localeCompare(b.image_key);if(sortEl.value==='unreviewed')return Number(isReviewed(a.image_key))-Number(isReviewed(b.image_key))||a.image_key.localeCompare(b.image_key);return a.image_key.localeCompare(b.image_key)});return rows}
function opt(value,current,label){return `<option value="${value}" ${current===value?'selected':''}>${label}</option>`}
function profilePanel(r,p){const c=r.captions[p],a=profileAnno(r.image_key,p),id=slug(r.image_key)+'_'+p;return `<section class="profile"><div class="profile-head"><h3>${p}</h3><div class="chips"><span class="chip">${c.word_count} words</span><span class="chip ${c.repair_action==='passthrough'?'':'warn'}">${esc(c.repair_action||'-')}</span></div></div><div class="caption">${esc(c.text)}</div><div class="grade"><div class="row"><label>Fit for LoRA purpose</label><select id="fit_${id}"><option value="">-- grade --</option>${opt('excellent',a.fit,'excellent')}${opt('good',a.fit,'good')}${opt('usable_with_issue',a.fit,'usable with issue')}${opt('bad',a.fit,'bad / not usable')}</select></div><div class="row"><label>Pose wording</label><select id="posegrade_${id}"><option value="">-- grade --</option>${opt('natural',a.pose_language,'natural')}${opt('mostly_natural',a.pose_language,'mostly natural')}${opt('anatomical_rigid',a.pose_language,'too anatomical / rigid')}${opt('wrong',a.pose_language,'wrong pose')}</select></div><div class="issues">${issues.map(x=>`<label><input type="checkbox" id="issue_${id}_${x}" ${(a.issues||[]).includes(x)?'checked':''}>${issueLabels[x]}</label>`).join('')}</div><div class="row"><label>Should pipeline fix this?</label><select id="fix_${id}"><option value="">--</option>${opt('no',a.pipeline_fix,'no')}${opt('yes',a.pipeline_fix,'yes')}${opt('unsure',a.pipeline_fix,'unsure')}</select></div><textarea class="notes" id="notes_${id}" placeholder="What is good/bad, especially pose naturalness or a major omission...">${esc(a.notes||'')}</textarea></div></section>`}
function poseRef(r){const phrases=(r.caption_ready_phrases||[]);const m=r.pose_modifiers||{};return `<section class="pose-ref"><h3>Governed Pose reference</h3><b>Projected:</b> ${esc(r.pose||'-')} · <b>best candidate:</b> ${esc(r.best_candidate_pose||'-')} · <b>pose-joint support:</b> ${esc(r.crop_support_percent??'-')}% · <b>reconstruction:</b> ${esc(r.reconstruction_match_percent??'-')}%<br><b>Modifiers:</b> lean=${esc(m.lean_severity||'-')}, shoulder tilt=${esc(m.shoulder_line_tilt_severity||'-')}, orientation=${esc(m.body_orientation||'-')}${phrases.length?`<ul class="pose-lines">${phrases.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:'<div style="color:#8f99a4;margin-top:4px">No caption-ready Pose phrase emitted.</div>'}</section>`}
function card(r){const a=annotation(r.image_key),sid=slug(r.image_key),selected=SELECTED.has(r.image_key),chips=[`pose: ${r.pose||'-'}`,`compact ${r.captions.compact.word_count}w`,`medium ${r.captions.medium.word_count}w`];return `<article class="card ${isReviewed(r.image_key)?'reviewed':''} ${selected?'packet-selected':''}" data-key="${esc(r.image_key)}"><div class="head"><div class="headleft"><label class="packet-select"><input type="checkbox" ${selected?'checked':''} onchange="toggleSelected('${esc(r.image_key)}',this.checked)">packet</label><h2>${esc(r.image_key)}</h2></div><div class="chips">${chips.map(x=>`<span class="chip">${esc(x)}</span>`).join('')}</div></div><div class="visuals"><figure><figcaption>Original</figcaption><img loading="lazy" src="/media/${encodeURI(r.original)}"></figure><figure><figcaption>DWPose + SAM3D overlay</figcaption><img loading="lazy" src="/media/${encodeURI(r.overlay)}"></figure></div>${poseRef(r)}<div class="captions">${profilePanel(r,'compact')}${profilePanel(r,'medium')}</div><section class="overall"><div class="overall-grid"><label><input type="checkbox" id="reviewed_${sid}" ${a.reviewed?'checked':''}> review complete</label><select id="pref_${sid}"><option value="">preferred caption --</option>${opt('compact',a.preferred_profile,'compact')}${opt('medium',a.preferred_profile,'medium')}${opt('tie',a.preferred_profile,'tie / either')}${opt('neither',a.preferred_profile,'neither')}</select><textarea id="overall_${sid}" placeholder="Overall comparison / anything the pipeline should learn...">${esc(a.overall_notes||'')}</textarea></div><div class="actions"><button onclick="quickPass('${esc(r.image_key)}')">Quick ✓ both good</button><button onclick="saveAnno('${esc(r.image_key)}')">Save review</button><button onclick="toggleRaw('${esc(r.image_key)}')">Raw record</button><span class="saved" id="saved_${sid}"></span></div></section><pre class="raw" id="raw_${sid}"></pre></article>`}
function render(){const rows=filtered(), reviewed=INDEX.records.filter(r=>isReviewed(r.image_key)).length, major=INDEX.records.filter(r=>hasMajor(r.image_key)).length, fixes=INDEX.records.filter(r=>wantsFix(r.image_key)).length;summaryEl.textContent=`Showing ${rows.length} / ${INDEX.record_count} · reviewed ${reviewed} · major issue ${major} · pipeline-fix ${fixes}`;gridEl.innerHTML=rows.map(card).join('');packetEl.textContent=`Export selected packet (${SELECTED.size})`;packetEl.disabled=SELECTED.size===0}
function collectProfile(key,p){const id=slug(key)+'_'+p;return {fit:el('fit_'+id)?.value||'',pose_language:el('posegrade_'+id)?.value||'',issues:issues.filter(x=>el('issue_'+id+'_'+x)?.checked),pipeline_fix:el('fix_'+id)?.value||'',notes:el('notes_'+id)?.value||''}}
async function saveAnno(key,rerender=true){const sid=slug(key),body={compact:collectProfile(key,'compact'),medium:collectProfile(key,'medium'),reviewed:el('reviewed_'+sid)?.checked||false,preferred_profile:el('pref_'+sid)?.value||'',overall_notes:el('overall_'+sid)?.value||''};const res=await fetch('/api/annotation/'+encodeURIComponent(key),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!res.ok){alert('Save failed');return}ANNO.records[key]=body;const saved=el('saved_'+sid);if(saved){saved.textContent='saved';setTimeout(()=>saved.textContent='',1000)}if(rerender)render()}
async function quickPass(key){const sid=slug(key);for(const p of ['compact','medium']){const id=sid+'_'+p;el('fit_'+id).value='good';el('posegrade_'+id).value='natural';el('fix_'+id).value='no';issues.forEach(x=>el('issue_'+id+'_'+x).checked=false)}el('reviewed_'+sid).checked=true;await saveAnno(key)}
function toggleSelected(key,on){if(on)SELECTED.add(key);else SELECTED.delete(key);render()}
async function toggleRaw(key){const id='raw_'+slug(key),out=el(id),r=INDEX.records.find(x=>x.image_key===key);if(!out.dataset.loaded){out.textContent=JSON.stringify(await (await fetch('/media/'+encodeURI(r.raw_json))).json(),null,2);out.dataset.loaded='1'}out.classList.toggle('open')}
[searchEl,poseEl,filterEl,sortEl].forEach(x=>{x.addEventListener('input',render);x.addEventListener('change',render)});el('selectVisible').onclick=()=>{filtered().forEach(r=>SELECTED.add(r.image_key));render()};el('clearSelected').onclick=()=>{SELECTED.clear();render()};el('export').onclick=async()=>{const r=await fetch('/api/export',{method:'POST'}),j=await r.json();alert(`Exported:\n${j.json}\n${j.csv}`)};packetEl.onclick=async()=>{if(!SELECTED.size)return;const r=await fetch('/api/export-selected-packet',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys:[...SELECTED]})}),j=await r.json();if(!r.ok){alert(j.error||'Packet export failed');return}window.open(j.url,'_blank');alert(`Selected packet exported:\n${j.html}\n${j.json}\n\nHTML is self-contained and printable.`)};load();
</script></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="qwen-training-caption-review-server")
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def _load_json(path: Path | None, default: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, dict) else default


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _safe_media(bundle: Path, rel: str) -> Path | None:
    try:
        candidate = (bundle / rel).resolve()
        candidate.relative_to(bundle.resolve())
    except (ValueError, OSError):
        return None
    return candidate


def _export(bundle: Path, index: dict[str, Any], annotations: dict[str, Any]) -> tuple[Path, Path]:
    rows: list[dict[str, Any]] = []
    annos = annotations.get("records") or {}
    for record in index.get("records") or []:
        key = str(record.get("image_key") or "")
        anno = annos.get(key) or {}
        for profile in ("compact", "medium"):
            grade = anno.get(profile) or {}
            caption = (record.get("captions") or {}).get(profile) or {}
            rows.append(
                {
                    "image_key": key,
                    "profile": profile,
                    "predicted_pose": record.get("pose"),
                    "caption_words": caption.get("word_count"),
                    "repair_action": caption.get("repair_action"),
                    "fit": grade.get("fit"),
                    "pose_language": grade.get("pose_language"),
                    "issues": grade.get("issues") or [],
                    "pipeline_fix": grade.get("pipeline_fix"),
                    "notes": grade.get("notes"),
                    "reviewed": bool(anno.get("reviewed")),
                    "preferred_profile": anno.get("preferred_profile"),
                    "overall_notes": anno.get("overall_notes"),
                    "caption": caption.get("text"),
                }
            )
    payload = {
        "schema_version": "training-caption-review-export-0.1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_schema_version": index.get("schema_version"),
        "records": rows,
    }
    json_path = bundle / "training_caption_review_export.json"
    csv_path = bundle / "training_caption_review_export.csv"
    _atomic_json(json_path, payload)
    fields = [
        "image_key", "profile", "predicted_pose", "caption_words", "repair_action",
        "fit", "pose_language", "issues", "pipeline_fix", "notes", "reviewed",
        "preferred_profile", "overall_notes", "caption",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["issues"] = ", ".join(flat["issues"])
            writer.writerow(flat)
    return json_path, csv_path


def _data_uri(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _profile_packet(record: dict[str, Any], anno: dict[str, Any], profile: str) -> str:
    c = (record.get("captions") or {}).get(profile) or {}
    a = anno.get(profile) or {}
    issues = ", ".join(a.get("issues") or []) or "-"
    return f'''<section class="profile"><h3>{html.escape(profile.title())} · {c.get("word_count","-")} words</h3><p class="caption">{html.escape(str(c.get("text") or ""))}</p><table><tr><th>Fit</th><td>{html.escape(str(a.get("fit") or "-"))}</td></tr><tr><th>Pose wording</th><td>{html.escape(str(a.get("pose_language") or "-"))}</td></tr><tr><th>Issues</th><td>{html.escape(issues)}</td></tr><tr><th>Pipeline fix?</th><td>{html.escape(str(a.get("pipeline_fix") or "-"))}</td></tr><tr><th>Notes</th><td>{html.escape(str(a.get("notes") or "-")).replace(chr(10),"<br>")}</td></tr></table></section>'''


def _packet_html(bundle: Path, records: list[dict[str, Any]], annotations: dict[str, Any]) -> str:
    annos = annotations.get("records") or {}
    cards: list[str] = []
    for record in records:
        key = str(record.get("image_key") or "")
        anno = annos.get(key) or {}
        original = _safe_media(bundle, str(record.get("original") or ""))
        overlay = _safe_media(bundle, str(record.get("overlay") or ""))
        phrases = record.get("caption_ready_phrases") or []
        phrase_html = "".join(f"<li>{html.escape(str(x))}</li>" for x in phrases) or "<li>None</li>"
        cards.append(f'''<article class="card"><h2>{html.escape(key)}</h2><div class="images"><figure><figcaption>Original</figcaption><img src="{_data_uri(original)}"></figure><figure><figcaption>DWPose + SAM3D</figcaption><img src="{_data_uri(overlay)}"></figure></div><section class="pose"><b>Governed pose:</b> {html.escape(str(record.get("pose") or "-"))} · <b>best candidate:</b> {html.escape(str(record.get("best_candidate_pose") or "-"))}<ul>{phrase_html}</ul></section><div class="profiles">{_profile_packet(record,anno,"compact")}{_profile_packet(record,anno,"medium")}</div><section class="overall"><b>Review complete:</b> {"yes" if anno.get("reviewed") else "no"} · <b>Preferred:</b> {html.escape(str(anno.get("preferred_profile") or "-"))}<p>{html.escape(str(anno.get("overall_notes") or "-")).replace(chr(10),"<br>")}</p></section></article>''')
    created = datetime.now(timezone.utc).isoformat()
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Training Caption Review Packet</title><style>body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#0b0d10;color:#edf1f5;margin:20px}}.card{{page-break-after:always;background:#14181d;border:1px solid #39424c;border-radius:10px;padding:14px;margin-bottom:22px}}.images,.profiles{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}figure{{margin:0}}figcaption{{font-size:12px;color:#aab3bd}}img{{width:100%;max-height:600px;object-fit:contain;background:#080a0d}}.pose,.overall,.profile{{border:1px solid #39424c;border-radius:8px;padding:10px;margin-top:10px}}.caption{{line-height:1.45}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #2b323a;padding:4px}}th{{width:110px;color:#aab3bd}}@media print{{body{{background:white;color:black;margin:8mm}}.card{{background:white;border-color:#aaa}}th{{color:#444}}}}</style></head><body><h1>Semantic V3 Training Caption Review</h1><p>Generated {html.escape(created)} · {len(records)} selected image(s)</p>{''.join(cards)}</body></html>'''


def _export_selected_packet(bundle: Path, keys: list[str]) -> tuple[Path, Path]:
    index = _load_json(bundle / "training_caption_review.index.json", {"records": []})
    annotations = _load_json(bundle / "training_caption_review_annotations.json", {"records": {}})
    by_key = {str(r.get("image_key")): r for r in index.get("records") or []}
    records = [by_key[key] for key in keys if key in by_key]
    if not records:
        raise ValueError("no_valid_selected_keys")
    payload = {
        "schema_version": "training-caption-review-selected-packet-0.1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "selected_keys": [str(r.get("image_key")) for r in records],
        "records": [
            {"record": r, "annotation": (annotations.get("records") or {}).get(str(r.get("image_key"))) or {}, "raw": _load_json(_safe_media(bundle, str(r.get("raw_json") or "")), {})}
            for r in records
        ],
    }
    json_path = bundle / "training_caption_review_selected_packet.json"
    html_path = bundle / "training_caption_review_selected_packet.html"
    _atomic_json(json_path, payload)
    html_path.write_text(_packet_html(bundle, records, annotations), encoding="utf-8")
    return html_path, json_path


def make_handler(bundle: Path):
    index_path = bundle / "training_caption_review.index.json"
    annotations_path = bundle / "training_caption_review_annotations.json"
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "TrainingCaptionReview/0.1"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[caption-review] {self.address_string()} - {fmt % args}")

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache" if "json" in content_type else "public, max-age=3600")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, value: dict[str, Any], status: int = 200) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/index":
                self._json(_load_json(index_path, {"records": [], "record_count": 0}))
                return
            if path == "/api/annotations":
                self._json(_load_json(annotations_path, {"schema_version": ANNOTATION_SCHEMA, "records": {}}))
                return
            if path.startswith("/media/"):
                target = _safe_media(bundle, unquote(path[len("/media/"):]))
                if target is None or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self._send(200, target.read_bytes(), mime)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _body(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                value = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return value if isinstance(value, dict) else None

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/api/annotation/"):
                key = unquote(path[len("/api/annotation/"):])
                body = self._body()
                if body is None:
                    self._json({"error": "annotation_must_be_object"}, 400)
                    return
                with lock:
                    data = _load_json(annotations_path, {"schema_version": ANNOTATION_SCHEMA, "records": {}})
                    data.setdefault("records", {})[key] = body
                    data["updated_at"] = datetime.now(timezone.utc).isoformat()
                    _atomic_json(annotations_path, data)
                self._json({"ok": True})
                return
            if path == "/api/export":
                with lock:
                    jp, cp = _export(bundle, _load_json(index_path, {"records": []}), _load_json(annotations_path, {"records": {}}))
                self._json({"ok": True, "json": str(jp), "csv": str(cp)})
                return
            if path == "/api/export-selected-packet":
                body = self._body()
                keys = body.get("keys") if body else None
                if not isinstance(keys, list):
                    self._json({"error": "keys_must_be_array"}, 400)
                    return
                try:
                    hp, jp = _export_selected_packet(bundle, [str(x) for x in keys])
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                self._json({"ok": True, "html": str(hp), "json": str(jp), "url": "/media/" + hp.name})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main() -> int:
    args = parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    index_path = bundle / "training_caption_review.index.json"
    if not index_path.is_file():
        raise SystemExit(f"Training caption review index not found: {index_path}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bundle))
    url = f"http://{args.host}:{args.port}/"
    print(f"Training Caption Review: {url}")
    print(f"Bundle: {bundle}")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
