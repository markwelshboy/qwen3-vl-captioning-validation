from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pose Review</title>
<style>
:root{color-scheme:dark;--bg:#0b0d10;--panel:#14181d;--panel2:#1b2026;--text:#edf1f5;--muted:#aab3bd;--line:#2b323a;--cyan:#36d7ff;--amber:#ffbf3f;--bad:#ff6b6b;--good:#6ee7a8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,sans-serif}.toolbar{position:sticky;top:0;z-index:10;background:rgba(11,13,16,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:12px 16px}.toolbar h1{font-size:20px;margin:0 0 10px}.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.controls input,.controls select,.controls button{background:#171b20;color:var(--text);border:1px solid #39424c;border-radius:7px;padding:7px 9px}.controls button{cursor:pointer}.legend{margin-left:auto;color:var(--muted);font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 10px}.cyan{background:var(--cyan)}.amber{background:var(--amber)}.summary{padding:8px 16px;color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(720px,1fr));gap:14px;padding:0 14px 28px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.card.reviewed{border-color:#496356}.head{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:10px 12px;background:var(--panel2);border-bottom:1px solid var(--line)}.head h2{font-size:16px;margin:0}.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{font-size:12px;background:#252c34;border-radius:999px;padding:3px 7px;color:#d9e0e7}.chip.warn{background:#4b3d1e}.visuals{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}.visuals figure{margin:0;min-width:0}.visuals figcaption{font-size:12px;color:var(--muted);padding:3px 2px}.visuals img{display:block;width:100%;height:auto;max-height:560px;object-fit:contain;background:#080a0d;border-radius:7px}.body{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px 12px 12px}.metrics{font-size:13px}.metrics table{width:100%;border-collapse:collapse}.metrics td{padding:3px 4px;border-bottom:1px solid #232a31}.metrics td:last-child{text-align:right;font-variant-numeric:tabular-nums}.scores{display:grid;grid-template-columns:auto 1fr auto;gap:4px 7px;align-items:center;margin:8px 0}.bar{height:7px;background:#252b31;border-radius:99px;overflow:hidden}.bar>i{display:block;height:100%;background:#75808b}.before .bar>i{background:#59616a}.review label{font-size:12px;color:var(--muted);display:block;margin-top:7px}.review input,.review select,.review textarea{width:100%;background:#11151a;color:var(--text);border:1px solid #39424c;border-radius:6px;padding:6px}.review textarea{min-height:66px;resize:vertical}.verdicts{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:6px}.verdicts button{background:#20262d;color:var(--text);border:1px solid #39424c;border-radius:6px;padding:6px;cursor:pointer}.verdicts button.active{outline:2px solid #8aa4bd}.review-actions{display:flex;gap:6px;margin-top:8px}.review-actions button,.jsonbtn{background:#20262d;color:var(--text);border:1px solid #39424c;border-radius:6px;padding:6px 8px;cursor:pointer}.saved{font-size:12px;color:var(--good);align-self:center}.raw{display:none;margin:0 12px 12px;max-height:420px;overflow:auto;background:#090b0e;border:1px solid var(--line);padding:10px;border-radius:7px;font-size:11px;white-space:pre-wrap}.raw.open{display:block}@media(max-width:850px){.grid{grid-template-columns:1fr}.visuals,.body{grid-template-columns:1fr}.legend{width:100%;margin:4px 0 0}}
</style>
</head>
<body>
<div class="toolbar"><h1>Pose QA Review</h1><div class="controls">
<input id="search" placeholder="image key">
<select id="pose"><option value="">all poses</option></select>
<select id="support"><option value="">all support classes</option></select>
<select id="filter"><option value="all">all records</option><option value="uncertain">uncertain only</option><option value="single">single-leg support</option><option value="kneel60">kneel candidate ≥60%</option><option value="strong">crop support ≥50%</option><option value="unreviewed">unreviewed only</option><option value="wrong">marked wrong</option></select>
<select id="sort"><option value="key">sort: key</option><option value="crop_desc">crop support ↓</option><option value="margin_asc">winner margin ↑</option><option value="kneel_desc">kneel score ↓</option><option value="recon_desc">reconstruction ↓</option></select>
<button id="export">Export JSON + CSV</button>
<span class="legend"><span class="dot cyan"></span>DWPose <span class="dot amber"></span>SAM3D</span>
</div></div>
<div id="summary" class="summary"></div><main id="grid" class="grid"></main>
<script>
let INDEX=null, ANNO={records:{}};
const searchEl=document.getElementById('search'), poseEl=document.getElementById('pose'), supportEl=document.getElementById('support'), filterEl=document.getElementById('filter'), sortEl=document.getElementById('sort'), exportEl=document.getElementById('export'), summaryEl=document.getElementById('summary'), gridEl=document.getElementById('grid');
const poses=['standing','crouching','squatting','sitting','reclined','kneeling','half_kneeling','other','unclear'];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>Number.isFinite(Number(v))?Number(v):0;
async function load(){INDEX=await (await fetch('/api/index')).json(); ANNO=await (await fetch('/api/annotations')).json(); initFilters(); render();}
function initFilters(){const p=[...new Set(INDEX.records.map(r=>r.pose).filter(Boolean))].sort(); poseEl.innerHTML='<option value="">all poses</option>'+p.map(x=>`<option>${esc(x)}</option>`).join(''); const s=[...new Set(INDEX.records.map(r=>r.support_class).filter(Boolean))].sort(); supportEl.innerHTML='<option value="">all support classes</option>'+s.map(x=>`<option>${esc(x)}</option>`).join('');}
function filtered(){let rows=[...INDEX.records];const q=searchEl.value.trim().toLowerCase();if(q)rows=rows.filter(r=>r.image_key.toLowerCase().includes(q));if(poseEl.value)rows=rows.filter(r=>r.pose===poseEl.value);if(supportEl.value)rows=rows.filter(r=>r.support_class===supportEl.value);const f=filterEl.value;if(f==='uncertain')rows=rows.filter(r=>r.pose==='uncertain');if(f==='single')rows=rows.filter(r=>r.support_state?.geometry_match);if(f==='kneel60')rows=rows.filter(r=>num(r.kneeling_candidate?.score_percent)>=60);if(f==='strong')rows=rows.filter(r=>num(r.crop_support_percent)>=50);if(f==='unreviewed')rows=rows.filter(r=>!ANNO.records?.[r.image_key]?.verdict);if(f==='wrong')rows=rows.filter(r=>ANNO.records?.[r.image_key]?.verdict==='wrong');const srt=sortEl.value;rows.sort((a,b)=>{if(srt==='crop_desc')return num(b.crop_support_percent)-num(a.crop_support_percent);if(srt==='margin_asc')return num(a.winner_margin_percent)-num(b.winner_margin_percent);if(srt==='kneel_desc')return num(b.kneeling_candidate?.score_percent)-num(a.kneeling_candidate?.score_percent);if(srt==='recon_desc')return num(b.reconstruction_match_percent)-num(a.reconstruction_match_percent);return a.image_key.localeCompare(b.image_key);});return rows;}
function scoreRows(r){const after=r.posture_score_percent||{}, before=r.posture_score_percent_before_support_topology||{};const names=[...new Set([...Object.keys(after),...Object.keys(before)])];return names.map(n=>`<div>${esc(n)}</div><div class="bar"><i style="width:${num(after[n])}%"></i></div><div>${num(after[n])}%</div>`).join('') + (Object.keys(before).length?`<div style="grid-column:1/-1;color:#87919b;margin-top:5px">before support topology</div>`+names.map(n=>`<div class="before">${esc(n)}</div><div class="bar before"><i style="width:${num(before[n])}%"></i></div><div>${num(before[n])}%</div>`).join(''):'');}
function relationText(r){return (r.relations||[]).map(x=>`${x.name}${x.side?':'+x.side:''}@${x.crop_support_percent??0}%`).join(', ')||'-';}
function card(r){const a=ANNO.records?.[r.image_key]||{};const single=r.support_state?.geometry_match;const kneel=num(r.kneeling_candidate?.score_percent);const chips=[`pose: ${r.pose}`,`best: ${r.best_candidate_pose}`,`crop: ${r.crop_support_percent??0}%`,r.support_class];if(single)chips.push(`single-leg ${r.support_state.candidate_support_side} / free ${r.support_state.candidate_free_leg}`);if(kneel>=60)chips.push(`kneel ${kneel}%`);return `<article class="card ${a.verdict?'reviewed':''}" data-key="${esc(r.image_key)}"><div class="head"><h2>${esc(r.image_key)}</h2><div class="chips">${chips.filter(Boolean).map((x,i)=>`<span class="chip ${(single||kneel>=60)&&i>=4?'warn':''}">${esc(x)}</span>`).join('')}</div></div><div class="visuals"><figure><figcaption>Original</figcaption><img loading="lazy" src="/media/${encodeURI(r.original)}"></figure><figure><figcaption>Combined overlay — cyan DWPose / amber SAM3D</figcaption><img loading="lazy" src="/media/${encodeURI(r.overlay)}"></figure></div><div class="body"><section class="metrics"><table><tr><td>Projected</td><td>${esc(r.pose)}</td></tr><tr><td>Best candidate</td><td>${esc(r.best_candidate_pose)}</td></tr><tr><td>Reconstruction</td><td>${r.reconstruction_match_percent??0}%</td></tr><tr><td>Crop support</td><td>${r.crop_support_percent??0}%</td></tr><tr><td>Crop coverage</td><td>${r.crop_coverage_percent??0}%</td></tr><tr><td>Winner margin</td><td>${r.winner_margin_percent??0}%</td></tr><tr><td>Support topology</td><td>${single?`single-leg: ${esc(r.support_state.candidate_support_side)} / free ${esc(r.support_state.candidate_free_leg)} @ ${r.support_state.crop_support_percent??0}%`:'-'}</td></tr><tr><td>Kneeling candidate</td><td>${kneel}%${r.kneeling_candidate?.geometry_match?' *':''}</td></tr><tr><td>Relations</td><td>${esc(relationText(r))}</td></tr></table><div class="scores">${scoreRows(r)}</div><button class="jsonbtn" onclick="toggleRaw('${esc(r.image_key)}')">Raw JSON</button></section><section class="review"><label>Verdict</label><div class="verdicts">${[['correct','✓ correct'],['approx','~ approx'],['wrong','✗ wrong'],['unclear','? unclear']].map(([v,t])=>`<button class="${a.verdict===v?'active':''}" onclick="setVerdict('${esc(r.image_key)}','${v}')">${t}</button>`).join('')}</div><label>Human pose</label><select id="pose_${esc(r.image_key)}"><option value="">--</option>${poses.map(p=>`<option value="${p}" ${a.human_pose===p?'selected':''}>${p}</option>`).join('')}</select><label>Modifiers / geometry notes</label><input id="mods_${esc(r.image_key)}" value="${esc((a.modifiers||[]).join(', '))}" placeholder="one_leg_raised, torso_bent_forward"><label>Notes</label><textarea id="notes_${esc(r.image_key)}">${esc(a.notes||'')}</textarea><div class="review-actions"><button onclick="saveAnno('${esc(r.image_key)}')">Save review</button><span id="saved_${esc(r.image_key)}" class="saved"></span></div></section></div><pre id="raw_${esc(r.image_key)}" class="raw"></pre></article>`;}
function render(){const rows=filtered();summaryEl.textContent=`Showing ${rows.length} / ${INDEX.record_count} records · reviewed ${Object.values(ANNO.records||{}).filter(x=>x.verdict).length}`;gridEl.innerHTML=rows.map(card).join('');}
async function setVerdict(key,v){ANNO.records[key]=ANNO.records[key]||{};ANNO.records[key].verdict=v;await saveAnno(key);}
async function saveAnno(key){const rec=ANNO.records[key]=ANNO.records[key]||{};const p=document.getElementById('pose_'+key),m=document.getElementById('mods_'+key),n=document.getElementById('notes_'+key);if(p)rec.human_pose=p.value;if(m)rec.modifiers=m.value.split(',').map(x=>x.trim()).filter(Boolean);if(n)rec.notes=n.value;const res=await fetch('/api/annotation/'+encodeURIComponent(key),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)});if(!res.ok){alert('Save failed');return;}const saved=document.getElementById('saved_'+key);if(saved){saved.textContent='saved';setTimeout(()=>saved.textContent='',1200);}ANNO=await (await fetch('/api/annotations')).json();render();}
async function toggleRaw(key){const el=document.getElementById('raw_'+key);if(!el.dataset.loaded){const r=INDEX.records.find(x=>x.image_key===key);el.textContent=JSON.stringify(await (await fetch('/media/'+encodeURI(r.raw_json))).json(),null,2);el.dataset.loaded='1';}el.classList.toggle('open');}
[searchEl,poseEl,supportEl,filterEl,sortEl].forEach(el=>{el.addEventListener('input',render);el.addEventListener('change',render);});
exportEl.onclick=async()=>{const r=await fetch('/api/export',{method:'POST'});const j=await r.json();alert(`Exported:\n${j.json}\n${j.csv}`)};
load();
</script></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="qwen-pose-review-server")
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def _load_json(path: Path, default: dict) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, dict) else default


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _safe_media(bundle: Path, rel: str) -> Path | None:
    candidate = (bundle / rel).resolve()
    try:
        candidate.relative_to(bundle)
    except ValueError:
        return None
    return candidate


def _export(bundle: Path, index: dict, annotations: dict) -> tuple[Path, Path]:
    rows = []
    annotation_rows = annotations.get("records") or {}
    for record in index.get("records") or []:
        key = str(record.get("image_key"))
        annotation = annotation_rows.get(key) or {}
        rows.append({
            "image_key": key,
            "predicted_pose": record.get("pose"),
            "best_candidate_pose": record.get("best_candidate_pose"),
            "crop_support_percent": record.get("crop_support_percent"),
            "support_class": record.get("support_class"),
            "winner_margin_percent": record.get("winner_margin_percent"),
            "reconstruction_match_percent": record.get("reconstruction_match_percent"),
            "single_leg_support": bool((record.get("support_state") or {}).get("geometry_match")),
            "support_side": (record.get("support_state") or {}).get("candidate_support_side"),
            "free_leg": (record.get("support_state") or {}).get("candidate_free_leg"),
            "kneeling_candidate_percent": (record.get("kneeling_candidate") or {}).get("score_percent"),
            "verdict": annotation.get("verdict"),
            "human_pose": annotation.get("human_pose"),
            "modifiers": annotation.get("modifiers") or [],
            "notes": annotation.get("notes"),
        })
    payload = {
        "schema_version": "pose-review-export-0.1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_schema_version": index.get("schema_version"),
        "records": rows,
    }
    json_path = bundle / "pose_review_export.json"
    csv_path = bundle / "pose_review_export.csv"
    _atomic_json(json_path, payload)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "image_key", "predicted_pose", "best_candidate_pose", "crop_support_percent",
            "support_class", "winner_margin_percent", "reconstruction_match_percent",
            "single_leg_support", "support_side", "free_leg", "kneeling_candidate_percent",
            "verdict", "human_pose", "modifiers", "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["modifiers"] = ", ".join(flat["modifiers"])
            writer.writerow(flat)
    return json_path, csv_path


def make_handler(bundle: Path):
    index_path = bundle / "pose_review.index.json"
    annotations_path = bundle / "pose_review_annotations.json"
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "PoseReview/0.1"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[pose-review] {self.address_string()} - {fmt % args}")

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Cache-Control",
                "no-cache" if content_type.startswith("application/json") else "public, max-age=3600",
            )
            self.end_headers()
            self.wfile.write(data)

        def _json(self, value: dict, status: int = 200) -> None:
            self._send(
                status,
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/index":
                self._json(_load_json(index_path, {"records": [], "record_count": 0}))
                return
            if path == "/api/annotations":
                self._json(_load_json(
                    annotations_path,
                    {"schema_version": "pose-review-annotations-0.1", "records": {}},
                ))
                return
            if path.startswith("/media/"):
                rel = unquote(path[len("/media/"):])
                target = _safe_media(bundle, rel)
                if target is None or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self._send(200, target.read_bytes(), content_type)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/api/annotation/"):
                key = unquote(path[len("/api/annotation/"):])
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._json({"error": "invalid_json"}, 400)
                    return
                if not isinstance(body, dict):
                    self._json({"error": "annotation_must_be_object"}, 400)
                    return
                with lock:
                    annotations = _load_json(
                        annotations_path,
                        {"schema_version": "pose-review-annotations-0.1", "records": {}},
                    )
                    annotations.setdefault("records", {})[key] = body
                    annotations["updated_at"] = datetime.now(timezone.utc).isoformat()
                    _atomic_json(annotations_path, annotations)
                self._json({"ok": True})
                return
            if path == "/api/export":
                with lock:
                    index = _load_json(index_path, {"records": []})
                    annotations = _load_json(annotations_path, {"records": {}})
                    json_path, csv_path = _export(bundle, index, annotations)
                self._json({"ok": True, "json": str(json_path), "csv": str(csv_path)})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main() -> int:
    args = parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    index = bundle / "pose_review.index.json"
    if not index.is_file():
        raise SystemExit(f"Pose review bundle not found: {index}")
    url = f"http://{args.host}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bundle))
    print(f"Pose review: {url}")
    print(f"Bundle: {bundle}")
    print("Ctrl-C to stop.")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
