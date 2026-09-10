from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the SAM3D/Qwen caption-refiner prototype.")
    parser.add_argument("bundle_dir", type=Path, help="Directory containing caption_refiner.index.json.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def _page(index: dict) -> str:
    payload = json.dumps(index, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Caption Refiner — SAM3D Pose Reference</title>
<style>
:root{{--bg:#0b0e12;--panel:#141922;--panel2:#10151d;--line:#29313d;--text:#eef2f6;--muted:#9aa7b5;--accent:#7cc7ff;--warn:#ffc76a;--good:#8ce3b0}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{position:sticky;top:0;z-index:10;background:rgba(11,14,18,.96);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:12px 18px;display:flex;align-items:center;gap:16px}}
header h1{{font-size:18px;margin:0}} header .meta{{color:var(--muted);font-size:13px}} header input{{margin-left:auto;background:#0d1218;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px;min-width:220px}}
main{{max-width:1680px;margin:18px auto 80px;padding:0 18px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:0 0 22px;overflow:hidden}}
.cardhead{{display:flex;align-items:center;gap:12px;padding:11px 14px;border-bottom:1px solid var(--line)}} .key{{font-weight:750;font-size:16px}} .badge{{border:1px solid #3a4654;color:#c4cfdb;border-radius:999px;padding:3px 8px;font-size:12px}} .badge.good{{color:var(--good);border-color:#315c43}} .badge.warn{{color:var(--warn);border-color:#6a5532}}
.visuals{{display:grid;grid-template-columns:minmax(280px,.8fr) minmax(560px,1.65fr);gap:1px;background:var(--line)}} .visual{{background:#090c10;min-height:320px;display:flex;align-items:center;justify-content:center;padding:10px}} .visual img{{max-width:100%;max-height:600px;object-fit:contain;display:block}}
.work{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}} .pane{{background:var(--panel2);padding:14px}} .pane h3{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#bbc5d0;margin:0 0 8px}}
textarea{{width:100%;min-height:150px;resize:vertical;background:#0b1017;color:var(--text);border:1px solid #303947;border-radius:9px;padding:11px 12px;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}
.pose textarea{{min-height:122px}} .preview{{background:#0d131a;border:1px solid #2c3541;border-radius:9px;padding:10px 12px;margin:8px 0;color:#d8e0e8;white-space:pre-wrap}} mark{{background:#755b16;color:#fff2b7;border-radius:3px;padding:0 2px}}
.sentences{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:9px}} .sentence{{background:#0e151e;border:1px solid #27313d;border-radius:8px;padding:8px}} .sentence b{{display:block;color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}}
.actions{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}} button{{background:#202a36;color:#edf3f9;border:1px solid #364456;border-radius:8px;padding:7px 10px;cursor:pointer}} button.primary{{background:#15344a;border-color:#285d7d;color:#cfeeff}} button:hover{{filter:brightness(1.15)}}
.final{{padding:14px;border-top:1px solid var(--line);background:#121821}} .final textarea{{min-height:170px}} .chips{{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}} .chip{{font-size:12px;border-radius:999px;padding:3px 8px;background:#2d2411;color:#ffd787;border:1px solid #5d4820}}
.details{{margin-top:10px;color:var(--muted)}} details summary{{cursor:pointer;color:#b9c5d1}} pre{{white-space:pre-wrap;background:#0b1016;padding:10px;border-radius:8px;overflow:auto}}
.status{{margin-left:auto;color:var(--muted);font-size:12px}} @media(max-width:900px){{.visuals,.work{{grid-template-columns:1fr}}.sentences{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Caption Refiner</h1><div class="meta">Existing JSON caption + SAM3D pose-card VLM</div><input id="search" placeholder="Filter image key…"></header>
<main id="app"></main>
<script id="data" type="application/json">{payload}</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent); const app=document.getElementById('app');
const edits={{}};
function esc(s){{return String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));}}
function splitSentences(text){{return String(text||'').trim().split(/(?<=[.!?])\s+/).filter(Boolean);}}
function spatialPreview(text){{let e=esc(text); return e.replace(/\b(frame\s+left|frame\s+right|image\s+left|image\s+right|left|right|behind|foreground|in front of)\b/gi,'<mark>$1</mark>');}}
function poseText(r){{return r.pose_vlm?.text||'';}}
function poseExisting(r){{const p=r.existing_pose_language||{{}}; return (p.caption_ready_phrases||[]).join(' ');}}
function card(r){{
 const ps=splitSentences(poseText(r)); const cs=splitSentences(r.existing_caption||''); const exact=r.pose_vlm?.exactly_two_sentences;
 const chips=(r.spatial_review_terms||[]).map(x=>`<span class="chip">${{esc(x)}}</span>`).join('');
 const diag=poseExisting(r);
 return `<article class="card" data-key="${{esc(r.image_key)}}"><div class="cardhead"><span class="key">${{esc(r.image_key)}}</span><span class="badge ${{exact?'good':'warn'}}">${{exact?'2-sentence pose ref':'pose ref needs review'}}</span><span class="badge">caption: ${{esc(r.caption_field||'?')}}</span><span class="status" id="status-${{esc(r.image_key)}}"></span></div>
 <div class="visuals"><div class="visual"><img src="/${{esc(r.image_asset)}}" loading="lazy"></div><div class="visual"><img src="/${{esc(r.pose_card_asset)}}" loading="lazy"></div></div>
 <div class="work"><section class="pane"><h3>Existing JSON-derived caption</h3><textarea readonly id="source-${{esc(r.image_key)}}">${{esc(r.existing_caption)}}</textarea><div class="preview">${{spatialPreview(r.existing_caption)}}</div><div class="chips">${{chips}}</div>
 <div class="sentences"><div class="sentence"><b>Current sentence 1</b>${{esc(cs[0]||'—')}}</div><div class="sentence"><b>Current sentence 2</b>${{esc(cs[1]||'—')}}</div></div></section>
 <section class="pane pose"><h3>Pose + framing reference</h3><textarea id="pose-${{esc(r.image_key)}}">${{esc(poseText(r))}}</textarea><div class="sentences"><div class="sentence"><b>Pose sentence 1</b>${{esc(ps[0]||'—')}}</div><div class="sentence"><b>Pose sentence 2</b>${{esc(ps[1]||'—')}}</div></div>
 <div class="actions"><button class="primary" onclick="replaceTwo('${{esc(r.image_key)}}')">Replace first 2 sentences</button><button onclick="replaceOne('${{esc(r.image_key)}}',0)">Replace sentence 1</button><button onclick="replaceOne('${{esc(r.image_key)}}',1)">Replace sentence 2</button><button onclick="appendPose('${{esc(r.image_key)}}')">Append pose ref</button></div>
 ${{diag?`<details class="details"><summary>Existing deterministic Pose language</summary><pre>${{esc(diag)}}</pre></details>`:''}}</section></div>
 <section class="final"><h3>Final editable caption</h3><textarea id="final-${{esc(r.image_key)}}">${{esc(r.existing_caption)}}</textarea><div class="actions"><button class="primary" onclick="saveOne('${{esc(r.image_key)}}')">Save edit</button><button onclick="restore('${{esc(r.image_key)}}')">Restore existing caption</button><button onclick="copyFinal('${{esc(r.image_key)}}')">Copy final</button></div></section></article>`;
}}
function getRecord(k){{return DATA.records.find(r=>r.image_key===k);}}
function replaceTwo(k){{const f=document.getElementById('final-'+k), p=document.getElementById('pose-'+k); let a=splitSentences(f.value), b=splitSentences(p.value); if(!b.length)return; f.value=[...b.slice(0,2),...a.slice(2)].join(' ');}}
function replaceOne(k,i){{const f=document.getElementById('final-'+k), p=document.getElementById('pose-'+k); let a=splitSentences(f.value), b=splitSentences(p.value); if(!b[i])return; while(a.length<=i)a.push(''); a[i]=b[i]; f.value=a.filter(Boolean).join(' ');}}
function appendPose(k){{const f=document.getElementById('final-'+k), p=document.getElementById('pose-'+k); f.value=(f.value.trim()+' '+p.value.trim()).trim();}}
function restore(k){{f=document.getElementById('final-'+k); f.value=getRecord(k).existing_caption||'';}}
async function copyFinal(k){{await navigator.clipboard.writeText(document.getElementById('final-'+k).value); document.getElementById('status-'+k).textContent='copied';}}
async function saveOne(k){{const body={{image_key:k,final_caption:document.getElementById('final-'+k).value,pose_reference:document.getElementById('pose-'+k).value}}; const res=await fetch('/api/save',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify(body)}}); const j=await res.json(); document.getElementById('status-'+k).textContent=j.ok?'saved':'save failed';}}
function render(){{const q=document.getElementById('search').value.toLowerCase().trim(); app.innerHTML=DATA.records.filter(r=>!q||String(r.image_key).toLowerCase().includes(q)).map(card).join('');}}
document.getElementById('search').addEventListener('input',render); render();
</script></body></html>'''


def main() -> int:
    args = parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    index_path = bundle / "caption_refiner.index.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing {index_path}")
    index = _read_json(index_path)
    edits_path = bundle / "caption_refiner.edits.json"

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = _page(index).encode("utf-8")
                return self._send(200, body, "text/html; charset=utf-8")
            rel = unquote(parsed.path.lstrip("/"))
            target = (bundle / rel).resolve()
            try:
                target.relative_to(bundle)
            except ValueError:
                return self._send(403, b"forbidden", "text/plain")
            if not target.is_file():
                return self._send(404, b"not found", "text/plain")
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self._send(200, target.read_bytes(), mime)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/save":
                return self._send(404, b'{"ok":false}', "application/json")
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                key = str(payload.get("image_key") or "").strip()
                if not key:
                    raise ValueError("missing image_key")
                edits = _read_json(edits_path) if edits_path.exists() else {"schema_version": "caption-refiner-edits-0.1", "records": {}}
                records = edits.setdefault("records", {})
                records[key] = {
                    "final_caption": str(payload.get("final_caption") or "").strip(),
                    "pose_reference": str(payload.get("pose_reference") or "").strip(),
                }
                _write_json(edits_path, edits)
                export_dir = bundle / "exported-captions"
                export_dir.mkdir(parents=True, exist_ok=True)
                (export_dir / f"{key}.txt").write_text(records[key]["final_caption"].strip() + "\n", encoding="utf-8")
                return self._send(200, b'{"ok":true}', "application/json")
            except Exception as exc:
                body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                return self._send(400, body, "application/json")

        def log_message(self, fmt: str, *args) -> None:
            print(f"[caption-refiner] {self.address_string()} - {fmt % args}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Caption refiner: http://{args.host}:{args.port}/")
    print(f"Bundle: {bundle}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
