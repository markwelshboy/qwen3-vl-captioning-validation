from __future__ import annotations

"""v0.7 local Pose Review UI for v0.12 support-area governance.

Adds the SAM3D side/support-plane diagnostic view, exposes the v0.12 support
model and forward-compensation decisions, and includes the third view in
selected-card packet exports.
"""

import html
from pathlib import Path
from typing import Any

from . import pose_review_server as base
from . import pose_review_server_06 as _v06  # noqa: F401  (applies packet export and prior UI)


EXTRA_UI = r'''
<style>
.visuals.visuals-three{grid-template-columns:repeat(3,1fr)}
@media(max-width:1100px){.visuals.visuals-three{grid-template-columns:1fr 1fr}.visuals.visuals-three figure:last-child{grid-column:1/-1}}
@media(max-width:850px){.visuals.visuals-three{grid-template-columns:1fr}.visuals.visuals-three figure:last-child{grid-column:auto}}
</style>
<script>
(function(){
  const controls=document.querySelector('.controls');
  const v12filter=document.createElement('select');
  v12filter.id='v12Filter';
  v12filter.innerHTML='<option value="">all v0.12 support states</option>'+
    '<option value="deferred">support veto deferred</option>'+
    '<option value="forward">forward compensated flexed stance</option>'+
    '<option value="headguard">head-support topology rejected</option>'+
    '<option value="lowfoot">foot feasibility <30%</option>';
  controls.insertBefore(v12filter, document.getElementById('filter'));

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(v12filter.value==='deferred') rows=rows.filter(r=>{
      const x=r.physical_governance?.support_veto_authority?.restored_low_stance_families||[];
      return x.length>0;
    });
    if(v12filter.value==='forward') rows=rows.filter(r=>num(r.physical_governance?.forward_compensation_refine?.advance_toward_support)>=0.35);
    if(v12filter.value==='headguard') rows=rows.filter(r=>r.head_support_topology_guard?.geometry_match===false);
    if(v12filter.value==='lowfoot') rows=rows.filter(r=>num(r.independent_support_diagnostic?.support_feasibility_score_percent)<30);
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    if(r.side_view){
      text=text.replace('class="visuals"','class="visuals visuals-three"');
      const marker='</figure></div><div class="body">';
      const extra=`</figure><figure><figcaption>SAM3D side + support-plane views</figcaption><img loading="lazy" src="/media/${encodeURI(r.side_view)}"></figure></div><div class="body">`;
      text=text.replace(marker,extra);
    }

    const s=r.independent_support_diagnostic||{};
    const g=s.geometry||{};
    const pg=r.physical_governance||{};
    const restore=pg.support_veto_authority||{};
    const fwd=pg.forward_compensation_refine||{};
    const head=r.head_support_topology_guard||{};
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#63e6be"><b>v0.12 support-area governance</b></td></tr>
      <tr><td>Support model</td><td>${esc(s.support_model_version||'-')}</td></tr>
      <tr><td>Foot-area feasibility</td><td>${s.support_feasibility_score_percent??0}%</td></tr>
      <tr><td>Support diagnostic crop</td><td>${s.crop_support_percent??0}% · hard-veto authority ${s.hard_veto_authority_available?'YES':'no'}</td></tr>
      <tr><td>Pelvis → support area</td><td>${g.pelvis_to_support_area_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Torso → support area</td><td>${g.torso_to_support_area_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Forward compensation</td><td>${g.shoulder_shift_toward_feet_shoulder_widths??'-'} sw · rescue ${Math.round(100*num(g.compensation_rescue_score))}%</td></tr>
      <tr><td>Deferred support veto</td><td>${esc((restore.restored_low_stance_families||[]).join(', ')||'-')}</td></tr>
      <tr><td>Forward flex refine</td><td>flex ${Math.round(100*num(fwd.bilateral_flexion))}% · advance ${Math.round(100*num(fwd.advance_toward_support))}%</td></tr>
      <tr><td>Head support topology</td><td>${head.available===false?'unavailable':head.geometry_match===false?'REJECTED':head.geometry_match===true?'passes':'-'}</td></tr>
      <tr><td>Palm / wrist → head</td><td>${head.palm_root_to_head_shoulder_widths??'-'} / ${head.wrist_to_head_shoulder_widths??'-'} sw</td></tr>`;
    return text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
  };

  v12filter.addEventListener('change',render);
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


_old_packet_html = _v06._packet_html


def _packet_html_v07(
    bundle: Path,
    records: list[dict[str, Any]],
    annotations: dict[str, Any],
    raw_by_key: dict[str, Any],
) -> str:
    text = _old_packet_html(bundle, records, annotations, raw_by_key)
    text = text.replace(
        ".images{display:grid;grid-template-columns:1fr 1fr;gap:10px}",
        ".images{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}",
    )
    for record in records:
        side_rel = str(record.get("side_view") or "")
        if not side_rel:
            continue
        side = base._safe_media(bundle, side_rel)
        if side is None or not side.is_file():
            continue
        uri = _v06._data_uri(side)
        key = html.escape(str(record.get("image_key") or ""))
        start = text.find(f"<h2>{key}</h2>")
        if start < 0:
            continue
        end = text.find("</figure></div>\n<div class=\"cols\">", start)
        if end < 0:
            continue
        insert = (
            "</figure><figure><figcaption>SAM3D side + support-plane views</figcaption>"
            f"<img src=\"{uri}\"></figure></div>\n<div class=\"cols\">"
        )
        text = text[:end] + insert + text[end + len("</figure></div>\n<div class=\"cols\">"):]
    return text


_v06._packet_html = _packet_html_v07


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
