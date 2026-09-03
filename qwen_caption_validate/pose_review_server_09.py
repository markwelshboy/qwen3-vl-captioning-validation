from __future__ import annotations

"""v0.9 local Pose Review UI for v0.14 targeted governance."""

import html
from pathlib import Path
from typing import Any

from . import pose_review_server as base
from . import pose_review_server_08 as _v08  # noqa: F401  (applies all prior UI)
from . import pose_review_server_06 as _v06


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const v14filter=document.createElement('select');
  v14filter.id='v14Filter';
  v14filter.innerHTML='<option value="">all v0.14 targeted states</option>'+
    '<option value="recline">whole-body recline override</option>'+
    '<option value="squatcomp">relative squat compensation applied</option>'+
    '<option value="headreject">proximal head-support rejected</option>';
  controls.insertBefore(v14filter, document.getElementById('filter'));

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(v14filter.value==='recline') rows=rows.filter(r=>r.whole_body_recline_override?.applied===true);
    if(v14filter.value==='squatcomp') rows=rows.filter(r=>r.relative_squat_compensation?.applied===true);
    if(v14filter.value==='headreject') rows=rows.filter(r=>r.head_support_proximal_guard?.geometry_match===false);
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const recl=r.whole_body_recline_override||{};
    const comp=r.relative_squat_compensation||{};
    const head=r.head_support_proximal_guard||{};
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#ff9f7a"><b>v0.14 targeted governance</b></td></tr>
      <tr><td>Whole-body recline</td><td>${recl.whole_body_recline_score_percent??0}% · ${recl.applied?'APPLIED':'no'} · sitting factor ${recl.sitting_factor??'-'}</td></tr>
      <tr><td>Recline inputs</td><td>body ${Math.round(100*num(recl.lower_body_recline_score))}% · retreat ${Math.round(100*num(recl.retreat_from_support))}% · external ${Math.round(100*num(recl.external_support_requirement))}%</td></tr>
      <tr><td>Relative squat compensation</td><td>${comp.applied?'APPLIED':'no'} · fraction ${comp.shoulder_compensation_fraction??'-'} · need ${Math.round(100*num(comp.compensation_need))}%</td></tr>
      <tr><td>Squat compensation factor</td><td>${comp.squat_factor??'-'} · sufficiency ${comp.compensation_sufficiency==null?'-':Math.round(100*num(comp.compensation_sufficiency))+'%'}</td></tr>
      <tr><td>Open-hand proximal support</td><td>${head.available===false?'unavailable':head.geometry_match===false?'REJECTED':head.geometry_match===true?'passes':'-'}</td></tr>
      <tr><td>Distal / palm / wrist</td><td>${head.hand_to_face_shoulder_widths??'-'} / ${head.palm_root_to_head_shoulder_widths??'-'} / ${head.wrist_to_head_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Palm / wrist excess</td><td>${head.palm_excess_over_distal_hand_shoulder_widths??'-'} / ${head.wrist_excess_over_distal_hand_shoulder_widths??'-'} sw</td></tr>`;
    return text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
  };

  v14filter.addEventListener('change',render);
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


_old_packet_html = _v06._packet_html


def _packet_html_v09(
    bundle: Path,
    records: list[dict[str, Any]],
    annotations: dict[str, Any],
    raw_by_key: dict[str, Any],
) -> str:
    text = _old_packet_html(bundle, records, annotations, raw_by_key)
    for record in records:
        key = str(record.get("image_key") or "")
        if not key:
            continue
        recl = record.get("whole_body_recline_override") or {}
        comp = record.get("relative_squat_compensation") or {}
        head = record.get("head_support_proximal_guard") or {}
        row = (
            "<tr><th>v0.14 targeted</th><td>"
            f"whole-recline {recl.get('whole_body_recline_score_percent',0)}% "
            f"({'applied' if recl.get('applied') else 'no'}) · "
            f"sit-factor {html.escape(str(recl.get('sitting_factor','-')))} · "
            f"comp {html.escape(str(comp.get('shoulder_compensation_fraction','-')))} · "
            f"squat-factor {html.escape(str(comp.get('squat_factor','-')))} · "
            f"head-proximal {'REJECT' if head.get('geometry_match') is False else 'pass' if head.get('geometry_match') is True else '-'}"
            "</td></tr>"
        )
        heading = f"<h2>{html.escape(key)}</h2>"
        start = text.find(heading)
        if start < 0:
            continue
        marker = "</table><section class=\"annotation\">"
        end = text.find(marker, start)
        if end < 0:
            continue
        text = text[:end] + row + text[end:]
    return text


_v06._packet_html = _packet_html_v09


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
