from __future__ import annotations

"""v0.8 local Pose Review UI for v0.13 seated/low-stance governance."""

import html
from pathlib import Path
from typing import Any

from . import pose_review_server as base
from . import pose_review_server_07 as _v07  # noqa: F401  (applies side-view UI)
from . import pose_review_server_06 as _v06


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const v13filter=document.createElement('select');
  v13filter.id='v13Filter';
  v13filter.innerHTML='<option value="">all v0.13 topology states</option>'+
    '<option value="seated">seated-flexion topology</option>'+
    '<option value="retreat">retreat low-stance counterevidence ≥45%</option>'+
    '<option value="deferred">deferred support rows reweighted</option>'+
    '<option value="squatrescue">forward-supported squat rescue</option>';
  controls.insertBefore(v13filter, document.getElementById('filter'));

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(v13filter.value==='seated') rows=rows.filter(r=>r.seated_low_stance_diagnostic?.seated_flexion_topology_match===true);
    if(v13filter.value==='retreat') rows=rows.filter(r=>num(r.seated_low_stance_diagnostic?.retreat_low_stance_counterevidence)>=0.45);
    if(v13filter.value==='deferred') rows=rows.filter(r=>(r.seated_low_stance_diagnostic?.deferred_support_rows_reweighted||[]).length>0);
    if(v13filter.value==='squatrescue') rows=rows.filter(r=>r.seated_low_stance_diagnostic?.forward_supported_squat_rescue===true);
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const d=r.seated_low_stance_diagnostic||{};
    const pg=r.physical_governance||{};
    const per=pg.per_pose||{};
    const crouch=per.crouching||{};
    const squat=per.squatting||{};
    const sit=per.sitting||{};
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#f0b86e"><b>v0.13 seated / low-stance topology</b></td></tr>
      <tr><td>Retreat / advance</td><td>${Math.round(100*num(d.retreat_from_support))}% / ${Math.round(100*num(d.advance_toward_support))}%</td></tr>
      <tr><td>Retreat low-stance counterevidence</td><td>${Math.round(100*num(d.retreat_low_stance_counterevidence))}%</td></tr>
      <tr><td>Thigh horizontal / torso upright</td><td>${Math.round(100*num(d.thigh_horizontal_score))}% / ${Math.round(100*num(d.torso_upright_score))}%</td></tr>
      <tr><td>Seated-flexion topology</td><td>${d.seated_flexion_topology_match?'MATCH':'no'} · ${d.seated_flexion_topology_score_percent??0}%</td></tr>
      <tr><td>Seated leaning-back candidate</td><td>${d.seated_leaning_back_candidate_percent??0}%</td></tr>
      <tr><td>Deferred support rows reweighted</td><td>${esc((d.deferred_support_rows_reweighted||[]).join(', ')||'-')}</td></tr>
      <tr><td>Crouch retreat factor</td><td>${crouch.v13_torso_retreat_factor??'-'}</td></tr>
      <tr><td>Squat retreat / seated factors</td><td>${squat.v13_torso_retreat_factor??'-'} / ${squat.v13_seated_topology_squat_factor??'-'}</td></tr>
      <tr><td>Sitting topology candidate</td><td>${sit.v13_seated_flexion_topology_candidate==null?'-':Math.round(100*num(sit.v13_seated_flexion_topology_candidate))+'%'}</td></tr>
      <tr><td>Forward-supported squat rescue</td><td>${d.forward_supported_squat_rescue?'YES':'no'}</td></tr>`;
    return text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
  };

  v13filter.addEventListener('change',render);
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


_old_packet_html = _v06._packet_html


def _packet_html_v08(
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
        d = record.get("seated_low_stance_diagnostic") or {}
        pg = record.get("physical_governance") or {}
        per = pg.get("per_pose") or {}
        crouch = per.get("crouching") or {}
        squat = per.get("squatting") or {}
        sit = per.get("sitting") or {}
        row = (
            "<tr><th>v0.13 topology</th><td>"
            f"retreat {int(round(100*float(d.get('retreat_from_support') or 0.0)))}% · "
            f"advance {int(round(100*float(d.get('advance_toward_support') or 0.0)))}% · "
            f"seat-topology {d.get('seated_flexion_topology_score_percent',0)}% · "
            f"crouch-factor {html.escape(str(crouch.get('v13_torso_retreat_factor','-')))} · "
            f"squat-factor {html.escape(str(squat.get('v13_torso_retreat_factor','-')))} / "
            f"{html.escape(str(squat.get('v13_seated_topology_squat_factor','-')))} · "
            f"sit-candidate {html.escape(str(sit.get('v13_seated_flexion_topology_candidate','-')))}"
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


_v06._packet_html = _packet_html_v08


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
