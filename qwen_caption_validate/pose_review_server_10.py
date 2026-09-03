from __future__ import annotations

"""v0.10 local Pose Review UI for v0.15 assertion authority + orientation."""

import html
from pathlib import Path
from typing import Any

from . import pose_review_server as base
from . import pose_review_server_09 as _v09  # noqa: F401  (applies all prior UI)
from . import pose_review_server_06 as _v06


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const v15filter=document.createElement('select');
  v15filter.id='v15Filter';
  v15filter.innerHTML='<option value="">all v0.15 assertion states</option>'+
    '<option value="withheld">withheld by assertion authority</option>'+
    '<option value="threequarter">~45° / three-quarter body turn</option>'+
    '<option value="profile">near-profile body turn</option>';
  controls.insertBefore(v15filter, document.getElementById('filter'));

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(v15filter.value==='withheld') rows=rows.filter(r=>r.assertion_authority?.withheld_by_v15===true);
    if(v15filter.value==='threequarter') rows=rows.filter(r=>r.body_orientation_diagnostic?.body_yaw_label==='three_quarter');
    if(v15filter.value==='profile') rows=rows.filter(r=>['near_profile','profile'].includes(r.body_orientation_diagnostic?.body_yaw_label));
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const a=r.assertion_authority||{};
    const o=r.body_orientation_diagnostic||{};
    const paths=a.recline_paths||{};
    const whole=paths.whole_body||{};
    const upper=paths.upper_body||{};
    const broad=paths.broad_crop||{};
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#b8a1ff"><b>v0.15 assertion authority + body orientation</b></td></tr>
      <tr><td>Public before / after</td><td>${a.public_pose_before??'-'} → ${a.public_pose_after??'-'} ${a.withheld_by_v15?'· WITHHELD':''}</td></tr>
      <tr><td>Assertion path</td><td>${a.selected_path??'-'} · ${a.selected_path_authority_percent??0}%</td></tr>
      <tr><td>Recline whole-body path</td><td>${whole.qualifies?'PASS':'no'} · score ${whole.score_percent??0}% · observed ${whole.path_authority_percent??0}%</td></tr>
      <tr><td>Recline upper-body path</td><td>${upper.qualifies?'PASS':'no'} · directional ${upper.directional_recline_score_percent??0}% · observed ${upper.path_authority_percent??0}%</td></tr>
      <tr><td>Recline broad-crop path</td><td>${broad.qualifies?'PASS':'no'} · observed ${broad.path_authority_percent??0}%</td></tr>
      <tr><td>Body yaw</td><td>${o.body_yaw_from_frontal_deg??'-'}° · ${o.body_yaw_label??'-'} · observed ${o.observed_authority_percent??0}%</td></tr>
      <tr><td>Shoulder / hip yaw</td><td>${o.shoulder_yaw_from_frontal_deg??'-'}° / ${o.hip_yaw_from_frontal_deg??'-'}°</td></tr>
      <tr><td>Shoulder / hip roll</td><td>${o.shoulder_roll_deg??'-'}° / ${o.hip_roll_deg??'-'}°</td></tr>
      <tr><td>Torso twist</td><td>${o.shoulder_hip_yaw_disagreement_deg??'-'}° ${o.possible_torso_twist?'· POSSIBLE TWIST':''}</td></tr>
      <tr><td>Orientation modifier</td><td>${o.suggested_modifier??'-'}</td></tr>`;
    return text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
  };

  v15filter.addEventListener('change',render);
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


_old_packet_html = _v06._packet_html


def _packet_html_v10(
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
        a = record.get("assertion_authority") or {}
        o = record.get("body_orientation_diagnostic") or {}
        row = (
            "<tr><th>v0.15 authority / orientation</th><td>"
            f"{html.escape(str(a.get('public_pose_before','-')))} → {html.escape(str(a.get('public_pose_after','-')))} · "
            f"path {html.escape(str(a.get('selected_path','-')))} "
            f"({a.get('selected_path_authority_percent',0)}%) · "
            f"{'WITHHELD · ' if a.get('withheld_by_v15') else ''}"
            f"yaw {html.escape(str(o.get('body_yaw_from_frontal_deg','-')))}° "
            f"[{html.escape(str(o.get('body_yaw_label','-')))}] · "
            f"modifier {html.escape(str(o.get('suggested_modifier') or '-'))}"
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


_v06._packet_html = _packet_html_v10


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
