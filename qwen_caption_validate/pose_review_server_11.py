from __future__ import annotations

"""v0.11 local Pose Review UI for v0.16 authority + posture modifiers."""

import html
from pathlib import Path
from typing import Any

from . import pose_review_server as base
from . import pose_review_server_10 as _v10  # noqa: F401  (applies all prior UI)
from . import pose_review_server_06 as _v06


# Older review layers called this value "Crop authority".  It is actually
# DWPose/SAM3D pose-joint corroboration inside the crop, not a segmentation-like
# estimate of how much of the visible human body is present.  The distinction is
# important for severe foreshortening such as poseblind-02_00200.
base.HTML = base.HTML.replace("Crop authority", "Pose-joint authority")


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const v16filter=document.createElement('select');
  v16filter.id='v16Filter';
  v16filter.innerHTML='<option value="">all v0.16 authority/modifier states</option>'+
    '<option value="crouchwithheld">crouch withheld: lower body unobserved</option>'+
    '<option value="heavylean">heavy / near-horizontal lean</option>'+
    '<option value="shouldertilt">strong shoulder-line tilt</option>'+
    '<option value="recovery">strong candidate needs Fusion recovery</option>';
  controls.insertBefore(v16filter, document.getElementById('filter'));

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(v16filter.value==='crouchwithheld') rows=rows.filter(r=>r.assertion_authority?.withheld_by_v16===true && r.assertion_authority?.selected_path==='reconstruction_only_crouch');
    if(v16filter.value==='heavylean') rows=rows.filter(r=>['heavy','near_horizontal'].includes(r.posture_modifier_diagnostic?.lean_severity));
    if(v16filter.value==='shouldertilt') rows=rows.filter(r=>['strong','very_strong'].includes(r.posture_modifier_diagnostic?.shoulder_line_tilt_severity));
    if(v16filter.value==='recovery') rows=rows.filter(r=>r.semantic_recovery?.needed===true);
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const a=r.assertion_authority||{};
    const crouch=a.crouch_paths?.hip_knee_chain||{};
    const m=r.posture_modifier_diagnostic||{};
    const obs=m.dwpose_observed_axes||{};
    const sem=r.semantic_recovery||{};
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#82d4ff"><b>v0.16 low-stance authority + posture modifiers</b></td></tr>
      <tr><td>Pose-joint authority</td><td>${a.crop_support_percent??0}% · this is joint corroboration, not literal visible-body extent</td></tr>
      <tr><td>Crouch hip / knee path</td><td>${crouch.qualifies?'PASS':'no'} · hip ${crouch.hip_authority_percent??0}% · thigh ${crouch.thigh_authority_percent??0}% · knee ${crouch.knee_authority_percent??0}% · path ${crouch.path_authority_percent??0}%</td></tr>
      <tr><td>v0.16 assertion</td><td>${a.public_pose_before_v16??a.public_pose_before??'-'} → ${a.public_pose_after??'-'} · ${a.selected_path??'-'} ${a.withheld_by_v16?'· WITHHELD':''}</td></tr>
      <tr><td>Torso inclination</td><td>${m.torso_inclination_from_vertical_deg??'-'}° · ${m.lean_severity??'-'} · ${m.lean_direction??'-'} · authority ${m.torso_inclination_authority_percent??0}% · ${m.torso_inclination_source??'-'}</td></tr>
      <tr><td>Shoulder declination</td><td>${m.shoulder_line_declination_deg??'-'}° · ${m.shoulder_line_tilt_severity??'-'} · lower ${m.lower_shoulder??'-'} · authority ${m.shoulder_line_declination_authority_percent??0}% · ${m.shoulder_line_declination_source??'-'}</td></tr>
      <tr><td>DWPose observed torso</td><td>${obs.torso_axis_from_image_down_deg??'-'}° · shoulders ${obs.shoulders_observed?'yes':'no'} · hips ${obs.hips_observed?'yes':'no'}</td></tr>
      <tr><td>Posture modifiers</td><td>${(m.suggested_modifiers||[]).join(', ')||'-'}${m.suggested_compound_pose_modifier?' · '+m.suggested_compound_pose_modifier:''}</td></tr>
      <tr><td>Fusion semantic recovery</td><td>${sem.needed?'YES · '+(sem.candidate_pose??'-')+' '+(sem.candidate_score_percent??0)+'% / margin '+(sem.winner_margin_percent??0)+'%':'no'}</td></tr>`;
    return text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
  };

  v16filter.addEventListener('change',render);
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


_old_packet_html = _v06._packet_html


def _packet_html_v11(
    bundle: Path,
    records: list[dict[str, Any]],
    annotations: dict[str, Any],
    raw_by_key: dict[str, Any],
) -> str:
    text = _old_packet_html(bundle, records, annotations, raw_by_key)
    text = text.replace("<th>Crop authority</th>", "<th>Pose-joint authority</th>")
    for record in records:
        key = str(record.get("image_key") or "")
        if not key:
            continue
        a = record.get("assertion_authority") or {}
        crouch = (a.get("crouch_paths") or {}).get("hip_knee_chain") or {}
        mod = record.get("posture_modifier_diagnostic") or {}
        sem = record.get("semantic_recovery") or {}
        modifiers = ", ".join(str(x) for x in (mod.get("suggested_modifiers") or [])) or "-"
        row = (
            "<tr><th>v0.16 authority / lean</th><td>"
            f"pose-joint-authority {a.get('crop_support_percent',0)}% · "
            f"crouch-path {crouch.get('path_authority_percent',0)}% "
            f"({'PASS' if crouch.get('qualifies') else 'no'}) · "
            f"{html.escape(str(a.get('public_pose_before_v16',a.get('public_pose_before','-'))))} → "
            f"{html.escape(str(a.get('public_pose_after','-')))} · "
            f"{'WITHHELD · ' if a.get('withheld_by_v16') else ''}"
            f"torso {html.escape(str(mod.get('torso_inclination_from_vertical_deg','-')))}° "
            f"[{html.escape(str(mod.get('lean_severity','-')))} / {html.escape(str(mod.get('lean_direction','-')))}] · "
            f"shoulder {html.escape(str(mod.get('shoulder_line_declination_deg','-')))}° · "
            f"mods {html.escape(modifiers)} · "
            f"Fusion {'YES:'+html.escape(str(sem.get('candidate_pose','-'))) if sem.get('needed') else 'no'}"
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


_v06._packet_html = _packet_html_v11


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
