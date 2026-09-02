from __future__ import annotations

"""v0.6 local Pose Review UI extension for v0.10 sitting/recline governance."""

from . import pose_review_server as base
from . import pose_review_server_05 as _v05  # noqa: F401  (applies prior UI extensions)


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const refine=document.createElement('select');
  refine.id='refineFilter6';
  refine.innerHTML='<option value="">all v0.10 recline governance</option>'+
    '<option value="upper">upper-body recline ≥55%</option>'+
    '<option value="sitreject">sitting hard rejected</option>'+
    '<option value="reclinepath">upper-body authority path used</option>'+
    '<option value="changed">v0.10 changed v0.9 best</option>';
  controls.insertBefore(refine, document.getElementById('filter'));

  const sort=document.getElementById('sort');
  sort.insertAdjacentHTML('beforeend',
    '<option value="upper_recline_desc">upper-body recline ↓</option>'+
    '<option value="headshoulder_desc">head→shoulder angle ↓</option>'+
    '<option value="pathauth_desc">upper-body authority ↓</option>');

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(refine.value==='upper') rows=rows.filter(r=>num(r.upper_body_recline_diagnostic?.score_percent)>=55);
    if(refine.value==='sitreject') rows=rows.filter(r=>r.physical_governance?.per_pose?.sitting?.hard_rejected);
    if(refine.value==='reclinepath') rows=rows.filter(r=>r.physical_governance?.authority?.authority_path==='upper_body_recline');
    if(refine.value==='changed') rows=rows.filter(r=>{
      const before=r.v09_best_candidate_before_sitting_recline_refine||'';
      const after=r.physical_governance?.governed_best_candidate_pose||'';
      return before&&after&&before!==after;
    });
    const s=sort.value;
    if(s==='upper_recline_desc') rows.sort((a,b)=>num(b.upper_body_recline_diagnostic?.score_percent)-num(a.upper_body_recline_diagnostic?.score_percent));
    if(s==='headshoulder_desc') rows.sort((a,b)=>num(b.upper_body_recline_diagnostic?.head_to_shoulders_axis_from_vertical_deg)-num(a.upper_body_recline_diagnostic?.head_to_shoulders_axis_from_vertical_deg));
    if(s==='pathauth_desc') rows.sort((a,b)=>num(b.upper_body_recline_diagnostic?.path_authority_percent)-num(a.upper_body_recline_diagnostic?.path_authority_percent));
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const u=r.upper_body_recline_diagnostic||{};
    const g=r.physical_governance||{};
    const inp=g.sitting_recline_inputs||{};
    const auth=g.authority||{};
    const sit=g.per_pose?.sitting||{};
    const recl=g.per_pose?.reclined||{};
    const reasons=(sit.hard_rejection_reasons||[]).join(', ');
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#d58cff"><b>v0.10 sitting / upper-body recline governance</b></td></tr>
      <tr><td>Upper-body recline</td><td>${u.score_percent??0}%</td></tr>
      <tr><td>Head → shoulders from vertical</td><td>${u.head_to_shoulders_axis_from_vertical_deg??'-'}°</td></tr>
      <tr><td>Shoulders → hips from vertical</td><td>${u.shoulder_to_hips_axis_from_vertical_deg??'-'}°</td></tr>
      <tr><td>Head → hips from vertical</td><td>${u.head_to_hips_axis_from_vertical_deg??'-'}°</td></tr>
      <tr><td>Upper-chain continuation</td><td>${u.horizontal_chain_continuation_cosine??'-'}</td></tr>
      <tr><td>Knees relative to hips</td><td>${esc(u.knee_position_relative_to_hips||'-')} (${u.mean_hip_to_knee_vertical_drop_shoulder_widths??'-'} sw)</td></tr>
      <tr><td>Upper-body authority</td><td>${u.path_authority_percent??0}%</td></tr>
      <tr><td>Combined recline</td><td>${Math.round(100*num(inp.combined_recline_score))}%</td></tr>
      <tr><td>Body flatness</td><td>${inp.body_flatness_ratio??'-'}</td></tr>
      <tr><td>External support</td><td>${Math.round(100*num(inp.external_support_requirement))}%</td></tr>
      <tr><td>Sitting governed</td><td>${sit.governed_score_percent??0}%${sit.hard_rejected?' HARD×':''}</td></tr>
      <tr><td>Sitting recline factor</td><td>${sit.sitting_recline_feasibility_factor??'-'}</td></tr>
      <tr><td>Sitting reject reason</td><td style="max-width:520px">${reasons?esc(reasons):'-'}</td></tr>
      <tr><td>Reclined governed</td><td>${recl.governed_score_percent??0}%</td></tr>
      <tr><td>Upper recline candidate</td><td>${recl.upper_body_recline_candidate==null?'-':Math.round(100*num(recl.upper_body_recline_candidate))+'%'}</td></tr>
      <tr><td>Authority path</td><td>${esc(auth.authority_path||'-')}</td></tr>
      <tr><td>v0.9 best → v0.10 best</td><td>${esc(r.v09_best_candidate_before_sitting_recline_refine||'-')} → ${esc(g.governed_best_candidate_pose||'-')}</td></tr>`;
    return text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
  };

  refine.addEventListener('change',render);
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
