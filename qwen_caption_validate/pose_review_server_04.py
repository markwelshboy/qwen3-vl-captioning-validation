from __future__ import annotations

"""v0.4 local Pose Review UI extension for v0.8 independent support diagnostics."""

from . import pose_review_server as base
from . import pose_review_server_03 as _v03  # noqa: F401  (applies prior UI extensions)


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const diag=document.createElement('select');
  diag.id='diagFilter4';
  diag.innerHTML='<option value="">all v0.8 support diagnostics</option>'+
    '<option value="external">external support review</option>'+
    '<option value="standconf">standing joint conflict</option>'+
    '<option value="lowfeas">low-stance infeasible</option>'+
    '<option value="bilatflex">bilateral flexion ≥45%</option>';
  controls.insertBefore(diag, document.getElementById('filter'));

  const sort=document.getElementById('sort');
  sort.insertAdjacentHTML('beforeend',
    '<option value="external_desc">external support ↓</option>'+
    '<option value="standconf_desc">standing joint conflict ↓</option>'+
    '<option value="supportfeas_asc">support feasibility ↑</option>'+
    '<option value="bilatflex_desc">bilateral flexion ↓</option>');

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(diag.value==='external') rows=rows.filter(r=>r.independent_support_diagnostic?.external_support_review_match);
    if(diag.value==='standconf') rows=rows.filter(r=>r.independent_support_diagnostic?.standing_joint_conflict_review_match);
    if(diag.value==='lowfeas') rows=rows.filter(r=>r.independent_support_diagnostic?.low_stance_feasibility_review_match);
    if(diag.value==='bilatflex') rows=rows.filter(r=>num(r.leg_state_diagnostic?.bilateral_flexion_score_percent)>=45);
    const s=sort.value;
    if(s==='external_desc') rows.sort((a,b)=>num(b.independent_support_diagnostic?.external_support_requirement_percent)-num(a.independent_support_diagnostic?.external_support_requirement_percent));
    if(s==='standconf_desc') rows.sort((a,b)=>num(b.independent_support_diagnostic?.standing_joint_conflict_percent)-num(a.independent_support_diagnostic?.standing_joint_conflict_percent));
    if(s==='supportfeas_asc') rows.sort((a,b)=>num(a.independent_support_diagnostic?.support_feasibility_score_percent)-num(b.independent_support_diagnostic?.support_feasibility_score_percent));
    if(s==='bilatflex_desc') rows.sort((a,b)=>num(b.leg_state_diagnostic?.bilateral_flexion_score_percent)-num(a.leg_state_diagnostic?.bilateral_flexion_score_percent));
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const l=r.leg_state_diagnostic||{};
    const sides=l.per_side||{};
    const left=sides.left||{};
    const right=sides.right||{};
    const s=r.independent_support_diagnostic||{};
    const g=s.geometry||{};
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#77c7ff"><b>v0.8 independent leg/support diagnostic</b></td></tr>
      <tr><td>Leg state</td><td>${esc(l.state||'-')}</td></tr>
      <tr><td>Left knee / hip</td><td>${left.knee_angle_deg??'-'}° / ${left.hip_angle_deg??'-'}°</td></tr>
      <tr><td>Right knee / hip</td><td>${right.knee_angle_deg??'-'}° / ${right.hip_angle_deg??'-'}°</td></tr>
      <tr><td>Bilateral flexion</td><td>${l.bilateral_flexion_score_percent??0}%</td></tr>
      <tr><td>Bilateral straight</td><td>${l.bilateral_straight_score_percent??0}%</td></tr>
      <tr><td>Leg asymmetry</td><td>${l.asymmetry_score_percent??0}%</td></tr>
      <tr><td>Leg-state crop support</td><td>${l.crop_support_percent??0}%</td></tr>
      <tr><td>Support candidate</td><td>${esc(s.candidate||'-')}</td></tr>
      <tr><td>Foot-support feasibility</td><td>${s.support_feasibility_score_percent??0}%</td></tr>
      <tr><td>Flexed foot-supported score</td><td>${s.foot_supported_flexed_stance_feasibility_percent??0}%</td></tr>
      <tr><td>External support requirement</td><td>${s.external_support_requirement_percent??0}%${s.external_support_review_match?' *':''}</td></tr>
      <tr><td>Standing joint conflict</td><td>${s.standing_joint_conflict_percent??0}%${s.standing_joint_conflict_review_match?' *':''}</td></tr>
      <tr><td>Pelvis → support segment</td><td>${g.pelvis_to_support_segment_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Torso → support segment</td><td>${g.torso_proxy_to_support_segment_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Pelvis → foot centroid</td><td>${g.pelvis_to_foot_centroid_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Shoulder shift toward feet</td><td>${g.shoulder_shift_toward_feet_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Shoulder compensation</td><td>${g.shoulder_compensation_fraction??'-'}</td></tr>
      <tr><td>Support diagnostic crop</td><td>${s.crop_support_percent??0}%</td></tr>`;
    return text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
  };

  diag.addEventListener('change',render);
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
