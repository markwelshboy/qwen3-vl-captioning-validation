from __future__ import annotations

"""v0.3 local Pose Review UI extension for v0.7 report-only diagnostics."""

from . import pose_review_server as base
from . import pose_review_server_02 as _v02  # noqa: F401  (applies v0.2 UI extensions)


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const diag=document.createElement('select');
  diag.id='diagFilter3';
  diag.innerHTML='<option value="">all diagnostics</option>'+
    '<option value="balance">balance conflict</option>'+
    '<option value="recline">recline review</option>'+
    '<option value="kneelctx">kneel context conflict</option>';
  controls.insertBefore(diag, document.getElementById('filter'));

  const sort=document.getElementById('sort');
  sort.insertAdjacentHTML('beforeend',
    '<option value="balance_conflict_desc">balance conflict ↓</option>'+
    '<option value="seated_disp_desc">seated displacement ↓</option>'+
    '<option value="recline_diag_desc">recline diagnostic ↓</option>'+
    '<option value="kneel_context_desc">kneel context conflict ↓</option>');

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(diag.value==='balance') rows=rows.filter(r=>r.support_balance_diagnostic?.review_match);
    if(diag.value==='recline') rows=rows.filter(r=>r.recline_diagnostic?.review_match);
    if(diag.value==='kneelctx') rows=rows.filter(r=>r.kneeling_context_diagnostic?.review_match);
    const s=sort.value;
    if(s==='balance_conflict_desc') rows.sort((a,b)=>num(b.support_balance_diagnostic?.low_stance_balance_conflict_percent)-num(a.support_balance_diagnostic?.low_stance_balance_conflict_percent));
    if(s==='seated_disp_desc') rows.sort((a,b)=>num(b.support_balance_diagnostic?.seated_displacement_score_percent)-num(a.support_balance_diagnostic?.seated_displacement_score_percent));
    if(s==='recline_diag_desc') rows.sort((a,b)=>num(b.recline_diagnostic?.score_percent)-num(a.recline_diagnostic?.score_percent));
    if(s==='kneel_context_desc') rows.sort((a,b)=>num(b.kneeling_context_diagnostic?.context_conflict_percent)-num(a.kneeling_context_diagnostic?.context_conflict_percent));
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const b=r.support_balance_diagnostic||{};
    const bg=b.geometry||{};
    const rc=r.recline_diagnostic||{};
    const kc=r.kneeling_context_diagnostic||{};
    const rows=`
      <tr><td colspan="2" style="padding-top:8px;color:#8fa0b0"><b>v0.7 report-only diagnostics</b></td></tr>
      <tr><td>Low-stance balance</td><td>${b.low_stance_balance_score_percent??0}%</td></tr>
      <tr><td>Seated displacement</td><td>${b.seated_displacement_score_percent??0}%</td></tr>
      <tr><td>Balance conflict</td><td>${b.low_stance_balance_conflict_percent??0}%${b.review_match?' *':''}</td></tr>
      <tr><td>Pelvis → support</td><td>${bg.pelvis_to_support_segment_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Shoulder → support</td><td>${bg.shoulder_to_support_segment_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Shoulder closure</td><td>${bg.shoulder_closure_toward_support_shoulder_widths??'-'} sw</td></tr>
      <tr><td>Balance crop support</td><td>${b.crop_support_percent??0}%</td></tr>
      <tr><td>Recline diagnostic</td><td>${rc.score_percent??0}%${rc.review_match?' *':''}</td></tr>
      <tr><td>Current recline / gap</td><td>${rc.current_posture_reclined_score_percent??0}% / +${rc.diagnostic_gap_percent??0}%</td></tr>
      <tr><td>Torso from vertical</td><td>${rc.torso_axis_from_vertical_deg??'-'}°</td></tr>
      <tr><td>Body flatness ratio</td><td>${rc.body_flatness_ratio??'-'}</td></tr>
      <tr><td>Recline crop support</td><td>${rc.crop_support_percent??0}%</td></tr>
      <tr><td>Kneel context conflict</td><td>${kc.context_conflict_percent??0}%${kc.review_match?' *':''}</td></tr>`;
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
