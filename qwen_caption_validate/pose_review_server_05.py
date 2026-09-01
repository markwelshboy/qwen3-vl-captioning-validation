from __future__ import annotations

"""v0.5 local Pose Review UI extension for v0.9 physical governance."""

from . import pose_review_server as base
from . import pose_review_server_04 as _v04  # noqa: F401  (applies prior UI extensions)


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const gov=document.createElement('select');
  gov.id='govFilter5';
  gov.innerHTML='<option value="">all v0.9 governance</option>'+
    '<option value="hardreject">has hard rejection</option>'+
    '<option value="withheld">authority withheld</option>'+
    '<option value="changed">governance changed candidate</option>'+
    '<option value="reclined">governed best reclined</option>';
  controls.insertBefore(gov, document.getElementById('filter'));

  const sort=document.getElementById('sort');
  sort.insertAdjacentHTML('beforeend',
    '<option value="authority_asc">pose authority ↑</option>'+
    '<option value="rejects_desc">hard rejections ↓</option>'+
    '<option value="governed_margin_desc">governed margin ↓</option>');

  function rejectedRows(r){
    const rows=r.physical_governance?.per_pose||{};
    return Object.entries(rows).filter(([_,v])=>v&&v.hard_rejected);
  }

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(gov.value==='hardreject') rows=rows.filter(r=>rejectedRows(r).length>0);
    if(gov.value==='withheld') rows=rows.filter(r=>r.physical_governance?.authority?.usable_as_projected_pose===false);
    if(gov.value==='changed') rows=rows.filter(r=>{
      const a=r.reconstruction_best_candidate_before_governance||'';
      const b=r.physical_governance?.governed_best_candidate_pose||'';
      return a&&b&&a!==b;
    });
    if(gov.value==='reclined') rows=rows.filter(r=>r.physical_governance?.governed_best_candidate_pose==='reclined');
    const s=sort.value;
    if(s==='authority_asc') rows.sort((a,b)=>num(a.physical_governance?.authority?.crop_support_percent)-num(b.physical_governance?.authority?.crop_support_percent));
    if(s==='rejects_desc') rows.sort((a,b)=>rejectedRows(b).length-rejectedRows(a).length);
    if(s==='governed_margin_desc') rows.sort((a,b)=>num(b.physical_governance?.governed_winner_margin_percent)-num(a.physical_governance?.governed_winner_margin_percent));
    return rows;
  };

  const originalCard=card;
  card=function(r){
    let text=originalCard(r);
    const g=r.physical_governance||{};
    const auth=g.authority||{};
    const per=g.per_pose||{};
    const raw=r.posture_score_percent_before_physical_governance||{};
    const governed=r.posture_score_percent||{};
    const rejects=rejectedRows(r).map(([name,row])=>{
      const reasons=(row.hard_rejection_reasons||[]).join(', ');
      return `${name}: ${reasons}`;
    });
    const scoreLine=['standing','crouching','squatting','sitting','reclined'].map(name=>{
      const row=per[name]||{};
      const suffix=row.hard_rejected?' HARD×':'';
      return `${name} ${raw[name]??0}%→${governed[name]??0}%${suffix}`;
    }).join(' | ');
    const rows=`
      <tr><td colspan="2" style="padding-top:10px;color:#ffad66"><b>v0.9 physical governance</b></td></tr>
      <tr><td>Raw reconstruction pose</td><td>${esc(r.reconstruction_pose_before_governance||'-')}</td></tr>
      <tr><td>Raw best candidate</td><td>${esc(r.reconstruction_best_candidate_before_governance||'-')}</td></tr>
      <tr><td>Governed before authority</td><td>${esc(g.governed_pose_before_authority||'-')}</td></tr>
      <tr><td>Governed best candidate</td><td>${esc(g.governed_best_candidate_pose||'-')} @ ${g.governed_best_score_percent??0}%</td></tr>
      <tr><td>Governed winner margin</td><td>${g.governed_winner_margin_percent??0}%</td></tr>
      <tr><td>Pose authority</td><td>${auth.crop_support_percent??0}% [${esc(auth.support_class||'-')}]</td></tr>
      <tr><td>Authority usable</td><td>${auth.usable_as_projected_pose?'yes':'NO'}${auth.withheld_reason?' — '+esc(auth.withheld_reason):''}</td></tr>
      <tr><td>Raw → governed</td><td style="max-width:520px">${esc(scoreLine)}</td></tr>
      <tr><td>Hard rejections</td><td style="max-width:520px">${rejects.length?esc(rejects.join(' ; ')):'-'}</td></tr>`;
    return text.replace('</table><div class="scores">', rows+'</table><div class="scores">');
  };

  gov.addEventListener('change',render);
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
