from __future__ import annotations

"""Filter/sort UI extension for the local Pose Review server v0.1."""

from . import pose_review_server as base


EXTRA_UI = r'''
<script>
(function(){
  const controls=document.querySelector('.controls');
  const anchor=document.getElementById('support');
  function add(tag,id,html,attrs={}){
    const el=document.createElement(tag); el.id=id;
    if(html!==null) el.innerHTML=html;
    for(const [k,v] of Object.entries(attrs)) el.setAttribute(k,v);
    anchor.insertAdjacentElement('afterend',el);
    return el;
  }
  const kneelMin2=add('input','kneelMin2',null,{type:'number',min:'0',max:'100',step:'1',placeholder:'kneel ≥ %'});
  kneelMin2.style.width='95px';
  const cropMax2=add('input','cropMax2',null,{type:'number',min:'0',max:'100',step:'1',placeholder:'crop ≤ %'});
  cropMax2.style.width='90px';
  const cropMin2=add('input','cropMin2',null,{type:'number',min:'0',max:'100',step:'1',placeholder:'crop ≥ %'});
  cropMin2.style.width='90px';
  const relation2=add('select','relation2','<option value="">all relations</option>');
  const best2=add('select','best2','<option value="">all best candidates</option>');

  const originalInit=initFilters;
  initFilters=function(){
    originalInit();
    const best=[...new Set(INDEX.records.map(r=>r.best_candidate_pose).filter(Boolean))].sort();
    best2.innerHTML='<option value="">all best candidates</option>'+best.map(x=>`<option>${esc(x)}</option>`).join('');
    const rel=[...new Set(INDEX.records.flatMap(r=>(r.relations||[]).map(x=>x.name)).filter(Boolean))].sort();
    relation2.innerHTML='<option value="">all relations</option>'+rel.map(x=>`<option>${esc(x)}</option>`).join('');
  };

  const originalFiltered=filtered;
  filtered=function(){
    let rows=originalFiltered();
    if(best2.value) rows=rows.filter(r=>r.best_candidate_pose===best2.value);
    if(relation2.value) rows=rows.filter(r=>(r.relations||[]).some(x=>x.name===relation2.value));
    if(cropMin2.value!=='') rows=rows.filter(r=>num(r.crop_support_percent)>=num(cropMin2.value));
    if(cropMax2.value!=='') rows=rows.filter(r=>num(r.crop_support_percent)<=num(cropMax2.value));
    if(kneelMin2.value!=='') rows=rows.filter(r=>num(r.kneeling_candidate?.score_percent)>=num(kneelMin2.value));
    return rows;
  };
  [best2,relation2,cropMin2,cropMax2,kneelMin2].forEach(el=>{
    el.addEventListener('input',render);
    el.addEventListener('change',render);
  });
})();
</script>
'''

base.HTML = base.HTML.replace("</body></html>", EXTRA_UI + "</body></html>")


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
