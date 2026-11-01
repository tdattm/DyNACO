"""Audit persisted second-pass events only; no native SRR calls."""
import json
from pathlib import Path
import numpy as np
from r3.common import ROOT,sha256,git

OUT=ROOT/'artifacts/srr_p0/v1'
def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def cost(x,t):return float(np.linalg.norm(x[t]-x[np.roll(t,-1)],axis=1).sum())
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    save('historical_hashes.json',{p:sha256(ROOT/p) for p in git('ls-files').splitlines() if (ROOT/p).is_file()})
    data=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy').astype(np.float64)
    folder=ROOT/'artifacts/r3/m1_additional_k32/v2';cases=[];events=[]
    for i in range(8):
        for seed in (31415,27182):
            for it in (499,999):
                key=f'i{i}_s{seed}_t{it}';p1=json.loads((folder/(key+'_pass1.json')).read_text());p2=json.loads((folder/(key+'_trace.json')).read_text())
                route=np.array(p1['route']);cl=p1['checklist'].copy();processed=set();local=[]
                for e in p2['events']:
                    assert e['node']==cl[e['cursor']];processed.add(e['node'])
                    before=cost(data[i],route)
                    for idx,node in e.get('route_updates',[]):route[idx]=node
                    assert len(set(route))==1000
                    after=cost(data[i],route)
                    if e['selected_gain_f32']>0:
                        u,v,a,b=e['selected_move'];inserted=[z['node'] for z in e['insertions'] if z['inserted']]
                        row=dict(case=key,cursor=e['cursor'],gain_f32=e['selected_gain_f32'],endpoints=[u,v,a,b],
                            unique_endpoints=len({u,v,a,b}),structurally_degenerate=len({u,v,a,b})<4,
                            shared_removed_edges=len({u,a}&{v,b})>0,
                            empty_patch=not bool(e.get('route_updates')),cost_change_f64=before-after,
                            robust_reduction=before-after>max(1e-8,1e-6*before),
                            insertions=e['insertions'],newly_scheduled_nodes=inserted,
                            observed_schedule_extension=bool(inserted))
                        local.append(row);events.append(row)
                    for z in e['insertions']:
                        assert z['already_processed']==(z['node'] in processed)
                        assert z['inserted']==(z['node'] not in cl)
                        if z['inserted']:cl.append(z['node'])
                assert list(route)==p2['route'] and cl==p2['checklist']
                cases.append(dict(case=key,accepted=len(local),empty=sum(r['empty_patch'] for r in local),
                    mutations=sum(not r['empty_patch'] for r in local),robust=sum(r['robust_reduction'] for r in local),
                    noop_extensions=sum(r['empty_patch'] and r['observed_schedule_extension'] for r in local)))
    totals={k:sum(r[k] for r in cases) for k in ('accepted','empty','mutations','robust','noop_extensions')}
    summary=dict(totals=totals,expected_counts_match=totals['accepted']==536 and totals['empty']==396 and totals['mutations']==140,
        noop_degenerate=sum(r['empty_patch'] and r['structurally_degenerate'] for r in events),
        noop_gain_range=[min(r['gain_f32'] for r in events if r['empty_patch']),max(r['gain_f32'] for r in events if r['empty_patch'])],
        noop_nonzero_float64_changes=sum(r['empty_patch'] and r['cost_change_f64']!=0 for r in events))
    save('m15_events.json',events);save('m15_cases.json',cases);save('m15_summary.json',summary)
    (OUT/'M15_REPORT.md').write_text(f'''# M1.5 saved-event audit

{json.dumps(summary,indent=2)}

All32 second-pass traces were applied as exact indexed route patches from saved
pass1 routes, with independent float64 lengths on source float32 coordinates.
Native positive-float32 accepted events, nonempty mutations and robust float64
reductions are distinct. Degeneracy/shared endpoint status is recorded per event,
not assumed as a universal cause. Tiny native positives can reflect cancellation
on overlapping edges. Empty patches have zero actual tour-length change.
Insertion attempts and observed checklist extensions are recorded exactly.
An extension changes the observed later node schedule; its causal effect on
quality cannot be inferred without an intervention. Historical M1 conclusions
remain unchanged; counts of accepted events must not be described as counts of
actual tour improvements. No native 2-opt correction or replay was performed.
''')
    print(json.dumps(summary))
if __name__=='__main__':main()
