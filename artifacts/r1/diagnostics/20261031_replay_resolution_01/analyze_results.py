"""Independent persisted-data audit and descriptive pilot summaries; no GNN."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
PILOT=OUT/'pilot'
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    rows=list(csv.DictReader((PILOT/'paired_observations.csv').open()))
    controls=load(PILOT/'controls.json'); manifest=load(PILOT/'manifest.json')
    assert len(rows)==48 and len(controls)==16
    assert len({(r['decision_id'],r['horizon_iterations']) for r in rows})==48
    for name,value in manifest['source_sha256'].items(): assert sha(ROOT/name)==value,name
    for entry in manifest['datasets']: assert sha(entry['path'])==entry['sha256']
    # Preserve and verify the old measurements, rather than rewriting them.
    old=load(ROOT/'artifacts/r1/diagnostics/20261031_v2_01/v1_audit.json')
    for name,value in old['v1_artifact_hashes'].items():
        assert sha(ROOT/'artifacts/r1/experiments/20261031_matched_v1'/name.replace('\\','/'))==value
    for control in controls:
        ident=control['decision_id']; a=load(PILOT/'runs'/f'{ident}_hold.json'); b=load(PILOT/'runs'/f'{ident}_hold_control.json'); c=load(PILOT/'runs'/f'{ident}_refresh.json')
        t=c['refresh_events'][-1]['iteration']
        assert a['native_guidance_hashes']==b['native_guidance_hashes']
        assert a['iteration_solution_hashes']==b['iteration_solution_hashes']
        assert a['final_search_state_hashes']==b['final_search_state_hashes']
        assert a['tour']==b['tour'] and a['trace']==b['trace']
        assert a['native_guidance_hashes'][:t]==c['native_guidance_hashes'][:t]
        assert a['iteration_solution_hashes'][:t]==c['iteration_solution_hashes'][:t]
        assert a['gnn_calls']+1==c['gnn_calls']==control['refresh_calls']
        assert [e['iteration'] for e in a['refresh_events']]==list(range(0,t,100))
        assert c['iterations_executed']==a['iterations_executed']==t+100
        for key,val in a['branch_state'].items():
            if key!='boundary_elapsed_seconds': assert c['branch_state'][key]==val
        for r in (a,b,c):
            tour=np.array(r['tour']); assert np.array_equal(np.sort(tour),np.arange(len(tour)))
            costs=[r['initial_cost']]+[v['best_so_far'] for v in r['iterations']]
            assert all(y<=x for x,y in zip(costs,costs[1:]))
        for row in [r for r in rows if r['decision_id']==ident]:
            h=int(row['horizon_iterations']); ac=a['iterations'][t+h-1]['best_so_far']; cc=c['iterations'][t+h-1]['best_so_far']
            assert ac==float(row['hold_cost']) and cc==float(row['refresh_cost'])
            assert abs(float(row['refresh_benefit_raw'])-(ac-cc))<1e-12
            assert abs(float(row['refresh_benefit_percent'])-100*(ac-cc)/ac)<1e-12
    for mode in ('default_02','strict'):
        d=load(OUT/mode/'diagnosis.json'); assert d['status']=='completed'
        for label in ('hold_a','hold_b'):
            events=load(OUT/mode/f'{label}_refreshes.json')
            for i,event in enumerate(events):
                snapshot=np.load(OUT/mode/f'{label}_refresh{i}.npz')
                assert event['model_eval'] and event['state_dict_sha256']==event['state_dict_after_sha256']
                for key,digest in event['hashes'].items(): assert hashlib.sha256(snapshot[key].tobytes()).hexdigest()==digest
                assert snapshot['native_prior'].tobytes()==snapshot['prior'].tobytes()
        if mode=='strict':
            assert all(all(v['byte_equal'] for v in c['arrays'].values()) for c in d['refresh_comparisons'])
    instance_rows=[]
    for n in (500,1000):
        for h in (25,50,100):
            for instance in sorted({r['instance_id'] for r in rows if int(r['n'])==n}):
                selected=[r for r in rows if int(r['n'])==n and int(r['horizon_iterations'])==h and r['instance_id']==instance]
                instance_rows.append(dict(n=n,horizon_iterations=h,instance_id=instance,decisions=len(selected),
                    mean_refresh_benefit_raw=statistics.mean(float(r['refresh_benefit_raw']) for r in selected),
                    mean_refresh_benefit_percent=statistics.mean(float(r['refresh_benefit_percent']) for r in selected)))
    write_csv(OUT/'instance_summary.csv',instance_rows)
    summary=[]
    for n in (500,1000):
        for h in (25,50,100):
            selected=[r for r in rows if int(r['n'])==n and int(r['horizon_iterations'])==h]
            instances=[r['mean_refresh_benefit_raw'] for r in instance_rows if r['n']==n and r['horizon_iterations']==h]
            values=[float(r['refresh_benefit_raw']) for r in selected]
            summary.append(dict(n=n,horizon_iterations=h,independent_instances=len(instances),paired_decisions=len(selected),
                mean_benefit_raw=statistics.mean(values),median_benefit_raw=statistics.median(values),
                mean_benefit_percent=statistics.mean(float(r['refresh_benefit_percent']) for r in selected),
                instance_mean_min_raw=min(instances),instance_mean_max_raw=max(instances),
                improved_decisions=sum(v>1e-6 for v in values),worsened_decisions=sum(v< -1e-6 for v in values),
                tied_within_1e_6=sum(abs(v)<=1e-6 for v in values),
                mean_marginal_gnn_seconds=statistics.mean(float(r['marginal_gnn_seconds']) for r in selected),
                mean_instrumented_suffix_wall_difference_seconds=statistics.mean(float(r['suffix_wall_difference_seconds']) for r in selected)))
    write_csv(OUT/'pilot_summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,n in zip(axes,(500,1000)):
        for instance in sorted({r['instance_id'] for r in instance_rows if r['n']==n}):
            selected=[r for r in instance_rows if r['n']==n and r['instance_id']==instance]
            ax.plot([r['horizon_iterations'] for r in selected],[r['mean_refresh_benefit_raw'] for r in selected],marker='o',label=instance.rsplit('-',1)[-1])
        ax.axhline(0,color='black',linewidth=.7); ax.set_title(f'TSP-{n}: two independent instances'); ax.set_xlabel('Continuation iterations'); ax.set_ylabel('HOLD minus REFRESH tour length'); ax.legend(title='Instance')
    fig.tight_layout(); fig.savefig(OUT/'pilot_instance_benefit.png',dpi=150); plt.close(fig)
    result=dict(status='PASS',accepted_decisions=16,raw_arm_runs=48,horizon_observations=48,
        exact_all_ant_prefixes=True,exact_native_prior_prefixes=True,exact_full_hold_controls=True,v1_artifacts_unchanged=True,
        strict_live_inference=True,recorded_guidance_replay_used=False,summary=summary)
    (OUT/'artifact_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
