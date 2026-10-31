"""Read-only v1 audit and observational refresh descriptions."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT/'artifacts/r1/experiments/20261031_matched_v1'

def write_csv(path, rows):
    if rows:
        with Path(path).open('x', newline='') as f:
            w=csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

def audit(out):
    out=Path(out); out.mkdir(parents=True, exist_ok=False)
    manifest=json.loads((V1/'benchmark_config.json').read_text())
    checks={name:hashlib.sha256(subprocess.check_output(['git','show','8ee944b:'+name],cwd=ROOT)).hexdigest()==digest for name,digest in manifest['source_sha256'].items()}
    assert all(checks.values()), checks
    records=[]; episodes=[]; all_events=[]
    for path in sorted((V1/'runs').glob('*.json')):
        r=json.loads(path.read_text()); trace=r['iterations']; events=r['refresh_events']
        assert len(trace)==r['iterations_executed']==1000 and len(events)==r['gnn_calls']
        significant=r['initial_cost']; last=0; last_at=[0]
        for row in trace:
            if row['best_so_far'] < significant-1e-6:
                significant=row['best_so_far']; last=row['iteration']
            last_at.append(last)
        previous=0; counts={}
        for e in events:
            t=e['iteration']; incumbent=r['initial_cost'] if t==0 else trace[t-1]['best_so_far']
            assert e['global_best']==incumbent and e['stagnation']==t-last_at[t]
            episode=last_at[t]; counts[episode]=counts.get(episode,0)+1
            all_events.append(dict(run_id=path.stem,n=r['n'],method=r['method'],iteration=t,
                phase='early' if t<333 else 'middle' if t<667 else 'late',reason=e['reason'],
                prior_age_iterations=t-previous,stagnation_iterations=t-last_at[t],episode_start=episode))
            for h in (25,50,100):
                if t+h>1000: continue
                after=trace[t+h-1]['best_so_far']
                records.append(dict(run_id=path.stem,n=r['n'],method=r['method'],iteration=t,
                    phase='early' if t<333 else 'middle' if t<667 else 'late',reason=e['reason'],
                    prior_age_iterations=t-previous,stagnation_iterations=t-last_at[t],episode_start=episode,
                    horizon_iterations=h,incumbent_at_refresh=incumbent,cost_after_horizon=after,
                    subsequent_improvement_raw=incumbent-after,subsequent_improvement_percent=100*(incumbent-after)/incumbent,
                    intervening_refreshes=sum(t<x['iteration']<t+h for x in events)))
            previous=t
        episodes.extend(dict(run_id=path.stem,n=r['n'],method=r['method'],episode_start=e,calls_in_episode=c) for e,c in counts.items())
    write_csv(out/'v1_refresh_events_all.csv',all_events)
    write_csv(out/'v1_refresh_observations.csv',records); write_csv(out/'v1_episode_calls.csv',episodes)
    summary=[]
    for n in (100,200,500,1000):
        for method in sorted({r['method'] for r in records}):
            selected=[r for r in records if r['n']==n and r['method']==method and r['horizon_iterations']==25]
            ep=[r for r in episodes if r['n']==n and r['method']==method]
            summary.append(dict(n=n,method=method,observed_events_with_25_iteration_followup=len(selected),
                initial_events=sum(r['reason']=='initial' for r in selected),
                stagnation_events=sum(r['reason']=='stagnation' for r in selected),
                max_interval_events=sum(r['reason']=='max_interval' for r in selected),
                fixed_interval_events=sum(r['reason']=='fixed_interval' for r in selected),
                mean_noninitial_interval=statistics.mean(r['prior_age_iterations'] for r in selected if r['iteration']>0),
                median_stagnation=statistics.median(r['stagnation_iterations'] for r in selected),
                late_events=sum(r['phase']=='late' for r in selected),
                episodes_with_multiple_calls=sum(r['calls_in_episode']>1 for r in ep),
                max_calls_in_episode=max(r['calls_in_episode'] for r in ep)))
    write_csv(out/'v1_retrospective_summary.csv',summary)
    (out/'v1_audit.json').write_text(json.dumps(dict(status='PASS',source_hashes=checks,
        original_base=manifest['base_commit'],experiment_HEAD=manifest['current_commit'],
        source_state='uncommitted working sources identified by seven exact byte hashes',
        packaging_commit='8ee944b8f5929bbadea8d69a00beac12f89779e6',
        v1_artifact_hashes={str(p.relative_to(V1)):hashlib.sha256(p.read_bytes()).hexdigest() for p in V1.rglob('*') if p.is_file()},
        caveat='Observational outcomes include other refreshes; no causal benefit inferred.'),indent=2)+'\n')

if __name__=='__main__':
    import sys
    audit(sys.argv[1])
