"""Figures and independent consistency audit for the completed observational stage."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from r1.v2_retrospective import write_csv,V1,ROOT

def main(directory):
    out=Path(directory); data=out/'retrospective_complete_02'
    events=list(csv.DictReader((data/'v1_refresh_events_all.csv').open()))
    episodes=list(csv.DictReader((data/'v1_episode_calls.csv').open()))
    observations=list(csv.DictReader((data/'v1_refresh_observations.csv').open()))
    prior=json.loads((out/'v1_audit.json').read_text())
    assert all(hashlib.sha256((V1/name.replace('\\','/')).read_bytes()).hexdigest()==digest for name,digest in prior['v1_artifact_hashes'].items())
    assert len(events)==1612 and sum(int(r['calls_in_episode']) for r in episodes)==1612
    for row in observations:
        before=float(row['incumbent_at_refresh']); after=float(row['cost_after_horizon'])
        assert abs(float(row['subsequent_improvement_raw'])-(before-after))<1e-12
        assert abs(float(row['subsequent_improvement_percent'])-100*(before-after)/before)<1e-12
        assert int(row['iteration'])+int(row['horizon_iterations'])<=1000
    summary=[]
    for n in (100,200,500,1000):
        for method in sorted({r['method'] for r in events}):
            selected=[r for r in events if int(r['n'])==n and r['method']==method]
            intervals=[int(r['prior_age_iterations']) for r in selected if int(r['iteration'])>0]
            ep=[r for r in episodes if int(r['n'])==n and r['method']==method]
            summary.append(dict(n=n,method=method,events=len(selected),interval_min=min(intervals),
                interval_q25=float(np.quantile(intervals,.25)),interval_median=statistics.median(intervals),
                interval_q75=float(np.quantile(intervals,.75)),interval_max=max(intervals),
                stagnation_median=statistics.median(int(r['stagnation_iterations']) for r in selected),
                early_events=sum(r['phase']=='early' for r in selected),middle_events=sum(r['phase']=='middle' for r in selected),
                late_events=sum(r['phase']=='late' for r in selected),
                calls_in_repeated_episodes=sum(int(r['calls_in_episode']) for r in ep if int(r['calls_in_episode'])>1),
                max_calls_per_episode=max(int(r['calls_in_episode']) for r in ep)))
    write_csv(out/'retrospective_summary_all.csv',summary)
    followup=[]
    for n in (100,200,500,1000):
        for method in sorted({r['method'] for r in observations}):
            for h in (25,50,100):
                selected=[r for r in observations if int(r['n'])==n and r['method']==method and int(r['horizon_iterations'])==h]
                followup.append(dict(n=n,method=method,horizon_iterations=h,observations=len(selected),
                    mean_subsequent_improvement_raw=statistics.mean(float(r['subsequent_improvement_raw']) for r in selected),
                    mean_subsequent_improvement_percent=statistics.mean(float(r['subsequent_improvement_percent']) for r in selected)))
    write_csv(out/'observational_followup_summary.csv',followup)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures'; figures.mkdir()
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,n in zip(axes.flat,(100,200,500,1000)):
        values=[int(r['prior_age_iterations']) for r in events if int(r['n'])==n and r['method']=='event_stagnation' and int(r['iteration'])>0]
        ax.hist(values,bins=np.arange(0,226,25),edgecolor='white'); ax.set_title(f'TSP-{n} event policy'); ax.set_xlabel('Prior age / refresh interval (iterations)'); ax.set_ylabel('Refresh count')
    fig.tight_layout(); fig.savefig(figures/'observed_refresh_intervals.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,n in zip(axes.flat,(100,200,500,1000)):
        selected=[r for r in summary if r['n']==n]; x=np.arange(len(selected)); bottom=np.zeros(len(selected))
        for phase in ('early','middle','late'):
            values=np.array([r[phase+'_events']/r['events'] for r in selected]); ax.bar(x,values,bottom=bottom,label=phase); bottom+=values
        ax.set_xticks(x,[r['method'] for r in selected],rotation=25,ha='right',fontsize=7); ax.set_title(f'TSP-{n}'); ax.set_ylabel('Fraction of refresh calls')
    axes.flat[0].legend(); fig.tight_layout(); fig.savefig(figures/'observed_refresh_phases.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,n in zip(axes.flat,(100,200,500,1000)):
        for method in sorted({r['method'] for r in followup}):
            selected=[r for r in followup if r['n']==n and r['method']==method]
            ax.plot([r['horizon_iterations'] for r in selected],[r['mean_subsequent_improvement_raw'] for r in selected],marker='o',label=method)
        ax.set_title(f'TSP-{n}: observational, not causal'); ax.set_xlabel('Follow-up iterations'); ax.set_ylabel('Mean subsequent improvement (raw length)')
    axes.flat[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(figures/'observational_followup.png',dpi=150); plt.close(fig)
    result=dict(status='PASS_OBSERVATIONAL_ARTIFACTS_ONLY',v1_artifacts_unchanged=True,events=len(events),
        followup_observations=len(observations),episode_records=len(episodes),
        causal_gate='BLOCKED: cached_prior_sha256 mismatch in identical HOLD replays',
        causal_decisions_executed=0,held_out_benchmarks_executed=0,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'r1').glob('*.py')})
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    import sys
    main(sys.argv[1])
