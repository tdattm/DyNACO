"""Persisted-data audit, instance-blocked descriptive prediction, coverage gate."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PILOT=OUT/'pilot'
def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_csv(name,rows):
    if rows:
        with (OUT/name).open('x',newline='') as f:
            w=csv.DictWriter(f,list(rows[0])); w.writeheader(); w.writerows(rows)
def corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return None if len(a)<3 or np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
def design(rows,model):
    x=[[1.,float(r['prior_age_iterations'])/50] for r in rows]
    if model>=2:
        x=[v+[float(int(r['phase'])==1),float(int(r['phase'])==2)] for v,r in zip(x,rows)]
    if model>=3: x=[v+[np.log1p(float(r['stagnation_iterations']))] for v,r in zip(x,rows)]
    return np.asarray(x)
def fit_predict(train,test,model):
    x=design(train,model); z=design(test,model)
    mu=x.mean(axis=0); sd=x.std(axis=0); mu[0]=0; sd[0]=1; sd[sd==0]=1
    xx=(x-mu)/sd; zz=(z-mu)/sd
    y=np.array([float(r['refresh_benefit_raw']) for r in train])
    coef,residual,rank,singular=np.linalg.lstsq(xx,y,rcond=None)
    return zz@coef,dict(rank=int(rank),columns=x.shape[1],condition=float(singular[0]/singular[-1]),
        stagnation_coefficient_per_log1p_unit=float(coef[-1]/sd[-1]) if model==3 else None)

def main():
    protocol=load(OUT/'pilot_protocol.json'); manifest=load(PILOT/'manifest.json')
    completion=load(PILOT/'completion.json')
    assert completion['decisions']==72 and completion['elapsed_seconds']<=protocol['resource_cap_seconds']
    assert sha(OUT/'pilot_protocol.json')==manifest['protocol_sha256']
    for name,digest in manifest['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    preservation=load(OUT/'preservation_manifest.json')
    assert all(sha(ROOT/name)==digest for name,digest in preservation['historical_artifact_sha256'].items())
    for entry in manifest['datasets']: assert sha(entry['path'])==entry['sha256']
    rows=list(csv.DictReader((PILOT/'paired_observations.csv').open())); controls=load(PILOT/'controls.json')
    assert len(rows)==216 and len(controls)==72
    assert len({(r['decision_id'],r['horizon_iterations']) for r in rows})==216
    instances=sorted({r['instance_id'] for r in rows}); assert len(instances)==4
    all_gnn=0; all_iterations=0; peak=0; recomputed_effects=[]; candidate_by_instance={}
    for control in controls:
        ident=control['decision_id']; related=[r for r in rows if r['decision_id']==ident]; t=int(related[0]['probe_iteration'])
        a,b,c=[load(PILOT/'runs'/f'{ident}_{arm}.json') for arm in ('hold','hold_control','refresh')]
        instance=related[0]['instance_id']
        candidate_by_instance.setdefault(instance,a['candidate_hash'])
        assert a['candidate_hash']==b['candidate_hash']==c['candidate_hash']==candidate_by_instance[instance]
        assert a['iteration_solution_hashes']==b['iteration_solution_hashes']
        assert a['native_guidance_hashes']==b['native_guidance_hashes']
        assert a['final_search_state_hashes']==b['final_search_state_hashes'] and a['tour']==b['tour']
        assert a['iteration_solution_hashes'][:t]==c['iteration_solution_hashes'][:t]
        assert a['native_guidance_hashes'][:t]==c['native_guidance_hashes'][:t]
        assert a['config']==b['config']==c['config'] and a['instance_seed']==b['instance_seed']==c['instance_seed']
        for key,value in a['branch_state'].items():
            if key!='boundary_elapsed_seconds': assert b['branch_state'][key]==c['branch_state'][key]==value
        assert a['gnn_calls']==b['gnn_calls']==t//100+1 and c['gnn_calls']==a['gnn_calls']+1
        pair=np.load(PILOT/'runs'/f'{ident}_prior_pair.npz')
        assert hashlib.sha256(pair['old'].tobytes()).hexdigest()==a['branch_state']['cached_prior_sha256']
        assert hashlib.sha256(pair['new'].tobytes()).hexdigest()==c['native_guidance_hashes'][t]
        delta=pair['new'].astype(float)-pair['old'].astype(float)
        expected_rms=float(np.sqrt(np.mean(delta**2)))
        recomputed_effects.append(dict(decision_id=ident,probe_iteration=t,
            native_float32_benefit_raw=a['final_cost']-c['final_cost'],
            independently_recomputed_float64_benefit_raw=a['recomputed_length']-c['recomputed_length'],
            float32_minus_float64_effect_raw=(a['final_cost']-c['final_cost'])-(a['recomputed_length']-c['recomputed_length'])))
        for r in (a,b,c):
            all_gnn+=r['gnn_calls']; all_iterations+=r['iterations_executed']; peak=max(peak,r['peak_gpu_bytes'])
            assert r['iterations_executed']==r['pheromone_updates']==t+100
            costs=[r['initial_cost']]+[v['best_so_far'] for v in r['iterations']]
            assert all(y<=x for x,y in zip(costs,costs[1:]))
            assert sorted(r['tour'])==list(range(1000))
            for key,value in r.items():
                if key.endswith('_seconds'): assert np.isfinite(value) and value>=0
        previous=a['iterations'][t-51]['best_so_far']; current=a['branch_state']['incumbent']
        for row in related:
            h=int(row['horizon_iterations']); hold=a['iterations'][t+h-1]['best_so_far']; refresh=c['iterations'][t+h-1]['best_so_far']
            assert float(row['hold_cost'])==hold and float(row['refresh_cost'])==refresh
            assert abs(float(row['refresh_benefit_raw'])-(hold-refresh))<1e-12
            assert abs(float(row['refresh_benefit_percent'])-100*(hold-refresh)/hold)<1e-12
            assert abs(float(row['recent_improvement_50_raw_per_iteration'])-(previous-current)/50)<1e-12
            assert abs(float(row['prior_change_rms'])-expected_rms)<1e-12
            assert float(row['additional_gnn_calls'])==1
    # New raw coordinates must differ from every previous same-scale diagnostic instance.
    new=np.load(manifest['datasets'][0]['path'])
    for directory in ('20261031_replay_resolution_01','20261031_v2_01'):
        for path in (OUT.parent/directory).rglob('*.npy'):
            old=np.load(path,allow_pickle=False)
            if old.ndim==3 and old.shape[1:]==(1000,2):
                assert all(not np.array_equal(x,y) for x in new for y in old)
    write_csv('final_length_precision_check.csv',recomputed_effects)
    assert peak<=protocol['allocated_vram_cap_bytes']
    primary=[r for r in rows if int(r['horizon_iterations'])==100]
    strata=[]
    for t in protocol['probe_iterations']:
        selected=[r for r in primary if int(r['probe_iteration'])==t]
        st=[int(r['stagnation_iterations']) for r in selected]
        varying=sum(len({int(r['stagnation_iterations']) for r in selected if r['instance_id']==i})>1 for i in instances)
        non_ties=sum(r['outcome'] in ('harm','improvement') for r in selected)
        strata.append(dict(probe_iteration=t,phase=selected[0]['phase_label'],prior_age_iterations=t%100,
            independent_instances=len({r['instance_id'] for r in selected}),decisions=len(selected),
            stagnation_min=min(st),stagnation_median=statistics.median(st),stagnation_max=max(st),
            distinct_stagnation_values=len(set(st)),instances_with_within_seed_variation=varying,
            range_and_variation_pass=max(st)-min(st)>=25 and len(set(st))>=3 and varying>=3,
            non_tied_decisions=non_ties,improvements=sum(r['outcome']=='improvement' for r in selected),
            harms=sum(r['outcome']=='harm' for r in selected),mean_benefit_raw=statistics.mean(float(r['refresh_benefit_raw']) for r in selected)))
    write_csv('coverage_by_age_phase.csv',strata)
    summaries=[]; by_instance=[]
    for h in protocol['horizons']:
        selected=[r for r in rows if int(r['horizon_iterations'])==h]
        vals=[float(r['refresh_benefit_raw']) for r in selected]
        summaries.append(dict(horizon_iterations=h,independent_instances=4,paired_decisions=72,
            mean_benefit_raw=statistics.mean(vals),median_benefit_raw=statistics.median(vals),
            mean_benefit_percent=statistics.mean(float(r['refresh_benefit_percent']) for r in selected),
            improvements=sum(r['outcome']=='improvement' for r in selected),harms=sum(r['outcome']=='harm' for r in selected),
            exact_ties=sum(r['outcome']=='exact_tie' for r in selected),near_ties=sum(r['outcome']=='near_tie' for r in selected),
            mean_extra_gnn_seconds=statistics.mean(float(r['marginal_gnn_seconds']) for r in selected)))
        for instance in instances:
            v=[float(r['refresh_benefit_raw']) for r in selected if r['instance_id']==instance]
            by_instance.append(dict(instance_id=instance,horizon_iterations=h,decisions=len(v),mean_benefit_raw=statistics.mean(v),
                mean_benefit_percent=statistics.mean(float(r['refresh_benefit_percent']) for r in selected if r['instance_id']==instance)))
    write_csv('outcome_summary.csv',summaries); write_csv('independent_instance_summary.csv',by_instance)
    folds=[]; predictions=[]
    for h in protocol['horizons']:
        selected=[r for r in rows if int(r['horizon_iterations'])==h]
        for held_out in instances:
            train=[r for r in selected if r['instance_id']!=held_out]; test=[r for r in selected if r['instance_id']==held_out]
            for model in (1,2,3):
                predicted,details=fit_predict(train,test,model)
                errors=np.array([float(r['refresh_benefit_raw']) for r in test])-predicted
                folds.append(dict(horizon_iterations=h,held_out_instance=held_out,model=model,mse_raw_length_squared=float(np.mean(errors**2)),
                    train_independent_instances=3,test_independent_instances=1,**details))
                predictions.extend(dict(horizon_iterations=h,held_out_instance=held_out,model=model,decision_id=r['decision_id'],observed_benefit_raw=float(r['refresh_benefit_raw']),predicted_benefit_raw=float(p),squared_error_raw=float(e*e)) for r,p,e in zip(test,predicted,errors))
    write_csv('instance_blocked_prediction_folds.csv',folds); write_csv('instance_blocked_predictions.csv',predictions)
    model_summary=[]
    for h in protocol['horizons']:
        for model in (1,2,3):
            selected=[r for r in folds if r['horizon_iterations']==h and r['model']==model]
            model_summary.append(dict(horizon_iterations=h,model=model,independent_test_instances=4,mean_instance_mse_raw_length_squared=statistics.mean(r['mse_raw_length_squared'] for r in selected),minimum_rank=min(r['rank'] for r in selected),maximum_condition=max(r['condition'] for r in selected)))
    write_csv('prediction_model_summary.csv',model_summary)
    associations=[]
    for h in protocol['horizons']:
        for t in protocol['probe_iterations']:
            selected=[r for r in rows if int(r['horizon_iterations'])==h and int(r['probe_iteration'])==t]
            benefit=[float(r['refresh_benefit_raw']) for r in selected]
            associations.append(dict(horizon_iterations=h,probe_iteration=t,independent_instances=4,decisions=len(selected),
                stagnation_benefit_correlation=corr([np.log1p(float(r['stagnation_iterations'])) for r in selected],benefit),
                recent_rate_benefit_correlation=corr([float(r['recent_improvement_50_raw_per_iteration']) for r in selected],benefit),
                prior_change_rms_benefit_correlation=corr([float(r['prior_change_rms']) for r in selected],benefit)))
    write_csv('within_stratum_descriptive_associations.csv',associations)
    primary_folds=[r for r in folds if r['horizon_iterations']==100]
    mse2=statistics.mean(r['mse_raw_length_squared'] for r in primary_folds if r['model']==2)
    mse3=statistics.mean(r['mse_raw_length_squared'] for r in primary_folds if r['model']==3)
    better=sum(next(r['mse_raw_length_squared'] for r in primary_folds if r['model']==3 and r['held_out_instance']==i)<next(r['mse_raw_length_squared'] for r in primary_folds if r['model']==2 and r['held_out_instance']==i) for i in instances)
    coefficients=[r['stagnation_coefficient_per_log1p_unit'] for r in primary_folds if r['model']==3]
    consistent_sign=max(sum(c>0 for c in coefficients),sum(c<0 for c in coefficients))>=3
    checks=dict(technical=True,four_independent_instances=len(instances)>=4,
        adequate_within_stratum_stagnation_variation=sum(r['range_and_variation_pass'] for r in strata)>=4,
        meaningful_non_tied_coverage=sum(r['non_tied_decisions'] for r in strata)>=18 and sum(r['non_tied_decisions']>=2 for r in strata)>=4,
        predictive_increment=mse3<=.9*mse2 and better>=3 and consistent_sign)
    gate=dict(status='ELIGIBLE_FOR_RESOURCE_CHECK' if all(checks.values()) else 'INCONCLUSIVE',checks=checks,
        passed_variation_strata=sum(r['range_and_variation_pass'] for r in strata),non_tied_primary=sum(r['non_tied_decisions'] for r in strata),
        non_tied_strata=sum(r['non_tied_decisions']>=2 for r in strata),model2_mse=mse2,model3_mse=mse3,improved_instance_folds=better,
        stagnation_coefficients=coefficients,interpretation='Frozen exploratory screening, not significance. No expansion if any condition fails.')
    (OUT/'scientific_coverage_gate.json').write_text(json.dumps(gate,indent=2,allow_nan=False))
    audit=dict(status='PASS',decisions=72,arm_runs=216,horizon_observations=216,total_native_iterations=all_iterations,
        actual_principal_GNN_calls=all_gnn,max_per_run_allocated_gpu_bytes=peak,historical_artifacts_unchanged=True,
        exact_prefixes=True,all_full_hold_negative_controls=True,source_and_dataset_hashes_match=True,protocol_sha256=sha(OUT/'pilot_protocol.json'))
    (OUT/'artifact_audit.json').write_text(json.dumps(audit,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
    for ax,t in zip(axes.flat,(125,425,725,175,475,775)):
        for number,instance in enumerate(instances):
            selected=[r for r in primary if r['instance_id']==instance and int(r['probe_iteration'])==t]
            ax.scatter([int(r['stagnation_iterations']) for r in selected],[float(r['refresh_benefit_raw']) for r in selected],label=instance.rsplit('-',1)[-1],s=28)
        ax.axhline(0,color='black',linewidth=.7); ax.set_title(f't={t}, prior age={t%100}'); ax.set_xlabel('Stagnation (iterations)'); ax.set_ylabel('HOLD minus REFRESH length')
    axes.flat[0].legend(title='Independent instance',fontsize=7); fig.suptitle('TSP-1K: 100-iteration horizon, descriptive within fixed phase/age cells'); fig.tight_layout(); fig.savefig(OUT/'stratified_stagnation_benefit.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for instance in instances:
        selected=[r for r in by_instance if r['instance_id']==instance]
        ax.plot([r['horizon_iterations'] for r in selected],[r['mean_benefit_raw'] for r in selected],marker='o',label=instance.rsplit('-',1)[-1])
    ax.axhline(0,color='black',linewidth=.7); ax.set_xlabel('Continuation horizon'); ax.set_ylabel('Mean HOLD minus REFRESH length'); ax.legend(title='Independent instance'); fig.tight_layout(); fig.savefig(OUT/'instance_benefit.png',dpi=150); plt.close(fig)
    print(json.dumps(dict(audit=audit,gate=gate,outcomes=summaries,models=model_summary),indent=2))

if __name__=='__main__': main()
