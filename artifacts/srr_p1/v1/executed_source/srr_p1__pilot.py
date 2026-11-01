"""Registered development feasibility only. Never creates or loads principal data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import sys,time,json,resource,signal
import numpy as np
from r3.common import ROOT,strict_cuda,model,sha256
sys.path.insert(0,str(ROOT/'.srr-build/native'))
import faco,utils,torch
from r3.state_capture import args
from srr_p1.runner import run,digest
OUT=ROOT/'artifacts/srr_p1/v1'
def save(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def main():
    assert json.loads((OUT/'correctness_gate.json').read_text())['status']=='PASS'
    plan=dict(development_instance=0,execution_seed=31415,ants=100,block=100,
        warmup='100 iterations per arm, unmeasured; excluded',
        repeats=3,full_iterations=1000,wall_budget=4.,
        ordering='For repeat r then arm0/1; measured then unmeasured if r even, reversed if odd. Then wall arm0/1.',
        caps=dict(overall_seconds=600,per_search_seconds=60,rss_bytes=3*1024**3,cuda_allocated_bytes=1024**3,threads=2),
        correctness='20-iteration original utils reference and telemetry invariance before performance runs',
        no_policy_tuning=True,principal_data_access=False)
    save('pilot_plan.json',plan);save('pilot_plan_hash.json',dict(sha256=sha256(OUT/'pilot_plan.json')))
    def deadline(*_):raise TimeoutError('600s pilot process cap')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    process_start=time.perf_counter();strict_cuda();torch.cuda.set_per_process_memory_fraction(.24,device=0)
    t=time.perf_counter();net,_,_=model();torch.cuda.synchronize();load_seconds=time.perf_counter()-t
    x=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy')[0]
    observed=[];instances=[]
    class Observe(faco.MFACO_TSP):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);instances.append(self)
        def sample(self,*a,**kw):
            o=super().sample(*a,**kw);observed.append((np.array(o[0],copy=True),digest(kw['prior'])));return o
    utils.set_seed(31415)
    utils.infer_instance('tsp',Observe,utils.build_pyg_data_tsp,net,x,32,100,True,args(2,10),collect_metrics=False,seed=31415)
    a=run(x,net,iterations=20,block=10)
    assert np.array_equal(a['costs'],np.array([r[0] for r in observed]))
    assert a['prior_hashes']==[r[1] for r in observed]
    for key,field in [('source','source_route'),('best','best_route'),('pheromone','pheromone_sparse_np')]:assert np.array_equal(a[key],getattr(instances[-1],field))
    for extra in (False,True):
        plain=run(x,net,extra,False,20,block=10);measured=run(x,net,extra,True,20,block=10)
        repeated=run(x,net,extra,True,20,block=10)
        for key in ('costs','tours','source','best','pheromone','prior_hashes','native_seed'):
            assert np.array_equal(plain[key],measured[key]) and np.array_equal(measured[key],repeated[key]),key
    save('runner_reference_gate.json',dict(status='PASS',original_utils_exact=True,iterations=20,ants=100,
        guidance_refresh=10,full_seed_contract_preserved=True,light_invariance_both_arms=True,repeats_exact=True))
    for arm in (False,True):run(x,net,arm,False,100)
    rows=[]
    def record(tag,result):
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(OUT/(tag+'.npz'),**arrays)
        row={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        row['tag']=tag;row['prior_hashes_sha256']=digest(np.array(row.pop('prior_hashes'),dtype='S64'))
        save(tag+'.json',row);rows.append(row);print(tag,result['iterations'],result['search_seconds'],flush=True)
    for repeat in range(3):
        for arm in (False,True):
            pair={}
            for light in ((True,False) if repeat%2==0 else (False,True)):
                result=run(x,net,arm,light);pair[light]=result
                record(f'iter_r{repeat}_arm{int(arm)}_light{int(light)}',result)
            for key in ('costs','tours','source','best','pheromone','prior_hashes','native_seed'):
                assert np.array_equal(pair[False][key],pair[True][key]),'Full-scale instrumentation invariance'
    for arm in (False,True):
        result=run(x,net,arm,True,iterations=100000,budget=4.)
        assert result['eligible_iterations']>0
        assert result['times'][result['primary_iteration']-1]<=4.
        record(f'wall_arm{int(arm)}',result)
    signal.alarm(0)
    save('pilot_summary.json',dict(status='PASS',plan_sha256=sha256(OUT/'pilot_plan.json'),
        checkpoint_load_seconds=load_seconds,process_seconds=time.perf_counter()-process_start,runs=rows,
        full_scale_invariance=True,full_trajectories=12,wall_trajectories=2,
        resource_pass=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        save('pilot_failure.json',dict(error=traceback.format_exc(),status='BLOCKED',no_retry=True));raise
