"""Bounded development regression/pilot worker; saved coordinates/prior only."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,resource,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,sys.argv[1])
import faco_opt
faco_opt.set_num_threads(2)
out=Path(sys.argv[2]);extra=sys.argv[3]=='1';measure=sys.argv[4]=='1'
data=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy')
with np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/i0_s31415/state_t500.npz') as z:
    prior=np.ascontiguousarray(z['current_prior'],dtype=np.float32)
results={};start=time.perf_counter()
for traced,parallel in ((False,True),(True,True),(True,False)):
    tag=f'{traced}_{parallel}';solver=faco_opt.MFACO_TSP(data[0],10,32,32,12,.1,1.,.05,True,False,True,True)
    if hasattr(solver,'configure_srr'):solver.configure_srr(extra,False,measure)
    solver.seed_rng(91)
    for step in range(3):
        result=solver.sample(require_prob=traced,prior=prior,parallel_traced=parallel)
        costs,flats=result[:2];results[f'{tag}_{step}_costs']=costs.copy();results[f'{tag}_{step}_flats']=np.asarray(flats)
        if traced:
            results[f'{tag}_{step}_logps']=result[3].copy()
            results[f'{tag}_{step}_raw']=np.asarray(result[6])
        for c,t in zip(costs,flats):
            assert t[0]==t[-1] and np.array_equal(np.sort(t[:-1]),np.arange(1000))
            x=data[0].astype(np.float64)[t[:-1]];length=np.linalg.norm(x-np.roll(x,-1,axis=0),axis=1).sum()
            assert abs(length-float(c))<2e-5*length
        best=int(np.argmin(costs));solver._update_pheromone_from_flat(flats[best],float(costs[best]))
        for field in ('source_route','best_route','pheromone_sparse_np','nn_list','heuristic_sparse_np'):
            results[f'{tag}_{step}_{field}']=np.array(getattr(solver,field),copy=True)
        results[f'{tag}_{step}_best_cost']=np.array(solver.best_cost)
        results[f'{tag}_{step}_source_cost']=np.array(solver.source_cost)
        if measure:
            calls,ev,accepted,mutations=solver.srr_metrics
            assert calls==(20 if extra else 10) and accepted>=mutations
            results[f'{tag}_{step}_telemetry']=np.array([calls,ev,accepted,mutations])
# Parallel traced uses the same per-ant RNG seeding as normal sampling.
for step in range(3):
    for f in ('costs','flats','source_route','best_route','pheromone_sparse_np'):
        assert np.array_equal(results[f'False_True_{step}_{f}'],results[f'True_True_{step}_{f}'])
results['guidance_input']=prior
# One frozen-checkpoint CUDA inference validates real guidance construction.
import torch,utils,faco
from r3.common import strict_cuda,model
strict_cuda();net,_,_=model()
aco=faco.MFACO_TSP(data[0],10,device='cuda:0',extend_ls=True,extra_srr_pass=extra,lightweight_srr=measure)
norm=(data[0]-data[0].min(axis=0))/(data[0].max(axis=0)-data[0].min(axis=0)).max()
graph=utils.build_pyg_data_tsp(aco,norm,'cuda:0',dynamic=True,ablation_pheromone=False,ablation_incumbent=False)
with torch.no_grad():prediction=net(graph)
torch.cuda.synchronize()
for k,v in graph.to_dict().items():
    if isinstance(v,torch.Tensor):results['graph_'+k]=v.cpu().numpy()
results['neural_prior']=prediction.cpu().numpy()
assert torch.cuda.max_memory_allocated()<512*1024*1024
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<3*1024**3
np.savez(out,**results)
(out.with_suffix('.json')).write_text(json.dumps(dict(status='PASS',seconds=time.perf_counter()-start,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,ants=10,iterations=3,
    paths=['normal parallel','traced parallel','traced serial'],extra=extra,measure=measure,
    binary=faco_opt.__file__,peak_cuda_allocated=torch.cuda.max_memory_allocated()),indent=2))
