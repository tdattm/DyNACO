"""Development-only runner; accepts saved coordinates, never generates data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import time,resource,hashlib
import numpy as np
import torch,utils
from r3.state_capture import solver

def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def run(x,net,extra=False,light=False,iterations=1000,budget=None,seed=31415,block=100):
    setup=time.perf_counter();utils.set_seed(seed)
    discarded=solver(x);discarded.seed_rng(seed);del discarded
    aco=solver(x);aco._cpp.configure_srr(extra,False,light)
    native_seed=int(np.random.randint(0,2**63-1));aco.seed_rng(native_seed)
    norm=(x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)).max()
    torch.cuda.synchronize();setup=time.perf_counter()-setup
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    times=[];tours=[];costs=[];priors=[];evals=calls=0;prior=None
    initial=np.array(aco.best_route,copy=True)
    with torch.no_grad():
        for it in range(iterations):
            if budget is not None and time.perf_counter()-start>=budget:break
            if it%block==0:
                graph=utils.build_pyg_data_tsp(aco,norm,'cuda:0',dynamic=True,ablation_pheromone=False,ablation_incumbent=False)
                prior=net(graph).view(aco.n,aco.k)
            current=(prior*(1-(it%block)/(block-1))).cpu().numpy()
            output=aco.sample(require_prob=False,prior=current,parallel_traced=True)
            c,flats=output[:2];idx=int(c.argmin());aco._update_pheromone_from_flat(flats[idx],float(c[idx]))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            times.append(elapsed);tours.append(np.array(aco.best_route,copy=True));costs.append(np.array(c,copy=True));priors.append(digest(current))
            if light:
                stats=aco._cpp.srr_metrics;calls+=stats[0];evals+=stats[1]
                assert stats[0]==100*(2 if extra else 1) and stats[2:]==[0,0]
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            assert rss<3*1024**3 and torch.cuda.max_memory_allocated()<1024**3,'resource cap'
            if elapsed>60:raise TimeoutError('per-run60s search cap')
            if budget is not None and elapsed>budget:break
    torch.cuda.synchronize();wall=time.perf_counter()-start
    eligible=[i for i,t in enumerate(times) if budget is None or t<=budget]
    primary=tours[eligible[-1]] if eligible else initial
    xd=x.astype(np.float64);length=lambda t:float(np.linalg.norm(xd[t]-xd[np.roll(t,-1)],axis=1).sum())
    for t in tours:assert np.array_equal(np.sort(t),np.arange(1000))
    assert abs(length(tours[-1])-aco.best_cost)<2e-5*length(tours[-1])
    return dict(extra=extra,light=light,seed=seed,native_seed=native_seed,iterations=len(times),
        eligible_iterations=len(eligible),budget=budget,setup_seconds=setup,search_seconds=wall,
        last_update_seconds=times[-1],overshoot_seconds=max(0,times[-1]-budget) if budget else 0,
        primary_iteration=(eligible[-1]+1) if eligible else 0,primary_length_f64=length(primary),
        calls=calls if light else None,evaluations=evals if light else None,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cuda_allocated=torch.cuda.max_memory_allocated(),cuda_reserved=torch.cuda.max_memory_reserved(),
        times=np.array(times),tours=np.array(tours),costs=np.array(costs),prior_hashes=priors,
        source=np.array(aco.source_route,copy=True),best=np.array(aco.best_route,copy=True),
        pheromone=np.array(aco.pheromone_sparse_np,copy=True))
