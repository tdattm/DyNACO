"""Original fixed H/S inference loop, checked against unchanged utils.infer_instance."""
import time
from types import SimpleNamespace
import numpy as np
import torch
import faco
import utils
from r3.diagnostics import canonical_tour,length

def args(H=10,mini_H=100):
    return SimpleNamespace(H=H,mini_H=mini_H,disable_heuristic=False,no_local_search=False,
        rho=.1,device='cuda:0',no_smooth_mmas=False,min_new_edges=12,no_extend_ls=False,
        no_normalized_heuristic=False,L=0,no_anneal=False,gamma=1.,min_gamma=0.,
        iter_log=True,iter_print=False,verify=False,timed=True)

def solver(coords,n_ants=100,k=32):
    return faco.MFACO_TSP(coords=coords,n_ants=n_ants,cand_list_size=k,backup_list_size=k,
        min_new_edges=12,decay=.1,alpha=1.,p_best=.05,use_local_search=True,extend_ls=True,
        smooth_mmas=True,enable_torch_sync=True,device='cuda:0',disable_heuristic=False,
        normalized_heuristic=True,fixed_steps=0,nls=False,T_nls=10)

def snapshot(aco,coords,iteration,prior=None):
    start=time.perf_counter()
    result={k:np.array(getattr(aco,k),copy=True) for k in ('source_route','best_route','nn_list','backup_list','pheromone_sparse_np','heuristic_sparse_np')}
    result['coordinates']=np.array(coords,copy=True)
    if prior is not None:result['current_prior']=prior.detach().cpu().numpy().copy()
    for kind in ('source','best'):
        t=canonical_tour(result[kind+'_route'],len(coords));v=length(coords,t)
        if abs(v-getattr(aco,kind+'_cost'))>2e-5*max(1,v):raise AssertionError('native cost mismatch')
        result[kind+'_length_f64']=np.array(v)
        result[kind+'_cost_native']=np.array(getattr(aco,kind+'_cost'))
    result['tau_min']=np.array(aco.tau_min);result['tau_max']=np.array(aco.tau_max)
    result['completed_iterations']=np.array(iteration)
    return result,time.perf_counter()-start

def run(coords,net,execution_seed,H=10,mini_H=100,capture=True,trace_iterations=(),on_snapshot=None,on_trace=None,deadline=None):
    utils.set_seed(execution_seed);config=args(H,mini_H)
    # Preserve upstream duplicate construction and discarded explicit seed.
    discarded=solver(coords);discarded.seed_rng(execution_seed);del discarded
    cmin=coords.min(axis=0);scale=(coords.max(axis=0)-cmin).max()
    if scale<1e-6:scale=1.
    norm=(coords-cmin)/scale
    aco=solver(coords);native_seed=int(np.random.randint(0,2**63-1));aco.seed_rng(native_seed)
    source_initial=aco.source_route.copy();costs_history=[];best_history=[];snapshots={};traces=[]
    native_seconds=0.;neural_seconds=0.;extract_seconds=0.;callback_seconds=0.
    def take(iteration,prior):
        nonlocal extract_seconds,callback_seconds
        state,seconds=snapshot(aco,coords,iteration,prior);extract_seconds+=seconds
        if on_snapshot:
            start=time.perf_counter();on_snapshot(state);callback_seconds+=time.perf_counter()-start
        else:snapshots[iteration]=state
    if capture:take(0,None)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.no_grad():
        for h in range(H):
            if deadline is not None and time.perf_counter()>deadline:raise TimeoutError('declared resource cap reached')
            torch.cuda.synchronize();tick=time.perf_counter()
            graph=utils.build_pyg_data_tsp(aco,norm,'cuda:0',dynamic=True,ablation_pheromone=False,ablation_incumbent=False)
            prior=net(graph).view(aco.n,aco.k)
            if not torch.isfinite(prior).all():raise AssertionError('nonfinite guidance')
            torch.cuda.synchronize();neural_seconds+=time.perf_counter()-tick
            for mini in range(mini_H):
                iteration=h*mini_H+mini
                ratio=mini/(mini_H-1) if mini_H>1 else 0.
                current=prior*(1.-ratio)
                prior_np=current.cpu().numpy()
                traced=iteration in trace_iterations
                source_before=aco.source_route.copy() if traced else None
                tick=time.perf_counter()
                output=aco.sample(require_prob=traced,prior=prior_np,parallel_traced=True)
                costs,flats=output[:2];idx=int(costs.argmin().item());best=float(costs[idx])
                aco._update_pheromone_from_flat(flats[idx],best)
                torch.cuda.synchronize();native_seconds+=time.perf_counter()-tick
                costs_history.append(np.asarray(costs).copy());best_history.append(aco.best_cost)
                # All returned ants checked on every iteration (permutation and cost at capture steps).
                if capture and (iteration+1 in (250,500,1000) or iteration==H*mini_H-1):
                    for cost,t in zip(costs,flats):
                        v=length(coords,t)
                        if abs(v-float(cost))>2e-5*max(1,v):raise AssertionError('sample cost mismatch')
                if traced:
                    tick=time.perf_counter();batch=output[4]
                    record=dict(iteration=iteration,source_before=source_before,
                        raw=np.asarray(output[6][0]).copy(),refined=np.asarray(output[1][0]).copy(),
                        trace=batch.to_trace_list()[0],touched_size=len(output[2][0]))
                    # Own copies of every view; trace object dies after this iteration.
                    record['trace']={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in record['trace'].items()}
                    if on_trace:on_trace(record)
                    else:traces.append(record)
                    extract_seconds+=time.perf_counter()-tick
                if capture and iteration+1 in (250,500,1000):take(iteration+1,current)
    torch.cuda.synchronize();wall=time.perf_counter()-start
    return dict(aco=aco,snapshots=snapshots,traces=traces,costs_history=np.asarray(costs_history),
        best_history=np.asarray(best_history),native_seed=native_seed,initial_source=source_initial,
        timing=dict(search_wall_seconds=wall,neural_graph_forward_seconds=neural_seconds,
                    sample_update_sync_seconds=native_seconds,state_extraction_seconds=extract_seconds,
                    snapshot_callback_seconds=callback_seconds),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        config=vars(config))
