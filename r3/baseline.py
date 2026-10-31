"""Bounded sequential CUDA baseline, native ranking, and scanner pilots."""
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import faco
import utils
from r3.common import ROOT,save_json,sha256,strict_cuda,model,git_command
from r3.state_capture import run,args,solver,snapshot
from r3.diagnostics import diagnose,audit_candidates

def save_npz(path,**arrays):
    with Path(path).open('xb') as stream:np.savez_compressed(stream,**arrays)

def main():
    out=ROOT/'artifacts/r3/baseline/verification_v1'
    out.mkdir(parents=True,exist_ok=False)
    strict_cuda();net,_,_=model();results=[]
    r1=json.loads((ROOT/'artifacts/r3/references/r1_artifacts__r1__datasets__dataset_manifest.json').read_text())
    try:
        for entry in r1['datasets']:
            n=entry['n'];path=out/f'pilot_tsp{n}.npy'
            data=subprocess.check_output(git_command('show','41d9a15b5f311d85118592550c7bdeaf71b3fb9e:'+entry['path']),cwd=ROOT)
            with path.open('xb') as f:f.write(data)
            if sha256(path)!=entry['sha256']:raise AssertionError('R1 dataset hash discrepancy')
            xs=np.load(path,allow_pickle=False)
            regenerated=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261031,n]))).random(xs.shape,dtype=np.float32)
            if not np.array_equal(xs,regenerated):raise AssertionError('R1 dataset generation discrepancy')
            x=xs[0]
            tick=time.perf_counter();smoke=run(x,net,31415,H=2,mini_H=10,trace_iterations=(0,19))
            duration=time.perf_counter()-tick
            owned={k:solver(x,n_ants=1,k=k) for k in (32,40,48,64)}
            nn={k:s.nn_list.copy() for k,s in owned.items()};audit=audit_candidates(x,nn)
            if not np.array_equal(owned[32].backup_list,nn[64][:,32:64]):raise AssertionError('backup not ranks33-64')
            # TSP1000 scan only after successful 100/200/500 pilots.
            initial=diagnose(x,smoke['initial_source'],nn[64],20261102+n)
            final=diagnose(x,smoke['aco'].source_route.copy(),nn[64],20261102+n)
            save_json(out/f'n{n}_scanner_initial.json',initial);save_json(out/f'n{n}_scanner_final.json',final)
            state,_=snapshot(smoke['aco'],x,20);save_npz(out/f'n{n}_smoke_final.npz',**state)
            item=dict(n=n,iterations=20,ants=100,k=32,execution_seed=31415,native_seed=smoke['native_seed'],
                dataset_sha256=sha256(path),candidate_audit=audit,best_cost=smoke['aco'].best_cost,
                timing=smoke['timing'],end_to_end_seconds=duration,all_checkpoint_ant_tours_valid=True,
                peak_cuda_bytes=smoke['peak_cuda_bytes'])
            results.append(item);save_json(out/f'n{n}_smoke.json',item);print(f'baseline smoke N={n} PASS wall={duration:.3f}s',flush=True)
        # Independent unchanged utils reference, full 1000 iterations on audited pilot data.
        x=np.load(out/'pilot_tsp1000.npy')[0];observed=[];owners=[]
        class Observe(faco.MFACO_TSP):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);owners.append(self)
            def sample(self,*a,**kw):
                result=super().sample(*a,**kw);observed.append(result[0].copy());return result
        utils.set_seed(31415);torch.cuda.synchronize();tick=time.perf_counter()
        reference=utils.infer_instance('tsp',Observe,utils.build_pyg_data_tsp,net,x,32,100,True,args(),collect_metrics=False,seed=31415)
        torch.cuda.synchronize();reference_seconds=time.perf_counter()-tick
        measured=run(x,net,31415,trace_iterations=(0,249,499,999))
        assert np.array_equal(observed,measured['costs_history'])
        for name in ('source_route','best_route','pheromone_sparse_np','heuristic_sparse_np','nn_list','backup_list'):
            assert np.array_equal(getattr(owners[-1],name),getattr(measured['aco'],name)),name
        repeat=run(x,net,31415,capture=False)
        assert np.array_equal(measured['costs_history'],repeat['costs_history'])
        assert np.array_equal(measured['aco'].source_route,repeat['aco'].source_route)
        for iteration,state in measured['snapshots'].items():save_npz(out/f'full1000_t{iteration}.npz',**state)
        save_npz(out/'full1000_regression_traces.npz',original=np.array(observed),captured=measured['costs_history'],repeat=repeat['costs_history'])
        full=dict(iterations=1000,H=10,mini_H=100,execution_seed=31415,native_seed=measured['native_seed'],
            exact_original_inference=True,exact_same_seed_repeat=True,exact_fast_vs_bounded_trace=True,
            checkpoint_validity=True,best_cost=measured['aco'].best_cost,reference_seconds=reference_seconds,
            timing=measured['timing'],peak_cuda_bytes=measured['peak_cuda_bytes'])
        save_json(out/'full1000_verification.json',full)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        save_json(out/'completion.json',dict(status='PASS',smokes=results,full=full,process_rss_high_water_bytes=rss,
            diagnostic_estimate_seconds=64*max(initial['scan_wall_seconds'],final['scan_wall_seconds']),
            source='original solver and checkpoint; independently audited R1 coordinates only; no R1 scheduler'))
        print('Full 1000-iteration original regression + repeat PASS',flush=True)
    except Exception as e:
        import traceback
        save_json(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_smokes=results))
        raise
if __name__=='__main__':main()
