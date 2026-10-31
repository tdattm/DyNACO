"""Execute the frozen Stage A once, preserving failed and partial data."""
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from r3.common import ROOT,BASE,save_json,sha256,strict_cuda,model,git
from r3.protocol import load,PATH
from r3.datasets import generate
from r3.state_capture import run,solver
from r3.baseline import save_npz
from r3.diagnostics import diagnose,audit_candidates,canonical_tour
from r3.replay import reconstruct

def main():
    protocol=load()
    completion=json.loads((ROOT/'artifacts/r3/baseline/verification_v1/completion.json').read_text())
    if completion['status']!='PASS':raise AssertionError('baseline gate incomplete')
    if git('diff',BASE,'--','faco.py','utils.py','net.py','src/mfaco_train.cpp','src/mfaco_train.h','src/binding.cpp','src/kd_tree.h'):
        raise AssertionError('production source changed')
    strict_cuda();net,_,_=model()
    # Warm kernels on pilot data; do not inspect reserved confirmation data.
    pilot=np.load(ROOT/'artifacts/r3/baseline/verification_v1/pilot_tsp1000.npy')[1]
    run(pilot,net,1234,H=1,mini_H=2,capture=False)
    out=ROOT/'artifacts/r3/experiments/stage_a_v1_01';out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+protocol['resources']['principal_wall_cap_seconds']
    records=[];data=None
    try:
        data=generate(out/'datasets')
        source_files=[*ROOT.glob('*.py'),*ROOT.glob('src/*.cpp'),*ROOT.glob('src/*.h'),*ROOT.glob('r3/**/*.py'),*ROOT.glob('r3/native/*.cpp'),*ROOT.glob('r3/*.sh'),*ROOT.glob('tests/*.py')]
        hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted(source_files)}
        save_json(out/'manifest.json',dict(base_sha=BASE,working_head=git('rev-parse','HEAD'),
            branch=git('branch','--show-current'),git_status=git('status','--short'),
            git_diff=git('diff','HEAD'),baseline_tree=git('rev-parse',BASE+'^{tree}'),
            source_sha256=hashes,source_tree_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            native_build_sha256={p.name:sha256(p) for p in (ROOT/'.r3-build/native').glob('*.so')},
            protocol_sha256=sha256(PATH),dataset_sha256=sha256(out/'datasets/tsp1000_seed20261101_float32.npy'),
            checkpoint_sha256=protocol['checkpoint_sha256'],protocol=protocol,
            timing_scope='monotonic perf_counter, CUDA synchronize; strict CUDA; principal wall includes all data generation/capture/diagnostics',
            actual_native_evaluation_trace='unavailable; initial checklist and raw relocation reconstructed exactly; all-node potential eligibility used for offline gate'))
        for instance,x in enumerate(data):
            knn_start=time.perf_counter()
            owned={k:solver(x,n_ants=1,k=k) for k in (32,40,48,64)}
            lists={k:s.nn_list.copy() for k,s in owned.items()};audit=audit_candidates(x,lists)
            if not np.array_equal(owned[32].backup_list,lists[64][:,32:64]):raise AssertionError('K32 backup mismatch')
            save_npz(out/f'native_candidates_i{instance}.npz',**{f'k{k}':a for k,a in lists.items()})
            audit['native_construction_and_reference_audit_seconds']=time.perf_counter()-knn_start
            save_json(out/f'candidate_audit_i{instance}.json',audit)
            del owned
            for seed in protocol['execution_seeds']:
                run_dir=out/f'i{instance}_s{seed}';run_dir.mkdir(exist_ok=False)
                states=[];raws=[];trace_results=[]
                def state_callback(state):
                    iteration=int(state['completed_iterations']);save_npz(run_dir/f'state_t{iteration}.npz',**state)
                    states.append(iteration)
                def trace_callback(record):
                    iteration=record['iteration'];trace=record['trace']
                    raw=canonical_tour(record['raw'],1000)
                    reconstructed,checklist,steps=reconstruct(record['source_before'],trace)
                    if not np.array_equal(reconstructed,raw):raise AssertionError('native raw-tour reconstruction mismatch')
                    if record['touched_size']!=0:raise AssertionError('binding touched-list semantics changed')
                    save_npz(run_dir/f'raw_before_iter{iteration}.npz',raw=raw,refined=record['refined'],source_before=record['source_before'],initial_checklist=np.array(checklist,dtype=np.int32))
                    serial_trace={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in trace.items()}
                    save_json(run_dir/f'perturbation_iter{iteration}.json',dict(trace=serial_trace,steps=steps,
                        raw_exact_match=True,initial_checklist=checklist,actual_dynamic_evaluations=None,actual_native_acceptances=None,
                        reason='native API exposes no evaluation/acceptance sequence; never infer from empty touched_list'))
                    raws.append(iteration);trace_results.append(dict(iteration=iteration,raw_exact_match=True,checklist_size=len(checklist)))
                result=run(x,net,seed,on_snapshot=state_callback,on_trace=trace_callback,
                           trace_iterations=protocol['execution_contract']['bounded_trace_iterations_zero_based'],deadline=deadline)
                if states!=protocol['snapshots_completed_iterations']:raise AssertionError('missing state')
                save_npz(run_dir/'trajectory.npz',ant_costs=result['costs_history'],global_best_cost=result['best_history'])
                row=dict(instance_id=instance,execution_seed=seed,native_seed=result['native_seed'],timing=result['timing'],
                    final_best_cost=result['aco'].best_cost,final_source_cost=result['aco'].source_cost,
                    peak_cuda_bytes=result['peak_cuda_bytes'],config=result['config'],trace_reconstruction=trace_results)
                del result
                for iteration in states:
                    with np.load(run_dir/f'state_t{iteration}.npz') as state:
                        for kind_index,kind in enumerate(('source','best')):
                            allocation=int(np.random.SeedSequence([20261103,instance,seed,iteration,kind_index]).generate_state(1)[0])
                            diagnostic=diagnose(x,state[kind+'_route'],lists[64],allocation)
                            diagnostic.update(instance_id=instance,execution_seed=seed,completed_iteration=iteration,tour_kind=kind)
                            save_json(run_dir/f'diagnostic_{kind}_t{iteration}.json',diagnostic)
                for iteration in raws:
                    with np.load(run_dir/f'raw_before_iter{iteration}.npz') as state:
                        allocation=int(np.random.SeedSequence([20261103,instance,seed,iteration,2]).generate_state(1)[0])
                        diagnostic=diagnose(x,state['raw'],lists[64],allocation,checklist=state['initial_checklist'])
                        diagnostic.update(instance_id=instance,execution_seed=seed,iteration_before_srr=iteration,tour_kind='raw_ant0')
                        save_json(run_dir/f'diagnostic_raw_before_iter{iteration}.json',diagnostic)
                row['process_rss_high_water_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                save_json(run_dir/'run.json',row);records.append(row)
                if row['process_rss_high_water_bytes']>protocol['resources']['max_process_rss_bytes'] or row['peak_cuda_bytes']>protocol['resources']['max_torch_cuda_allocated_bytes']:
                    raise TimeoutError('declared memory cap reached')
                if time.perf_counter()>deadline:raise TimeoutError('declared principal wall cap reached')
                print(f'completed instance={instance} seed={seed} best={row["final_best_cost"]:.8f} principal_elapsed={time.perf_counter()-start:.2f}s',flush=True)
        save_json(out/'collection_complete.json',dict(status='COMPLETE',runs=len(records),states=64,
            diagnostic_frozen_tours=128,raw_tours=64,elapsed_seconds=time.perf_counter()-start,
            process_rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        from r3.analysis import analyze
        analyze(out,principal_start=start)
    except Exception as error:
        save_json(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_runs=len(records),
            elapsed_seconds=time.perf_counter()-start,resource_failure=isinstance(error,TimeoutError)))
        from r3.gate import evaluate
        save_json(out/'gate.json',evaluate([],False,False,not isinstance(error,TimeoutError)))
        raise
if __name__=='__main__':main()
