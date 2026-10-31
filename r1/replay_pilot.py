"""Bounded strict-CUDA live-inference paired diagnostic, never a v1 benchmark."""
import json
from pathlib import Path
import random
import subprocess
import time
import traceback
import torch
from net import Net
from r1.datasets import ensure_dataset,sha256
from r1.engine import run,RunConfig,weight_hash
from r1.replay_determinism import strict_cuda,settings
from r1.replay_diagnose import CHECKPOINT,ROOT
from r1.v2_replay import branch,assert_prefix,observations
from r1.v2_retrospective import write_csv

def save(path,data):
    with Path(path).open('x') as f: json.dump(data,f,indent=2,allow_nan=False)

def main(directory):
    out=Path(directory); protocol=json.loads((out/'pilot_protocol.json').read_text())
    gate=json.loads((out/'validation_gate.json').read_text()); assert gate['status']=='PASS'
    strict_cuda(); target=out/'pilot'; target.mkdir(); (target/'runs').mkdir()
    start=time.perf_counter()
    model=Net(feats=2,edge_feats=6,logit_net=True).cuda().eval()
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    original_weights=weight_hash(model); datasets={}; entries=[]
    for n in protocol['sizes']:
        coords,entry=ensure_dataset(target/'datasets',n,protocol['instances_per_size'],protocol['dataset_seed'])
        datasets[n]=coords; entries.append(entry)
        run(coords[0],model,RunConfig(total_iterations=10,fixed_s=5))
    save(target/'manifest.json',dict(protocol=protocol,protocol_sha256=sha256(out/'pilot_protocol.json'),
        settings=settings(),model_weights_sha256=original_weights,checkpoint_sha256=sha256(CHECKPOINT),datasets=entries,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'r1').glob('*.py')},
        backend='unchanged MFACO_TSP C++ OpenMP',threads=2,device='cuda:0',ants=100,k=32,rho=.1,min_new_edges=12,
        timing='strict CUDA live inference with diagnostic hashes; full prefix replay and suffix diagnostic overhead included; not ordinary runtime',
        gpu_before=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True)))
    rng=random.Random(9237); rows=[]; controls=[]
    jobs=[(n,i,seed,t) for n in protocol['sizes'] for i in range(protocol['instances_per_size']) for seed in protocol['solver_seeds'] for t in protocol['probe_iterations']]
    try:
        for number,(n,i,seed,t) in enumerate(jobs):
            if time.perf_counter()-start>protocol['resource_cap_seconds']: raise RuntimeError('Pilot cap reached')
            ident=f'n{n}_i{i}_seed{seed}_t{t}'; arms={}; order=['hold','hold_control','refresh']; rng.shuffle(order)
            for arm in order:
                calls=[]; hook=model.register_forward_hook(lambda *args:calls.append(1))
                try: result=branch(datasets[n][i],model,seed,t,arm=='refresh')
                finally: hook.remove()
                assert len(calls)==result['gnn_calls'] and weight_hash(model)==original_weights
                save(target/'runs'/f'{ident}_{arm}.json',result); arms[arm]=result
            a,b,c=arms['hold'],arms['hold_control'],arms['refresh']
            assert_prefix(a,b,t+100); assert_prefix(a,c,t)
            assert a['trace']==b['trace'] and a['tour']==b['tour'] and a['final_search_state_hashes']==b['final_search_state_hashes']
            observations_here=observations(c,a,t)
            rows.extend(dict(decision_id=ident,n=n,instance_id=f'tsp{n}-{protocol["dataset_seed"]}-v1-{i:03}',solver_seed=seed,**r) for r in observations_here)
            controls.append(dict(decision_id=ident,prefix_exact=True,hold_full_trajectory_exact=True,hold_final_state_exact=True,
                hold_calls=a['gnn_calls'],refresh_calls=c['gnn_calls'],order=order))
            print(f'{number+1}/{len(jobs)} {ident}: raw benefits={[r["refresh_benefit_raw"] for r in observations_here]}',flush=True)
    except Exception:
        save(target/'failure.json',dict(error=traceback.format_exc(),completed_decisions=len(controls))); raise
    finally:
        write_csv(target/'paired_observations.csv',rows); save(target/'controls.json',controls)
    save(target/'completion.json',dict(decisions=len(controls),elapsed_seconds=time.perf_counter()-start,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),weights_unchanged=weight_hash(model)==original_weights,
        gpu_after=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True)))

if __name__=='__main__':
    import sys
    main(sys.argv[1])
