"""Execute a predeclared diagnostic protocol; halt on replay mismatch or cap."""
import json
from pathlib import Path
import random
import subprocess
import time
import traceback
import numpy as np
import torch
from net import Net
from r1.datasets import ensure_dataset,sha256
from r1.engine import run,RunConfig,weight_hash
from r1.v2_replay import branch,assert_prefix,observations
from r1.v2_retrospective import write_csv,ROOT

def save(path,data):
    with Path(path).open('x') as f: json.dump(data,f,indent=2,allow_nan=False)

def main(out):
    out=Path(out); protocol=json.loads((out/'protocol.json').read_text())
    gate=json.loads((out/'technical_gate.json').read_text())
    if gate['status']!='PASS': raise RuntimeError('Paired replay gate has not passed; research execution blocked')
    save(out/'protocol_execution_hash.json',dict(sha256=sha256(out/'protocol.json')))
    (out/'runs').mkdir(); start=time.perf_counter()
    checkpoint=ROOT/'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt'
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Net(feats=2,edge_feats=6,logit_net=True).cuda().eval(); model.load_state_dict(ckpt['model_state_dict'],strict=True)
    initial=weight_hash(model)
    datasets={}; entries=[]
    for n in protocol['sizes']:
        coords,entry=ensure_dataset(out/'datasets',n,protocol['instances_per_size'],protocol['dataset_seed'])
        datasets[n]=coords; entries.append(entry)
        run(coords[0],model,RunConfig(total_iterations=10,fixed_s=5))
    save(out/'manifest.json',dict(protocol=protocol,checkpoint_sha256=sha256(checkpoint),weights_sha256=initial,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'r1').glob('*.py')},datasets=entries,
        backend='unchanged MFACO_TSP C++ OpenMP',threads=2,device='cuda:0',ants=100,k=32,rho=.1,min_new_edges=12,
        annealing='no_anneal',injection=False,
        gpu_start=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True),
        provenance=json.loads((out/'v1_audit.json').read_text())['source_state']))
    jobs=[(n,i,seed,t) for n in protocol['sizes'] for i in range(protocol['instances_per_size']) for seed in protocol['solver_seeds'] for t in protocol['probe_iterations']]
    pilot=[j for j in jobs if j[1]==0 and j[2]==protocol['solver_seeds'][0] and j[3] in (175,725)]
    jobs=pilot+[j for j in jobs if j not in pilot]
    rng=random.Random(9237); rows=[]; controls=[]; elapsed=[]
    try:
        for number,(n,i,seed,t) in enumerate(jobs):
            if time.perf_counter()-start>protocol['resource_cap_seconds']: raise RuntimeError('Resource cap reached')
            ident=f'n{n}_i{i}_seed{seed}_t{t}'; arms={}; order=['refresh','hold','hold_control']; rng.shuffle(order)
            tick=time.perf_counter()
            for arm in order:
                r=branch(datasets[n][i],model,seed,t,arm=='refresh')
                assert weight_hash(model)==initial
                save(out/'runs'/f'{ident}_{arm}.json',r); arms[arm]=r
            assert_prefix(arms['hold'],arms['hold_control'],t)
            assert arms['hold']['trace']==arms['hold_control']['trace'] and arms['hold']['tour']==arms['hold_control']['tour']
            assert arms['hold']['gnn_calls']==arms['hold_control']['gnn_calls']
            new=observations(arms['refresh'],arms['hold'],t)
            rows.extend(dict(decision_id=ident,n=n,instance_id=f'tsp{n}-{protocol["dataset_seed"]}-v1-{i:03}',solver_seed=seed,**r) for r in new)
            controls.append(dict(decision_id=ident,prefix_exact=True,hold_control_exact=True,gnn_calls_hold=arms['hold']['gnn_calls'],gnn_calls_refresh=arms['refresh']['gnn_calls'],order=order))
            elapsed.append(time.perf_counter()-tick)
            print(f'{number+1}/{len(jobs)} {ident} benefits={[round(r["refresh_benefit_raw"],6) for r in new]}',flush=True)
            if number+1==len(pilot):
                estimate=sum(elapsed)/len(elapsed)*len(jobs)*1.5
                save(out/'pilot.json',dict(decisions=len(pilot),observed_seconds=sum(elapsed),projected_seconds_with_50pct_margin=estimate,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
                if estimate>protocol['resource_cap_seconds']: raise RuntimeError('Pilot estimate exceeds resource cap; no scaling')
    except Exception:
        save(out/'failure.json',dict(error=traceback.format_exc(),completed_decisions=len(controls),planned_decisions=len(jobs)))
        raise
    finally:
        write_csv(out/'paired_observations.csv',rows); save(out/'controls.json',controls)
    save(out/'completion.json',dict(decisions=len(controls),planned_decisions=len(jobs),seconds=time.perf_counter()-start,weights_unchanged=weight_hash(model)==initial))

if __name__=='__main__':
    import sys
    main(sys.argv[1])
