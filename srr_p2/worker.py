"""Only launched by execute after explicit approval; no generator exists here."""
import os,sys,time,json,traceback
from pathlib import Path
from srr_p2.core import ROOT,read,save,sha,verify_registration,dataset,schedule,validate,complete,journal_record
from m2_safety.guard import require

def main():
    reg,rh,dm,out=sys.argv[1:];out=Path(out)
    require('worker',reg,None,dm,out)
    launch=read(out/'launch.json');auth=launch['authorization']
    if auth.get('action')!='APPROVE_M2_PRINCIPAL_192' or auth.get('registration_sha256')!=rh or auth.get('dataset_manifest_sha256')!=sha(dm):raise ValueError('worker authorization gate')
    def phase(name):
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase=name,monotonic=time.monotonic()))+'\n');f.flush()
    phase('initialization');r=verify_registration(reg,rh);dataset(dm,rh)
    sys.path.insert(0,str(ROOT/r['binary_dir']))
    sys.path.insert(1,str(Path(r['diagnostic_binary']).parent))
    import numpy as np,torch,faco_opt
    from r3.common import strict_cuda,model
    from srr_p2.runner import run
    env=r['environment']
    if (sys.version,np.__version__,torch.__version__,torch.version.cuda)!=(env['python'],env['numpy'],env['torch'],env['cuda']):raise ValueError('environment drift')
    if sha(faco_opt.__file__)!=r['files'][r['production_binary']]:raise ValueError('loaded binary drift')
    strict_cuda();torch.cuda.set_per_process_memory_fraction(.24,device=0)
    if torch.cuda.get_device_name()!=env['gpu']:raise ValueError('GPU identity drift')
    t=time.monotonic();net,_,_=model();torch.cuda.synchronize();load=time.monotonic()-t
    if torch.cuda.max_memory_allocated()>=1024**3:raise MemoryError('CUDA initialization/model cap')
    save(out/'startup.json',dict(checkpoint_load_seconds=load,cuda_allocated=torch.cuda.max_memory_allocated(),cuda_reserved=torch.cuda.max_memory_reserved(),threads=torch.get_num_threads()))
    # Explicitly observed development-only coordinates, never principal warmup.
    dev=np.load(ROOT/r['development_file'],allow_pickle=False)[0]
    for arm in (0,1):
        t=time.monotonic();v=run(dev,net,bool(arm),False,100,phase=phase)
        save(out/f'warmup_a{arm}.json',dict(seconds=time.monotonic()-t,search_seconds=v['search_seconds'],iterations=v['iterations']))
    rows=[]
    for identity in schedule():
        phase('setup');verify_registration(reg,rh);x=dataset(dm,rh)[identity['instance']]
        d=out/identity['id'];d.mkdir(exist_ok=False)
        save(d/'manifest.json',dict(**identity,registration_sha256=rh,dataset_manifest_sha256=sha(dm),coordinate_sha256=__import__('hashlib').sha256(x.tobytes()).hexdigest()))
        with (d/'completed_iterations.bin').open('xb',buffering=0) as journal:
            def checkpoint(seconds,tour,calls,evaluations):
                journal.write(journal_record(seconds,tour,calls,evaluations))
            result=run(x,net,bool(identity['arm']),True,1000 if identity['mode']=='iterations' else 100000,budget=None if identity['mode']=='iterations' else 4.,seed=identity['seed'],phase=phase,checkpoint=checkpoint)
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(d/'raw.npz',**arrays)
        scalar={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        save(d/'raw.json',scalar)
        row=validate(x,scalar,arrays,identity);rows.append(row)
        save(d/'validated.json',dict(**row,raw_sha256=sha(d/'raw.npz'),json_sha256=sha(d/'raw.json')))
    phase('analysis');save(out/'analysis.json',complete(rows))
    save(out/'study_status.json',dict(status='COMPLETE',runs=len(rows),checkpoint_load_seconds=load))

if __name__=='__main__':
    require('worker',sys.argv[1],None,sys.argv[3],Path(sys.argv[4]))
    try:main()
    except BaseException as e:
        out=Path(sys.argv[-1])
        resource=isinstance(e,(MemoryError,TimeoutError)) or 'out of memory' in str(e).lower() or str(e)=='resource cap'
        save(out/'study_status.json',dict(status='RESOURCE-INCONCLUSIVE' if resource else 'INCONCLUSIVE',error=traceback.format_exc(),no_retry=True))
        raise
