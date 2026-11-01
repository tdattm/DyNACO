"""Development archives and failure injection only; no principal coordinates."""
import sys,tempfile,subprocess,time,json,os
from pathlib import Path
import numpy as np
from srr_p2.core import *

OUT=ROOT/'artifacts/srr_p2/v1/final_validation'
def main():
    OUT.mkdir(exist_ok=False)
    checks=[]
    def check(name,fn):fn();checks.append(name);print('PASS',name,flush=True)
    def reject(fn):
        try:fn()
        except (ValueError,AssertionError,FileNotFoundError,KeyError):return
        raise AssertionError('invalid input accepted')
    check('dispatch_192_order_unique',lambda:assertion(len(schedule())==192 and len({r['id'] for r in schedule()})==192 and schedule()[0]['id']=='iterations_i00_s31415_a0' and schedule()[2]['arm']==1 and schedule()[96]['mode']=='wall'))
    check('native_seed_contract',lambda:assertion(int(np.random.RandomState(31415).randint(0,2**63-1))==2689878502059415719))
    check('deadline_last_eligible',lambda:assertion(eligible([3.8,4.,4.01],4.)==1))
    check('no_eligible_rejected',lambda:reject(lambda:eligible([4.01],4.)))
    check('nonfinite_timestamp_rejected',lambda:reject(lambda:eligible([float('nan')],4.)))
    x=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy',allow_pickle=False)[0]
    for arm in (0,1):
        base=ROOT/f'artifacts/srr_p1/v1/wall_arm{arm}'
        a=dict(np.load(str(base)+'.npz',allow_pickle=False));row=read(str(base)+'.json')
        identity=dict(id=f'wall_i00_s31415_a{arm}',mode='wall',instance=0,seed=31415,arm=arm)
        check(f'P1_wall_actual_global_best_f64_arm{arm}',lambda:validate(x,row,a,identity))
        check(f'malformed_record_arm{arm}',lambda:reject(lambda:validate(x,dict(row,primary_iteration=999999),a,identity)))
    check('missing_grid_rejected',lambda:reject(lambda:complete([])))
    fixture=[dict(**i,length_f64=100.-i['arm']) for i in schedule()]
    check('complete_numerical_fixture',lambda:assertion(complete(fixture)['decision']=='GO'))
    check('duplicate_grid_rejected',lambda:reject(lambda:complete(fixture+[fixture[0]])))
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        # Repeated observed development coordinates are metadata fixtures ONLY.
        f=d/'fixture.npy';np.save(f,np.broadcast_to(x,(16,1000,2)).copy())
        m=dict(role='M2_PRINCIPAL_SEALED',registration_sha256='fixture',file=f.name,sha256=sha(f))
        save(d/'seal.json',m)
        check('sealed_fixture_hash_accept',lambda:assertion(dataset(d/'seal.json','fixture').shape==(16,1000,2)))
        check('wrong_registration_rejected',lambda:reject(lambda:dataset(d/'seal.json','bad')))
        save(d/'badseal.json',dict(m,sha256='bad'))
        check('wrong_content_hash_rejected',lambda:reject(lambda:dataset(d/'badseal.json','fixture')))
        check('missing_dataset_rejected',lambda:reject(lambda:dataset(d/'missing','fixture')))
        for tag,array in [('count',x[None]),('dtype',np.broadcast_to(x,(16,1000,2)).astype(np.float64)),('nonfinite',np.full((16,1000,2),np.nan,dtype=np.float32))]:
            np.save(f,array);(d/'seal.json').unlink();save(d/'seal.json',dict(m,sha256=sha(f)))
            check('bad_dataset_'+tag,lambda:reject(lambda:dataset(d/'seal.json','fixture')))
        for name,cap,aggregate,rss in [('blocked_initialization',.05,10,3*1024**3),('aggregate',10,.05,3*1024**3),('rss',10,10,1)]:
            ev=d/(name+'.jsonl');partial=d/(name+'.partial');partial.write_text('preserved')
            proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True)
            result=monitor(proc,ev,time.monotonic()+aggregate,setup_cap=cap,rss_cap=rss)
            check('watchdog_'+name,lambda:assertion(result['status']=='RESOURCE-INCONCLUSIVE' and proc.poll() is not None and partial.read_text()=='preserved'))
        ev=d/'search.jsonl';save_event=dict(phase='search',monotonic=time.monotonic());ev.write_text(json.dumps(save_event)+'\n')
        proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True)
        r=monitor(proc,ev,time.monotonic()+10,search_cap=.05)
        check('blocked_native_search_watchdog',lambda:assertion(r['status']=='RESOURCE-INCONCLUSIVE'))
        proc=subprocess.run([sys.executable,'-m','srr_p2.execute'],capture_output=True)
        check('no_default_execution',lambda:assertion(proc.returncode!=0))
        claim=d/'claim';save(claim,dict(attempt=1))
        try:save(claim,dict(attempt=2))
        except FileExistsError:checks.append('persistent_duplicate_claim_rejected')
        else:raise AssertionError('duplicate claim accepted')
        import fcntl
        with (d/'gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            proc=subprocess.run([sys.executable,'-c',"import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",str(d/'gpu.lock')],capture_output=True)
            check('concurrent_gpu_lock_rejected',lambda:assertion(proc.returncode!=0))
    # Compare phase-hook runner to both saved P1 full trajectories, once each.
    import torch
    from r3.common import strict_cuda,model
    from srr_p2.runner import run
    strict_cuda();torch.cuda.set_per_process_memory_fraction(.24,device=0);net,_,_=model()
    for arm in (0,1):
        phases=[];t=time.monotonic()
        with (OUT/f'development_arm{arm}.bin').open('xb',buffering=0) as journal:
            v=run(x,net,bool(arm),True,phase=phases.append,checkpoint=lambda s,t,c,e:journal.write(journal_record(s,t,c,e)))
        reference=dict(np.load(ROOT/f'artifacts/srr_p1/v1/iter_r0_arm{arm}_light1.npz',allow_pickle=False))
        for key in ('tours','costs','source','best','pheromone'):assert np.array_equal(v[key],reference[key]),key
        journal=np.fromfile(OUT/f'development_arm{arm}.bin',dtype=JOURNAL_DTYPE)
        assert len(journal)==1000 and np.array_equal(journal['tour'],v['tours']) and np.array_equal(journal['seconds'],v['times'])
        assert journal['calls'][-1]==v['calls'] and journal['evaluations'][-1]==v['evaluations']
        scalar={k:z for k,z in v.items() if not isinstance(z,np.ndarray)}
        scalar['phases']=phases;scalar['total_seconds']=time.monotonic()-t
        save(OUT/f'development_arm{arm}.json',scalar)
        np.savez_compressed(OUT/f'development_arm{arm}.npz',**{k:z for k,z in v.items() if isinstance(z,np.ndarray)})
        check(f'P1_exact_1000_iteration_arm{arm}',lambda:assertion(phases==['setup','search','serialization']))
    # Test phase-only derivation has no semantic drift from P1.
    p1=(ROOT/'srr_p1/runner.py').read_text()
    p2=(ROOT/'srr_p2/runner.py').read_text().replace('P1 runner with phase-only watchdog hooks; no algorithm changes.','Development-only runner; accepts saved coordinates, never generates data.').replace(',phase=lambda p: None,checkpoint=lambda *args: None','').replace("phase('setup');",'').replace("phase('search');",'').replace("    phase('serialization')\n",'').replace('            checkpoint(elapsed,tours[-1],calls,evals)\n','')
    p2=p2.replace('initial=initial,times=','times=')
    check('runner_only_phase_hooks',lambda:assertion(p1==p2))
    save(OUT/'tests.json',dict(status='PASS',checks=checks,principal_data_access=False,principal_execution=False,development_full_runs=2))
def assertion(ok):
    if not ok:raise AssertionError('check failed')
if __name__=='__main__':main()
