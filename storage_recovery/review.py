"""Development archives and synthetic disk conditions only. No principal entry point."""
import math,json,os,sys,time,tempfile,subprocess,hashlib,errno,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from srr_p2.core import ROOT,read,save,sha,verify_registration,JOURNAL_DTYPE
from storage_recovery.supervisor import POLICY,admit,watch
OUT=ROOT/'artifacts/m2_storage_recovery/v1'
RH='5d2d8942f900d4d1a4ef89e069668c3d0290d17c628d058150ffe6998d5bca45'
PH='574cd6781e7a026951521e12bfdef28f5790752289c029f21b8db0f4e5e5dedd'
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    from r3.common import git
    historical=git('ls-files').splitlines()
    before={p:sha(ROOT/p) for p in historical}
    r=verify_registration(ROOT/'artifacts/srr_p2/v1/registration.json',RH)
    assert sha(ROOT/'artifacts/srr_p2/v1/FINAL_PROTOCOL.md')==PH
    old=read(ROOT/'artifacts/m2_principal/v1/decision.json')
    assert old['valid_runs']==0 and not old['dataset_generated'] and not old['launch_claim_created']
    olddir=ROOT/'artifacts/m2_principal/v1'
    assert not (olddir/'execution').exists() and not list(olddir.glob('*.npy'))
    claims=[ROOT/'.srr-build'/('principal_'+RH+'.claim'),Path('/mnt/d/GitRepository/DyNACO-srr-p2/.srr-build')/('principal_'+RH+'.claim')]
    assert not any(p.exists() for p in claims)
    per=JOURNAL_DTYPE.itemsize+4000+400+160
    assert per==8584 and 2*96*(1000+100000)*per==166460928000
    dev={};arrays={}
    for arm in (0,1):
        p=ROOT/f'artifacts/srr_p2/v1/final_validation/development_arm{arm}'
        dev[str(arm)]={ext:Path(str(p)+'.'+ext).stat().st_size for ext in ('json','npz','bin')}
        with np.load(str(p)+'.npz',allow_pickle=False) as z:
            arrays[str(arm)]={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype),bytes=z[k].nbytes) for k in z.files}
        assert dev[str(arm)]['bin']==1000*4024
    wall=[read(ROOT/f'artifacts/srr_p1/v1/wall_arm{a}.json') for a in (0,1)]
    rate=max(v['iterations']/4. for v in wall)
    envelope=math.ceil(4*rate*4)+1
    fixed=1024**2;top=64*1024**2;byte_rate=10240
    payload=96*(1000+envelope)*byte_rate+192*fixed+top
    expected=96*(1000+max(v['iterations'] for v in wall))*max(sum(v.values()) for v in dev.values())/1000
    model=dict(old_per_iteration_bytes=per,old_estimate_bytes=83230464000,old_required_bytes=166460928000,
        actual_development_file_bytes=dev,npz_array_layout=arrays,journal_record_bytes=4024,
        expected_development_scaled_bytes=math.ceil(expected),observed_wall_attempts=[v['iterations'] for v in wall],
        observed_max_attempts_per_second=rate,throughput_multiplier=4,wall_envelope_iterations=envelope,
        provision_bytes_per_iteration=byte_rate,fixed_per_case_bytes=fixed,top_level_bytes=top,
        conservative_payload_bytes=payload,delivery_copy_factor=2,free_floor_bytes=POLICY['free_floor_bytes'],
        admission_required_bytes=2*payload+POLICY['free_floor_bytes'],
        theoretical_format_envelope_bytes=96*(1000+100000)*byte_rate+192*fixed+top,
        deterministic_total_filesystem_guarantee=False,wall_loop_ceiling_unchanged=100000,
        runtime_policy=POLICY)
    save(OUT/'storage_model.json',model)
    tests=[]
    def ok(name,value):
        if not value:raise AssertionError(name)
        tests.append(dict(test=name,status='PASS'))
    ok('original_formula_exact',2*96*101000*per==166460928000)
    ok('development_sizes_independent',dev==read(ROOT/'artifacts/m2_principal/v1/storage_audit.json')['development_1000_iteration_file_bytes'])
    ok('simulated_admission_pass',admit(model['admission_required_bytes']+1,model))
    ok('simulated_admission_reject',not admit(model['admission_required_bytes']-1,model))
    prefixes=[]
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        d=Path(td)
        for scenario in ('floor','growth','enospc','write_error'):
            t=d/scenario;t.mkdir()
            code="from pathlib import Path; import time,subprocess,sys; p=Path(sys.argv[1]); (p/'prefix.bin').write_bytes(b'preserved-journal-prefix'); "
            if scenario=='write_error':code+="(p/'worker.log').write_text('[Errno 28] No space left on device')"
            else:code+="c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True); (p/'nested.pid').write_text(str(c.pid)); time.sleep(20)"
            proc=subprocess.Popen([sys.executable,'-c',code,str(t)],start_new_session=True)
            until=time.monotonic()+5
            while not (t/('worker.log' if scenario=='write_error' else 'nested.pid')).exists():
                if time.monotonic()>until:raise TimeoutError('synthetic child failed')
                time.sleep(.01)
            calls=[0]
            def fake(path):
                calls[0]+=1
                if scenario=='enospc':raise OSError(errno.ENOSPC,'simulated only')
                return dict(free=1 if scenario=='floor' and calls[0]>1 else 10**15,persisted_bytes=10**15 if scenario=='growth' and calls[0]>1 else 0,filesystem_device=t.stat().st_dev)
            result=watch(proc,t,d/(scenario+'.json'),sampler=fake)
            expected_reason={'floor':'DISK_FREE_FLOOR','growth':'DISK_GROWTH_RESERVE','enospc':'DISK_WRITE_ERROR','write_error':'DISK_WRITE_ERROR'}[scenario]
            ok('stop_classification_'+scenario,result['status']=='RESOURCE-INCONCLUSIVE' and result['reason']==expected_reason)
            ok('partial_journal_preserved_'+scenario,(t/'prefix.bin').read_bytes()==b'preserved-journal-prefix')
            prefixes.append(dict(scenario=scenario,prefix_sha256=sha(t/'prefix.bin'),result=result))
            if (t/'nested.pid').exists():
                pid=int((t/'nested.pid').read_text());time.sleep(.05)
                stat=Path(f'/proc/{pid}/stat')
                ok('nested_worker_session_stopped_'+scenario,not stat.exists() or stat.read_text().split()[2]=='Z')
    # Verify protection code bytes rather than invoking any principal command.
    for p in ('srr_p2/core.py','srr_p2/execute.py','srr_p2/worker.py','srr_p2/runner.py'):
        ok('frozen_protection_unchanged_'+p,sha(ROOT/p)==r['files'][p])
    import fcntl
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        lock=Path(td)/'lock'
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            v=subprocess.run([sys.executable,'-c',"import fcntl,sys; f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",str(lock)],capture_output=True)
            ok('exclusive_lock_mechanics',v.returncode!=0)
    ok('all_historical_bytes_unchanged',before=={p:sha(ROOT/p) for p in historical})
    verify_registration(ROOT/'artifacts/srr_p2/v1/registration.json',RH)
    ok('no_principal_data_or_execution',not (olddir/'execution').exists() and not any(p.exists() for p in claims))
    save(OUT/'safety_tests.json',dict(status='PASS',tests=tests,synthetic_failure_evidence=prefixes,principal_commands_invoked=0,principal_coordinates_accessed=False))
    save(OUT/'registration_integrity.json',dict(status='PASS',registration_sha256=RH,protocol_sha256=PH,registered_files_verified=len(r['files']),historical_files_verified=len(before),prior_result='M2 RESOURCE-INCONCLUSIVE',principal_valid_runs=0,principal_dataset_present=False,claims_present=False))
    disk=shutil.disk_usage(OUT)
    save(OUT/'resource_review.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),filesystem_path=str(OUT.resolve()),filesystem_device=OUT.stat().st_dev,current_free_bytes=disk.free,proposed_required_bytes=model['admission_required_bytes'],proposed_admission_now=admit(disk.free,model),historical_free_bytes=22120095744,historical_capacity_would_meet_proposal=admit(22120095744,model),no_launch_permission=True,remaining_review='New human approval of operational amendment plus new bound principal authorization and fresh full preflight'))
    save(OUT/'source_manifest.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'storage_recovery').glob('*.py')},historical_input_hashes=before))
    print(json.dumps(dict(status='PASS',model=model,free_now=disk.free,tests=len(tests)),indent=2))
if __name__=='__main__':main()
