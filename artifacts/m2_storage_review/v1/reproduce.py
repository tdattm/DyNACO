"""S2 reviewer-only hash/math and tiny synthetic process probes. No principal CLI."""
import sys,os,json,math,time,tempfile,subprocess,signal,hashlib,shutil,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,v):
    with (OUT/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'
def main():
    from storage_recovery.supervisor import watch,terminate_tree,admit,sample
    rh='5d2d8942f900d4d1a4ef89e069668c3d0290d17c628d058150ffe6998d5bca45'
    ph='574cd6781e7a026951521e12bfdef28f5790752289c029f21b8db0f4e5e5dedd'
    reg=ROOT/'artifacts/srr_p2/v1/registration.json';r=read(reg)
    mismatch=[p for p,h in r['files'].items() if sha(ROOT/p)!=h]
    historical=read(ROOT/'artifacts/m2_storage_recovery/v1/source_manifest.json')['historical_input_hashes']
    old_drift=[p for p,h in historical.items() if sha(ROOT/p)!=h]
    artifact=read(ROOT/'artifacts/m2_storage_recovery/v1/artifact_manifest.json')['files']
    artifact_drift=[p for p,h in artifact.items() if sha(ROOT/'artifacts/m2_storage_recovery/v1'/p)!=h['sha256']]
    s1_source=read(ROOT/'artifacts/m2_storage_recovery/v1/source_manifest.json')['files']
    source_drift=[p for p,h in s1_source.items() if sha(ROOT/p)!=h]
    assert not mismatch+old_drift+artifact_drift+source_drift
    assert sha(reg)==rh and sha(ROOT/'artifacts/srr_p2/v1/FINAL_PROTOCOL.md')==ph
    claims=[ROOT/'.srr-build'/('principal_'+rh+'.claim'),Path('/mnt/d/GitRepository/DyNACO-srr-p2/.srr-build')/('principal_'+rh+'.claim')]
    assert not any(p.exists() for p in claims)
    old=read(ROOT/'artifacts/m2_principal/v1/decision.json');assert old['valid_runs']==0 and not old['dataset_generated']
    amendment=ROOT/'artifacts/m2_storage_recovery/v1/OPERATIONAL_AMENDMENT.md'
    save('registration_integrity.json',dict(status='PASS',registration_sha256=sha(reg),protocol_sha256=ph,amendment_sha256=sha(amendment),registered_files_verified=len(r['files']),historical_files_verified=len(historical),S1_artifacts_verified=len(artifact),S1_source_hashes_verified=len(s1_source),drift=[],claims_present=False,principal_valid_runs=0,principal_data_accessed=False))
    import numpy as np
    sizes={};layouts={}
    for arm in (0,1):
        p=ROOT/f'artifacts/srr_p2/v1/final_validation/development_arm{arm}'
        sizes[str(arm)]={ext:Path(str(p)+'.'+ext).stat().st_size for ext in ('npz','json','bin')}
        with np.load(str(p)+'.npz',allow_pickle=False) as z:layouts[str(arm)]={k:dict(dtype=str(z[k].dtype),shape=list(z[k].shape),bytes=z[k].nbytes) for k in z.files}
    journal=struct.calcsize('<dqq1000i');assert journal==4024
    walls=[read(ROOT/f'artifacts/srr_p1/v1/wall_arm{a}.json')['iterations'] for a in (0,1)]
    n=math.ceil(4*max(walls)/4*4)+1
    payload=96*(1000+n)*10240+192*1048576+67108864;required=2*payload+4*1024**3
    oldrequired=2*96*(1000+100000)*(journal+4000+400+160)
    m=read(ROOT/'artifacts/m2_storage_recovery/v1/storage_model.json')
    assert required==m['admission_required_bytes']==17817796608 and oldrequired==166460928000
    assert sizes==m['actual_development_file_bytes']
    save('storage_recalculation.json',dict(status='REPRODUCED',journal_bytes=journal,old_required_bytes=oldrequired,development_file_bytes=sizes,npz_layout=layouts,observed_wall_attempts=walls,wall_envelope_iterations=n,conservative_payload_bytes=payload,proposed_required_bytes=required,classification='Conditional engineering admission estimate; NOT deterministic upper bound',S1_observed_free_bytes=21037568000,S1_free_margin_bytes=21037568000-required,S1_margin_percent_of_required=100*(21037568000-required)/required,historical_preflight_margin_bytes=22120095744-required,current_free_bytes=shutil.disk_usage(OUT).free,current_reading_usable_for_future_launch=False))
    probes=[]
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        d=Path(td)
        def child():
            return subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
        def healthy(path):return dict(free=10**15,persisted_bytes=0,filesystem_device=d.stat().st_dev)
        p=child();prefix=d/'prefix.bin';prefix.write_bytes(b'partial-journal')
        low=lambda path:dict(free=1,persisted_bytes=prefix.stat().st_size,filesystem_device=d.stat().st_dev)
        result=watch(p,d,d/'low.json',sampler=low)
        assert result['status']=='RESOURCE-INCONCLUSIVE' and prefix.read_bytes()==b'partial-journal' and p.poll() is not None
        probes.append(dict(test='standalone_low_disk_stop_and_prefix',observation='REPRODUCED_PASS',result=result))
        p=child()
        def broken(path):raise RuntimeError('synthetic supervisor failure')
        try:
            try:watch(p,d,d/'broken.json',sampler=broken)
            except RuntimeError:pass
            survived=p.poll() is None
            assert survived
            probes.append(dict(test='non_OSError_supervisor_failure',observation='UNSAFE_REPRODUCED',child_survived=True,resource_status_written=(d/'broken.json').exists()))
        finally:terminate_tree(p)
        pidfile=d/'nested.pid'
        launcher=subprocess.Popen([sys.executable,'-c',"import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True);Path(sys.argv[1]).write_text(str(p.pid))",str(pidfile)],start_new_session=True)
        try:
            launcher.wait(timeout=5);pid=int(pidfile.read_text())
            result=watch(launcher,d,d/'orphan.json',sampler=healthy)
            survived=alive(pid);assert survived and result['status']=='CHILD_EXITED'
            probes.append(dict(test='launcher_exit_with_separate_worker_session',observation='UNSAFE_REPRODUCED',worker_survived=True,result=result))
        finally:
            if pidfile.exists():
                try:os.killpg(int(pidfile.read_text()),signal.SIGKILL)
                except ProcessLookupError:pass
    save('reviewer_probes.json',dict(method='New synthetic probes in this Codex review turn, not external independent review',probes=probes,principal_commands_invoked=0,coordinates_generated_or_accessed=False,cleanup='All synthetic process groups terminated by reviewer harness after observation'))
    reported=read(ROOT/'artifacts/m2_storage_recovery/v1/safety_tests.json')['tests']
    save('S1_test_review.json',dict(reported_count=len(reported),reported_pass_count=sum(t['status']=='PASS' for t in reported),reviewed_tests=[dict(**t,S2_scope='Arithmetic/storage independently recomputed; low-disk prefix independently reprobed. Remaining items reviewed in code/evidence, not claimed rerun.') for t in reported],missing=['Mandatory admission/watchdog launch wiring','Watchdog non-OSError failure and loss-of-parent handling','Real filesystem sampler integration','Timed serialization growth and external writer coverage','Bound amendment approval enforcement','Single official stop-status arbitration']))
    print('S2 hash/math/probes completed; unsafe conditions reproduced; no principal action')
if __name__=='__main__':main()
