"""Package an unapproved executable-registration candidate and read-only audit."""
import sys,os,json,hashlib,subprocess,importlib.metadata,platform,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from m2_safety.common import sha,save,read,implementation,SCIENTIFIC,utc
from m2_safety.launch import identity
from r3.common import git
OUT=Path(__file__).resolve().parent
def main():
    assert read(OUT/'SAFETY_TEST_SUMMARY.json')['status']=='PASS'
    assert read(OUT/'SAVED_STATE_REGRESSION.json')['status']=='PASS'
    assert '12 passed' in (OUT/'existing_pytest.log').read_text()
    assert '5 passed' in (OUT/'namespace_development_regression.log').read_text()
    assert read(OUT/'PINNED_BOOTSTRAP_TESTS.json')['status']=='PASS'
    regpath=ROOT/'artifacts/srr_p2/v1/registration.json';assert sha(regpath)==SCIENTIFIC
    old=read(regpath)
    changed=[p for p,h in old['files'].items() if sha(ROOT/p)!=h]
    assert sorted(changed)==['srr_p2/execute.py','srr_p2/worker.py']
    history=read(ROOT/'artifacts/m2_storage_recovery/v1/source_manifest.json')['historical_input_hashes']
    drift=[p for p,h in history.items() if sha(ROOT/p)!=h];assert sorted(drift)==sorted(changed)
    for p,v in read(ROOT/'artifacts/m2_storage_review/v1/source_and_artifact_manifest.json')['files'].items():assert sha(ROOT/'artifacts/m2_storage_review/v1'/p)==v['sha256']
    claims=[ROOT/'.srr-build'/('principal_'+SCIENTIFIC+'.claim'),Path('/mnt/d/GitRepository/DyNACO-srr-p2/.srr-build')/('principal_'+SCIENTIFIC+'.claim')]
    assert not any(p.exists() for p in claims)
    live=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                args=(p/'cmdline').read_bytes().split(b'\0')
                match=any(a in (b'm2_safety.init',b'm2_safety.bootstrap',b'm2_safety.dummy',b'm2_safety.sampler') or a.startswith(b'from m2_safety.launch import launch;') for a in args)
                if match:
                    row=identity(int(p.name))
                    if row['state']!='Z':live.append(row)
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not live,live
    save(OUT/'FINAL_PROCESS_SCAN.json',dict(status='PASS',at=utc(),live_safety_workers=live,scope='Host /proc scan for every dummy bootstrap/init/sampler/launcher/descendant module after all tests; no signals sent'))
    candidate=dict(old);candidate['files']=dict(old['files'])
    for p in changed:candidate['files'][p]=sha(ROOT/p)
    for p in (ROOT/'m2_safety').glob('*.py'):candidate['files'][str(p.relative_to(ROOT))]=sha(p)
    amendment=OUT/'OPERATIONAL_AMENDMENT_V2.md'
    candidate['files'][str(amendment.relative_to(ROOT))]=sha(amendment)
    candidate['files']['/usr/bin/unshare']=sha('/usr/bin/unshare')
    candidate.update(status='CANDIDATE_PENDING_INDEPENDENT_RE_REGISTRATION_REVIEW',source_commit=git('rev-parse','HEAD'),scientific_registration_sha256=SCIENTIFIC,previous_executable_registration_sha256=SCIENTIFIC,implementation_sha256=implementation(),operational_amendment_sha256=sha(amendment),candidate_created_at=utc(),approval_status='NOT APPROVED',diagnostic_binary=str(ROOT/'artifacts/srr_p2/v1/native/r3_diagnostic.cpython-313-x86_64-linux-gnu.so'))
    candidate['superseded_unapproved_candidate_sha256']=sha(OUT/'pre_nonblocking_candidate/registration_candidate.json')
    save(OUT/'registration_candidate.json',candidate)
    (OUT/'registration_candidate.sha256').open('x').write(sha(OUT/'registration_candidate.json')+'\n')
    from srr_p2.core import verify_registration
    verify_registration(OUT/'registration_candidate.json',sha(OUT/'registration_candidate.json'))
    compiler='/mnt/d/GitRepository/DyNACO/.r1-env/sysroot/usr/bin/x86_64-linux-gnu-g++-13'
    env=dict(python=sys.version,numpy=importlib.metadata.version('numpy'),torch=importlib.metadata.version('torch'),kernel=platform.platform(),unshare=subprocess.check_output(['/usr/bin/unshare','--version'],text=True),unshare_sha256=sha('/usr/bin/unshare'),compiler=subprocess.check_output([compiler,'--version'],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True),filesystem_free_bytes=shutil.disk_usage(OUT).free)
    save(OUT/'source_environment_manifest.json',dict(source_commit=candidate['source_commit'],implementation_sha256=implementation(),amendment_sha256=sha(amendment),files={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'m2_safety').glob('*')) if p.is_file()},environment=env,old_registration_sha256=SCIENTIFIC,new_candidate_sha256=sha(OUT/'registration_candidate.json'),expected_pinned_delta=changed,other_historical_hashes_verified=len(history)-2,approvals_created=0,principal_runs=0))
    save(OUT/'decision.json',dict(decision='M2-S2F READY FOR INDEPENDENT SAFETY RE-REVIEW',needs_independent_re_registration_review=True,operational_amendment_approved=False,dataset_permission_approved=False,execution_authorization_issued=False,principal_data_generated=False,principal_runs=0))
    save(OUT/'artifact_manifest.json',dict(files={str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json'}))
    print('PASS candidate hash/source verification and final process scan; no approval issued')
if __name__=='__main__':main()
