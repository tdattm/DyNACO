"""Explicit principal launch only. No default action and no data generation."""
import argparse,os,sys,time,subprocess,fcntl
from pathlib import Path
from srr_p2.core import ROOT,read,save,sha,verify_registration,dataset,monitor
from m2_safety.guard import require

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',required=True,type=Path)
    p.add_argument('--authorization',required=True,type=Path)
    p.add_argument('--dataset-manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();start=time.monotonic()
    safety_fd=require('execute',a.registration,a.authorization,a.dataset_manifest,a.output)
    # Cross-output, cross-session lock on this registered checkout/GPU workload.
    lock=(ROOT/'.srr-build/principal_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auth=read(a.authorization);rh=sha(a.registration)
    if auth.get('action')!='APPROVE_M2_PRINCIPAL_192' or auth.get('registration_sha256')!=rh or auth.get('dataset_manifest_sha256')!=sha(a.dataset_manifest):raise ValueError('explicit bound principal approval required')
    if not auth.get('approved_by') or not auth.get('approved_at'):raise ValueError('approval provenance required')
    # All potentially expensive preflight hashing/loading occurs in the watched
    # worker, before model initialization or any solver work.
    save(ROOT/'.srr-build'/('principal_'+rh+'.claim'),dict(output=str(a.output.resolve()),authorization_sha256=sha(a.authorization)))
    # Atomic directory creation is the run lock. Never resume, replace, or reuse.
    a.output.mkdir(parents=False,exist_ok=False)
    save(a.output/'launch.json',dict(authorization=auth,registration_sha256=rh,dataset_manifest_sha256=sha(a.dataset_manifest),aggregate_start_monotonic=start))
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (a.output/'worker.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,'-m','srr_p2.worker',str(a.registration.resolve()),rh,str(a.dataset_manifest.resolve()),str(a.output.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),safety_fd))
        result=monitor(proc,a.output/'events.jsonl',start+2700)
    if (a.output/'study_status.json').exists() and result['status']!='RESOURCE-INCONCLUSIVE':
        result=read(a.output/'study_status.json')
    save(a.output/'watchdog.json',dict(**result,aggregate_seconds=time.monotonic()-start))
    if result['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
