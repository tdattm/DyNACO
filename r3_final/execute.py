"""Dataset seal, sequential worker and bounded process supervisor."""
import contextlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

_IMPORT_START = time.perf_counter()
import numpy as np
import torch
import faco_opt
from r3_final.core import *


def source_paths():
    return [*[ROOT/'r3_final'/p for p in ('__init__.py','core.py','execute.py')],
            *[ROOT/p for p in ('src/mfaco_train.cpp','src/mfaco_train.h','src/binding.cpp','src/setup.py',
                               'src/kd_tree.h','faco.py','net.py','utils.py','r3/state_capture.py','r3/common.py')]]


def sources():
    return {str(p.relative_to(ROOT)):sha256(p) for p in source_paths()}


def check_frozen():
    manifest=json.loads((OUT/'implementation_manifest.json').read_text())
    assert sources()==manifest['source_hashes'], 'scientific source changed after seal'
    assert sha256(faco_opt.__file__)==manifest['native_binary_sha256']
    assert sha256(CHECKPOINT)==manifest['checkpoint_sha256']
    assert sha256(OUT/'SCIENTIFIC_PROTOCOL.md')==(OUT/'scientific_protocol.sha256').read_text().split()[0]
    dataset=json.loads((OUT/'dataset_manifest.json').read_text())
    assert sha256(OUT/'principal_coords.npy')==dataset['sha256']


def seal():
    assert json.loads((OUT/'test_results.json').read_text())['status']=='PASS'
    final_check=json.loads((OUT/'development/final_binary_verification.json').read_text())
    assert final_check['status']=='PASS' and final_check['native_binary_sha256']==sha256(faco_opt.__file__)
    assert json.loads((OUT/'development/untimed_srr_diagnostics.json').read_text())['exact_native_replays']==9
    assert not (OUT/'dataset_manifest.json').exists() and not (OUT/'principal_coords.npy').exists()
    assert not git('status','--porcelain','--','r3_final','src','faco.py','utils.py','net.py'), 'commit implementation first'
    digest=sha256(OUT/'SCIENTIFIC_PROTOCOL.md')
    (OUT/'scientific_protocol.sha256').write_text(digest+'  SCIENTIFIC_PROTOCOL.md\n',encoding='utf-8')
    implementation=dict(executed_source_commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
                        reference_commit='e3eae0055e61789e75a445d61dfa75238ec7ef27',source_hashes=sources(),
                        native_binary_path=str(faco_opt.__file__),native_binary_sha256=sha256(faco_opt.__file__),
                        checkpoint_sha256=sha256(CHECKPOINT),protocol_sha256=digest,sealed_utc=now())
    atomic_json(OUT/'implementation_manifest.json',implementation)
    atomic_json(OUT/'reproducibility/source_hashes.json',implementation['source_hashes'])
    strict_cuda()
    inventory=dict(utc=now(),python=sys.version,python_executable=sys.executable,numpy=np.__version__,
                   pytorch=torch.__version__,torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                   platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
                   gpu=torch.cuda.get_device_name(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                   nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),threads=torch.get_num_threads(),
                   env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')},
                   disk_free_bytes=shutil.disk_usage(OUT).free,
                   packages=sorted(f'{d.metadata["Name"]}=={d.version}' for d in __import__('importlib.metadata',fromlist=['distributions']).distributions()))
    atomic_json(OUT/'environment_manifest.json',inventory)
    # The prescribed single random call; no principal coordinates existed in development.
    coords=np.random.Generator(np.random.PCG64(202612070301)).random((16,1000,2),dtype=np.float32)
    assert coords.flags.c_contiguous
    with (OUT/'principal_coords.npy').open('xb') as f:np.save(f,coords,allow_pickle=False)
    atomic_json(OUT/'dataset_manifest.json',dict(sha256=sha256(OUT/'principal_coords.npy'),
                numpy=np.__version__,generator='Generator(PCG64(202612070301)); one random((16,1000,2),dtype=float32) call',
                shape=list(coords.shape),dtype=str(coords.dtype),c_contiguous=True,created_utc=now(),
                per_instance_sha256=[array_hash(x) for x in coords],executed_source_commit=implementation['executed_source_commit'],
                native_binary_sha256=implementation['native_binary_sha256'],checkpoint_sha256=sha256(CHECKPOINT)))
    atomic_json(OUT/'execution_manifest.json',dict(status='SEALED',planned=list(identities()),runs=[],created_utc=now()))
    print('Dataset/protocol/implementation SEALED',flush=True)


def worker():
    import_seconds=time.perf_counter()-_IMPORT_START
    load_start=time.perf_counter()
    strict_cuda();check_frozen()
    net,_,_=model()
    checkpoint_setup_seconds=time.perf_counter()-load_start
    dev=np.load(ROOT/'artifacts/r3/baseline/verification_v1/pilot_tsp1000.npy')[0]
    warms=[]
    for arm in ARMS:
        tick=time.perf_counter();s,_,_=run(dev,net,12345,arm,'iterations',iterations=100)
        warms.append(dict(arm=arm,total_seconds=time.perf_counter()-tick,search_seconds=s['search_seconds']))
    warmfile=OUT/'raw_results'/f'warmup_{time.time_ns()}.json'
    atomic_json(warmfile,dict(utc=now(),development_only=True,records=warms,
                             imports_seconds=import_seconds,checkpoint_setup_seconds=checkpoint_setup_seconds))
    data=np.load(OUT/'principal_coords.npy',allow_pickle=False)
    for identity in identities():
        runroot=OUT/'raw_results'/identity['run_id']
        done=runroot/'completion.json'
        if done.exists():
            comp=json.loads(done.read_text())
            for filename,digest in comp['files'].items():assert sha256(runroot/filename)==digest,'corrupted completion'
            continue
        attempts=list(runroot.glob('attempt_*')) if runroot.exists() else []
        if len(attempts)>=2:raise RuntimeError('bounded retry exhausted')
        if attempts:
            previous=attempts[-1]/'failure.json'
            if previous.exists() and not json.loads(previous.read_text())['retryable']:
                raise RuntimeError('nonretryable scientific failure')
        attempt=runroot/f'attempt_{len(attempts)+1}'
        attempt.mkdir(parents=True,exist_ok=False)
        atomic_json(OUT/'current_run.json',dict(**identity,attempt=len(attempts)+1,started_utc=now()))
        atomic_json(attempt/'started.json',dict(**identity,started_utc=now(),source_commit=git('rev-parse','HEAD')))
        try:
            with (attempt/'stdout.log').open('w') as stdout,(attempt/'stderr.log').open('w') as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                check_frozen()
                if shutil.disk_usage(OUT).free<1024**3:raise OSError('disk reserve below1GiB')
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv'],text=True)
                atomic_json(attempt/'gpu_before.json',dict(utc=now(),output=gpu))
                tick=time.perf_counter()
                summary,arrays,aco=run(data[identity['instance']],net,identity['seed'],identity['arm'],identity['mode'])
                summary.update(identity);summary['attempt']=len(attempts)+1
                summary['warmup_file']=str(warmfile.relative_to(OUT))
                if summary['rss_peak_bytes']>4*1024**3 or summary['peak_cuda_bytes']>3.5*1024**3:
                    raise MemoryError('resource ceiling')
                serial=time.perf_counter()
                with (attempt/'trajectory.npz').open('xb') as f:np.savez_compressed(f,**arrays)
                summary['serialization_seconds']=time.perf_counter()-serial
                summary['total_run_seconds']=time.perf_counter()-tick
                summary['completed_utc']=now()
                atomic_json(attempt/'result.json',summary)
                del aco,arrays
                print(json.dumps(summary,ensure_ascii=False),flush=True)
            files={str(p.relative_to(runroot)):sha256(p) for p in attempt.iterdir() if p.is_file()}
            atomic_json(done,dict(identity=identity,status='VALID',attempt=len(attempts)+1,files=files))
            print(f'VALID {identity["run_id"]} iterations={summary["completed_iterations"]} search={summary["search_seconds"]:.3f}s',flush=True)
        except BaseException as error:
            retryable=isinstance(error,(OSError,torch.cuda.OutOfMemoryError))
            atomic_json(attempt/'failure.json',dict(identity=identity,utc=now(),error=repr(error),traceback=traceback.format_exc(),retryable=retryable))
            raise
        manifest=json.loads((OUT/'execution_manifest.json').read_text())
        manifest.update(status='RUNNING',runs=[dict(identity=z['identity'],status=z['status'],attempt=z['attempt']) for p in sorted((OUT/'raw_results').glob('*/completion.json')) for z in [json.loads(p.read_text())]],updated_utc=now())
        atomic_json(OUT/'execution_manifest.json',manifest)
    manifest=json.loads((OUT/'execution_manifest.json').read_text())
    manifest.update(status='COMPLETE',valid_runs=len(manifest['runs']),completed_utc=now())
    assert manifest['valid_runs']==192
    atomic_json(OUT/'execution_manifest.json',manifest)


def supervise():
    import fcntl
    lock=(OUT/'principal.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'raw_results').mkdir(exist_ok=True)
    log=OUT/'reproducibility/execution_log.md'
    log.parent.mkdir(exist_ok=True)
    def record(message):
        with log.open('a',encoding='utf-8') as f:f.write(f'- {now()} {message}\n')
    check_frozen()
    record('Supervisor started; one sequential GPU worker, watchdog300s, disk reserve1GiB, RSS ceiling4GiB.')
    for restart in range(3):
        with (OUT/f'supervisor_{restart}.stdout.log').open('w') as stdout,(OUT/f'supervisor_{restart}.stderr.log').open('w') as stderr:
            proc=subprocess.Popen([sys.executable,'-m','r3_final.execute','worker'],stdout=stdout,stderr=stderr,start_new_session=True)
            last=None;progress=time.monotonic();reason=None
            while proc.poll() is None:
                time.sleep(2)
                current=(OUT/'current_run.json').read_bytes() if (OUT/'current_run.json').exists() else None
                if current!=last:last=current;progress=time.monotonic()
                rssfile=Path(f'/proc/{proc.pid}/status')
                if rssfile.exists():
                    status=rssfile.read_text()
                    rss=int(status.split('VmRSS:')[1].split()[0])*1024 if 'VmRSS:' in status else 0
                else:rss=0
                if time.monotonic()-progress>300:reason='watchdog300s'
                elif shutil.disk_usage(OUT).free<1024**3:reason='disk reserve'
                elif rss>4*1024**3:reason='RSS ceiling'
                if reason:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    record('Worker terminated: '+reason)
                    break
            if proc.returncode==0:
                record('All192 planned principal runs completed.');return
            record(f'Worker exit={proc.returncode}; restart={restart}; all attempt evidence preserved.')
            current=json.loads((OUT/'current_run.json').read_text()) if (OUT/'current_run.json').exists() else None
            if current:
                folder=OUT/'raw_results'/current['run_id']/f'attempt_{current["attempt"]}'
                failure=folder/'failure.json'
                if not failure.exists():atomic_json(failure,dict(utc=now(),retryable=True,error=reason or 'worker process terminated',returncode=proc.returncode))
                if not json.loads(failure.read_text())['retryable']:break
    raise RuntimeError('principal incomplete; preserve evidence and issue technical investment NO-GO')


if __name__=='__main__':
    {'seal':seal,'worker':worker,'supervise':supervise}[sys.argv[1]]()
