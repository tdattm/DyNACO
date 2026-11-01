"""PID-namespace init: kernel containment owner and mandatory disk heartbeat."""
import os,sys,time,json,signal,subprocess,selectors,shutil
from pathlib import Path
from .common import *
from .approval import validate_bundle

class Stop(BaseException):pass
def signal_stop(number,frame):raise Stop('OWNER_SIGNAL_'+str(number))
def message(stream,buffer):
    if b'\n' not in buffer:
        try:chunk=os.read(stream.fileno(),8192)
        except BlockingIOError:return None,buffer
        if not chunk:raise RuntimeError('disk watchdog exited')
        buffer+=chunk
    if len(buffer)>8192:raise RuntimeError('disk watchdog message too large')
    if b'\n' not in buffer:return None,buffer
    line,buffer=buffer.split(b'\n',1)
    return json.loads(line),buffer
def main():
    if os.getpid()!=1:raise RuntimeError('must run as namespace init')
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,signal_stop)
    specfile=Path(sys.argv[1]);spec=read(specfile);record=Path(spec['record']);state=State();started=False;complete=False
    root=Path(spec['output']);fixture=spec['mode']=='fixture'
    log=(record/'namespace.log').open('x')
    try:
        # All gating precedes child creation, GPU lock or persistent claim.
        if not fixture:
            validate_bundle(spec)
            from srr_p2.core import verify_registration,dataset
            verify_registration(spec['registration'],sha(spec['registration']))
            dataset(spec['dataset_manifest'],sha(spec['registration']))
        if root.exists():raise ValueError('output already exists; no resume')
        free=shutil.disk_usage(root.parent).free
        if fixture and spec['scenario']=='insufficient':free=1
        if free<ADMISSION:
            state.fail('BLOCKED','DISK_ADMISSION',free_bytes=free,required_bytes=ADMISSION);return
        device=root.parent.stat().st_dev
        save(record/'admission.json',dict(free_bytes=free,required_bytes=ADMISSION,device=device,at=utc(),monotonic=time.monotonic()))
        if not fixture:
            # Sealed capability can only be inherited by the contained launch.
            grant=sealed_grant(dict(mode='principal',spec_file=str(specfile),spec_sha256=sha(specfile),implementation_sha256=implementation(),pid_namespace=Path('/proc/1/ns/pid').stat().st_ino,registration_sha256=sha(spec['registration']),authorization_file=str(Path(spec['execution_authorization']).resolve()),authorization_sha256=sha(spec['execution_authorization']),dataset_manifest_sha256=sha(spec['dataset_manifest']),output=str(root.resolve())))
        else:grant=None
        sampler=subprocess.Popen([sys.executable,'-m','m2_safety.sampler',str(specfile)],stdout=subprocess.PIPE,stderr=log,start_new_session=True)
        os.set_blocking(sampler.stdout.fileno(),False);buffer=b''
        selector=selectors.DefaultSelector();selector.register(sampler.stdout,selectors.EVENT_READ)
        # A valid first sample is required BEFORE worker creation.
        previous=None;ready_deadline=time.monotonic()+2
        while previous is None:
            if time.monotonic()>ready_deadline:raise RuntimeError('disk sampler startup heartbeat missing')
            if selector.select(timeout=.1):previous,buffer=message(sampler.stdout,buffer)
        if 'error' in previous or previous['device']!=device or previous['free']<ADMISSION:raise RuntimeError('disk sampler startup refused')
        last=time.monotonic()
        if time.monotonic()-read(record/'admission.json')['monotonic']>2:raise RuntimeError('stale admission')
        # Fresh sample immediately before invoking existing lock/claim entry.
        if shutil.disk_usage(root.parent).free<ADMISSION:raise RuntimeError('disk admission changed before launch')
        env=os.environ.copy()
        if grant is not None:env['DYNACO_SAFETY_FD']=str(grant)
        if fixture:command=[sys.executable,'-m','m2_safety.dummy',str(specfile)]
        else:command=[sys.executable,'-m','srr_p2.execute','--registration',spec['registration'],'--authorization',spec['execution_authorization'],'--dataset-manifest',spec['dataset_manifest'],'--output',str(root)]
        fds=([grant] if grant is not None else [])+([spec['escape_fd']] if fixture and 'escape_fd' in spec else [])
        launcher=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True,pass_fds=tuple(fds))
        started=True;start=time.monotonic()
        save(record/'worker_started.json',dict(pid=launcher.pid,ppid=1,pgid=os.getpgid(launcher.pid),sid=os.getsid(launcher.pid),pid_namespace=Path('/proc/1/ns/pid').stat().st_ino,at=utc()))
        while True:
            if fixture and spec['scenario']=='supervisor_exception' and (record/'inject').exists():raise RuntimeError('synthetic init supervisor failure')
            if time.monotonic()-spec['aggregate_start_monotonic']>2700:state.fail('RESOURCE-INCONCLUSIVE','AGGREGATE_CAP');break
            events=selector.select(timeout=.1)
            sample=None
            if events or b'\n' in buffer:sample,buffer=message(sampler.stdout,buffer)
            if sample is not None:
                now=time.monotonic()
                if 'error' in sample:
                    state.fail('RESOURCE-INCONCLUSIVE','DISK_WRITE_ERROR' if sample.get('errno') in (28,122) else 'DISK_MONITOR_FAILURE',error=sample['error']);break
                if sample['device']!=device:raise RuntimeError('output filesystem changed')
                dt=max(sample['t']-previous['t'],.001)
                rate=max(0,(sample['bytes']-previous['bytes'])/dt,(previous['free']-sample['free'])/dt)
                threshold=FLOOR+max(BURST,2*rate)
                with (record/'disk_observations.jsonl').open('a') as f:f.write(json.dumps(dict(**sample,threshold=threshold))+'\n')
                if sample['free']<=FLOOR or sample['free']<threshold:
                    state.fail('RESOURCE-INCONCLUSIVE','DISK_FREE_FLOOR' if sample['free']<=FLOOR else 'DISK_GROWTH_RESERVE');break
                previous=sample;last=now
            if time.monotonic()-last>2:raise RuntimeError('disk watchdog heartbeat expired')
            if launcher.poll() is not None:
                if fixture:
                    complete=launcher.returncode==0 and (root/'fixture_complete.json').exists()
                    if not complete:
                        text=(root/'worker.log').read_text() if (root/'worker.log').exists() else ''
                        state.fail('RESOURCE-INCONCLUSIVE' if '[Errno 28]' in text else 'INCONCLUSIVE','DISK_WRITE_ERROR' if '[Errno 28]' in text else 'LAUNCHER_EXIT_BEFORE_COMPLETE')
                else:
                    wd=read(root/'watchdog.json') if (root/'watchdog.json').exists() else {}
                    text='\n'.join(p.read_text(errors='replace')[-32768:] for p in (root/'worker.log',root/'study_status.json') if p.exists())
                    if any(s in text for s in ('[Errno 28]','[Errno 122]')):state.fail('RESOURCE-INCONCLUSIVE','DISK_WRITE_ERROR')
                    elif wd.get('status')=='RESOURCE-INCONCLUSIVE':state.fail('RESOURCE-INCONCLUSIVE','P2_RESOURCE_STOP',origin=wd)
                    elif launcher.returncode==0 and wd.get('status')=='COMPLETE':
                        from srr_p2.core import schedule
                        complete=all((root/r['id']/'validated.json').is_file() for r in schedule()) and read(root/'study_status.json').get('runs')==192
                    if not complete and state.failure is None:state.fail('INCONCLUSIVE','P2_INCOMPLETE_OR_CORRECTNESS')
                break
    except Stop as e:state.fail('RESOURCE-INCONCLUSIVE' if started else 'BLOCKED',str(e))
    except BaseException as e:state.fail('RESOURCE-INCONCLUSIVE' if started else 'BLOCKED','SAFETY_MONITOR_FAILURE' if started else 'PREFLIGHT_REFUSAL',error=type(e).__name__+': '+str(e))
    finally:
        try:save(record/'init_result.json',dict(**state.result(complete),started=started))
        except BaseException as e:print('init result persistence failure: '+repr(e),file=log,flush=True)
        log.close()
        # No process-group guessing. Exiting PID1 makes Linux kill EVERY
        # remaining namespace member, including separate sessions/descendants.
        os._exit(0 if complete and state.failure is None else 1)
if __name__=='__main__':main()
