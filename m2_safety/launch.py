"""Only supported principal entry point. No data generator or default launch."""
import os,sys,time,json,signal,subprocess,argparse,shutil
from pathlib import Path
from .common import *
from .approval import validate_bundle

def identity(pid):
    text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split()
    return dict(pid=pid,ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),state=fields[0],start_ticks=int(fields[19]),pid_namespace=Path(f'/proc/{pid}/ns/pid').stat().st_ino)
def members(inode):
    rows=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                v=identity(int(p.name))
                if v['pid_namespace']==inode and v['state']!='Z':rows.append(v)
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return rows
def exact_signal(fd,number):signal.pidfd_send_signal(fd,number)
def owned_pidfd(owner):
    fd=os.pidfd_open(owner['pid'])
    try:
        current=identity(owner['pid'])
        if any(current[k]!=owner[k] for k in ('start_ticks','pid_namespace')):raise RuntimeError('PID ownership changed')
        return fd
    except BaseException:os.close(fd);raise
def reserve_attempt(parent,attempt):
    # Separate safety claim: no resume, even if stopped before native P2 claim.
    if not attempt or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in attempt):raise ValueError('unsafe attempt id')
    save(Path(parent)/(attempt+'.safety_claim'),dict(attempt=attempt,at=utc()))
def reconcile(record):
    record=Path(record);owner=read(record/'namespace_owner.json') if (record/'namespace_owner.json').exists() else None
    if owner and members(owner['pid_namespace']):return dict(status='RUNNING',reason='containment_still_live')
    if (record/'authoritative_status.json').exists():return read(record/'authoritative_status.json')
    if (record/'init_result.json').exists():result=read(record/'init_result.json')
    else:result=dict(status='RESOURCE-INCONCLUSIVE' if (record/'worker_started.json').exists() else 'BLOCKED',reason='SAFETY_OWNER_LOST',cleanup_errors=[])
    result['containment_live_processes']=[]
    save(record/'authoritative_status.json',result);return result

class Interrupted(BaseException):pass
def launch(specfile):
    aggregate_start=time.monotonic()
    specfile=Path(specfile).resolve();spec=read(specfile);fixture=spec.get('mode')=='fixture'
    if not fixture:validate_bundle(spec)
    output=Path(spec['output']).resolve();record=Path(spec['record']).resolve()
    if output.exists() or record.exists():raise ValueError('unused output/record required')
    if not fixture and shutil.disk_usage(output.parent).free<ADMISSION:raise ValueError('BLOCKED: DISK_ADMISSION')
    if not fixture:reserve_attempt(ROOT/'.srr-build',spec['attempt'])
    record.mkdir();save(record/'controller.json',dict(**identity(os.getpid()),at=utc(),mode=spec['mode']))
    runtime_spec=record/'runtime_spec.json'
    runtime=dict(spec);runtime.update(aggregate_start_monotonic=aggregate_start,input_spec_sha256=sha(specfile))
    save(runtime_spec,runtime)
    state=State();proc=None;fd=None;nsfd=None;owner=None
    def interrupted(number,frame):raise Interrupted('CONTROLLER_SIGNAL_'+str(number))
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,interrupted)
    try:
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8');env.pop('DYNACO_SAFETY_FD',None)
        with (record/'bootstrap.log').open('x') as log:
            inherited=tuple([spec['escape_fd']]) if fixture and 'escape_fd' in spec else ()
            proc=subprocess.Popen([sys.executable,'-m','m2_safety.bootstrap',str(os.getpid()),str(runtime_spec)],env=env,stdout=log,stderr=log,start_new_session=True,pass_fds=inherited)
            fd=os.pidfd_open(proc.pid);save(record/'bootstrap_identity.json',identity(proc.pid))
            deadline=aggregate_start+2700
            while proc.poll() is None:
                if owner is None:
                    try:
                        kids=Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
                        for k in kids:
                            v=identity(int(k))
                            if v['pid_namespace']!=identity(os.getpid())['pid_namespace']:
                                owner=v;nsfd=os.open(f'/proc/{k}/ns/pid',os.O_RDONLY);save(record/'namespace_owner.json',v);break
                    except (FileNotFoundError,ProcessLookupError):pass
                if time.monotonic()>deadline:raise Interrupted('OUTER_AGGREGATE_CAP')
                if not (record/'worker_started.json').exists() and time.monotonic()-aggregate_start>120:raise Interrupted('OUTER_SETUP_CAP')
                time.sleep(.02)
    except BaseException as e:
        state.fail('RESOURCE-INCONCLUSIVE' if (record/'worker_started.json').exists() else 'BLOCKED',str(e))
    finally:
        if proc is not None and proc.poll() is None:
            try:exact_signal(fd,signal.SIGKILL);proc.wait(timeout=5)
            except BaseException as e:state.cleanup.append(dict(error='containment bootstrap termination: '+repr(e)))
        if owner:
            until=time.monotonic()+5
            while members(owner['pid_namespace']) and time.monotonic()<until:time.sleep(.02)
            remaining=members(owner['pid_namespace'])
            if remaining:state.fail('RESOURCE-INCONCLUSIVE','CONTAINMENT_EXIT_UNVERIFIED',remaining=remaining)
        else:remaining=[]
        if state.failure is not None:
            # Never let cleanup errors turn a resource stop into COMPLETE.
            result=state.result();result['containment_live_processes']=remaining
            if (record/'init_result.json').exists():
                earlier=read(record/'init_result.json')
                if earlier.get('status')!='COMPLETE' and earlier.get('origin_monotonic',float('inf'))<result.get('origin_monotonic',float('inf')):
                    earlier['cleanup_errors']=earlier.get('cleanup_errors',[])+[result];result=earlier
                    result['containment_live_processes']=remaining
            if not (record/'authoritative_status.json').exists():save(record/'authoritative_status.json',result)
        else:result=reconcile(record)
        if nsfd is not None:os.close(nsfd)
        if fd is not None:os.close(fd)
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    try:
        if read(a.spec).get('mode')!='principal':raise ValueError('fixture mode is not a principal launch path')
        result=launch(a.spec)
    except BaseException as e:result=dict(status='BLOCKED',reason='PREFLIGHT_REFUSAL',error=type(e).__name__+': '+str(e))
    print(json.dumps(result))
    if result['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
