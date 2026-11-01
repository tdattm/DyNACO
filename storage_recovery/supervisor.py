"""External operational disk supervisor proposal. No CLI, generator or solver imports."""
import os,json,time,signal,subprocess,shutil,errno
from pathlib import Path

GIB=1024**3
POLICY=dict(poll_seconds=.25,free_floor_bytes=4*GIB,serialization_burst_bytes=100000*10240+1024**2,reaction_seconds=2.)
def admit(free,model):return free>=model['admission_required_bytes']
def sample(path):
    path=Path(path).resolve();v=shutil.disk_usage(path)
    files=[p for p in path.rglob('*') if p.is_file()]
    return dict(free=v.free,persisted_bytes=sum(p.stat().st_size for p in files),filesystem_device=path.stat().st_dev)
def reason(now,previous,dt,policy):
    rate=max(0,(now['persisted_bytes']-previous['persisted_bytes'])/max(dt,.001))
    free_rate=max(0,(previous['free']-now['free'])/max(dt,.001))
    threshold=policy['free_floor_bytes']+max(policy['serialization_burst_bytes'],max(rate,free_rate)*policy['reaction_seconds'])
    return ('DISK_FREE_FLOOR' if now['free']<=policy['free_floor_bytes'] else 'DISK_GROWTH_RESERVE' if now['free']<threshold else None),rate,threshold
def terminate_tree(proc):
    # P2 execute and worker intentionally start distinct sessions. Killing only
    # the launcher's group would leave its worker alive; collect both first.
    todo=[proc.pid];pids=[]
    while todo:
        pid=todo.pop();pids.append(pid)
        try:todo.extend(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
        except FileNotFoundError:pass
    groups=set()
    for pid in pids:
        try:groups.add(os.getpgid(pid))
        except ProcessLookupError:pass
    groups.discard(os.getpgrp())
    # Stop deepest groups first, retaining the parent long enough to locate all.
    for group in sorted(groups,reverse=True):
        try:os.killpg(group,signal.SIGKILL)
        except ProcessLookupError:pass
    proc.wait()
def watch(proc,output,report,policy=POLICY,sampler=sample):
    """Observe an already started operational child; never creates a launch claim.

    Report lives outside frozen execution outputs. Caller keeps existing P2
    locks/auth/claim checks and 2700s watchdog; this adds no resume capability.
    sampler injection is for synthetic tests, never for a real approved launch.
    """
    output=Path(output);report=Path(report);events=[];last_t=time.monotonic()
    try:
        previous=sampler(output)
        while True:
            t=time.monotonic();v=sampler(output)
            why,rate,threshold=reason(v,previous,t-last_t,policy)
            events.append(dict(monotonic=t,**v,growth_bytes_per_second=rate,stop_threshold_bytes=threshold))
            if why:
                terminate_tree(proc);result=dict(status='RESOURCE-INCONCLUSIVE',reason=why,valid_partial_runs=0)
                break
            if proc.poll() is not None:
                # Inspect only known status/log files; never parse principal data.
                text='\n'.join(p.read_text(errors='replace')[-32768:] for p in (output/'worker.log',output/'study_status.json') if p.exists())
                disk_error=any(s in text for s in ('[Errno 28]','[Errno 122]','No space left on device','Disk quota exceeded'))
                result=dict(status='RESOURCE-INCONCLUSIVE' if disk_error else 'CHILD_EXITED',reason='DISK_WRITE_ERROR' if disk_error else 'delegate_to_frozen_P2_status',returncode=proc.returncode)
                break
            previous=v;last_t=t;time.sleep(policy['poll_seconds'])
    except OSError as e:
        terminate_tree(proc)
        result=dict(status='RESOURCE-INCONCLUSIVE',reason='DISK_WRITE_ERROR' if e.errno in (errno.ENOSPC,errno.EDQUOT) else 'DISK_MONITOR_IO_ERROR',errno=e.errno)
    result['observations']=events
    try:
        with report.open('x') as f:json.dump(result,f,indent=2)
    except OSError:
        # Floor/reserve cannot guarantee persistence after an unrelated abrupt
        # filesystem loss. Parent stderr is the final independent evidence path.
        import sys
        print(json.dumps(result),file=sys.stderr);raise
    return result
