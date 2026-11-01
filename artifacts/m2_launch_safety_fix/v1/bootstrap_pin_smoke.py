"""Exact-unshare pin and early interrupted-launch test; dummy processes only."""
import sys,os,time,signal,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from m2_safety.common import save,sha
from m2_safety.launch import members,identity,exact_signal,reconcile
OUT=Path(__file__).resolve().parent
def main():
    binary=Path('/usr/bin/unshare');assert b'poll parent pidfd failed' in binary.read_bytes()
    rows=[];base=OUT/'pinned_bootstrap_checks';base.mkdir()
    for case in ('normal','early_controller_kill','controller_kill'):
        p=base/case;p.mkdir();record=p/'record';output=p/'output'
        spec=p/'spec.json';save(spec,dict(mode='fixture',scenario='normal' if case=='normal' else 'native_stall',record=str(record),output=str(output)))
        with (p/'driver.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-c','from m2_safety.launch import launch;import sys;print(launch(sys.argv[1]))',str(spec)],stdout=log,stderr=log,start_new_session=True)
            fd=os.pidfd_open(proc.pid);held=None
            try:
                until=time.monotonic()+10
                ready=(record/'bootstrap_identity.json') if case=='early_controller_kill' else output/'descendant.json'
                while not ready.exists():
                    if time.monotonic()>until:raise TimeoutError(case)
                    time.sleep(.001)
                owner=json.loads((record/'namespace_owner.json').read_text()) if (record/'namespace_owner.json').exists() else None
                if owner:held=os.open(f"/proc/{owner['pid']}/ns/pid",os.O_RDONLY)
                bootstrap=json.loads((record/'bootstrap_identity.json').read_text());bpidfd=os.pidfd_open(bootstrap['pid'])
                if case=='normal':(record/'inject').write_text('ready')
                else:exact_signal(fd,signal.SIGKILL)
                proc.wait(timeout=5)
                until=time.monotonic()+5
                while time.monotonic()<until:
                    try:running=identity(bootstrap['pid'])['state']!='Z'
                    except (FileNotFoundError,ProcessLookupError):running=False
                    if not running and (not owner or not members(owner['pid_namespace'])):break
                    time.sleep(.01)
                assert not running and (not owner or not members(owner['pid_namespace']))
                row=dict(case=case,status='PASS',controller_pid=proc.pid,controller_exit=proc.returncode,owner=owner,bootstrap_exited=True,remaining_members=members(owner['pid_namespace']) if owner else [],result=reconcile(record))
                rows.append(row);os.close(bpidfd)
            finally:
                if proc.poll() is None:exact_signal(fd,signal.SIGKILL);proc.wait(timeout=5)
                os.close(fd)
                if held is not None:os.close(held)
    save(OUT/'PINNED_BOOTSTRAP_TESTS.json',dict(status='PASS',unshare_sha256=sha(binary),pidfd_race_check_string_present=True,tests=rows,principal_actions=0))
if __name__=='__main__':main()
