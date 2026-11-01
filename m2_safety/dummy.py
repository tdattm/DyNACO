"""Harmless lifecycle fixture only: tiny prefix and sleeping descendants."""
import sys,os,json,time,subprocess,ctypes
from pathlib import Path
def identity():return dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgid(0),sid=os.getsid(0),pid_namespace=Path('/proc/self/ns/pid').stat().st_ino)
def main():
    if sys.argv[1]=='descendant':
        p=Path(sys.argv[2]);p.write_text(json.dumps(identity()))
        if len(sys.argv)>3:
            fd=int(sys.argv[3]);lib=ctypes.CDLL(None,use_errno=True)
            result=lib.setns(fd,0x20000000)
            (p.parent/'escape.json').write_text(json.dumps(dict(result=result,errno=ctypes.get_errno())))
        time.sleep(30);return
    spec=json.loads(Path(sys.argv[1]).read_text());root=Path(spec['output']);root.mkdir()
    (root/'prefix.bin').write_bytes(b'preserved-completed-and-partial-journal')
    (root/'identity.json').write_text(json.dumps(identity()))
    outerfd=spec.get('escape_fd')
    args=[sys.executable,'-m','m2_safety.dummy','descendant',str(root/'descendant.json')]
    if outerfd is not None:args.append(str(outerfd))
    subprocess.Popen(args,start_new_session=True,pass_fds=(outerfd,) if outerfd is not None else ())
    if spec.get('nested_namespace'):
        subprocess.Popen(['unshare','--pid','--fork','--mount-proc',sys.executable,'-m','m2_safety.dummy','descendant',str(root/'nested_descendant.json')],start_new_session=True)
    while not (Path(spec['record'])/'inject').exists():time.sleep(.01)
    if spec['scenario']=='serialization_failure':
        (root/'worker.log').write_text('OSError: [Errno 28] No space left on device during synthetic serialization')
        time.sleep(.5);raise OSError(28,'synthetic serialization failure')
    if spec['scenario']=='parent_launcher_exit':time.sleep(.5);return
    if spec['scenario']=='normal':time.sleep(.3);(root/'fixture_complete.json').write_text('{"complete":true}');return
    if spec['scenario']=='launcher_term':
        # A separate session mirrors the legacy P2 worker topology.
        time.sleep(30);return
    # This sleep also simulates native work that cannot service Python callbacks.
    time.sleep(30)
if __name__=='__main__':main()
