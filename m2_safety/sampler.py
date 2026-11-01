"""Separate bounded-heartbeat disk sampler; no solver imports."""
import sys,os,time,json,shutil
from pathlib import Path
def main():
    spec=json.loads(Path(sys.argv[1]).read_text());root=Path(spec['output']);parent=root.parent
    fixture=spec.get('scenario') if spec['mode']=='fixture' else None
    for i in range(100000):
        try:
            inject=(Path(spec['record'])/'inject').exists()
            if fixture=='watchdog_crash' and inject:raise RuntimeError('synthetic sampler crash')
            if fixture=='watchdog_ioerror' and inject:raise OSError(28,'synthetic disk error')
            if fixture=='watchdog_stall' and inject:time.sleep(30)
            if fixture=='watchdog_partial' and inject:
                os.write(sys.stdout.fileno(),b'{"partial":');time.sleep(30)
            files=[p for p in root.rglob('*') if p.is_file()] if root.exists() else []
            if root.is_symlink():raise RuntimeError('output root became symlink')
            target=root if root.exists() else parent
            free=shutil.disk_usage(target).free
            if fixture in ('low_disk','serialization_low_disk') and inject:free=1
            row=dict(t=time.monotonic(),free=free,bytes=sum(p.stat().st_size for p in files),device=target.stat().st_dev)
            print(json.dumps(row),flush=True);time.sleep(.25)
        except BaseException as e:
            print(json.dumps(dict(error=type(e).__name__+': '+str(e),errno=getattr(e,'errno',None))),flush=True);raise
if __name__=='__main__':main()
