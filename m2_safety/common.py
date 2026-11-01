"""Operational identity and first-failure state; never generates data."""
import hashlib,json,os,fcntl,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
SCIENTIFIC='5d2d8942f900d4d1a4ef89e069668c3d0290d17c628d058150ffe6998d5bca45'
ADMISSION=17817796608
FLOOR=4*1024**3
BURST=1025048576
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def implementation():
    paths=sorted((ROOT/'m2_safety').glob('*.py'))+[ROOT/'srr_p2/execute.py',ROOT/'srr_p2/worker.py']
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},sort_keys=True).encode()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def sealed_grant(value):
    fd=os.memfd_create('dynaco-safety-grant',os.MFD_ALLOW_SEALING)
    os.write(fd,json.dumps(value).encode());os.lseek(fd,0,0)
    fcntl.fcntl(fd,fcntl.F_ADD_SEALS,fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE)
    return fd
class State:
    def __init__(self):self.failure=None;self.cleanup=[]
    def fail(self,status,reason,**detail):
        if self.failure is None:self.failure=dict(status=status,reason=reason,origin_monotonic=time.monotonic(),**detail)
        else:self.cleanup.append(dict(status=status,reason=reason,**detail))
    def result(self,complete=False):return dict(**(self.failure or dict(status='COMPLETE' if complete else 'INCONCLUSIVE',reason='validated_full_grid' if complete else 'incomplete_exit')),cleanup_errors=self.cleanup)
