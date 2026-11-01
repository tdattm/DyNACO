"""Registration and validation. No dataset generator and no solver imports."""
import hashlib,json,os,time,subprocess,signal
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(31415,27182,16180)
JOURNAL_DTYPE=np.dtype([('seconds','<f8'),('calls','<i8'),('evaluations','<i8'),('tour','<i4',(1000,))])
def journal_record(seconds,tour,calls,evaluations):
    r=np.empty(1,dtype=JOURNAL_DTYPE)
    r['seconds']=seconds;r['tour']=tour;r['calls']=calls;r['evaluations']=evaluations
    return r.tobytes()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def schedule():
    rows=[]
    for mode in ('iterations','wall'):
        for i in range(16):
            for j,s in enumerate(SEEDS):
                for a in ((0,1) if (i+j)%2==0 else (1,0)):
                    rows.append(dict(id=f'{mode}_i{i:02d}_s{s}_a{a}',mode=mode,instance=i,seed=s,arm=a))
    return rows
def verify_registration(path,expected):
    if sha(path)!=expected:raise ValueError('registration hash drift')
    r=read(path)
    from r3.common import git
    git('merge-base','--is-ancestor',r['source_commit'],'HEAD')
    for p,h in r['files'].items():
        if sha(ROOT/p)!=h:raise ValueError('source/protocol/binary/checkpoint drift: '+p)
    if read(ROOT/r['schedule'])!=schedule():raise ValueError('dispatch drift')
    return r
def dataset(path,registration_hash):
    m=read(path)
    if m['role']!='M2_PRINCIPAL_SEALED' or m['registration_sha256']!=registration_hash:raise ValueError('unsealed dataset')
    p=Path(path).parent/m['file']
    if sha(p)!=m['sha256']:raise ValueError('dataset hash drift')
    x=np.load(p,allow_pickle=False)
    if x.shape!=(16,1000,2) or x.dtype!=np.dtype('float32') or not x.flags.c_contiguous or not np.isfinite(x).all():raise ValueError('dataset shape/dtype/finiteness')
    if (x<0).any() or (x>=1).any():raise ValueError('uniform unit-square bounds')
    return x
def eligible(times,budget):
    t=np.asarray(times)
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or (t<=0).any() or (np.diff(t)<0).any():raise ValueError('malformed timestamps')
    ids=np.flatnonzero(t<=budget) if budget is not None else np.arange(len(t))
    if not len(ids):raise ValueError('no eligible completed iteration')
    return int(ids[-1])
def validate(x,row,a,identity):
    n=len(a['times']);budget=4. if identity['mode']=='wall' else None
    idx=eligible(a['times'],budget)
    if budget is None and n!=1000:raise ValueError('incomplete iteration run')
    if budget is not None and n>=100000 and a['times'][-1]<budget:raise ValueError('wall loop ceiling reached')
    if a['tours'].shape!=(n,1000) or not np.issubdtype(a['tours'].dtype,np.integer):raise ValueError('tour shape/type')
    if not np.all(np.sort(a['tours'],axis=1)==np.arange(1000)):raise ValueError('invalid permutation/implicit cyclic closure')
    if not np.array_equal(a['best'],a['tours'][-1]):raise ValueError('global best mismatch')
    if row['primary_iteration']!=idx+1 or row['iterations']!=n or row['eligible_iterations']!=idx+1:raise ValueError('credited iteration mismatch')
    if row['seed']!=identity['seed'] or row['extra']!=bool(identity['arm']) or row['light'] is not True or row['budget']!=budget:raise ValueError('run contract')
    if row['native_seed']!=int(np.random.RandomState(identity['seed']).randint(0,2**63-1)):raise ValueError('native seed drift')
    if not np.isfinite(a['costs']).all() or not np.isfinite(a['pheromone']).all():raise ValueError('nonfinite native state')
    if row['calls']!=100*(1+identity['arm'])*n or not isinstance(row['evaluations'],int) or row['evaluations']<0:raise ValueError('native counts')
    if row['cuda_allocated']>=1024**3 or row['peak_rss_bytes']>=3*1024**3 or row['search_seconds']>60:raise MemoryError('resource observation')
    xd=x.astype(np.float64);t=a['tours'][idx]
    length=float(np.linalg.norm(xd[t]-xd[np.roll(t,-1)],axis=1).sum())
    if not np.isfinite(length) or length!=row['primary_length_f64']:raise ValueError('independent float64 cost')
    return dict(**identity,length_f64=length)
def complete(rows):
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!={r['id'] for r in schedule()}:raise ValueError('missing/duplicate/extra grid')
    from srr_p1.analysis import aggregate
    out=aggregate(rows,range(16))
    go=all(v['median']>=.001 and v['bootstrap95'][0]>0 and v['positive_instances']>=12 for v in out.values())
    return dict(decision='GO' if go else 'NO-GO',modes=out)
def monitor(proc,events,deadline,setup_cap=120,search_cap=60,rss_cap=3*1024**3):
    """Parent watchdog does not depend on child Python/CUDA responsiveness."""
    phase='setup';phase_start=time.monotonic();offset=0
    while True:
        if events.exists():
            lines=events.read_text().splitlines()
            for line in lines[offset:]:
                try:e=json.loads(line)
                except json.JSONDecodeError:break
                offset+=1
                phase=e['phase'];phase_start=e['monotonic']
        now=time.monotonic()
        why=None
        if now>deadline:why='aggregate cap'
        elif now-phase_start>(search_cap if phase=='search' else setup_cap):why=phase+' cap'
        try:
            status=Path(f'/proc/{proc.pid}/status').read_text()
            peak=max(int(l.split()[1])*1024 for l in status.splitlines() if l.startswith(('VmRSS:','VmHWM:')))
            if peak>=rss_cap:why='RSS cap'
        except (FileNotFoundError,ValueError,ProcessLookupError):pass
        if why:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            return dict(status='RESOURCE-INCONCLUSIVE',reason=why)
        if proc.poll() is not None:
            return dict(status='COMPLETE' if proc.returncode==0 else 'INCONCLUSIVE',returncode=proc.returncode)
        time.sleep(.01)
