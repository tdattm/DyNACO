"""Independent archive checks; no solver imports, data generation or launch."""
from pathlib import Path
import tempfile,copy,json,subprocess,sys
import numpy as np
from srr_p2.core import *

def main():
    out=ROOT/'artifacts/srr_p2/v1';rp=out/'registration.json';rh=sha(rp)
    r=verify_registration(rp,rh)
    checks=[]
    for arm in (0,1):
        d=out/'final_validation'/f'development_arm{arm}'
        row=read(str(d)+'.json');a=dict(np.load(str(d)+'.npz',allow_pickle=False))
        old=dict(np.load(ROOT/f'artifacts/srr_p1/v1/iter_r0_arm{arm}_light1.npz',allow_pickle=False))
        for k in ('tours','costs','source','best','pheromone'):assert np.array_equal(a[k],old[k])
        x=np.load(ROOT/r['development_file'],allow_pickle=False)[0]
        validate(x,row,a,dict(id='development_only',mode='iterations',instance=0,seed=31415,arm=arm))
        j=np.fromfile(str(d)+'.bin',dtype=JOURNAL_DTYPE)
        assert np.array_equal(j['tour'],a['tours']) and np.array_equal(j['seconds'],a['times'])
        assert j['calls'][-1]==row['calls'] and j['evaluations'][-1]==row['evaluations']
        checks.append('independent_archive_and_journal_arm'+str(arm))
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'registration.json';v=copy.deepcopy(r)
        v['files']['srr_p2/core.py']='bad';save(p,v)
        try:verify_registration(p,sha(p))
        except ValueError:checks.append('source_drift_rejected')
        else:raise AssertionError('drift accepted')
        auth=Path(td)/'deny.json';save(auth,dict(action='NOT_APPROVED'))
        output=Path(td)/'must_not_exist'
        denied=subprocess.run([sys.executable,'-m','srr_p2.execute','--registration',str(rp),'--authorization',str(auth),'--dataset-manifest',str(Path(td)/'absent.json'),'--output',str(output)],capture_output=True)
        assert denied.returncode!=0 and not output.exists()
        checks.append('explicit_principal_gate_rejects_without_launch')
        try:verify_registration(rp,'bad')
        except ValueError:checks.append('registration_drift_rejected')
        else:raise AssertionError('drift accepted')
    assert read(out/'final_validation/tests.json')['status']=='PASS'
    assert len(read(out/'dispatch.json'))==192
    # Evidence input checks from the reviewed P1 namespace; read bytes only.
    historical={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'artifacts/srr_p1/v1').rglob('*') if p.is_file()}
    source_archive=out/'source_archive'
    for p,h in r['files'].items():
        if (source_archive/p).is_file():assert sha(source_archive/p)==h
    save(out/'independent_audit.json',dict(status='PASS',review='Separate archive/validator implementation in same agent; no external scientific review claimed',checks=checks,registration_sha256=rh,historical_p1_files_sha256=historical,principal_data_access=False,principal_execution=False))
    save(out/'output_manifest.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_manifest.json'}))
if __name__=='__main__':main()
