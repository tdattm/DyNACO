"""Saved-state and frozen baseline gates; must pass before performance pilot."""
import sys,json,subprocess,os
from pathlib import Path
import numpy as np
from r3.common import ROOT
sys.path.insert(0,str(ROOT/'.srr-build/test'))
import faco_opt
OUT=ROOT/'artifacts/srr_p1/v1'
def main():
    data=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy')
    folder=ROOT/'artifacts/r3/m1_additional_k32/v2';cases=[]
    for i in range(8):
        with np.load(ROOT/f'artifacts/r3/experiments/stage_a_v1_01/native_candidates_i{i}.npz') as z:nn=z['k32'].copy()
        for seed in (31415,27182):
            for it in (499,999):
                key=f'i{i}_s{seed}_t{it}'
                with np.load(ROOT/f'artifacts/r3/experiments/stage_a_v1_01/i{i}_s{seed}/raw_before_iter{it}.npz') as z:raw=z['raw'][:1000].tolist();cl=z['initial_checklist'].tolist()
                p1=json.loads((folder/(key+'_pass1.json')).read_text());p2=json.loads((folder/(key+'_trace.json')).read_text())
                for extra in (False,True):
                    expected=p2 if extra else p1
                    reference=[p1,p2] if extra else [p1]
                    ev=sum(len(e['evaluations']) for r in reference for e in r['events'])
                    accepted=sum(e['selected_gain_f32']>0 for r in reference for e in r['events'])
                    mutations=sum(bool(e.get('route_updates')) for r in reference for e in r['events'])
                    for detailed,light in ((False,False),(True,False),(False,True)):
                        s=faco_opt.MFACO_TSP(data[i],1,32,32,12,.1,1.,.05,True,False,True,True)
                        np.copyto(s.nn_list,nn);s.configure_srr(extra,detailed,light)
                        route,check=s.test_srr_schedule(raw,cl)
                        assert route==expected['route'] and check==expected['checklist']
                        if detailed:assert s.srr_metrics==[2 if extra else 1,ev,accepted,mutations]
                        if light:assert s.srr_metrics==[2 if extra else 1,ev,0,0]
                cases.append(dict(case=key,status='PASS',original_noop_counts_preserved=True))
    (OUT/'saved_state_gates.json').write_text(json.dumps(cases,indent=2))
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT);env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for tag,extra,light,old in [('disabled',False,False,'frozen_baseline'),('enabled_light',True,True,'enabled')]:
        with (OUT/(tag+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'srr_p1/worker.py'),str(ROOT/'.srr-build/test'),str(OUT/(tag+'.npz')),str(int(extra)),str(int(light))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        with np.load(OUT/(tag+'.npz')) as actual,np.load(ROOT/f'artifacts/srr_p0/v1/verification_v3/{old}.npz') as expected:
            assert all(np.array_equal(actual[k],expected[k]) for k in expected.files),tag
    (OUT/'correctness_gate.json').write_text(json.dumps(dict(status='PASS',cases=32,disabled_frozen_baseline_exact=True,
        light_no_route_copies=True,light_vs_unmeasured_exact=True,detailed_preserved=True,
        normal_parallel_traced_parallel_serial_checked=True,neural_input_output_exact=True),indent=2))
    print('CORRECTNESS PASS')
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        (OUT/'correctness_failure.json').write_text(traceback.format_exc());raise
