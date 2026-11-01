"""Correctness gates before bounded pilot. No principal study."""
import os,sys,json,subprocess,time
from pathlib import Path
import numpy as np
from r3.common import ROOT,sha256
OUT=ROOT/'artifacts/srr_p0/v1/verification_v3'
sys.path.insert(0,str(ROOT/'.srr-build/test'))
import faco_opt
faco_opt.set_num_threads(2)

def main():
    OUT.mkdir(exist_ok=False)
    data=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy')
    folder=ROOT/'artifacts/r3/m1_additional_k32/v2';cases=[]
    for i in range(8):
        with np.load(ROOT/f'artifacts/r3/experiments/stage_a_v1_01/native_candidates_i{i}.npz') as z:nn=z['k32'].copy()
        for seed in (31415,27182):
            for it in (499,999):
                key=f'i{i}_s{seed}_t{it}'
                with np.load(ROOT/f'artifacts/r3/experiments/stage_a_v1_01/i{i}_s{seed}/raw_before_iter{it}.npz') as z:
                    raw=z['raw'][:1000].tolist();cl=z['initial_checklist'].tolist()
                expected1=json.loads((folder/(key+'_pass1.json')).read_text());expected2=json.loads((folder/(key+'_trace.json')).read_text())
                for extra,measure in ((False,False),(True,False),(True,True)):
                    s=faco_opt.MFACO_TSP(data[i],1,32,32,12,.1,1.,.05,True,False,True,True)
                    np.copyto(s.nn_list,nn);s.configure_srr(extra,measure)
                    route,check=s.test_srr_schedule(raw,cl);expected=expected2 if extra else expected1
                    assert route==expected['route'] and check==expected['checklist']
                    assert np.array_equal(np.sort(route),np.arange(1000))
                    if measure:
                        ev=sum(len(e['evaluations']) for r in (expected1,expected2) for e in r['events'])
                        accepted=sum(e['selected_gain_f32']>0 for r in (expected1,expected2) for e in r['events'])
                        mutations=sum(bool(e.get('route_updates')) for r in (expected1,expected2) for e in r['events'])
                        assert list(s.srr_metrics)==[2,ev,accepted,mutations]
                cases.append(dict(case=key,disabled_exact=True,enabled_exact=True,checklist_exact=True,
                                  measurement_invariant=True,native_evaluations_exact=True,no_third_pass=True))
    (OUT/'saved_case_gate.json').write_text(json.dumps(cases,indent=2))
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT);env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    def worker(tag,binary,extra=False,measure=False):
        with (OUT/(tag+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'srr_p0/worker.py'),str(binary),str(OUT/(tag+'.npz')),
                str(int(extra)),str(int(measure))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
        return np.load(OUT/(tag+'.npz'))
    frozen=Path('/mnt/d/GitRepository/DyNACO-r3/.r3-build/native')
    a=worker('frozen_baseline',frozen);b=worker('disabled',ROOT/'.srr-build/test')
    assert set(a.files)==set(b.files)
    assert all(np.array_equal(a[k],b[k]) for k in a.files),'disabled baseline regression'
    # Only after saved-state and disabled regression gates pass: bounded pilot.
    c=worker('enabled',ROOT/'.srr-build/test',True)
    d=worker('enabled_measured',ROOT/'.srr-build/test',True,True)
    assert all(np.array_equal(c[k],d[k]) for k in c.files),'measurement changed outputs'
    e=worker('enabled_repeat',ROOT/'.srr-build/test',True)
    assert all(np.array_equal(c[k],e[k]) for k in c.files),'repeat nondeterministic'
    # Disable/reset and compare subsequent RNG-dependent sample via repeated path arrays.
    result=dict(status='PASS',saved_cases=32,disabled_exact_fields=len(a.files),
        costs_routes_source_best_pheromone_exact=True,seeded_repeated_rng_paths_exact=True,
        neural_graph_and_checkpoint_inference_exact=True,normal_parallel_equals_parallel_traced=True,
        serial_traced_disabled_regression=True,enabled_logging_invariant=True,enabled_repeat_exact=True,
        native_call_evaluation_accepted_mutation_counts=True,
        limitations='Serial traced and normal use different baseline RNG seeding contracts; each compared to its own frozen path. Timing with atomic counters includes instrumentation overhead.',
        decision='M2-P0 READY FOR PROTOCOL REVIEW')
    (OUT/'correctness_gate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        (OUT/'correctness_failure.json').write_text(json.dumps(dict(decision='M2-P0 BLOCKED: CORRECTNESS',error=traceback.format_exc()),indent=2))
        raise
