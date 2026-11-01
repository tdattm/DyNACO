"""Frozen M1: one additional unchanged K32 pass on saved records only."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import csv
import json
import resource
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from r3.common import ROOT,sha256,git
from r3.m0.validate import validate_trace, m0_native
from r3.m1_v2_compare import assert_equal,canonical,test_comparator
from r3.replay import reconstruct
from r3.diagnostics import canonical_tour,length

OUT=ROOT/'artifacts/r3/m1_additional_k32/v2'
M0=ROOT/'artifacts/r3/m0_srr_replay'
SOURCE=ROOT/'artifacts/r3/experiments/stage_a_v1_01'

def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def freeze():
    assert git('branch','--show-current')=='feature/r3-budget-aware-candidate-graph'
    for commit in ('35c33701291b77d9300d91921f649be17688f91a','1e6884fed9eb349ac7c3a2bfeab3f975131a6e95'):
        git('merge-base','--is-ancestor',commit,'HEAD')
    OUT.mkdir(parents=True,exist_ok=False)
    v1=ROOT/'artifacts/r3/m1_additional_k32/v1'
    protocol=json.loads((v1/'protocol.json').read_text())
    protocol['version']=2
    protocol['technical_amendment']='Strict full JSON representation comparison; comparator tests and all32 control-only preflight before intervention. Same scientific design.'
    protocol['resource_scope']='300 seconds across comparator tests, control-only preflight, principal calls, logging verification and independent validation; one worker,1GB RSS. Conservative1GB virtual-memory ceiling.'
    save('protocol.json',protocol)
    (OUT/'protocol.sha256').write_text(sha256(OUT/'protocol.json')+'\n')
    save('technical_amendment.json',dict(v1_commit='1e6884fed9eb349ac7c3a2bfeab3f975131a6e95',
         v1_protocol_sha256=sha256(v1/'protocol.json'),v2_protocol_sha256=sha256(OUT/'protocol.json'),
         v1_decision='INCONCLUSIVE; no second-pass outcome',scientific_design_changed=False,
         repair='Canonical JSON serialization with allow_nan=False and every field retained; strict complete representation equality',
         additional_checks=['10 deterministic comparator tests, no native replay','32 control-only preflight cases with independent trace validation'],
         failure_policy='Stop and preserve evidence; no v3 retry'))
    inputs={name:sha256(ROOT/name) for name in git('ls-files').splitlines() if (ROOT/name).is_file()}
    for name in ('r3/m1_v2.py','r3/m1_v2_compare.py'):inputs[name]=sha256(ROOT/name)
    original=json.loads((M0/'manifest.json').read_text())
    assert sha256(m0_native.__file__)==original['binary_sha256']
    assert all(sha256(ROOT/name)==h for name,h in original['input_sha256'].items())
    save('input_manifest.json',dict(head=git('rev-parse','HEAD'),input_sha256=inputs,
         protocol_sha256=sha256(OUT/'protocol.json'),binary_sha256=sha256(m0_native.__file__),
         binary_path=m0_native.__file__))
    print('FROZEN',sha256(OUT/'protocol.json'),flush=True)

def run():
    start=STARTED
    manifest=json.loads((OUT/'input_manifest.json').read_text())
    assert sha256(OUT/'protocol.json')==manifest['protocol_sha256']
    assert all(sha256(ROOT/p)==h for p,h in manifest['input_sha256'].items())
    assert sha256(m0_native.__file__)==manifest['binary_sha256']
    original=json.loads((M0/'manifest.json').read_text())
    assert manifest['binary_sha256']==original['binary_sha256']
    # Enforce a stricter virtual-memory ceiling as well as measuring peak RSS.
    resource.setrlimit(resource.RLIMIT_AS,(1000000000,1000000000))
    def timeout(*_):raise TimeoutError('300s experiment wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(300-(time.perf_counter()-STARTED))))
    records=[];failure=None;begun=0;physical_calls=PREFLIGHT_CALLS
    plans=[(i,s,it) for i in range(8) for s in (31415,27182) for it in (499,999)]
    try:
        data=np.load(SOURCE/'datasets/tsp1000_seed20261101_float32.npy',allow_pickle=False)
        for i,seed,it in plans:
            key=f'i{i}_s{seed}_t{it}';begun+=1
            r=dict(case=key,instance=i,seed=seed,iteration=it,status='STARTED')
            try:
                folder=SOURCE/f'i{i}_s{seed}'
                with np.load(folder/f'raw_before_iter{it}.npz') as z:
                    raw=canonical_tour(z['raw'],1000);ref=canonical_tour(z['refined'],1000)
                    source=canonical_tour(z['source_before'],1000);cl=z['initial_checklist'].astype(np.int32)
                    saved_ref=z['refined'].copy()
                with np.load(SOURCE/f'native_candidates_i{i}.npz') as z:
                    nn=np.ascontiguousarray(z['k32'],dtype=np.int32);nn64=np.ascontiguousarray(z['k64'],dtype=np.int32)
                trace=json.loads((folder/f'perturbation_iter{it}.json').read_text())['trace']
                rr,cc,_=reconstruct(source,trace)
                assert np.array_equal(rr,raw) and np.array_equal(cc,cl)
                p1=m0_native.run(data[i],raw.tolist(),cl.tolist(),nn,True);physical_calls+=1
                old=json.loads((M0/(key+'_events.json')).read_text())
                save(key+'_pass1.json',p1)
                assert_equal(p1,old)
                assert np.array_equal(p1['route'],ref)
                if len(saved_ref)==1001:assert np.array_equal(np.r_[p1['route'],p1['route'][0]],saved_ref)
                checks1=validate_trace(data[i],raw,cl,nn,p1)
                tick=time.perf_counter()
                p2=m0_native.run(data[i],p1['route'],p1['checklist'],nn,False);physical_calls+=1
                elapsed=time.perf_counter()-tick
                logged=m0_native.run(data[i],p1['route'],p1['checklist'],nn,True);physical_calls+=1
                save(key+'_trace.json',logged) # complete compact JSON, no event truncation
                assert_equal({k:p2[k] for k in ('route','checklist','total_change_f32')},{k:logged[k] for k in ('route','checklist','total_change_f32')})
                checks2=validate_trace(data[i],np.array(p1['route'],dtype=np.int32),p1['checklist'],nn,logged)
                assert checks1['processed_nodes']+checks2['processed_nodes']<=2000
                assert checks1['candidate_evaluations']+checks2['candidate_evaluations']<=128000
                baseline=length(data[i],p1['route']);after=length(data[i],p2['route']);gain=baseline-after
                assert np.array_equal(np.sort(p2['route']),np.arange(1000))
                import r3_diagnostic
                scan=r3_diagnostic.scan(data[i],np.asarray(p2['route'],dtype=np.int32),nn64,
                    np.full((1,1000),32,dtype=np.int32),np.arange(1000,dtype=np.int32),0)
                r.update(status='PASS',baseline_length_f64=baseline,second_length_f64=after,
                    improvement=gain,relative_improvement=gain/baseline,
                    robust_improvement=bool(gain>max(1e-8,1e-6*baseline)),
                    pass1_processed=checks1['processed_nodes'],pass2_processed=checks2['processed_nodes'],
                    pass1_evaluations=checks1['candidate_evaluations'],pass2_evaluations=checks2['candidate_evaluations'],
                    pass1_accepted=checks1['accepted_moves'],pass2_accepted=checks2['accepted_moves'],
                    pass2_processed_requeue_attempts=checks2['processed_node_reinsertion_attempts'],
                    second_pass_unlogged_seconds=elapsed,extra_evaluations=checks2['candidate_evaluations'],
                    extra_eval_ratio=checks2['candidate_evaluations']/checks1['candidate_evaluations'] if checks1['candidate_evaluations'] else None,
                    total_eval_ratio=(checks1['candidate_evaluations']+checks2['candidate_evaluations'])/checks1['candidate_evaluations'] if checks1['candidate_evaluations'] else None,
                    offline_final_robust_k32=scan['membership_count'][0],
                    offline_final_robust_off_k32=scan['robust_improving_moves']-scan['membership_count'][0],
                    pass1_m0_exact=True,logging_invariance=True,all_moves_validated=True,
                    peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                assert r['peak_process_rss_bytes']<=1000000000
                assert time.perf_counter()-STARTED<=300
            except Exception:
                r['status']='FAILED';r['error']=traceback.format_exc();failure=r['error']
            records.append(r);save(key+'_validation.json',r)
            print(key,r['status'],r.get('improvement'),flush=True)
            if failure:break
    except Exception:
        failure=traceback.format_exc()
    finally:
        signal.alarm(0)
    present={r['case'] for r in records}
    for i,s,it in plans:
        key=f'i{i}_s{s}_t{it}'
        if key not in present:records.append(dict(case=key,instance=i,seed=s,iteration=it,status='NOT_RUN'))
    wall=time.perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    preserved=all(sha256(ROOT/p)==h for p,h in manifest['input_sha256'].items())
    resource_ok=wall<=300 and rss<=1000000000
    correct=not failure and all(r['status']=='PASS' for r in records) and preserved
    signal_instances=sorted({r['instance'] for r in records if r.get('robust_improvement')})
    decision=('M1 RESOURCE-INCONCLUSIVE' if not resource_ok or (failure and ('MemoryError' in failure or 'TimeoutError' in failure))
              else 'M1 INCONCLUSIVE' if not correct else 'M1 EXPLORATORY SIGNAL' if len(signal_instances)>=4 else 'M1 NO SIGNAL')
    save('validation_records.json',records)
    save('test_resource_audit.json',dict(correctness_pass=correct,resource_pass=resource_ok,
        experiment_wall_seconds=wall,peak_process_rss_bytes=rss,rss_measurement='Linux ru_maxrss *1024; process lifetime peak, conservative',
        virtual_memory_enforced_bytes=1000000000,cpu_workers=1,physical_native_invocations=physical_calls,preflight_native_invocations=PREFLIGHT_CALLS,
        sequential_passes_per_treatment_path=2,verification_repeats_from_pass1_state=True,
        preserved_input_hashes=preserved,cases_passed=sum(r['status']=='PASS' for r in records),
        cases_failed=sum(r['status']=='FAILED' for r in records),cases_not_run=sum(r['status']=='NOT_RUN' for r in records),
        failure=failure,decision=decision,signal_instances=signal_instances))
    with (OUT/'paired_metrics.csv').open('x',newline='') as f:
        keys=list(dict.fromkeys(k for r in records for k in r));writer=csv.DictWriter(f,keys,lineterminator='\n');writer.writeheader();writer.writerows(records)
    report(records,decision,signal_instances,wall,rss,correct)
    print(decision,flush=True)

def report(records,decision,signal_instances,wall,rss,correct):
    metrics=['improvement','relative_improvement','pass1_processed','pass2_processed','pass1_evaluations',
             'pass2_evaluations','pass1_accepted','pass2_accepted','pass2_processed_requeue_attempts',
             'extra_eval_ratio','total_eval_ratio','second_pass_unlogged_seconds','offline_final_robust_k32']
    instances=[]
    if correct:
        for i in range(8):
            group=[r for r in records if r['instance']==i]
            item=dict(instance=i,robust_cases=sum(r['robust_improvement'] for r in group))
            for k in metrics:item[k]=statistics.mean(statistics.mean(r[k] for r in group if r['seed']==s) for s in (31415,27182))
            instances.append(item)
    dispersion={k:dict(mean=statistics.mean(r[k] for r in instances),median=statistics.median(r[k] for r in instances),
        minimum=min(r[k] for r in instances),maximum=max(r[k] for r in instances),sample_sd=statistics.stdev(r[k] for r in instances)) for k in metrics} if instances else {}
    save('instance_summary.json',dict(instances=instances,between_instance_dispersion=dispersion,decision=decision,
        signal_instances=signal_instances if correct else None,nesting='phases within seeds within independent instances'))
    passed=sum(r['status']=='PASS' for r in records)
    outcome=(f'{len(signal_instances)}/8 instances contain a robustly improved case; all32 correct.' if correct
             else 'Signal criterion not assessed. Missing cases have no numerical improvement estimate.')
    table='\n'.join(f"| {r['instance']} | {r['robust_cases']} | {r['improvement']:.9f} | {r['relative_improvement']:.9f} | {r['pass2_evaluations']:.2f} | {r['extra_eval_ratio']:.3f} | {r['pass2_accepted']:.2f} | {r['offline_final_robust_k32']:.2f} |" for r in instances)
    recommendation=('Conditionally design a new independently registered equal-compute controlled study on new independent data.' if decision=='M1 EXPLORATORY SIGNAL'
        else 'Stop the SRR investigation on this evidence.' if decision=='M1 NO SIGNAL'
        else 'Stop at this technical failure; retain evidence and do not retry v3.')
    text=f'''# M1 v2 technical amendment and controlled restart

**{decision}**. {passed}/32 principal cases passed. {outcome}
M1 v1 remains INCONCLUSIVE due to a harness error before any second pass;
it was not a negative mechanics finding. Stage A remains NO-GO, secondary
review INCONCLUSIVE, and M0 VALIDATED. These historical artifacts are unchanged.

## Registration, comparator and control-only preflight

The scientific fields of v1 were copied unchanged into the v2 protocol before
any replay. The technical amendment records strict complete JSON serialization
with nonfinite values rejected, focused comparator tests, and an all32 first-pass
preflight. No fields are dropped, converted to strings, or compared approximately.
Sorting dictionary keys affects only representation; event/route/checklist array
order and float values remain exact. Tests detect route/checklist/gain mutations,
event deletion/order/ranks, unexpected fields and nonfinite values, and cover a
complete representative M0 event record with native tuple-shaped route updates.
See comparator_tests.json and preflight_verdict.json for actual gate results.
The principal stage is entered only after all32 preflight cases pass. Every
attempted preflight has a persisted control object and per-case validation.

Principal execution revalidates pass1 against the whole M0 object, then runs one
unchanged K32 pass2 from its final route/checklist. The unlogged pass2 is timed;
the logged verification starts from the same pass1 state. All32 principal records
are retained, including failure/NOT_RUN statuses if any. Independent validate_trace
checks float32 evaluations, selection order, every accepted move, float64 edge/cost
delta, permutation and checklist evolution. The unchanged logged C++ path checks
inverse positions after accepted moves. Correctness flags in the case records
describe only checks actually reached. No solver source, M0 code or binary changed.

## Paired results (instance means)

| Instance | Robust cases /4 | Absolute gain | Relative gain | Extra evaluations | Extra/baseline ratio | Pass2 accepted | Offline K32 residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

paired_metrics.csv retains all32 cases. instance_summary.json averages two phases
within seed, then two seeds within each independent TSP instance, and reports
mean/median/range/sample SD across eight instance means. Moves, seeds and phases
are not independent problem instances. Float64 robust improvement exceeds
max(1e-8,1e-6*baseline length). Native accepted moves use the unchanged positive
float32 threshold and need not exceed that robust threshold. Offline final-tour
K32 residuals use robust canonical float64 2-opt moves and are not accepted events.

## Resource audit and interpretation

Elapsed experiment wall time including comparator tests, preflight, principal
calls, logging verification, checks and trace serialization: {wall:.6f}s.
Peak process RSS: {rss} bytes (Linux ru_maxrss *1024, process-lifetime peak).
One CPU worker and a conservative1GB virtual-memory ceiling were configured.
The audit accounts for all physical calls: 32 control-only preflight calls plus
three principal invocations per completed case (two sequential passes and a
pass2 logging duplicate from identical pass1 state). No third sequential pass
is allowed. Path limits remain2N anchors and4NK evaluations; all work is charged
against the300s wall budget and1GB RSS cap. Unlogged pass2 timing excludes logging
and serialization but includes pybind copies/conversion. It is not a speed benchmark.
See test_resource_audit.json for measured counts and failures. Final report and
delivery hashing are recorded after the experiment timer, outside search timing.

Cursor reset revisits all final-checklist nodes on an evolved tour. Any measured
gain cannot be attributed specifically to duplicate suppression, pruning,
candidate limitations, neural guidance or original-paper claims. These selected
ant0 records are not best-of-100 source trajectories and establish no end-to-end
DyNACO effect. No Stage B, solver-level experiment, expansion, M2 or new research
branch was implemented or run.

**Recommendation: {recommendation}** If a new study is designed, register
new independent instances and paired solver seeds, original K32 control and an
equal-compute revisit arm with common evaluation/time budgets. Freeze final and
anytime best-of-100 quality metrics, resource accounting, instance-level uncertainty
and scientific GO/NO-GO criteria before data. Keep the reserved confirmation set sealed.
'''
    with (OUT/'M1_V2_RESEARCH_REPORT.md').open('x') as f:f.write(text)
    with (OUT/'HANDOFF_SUPPLEMENT.md').open('x') as f:f.write(f'{decision}. {outcome}\n\n{recommendation}\n\nStage A NO-GO; secondary INCONCLUSIVE; M0 VALIDATED; M1v1 INCONCLUSIVE (technical). No Stage B or solver-level authorization. No push.\n')

def execute():
    global STARTED,PREFLIGHT_CALLS
    STARTED=time.perf_counter();PREFLIGHT_CALLS=0
    resource.setrlimit(resource.RLIMIT_AS,(1000000000,1000000000))
    def timeout(*_):raise TimeoutError('300s overall experiment wall budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    pre=[];error=None
    try:
        tests=test_comparator();save('comparator_tests.json',tests)
        assert tests['passed'],'Comparator tests failed'
        manifest=json.loads((OUT/'input_manifest.json').read_text())
        assert all(sha256(ROOT/p)==h for p,h in manifest['input_sha256'].items())
        assert sha256(m0_native.__file__)==manifest['binary_sha256']
        data=np.load(SOURCE/'datasets/tsp1000_seed20261101_float32.npy',allow_pickle=False)
        for i in range(8):
            with np.load(SOURCE/f'native_candidates_i{i}.npz') as c:nn=np.ascontiguousarray(c['k32'],dtype=np.int32)
            for seed in (31415,27182):
                folder=SOURCE/f'i{i}_s{seed}'
                for it in (499,999):
                    key=f'i{i}_s{seed}_t{it}';record=dict(case=key,status='STARTED')
                    try:
                        with np.load(folder/f'raw_before_iter{it}.npz') as z:
                            raw=canonical_tour(z['raw'],1000);saved_ref=z['refined'].copy();ref=canonical_tour(saved_ref,1000)
                            source=canonical_tour(z['source_before'],1000);cl=z['initial_checklist'].astype(np.int32)
                        trace=json.loads((folder/f'perturbation_iter{it}.json').read_text())['trace']
                        rr,cc,_=reconstruct(source,trace)
                        assert np.array_equal(rr,raw) and np.array_equal(cc,cl)
                        output=m0_native.run(data[i],raw.tolist(),cl.tolist(),nn,True);PREFLIGHT_CALLS+=1
                        save('preflight_'+key+'_control.json',output)
                        expected=json.loads((M0/(key+'_events.json')).read_text());assert_equal(output,expected)
                        assert np.array_equal(output['route'],ref)
                        if len(saved_ref)==1001:assert np.array_equal(np.r_[output['route'],output['route'][0]],saved_ref)
                        checks=validate_trace(data[i],raw,cl,nn,output)
                        record.update(status='PASS',complete_event_comparison=True,exact_array_and_closure=True,
                            final_checklist_exact=True,float32_delta_exact=True,independent_checks=checks)
                    except Exception:
                        record.update(status='FAILED',error=traceback.format_exc());pre.append(record);raise
                    pre.append(record);print('PREFLIGHT',key,'PASS',flush=True)
        assert len(pre)==32 and all(r['status']=='PASS' for r in pre)
    except Exception:error=traceback.format_exc()
    save('preflight_records.json',pre)
    save('preflight_verdict.json',dict(passed=error is None,cases_passed=sum(r['status']=='PASS' for r in pre),
        cases_attempted=len(pre),cases_not_run=32-len(pre),native_invocations=PREFLIGHT_CALLS,
        second_pass_invocations=0,error=error,elapsed_seconds=time.perf_counter()-STARTED))
    if error:
        signal.alarm(0)
        records=[dict(case=f'i{i}_s{s}_t{it}',instance=i,seed=s,iteration=it,status='NOT_RUN')
                 for i in range(8) for s in (31415,27182) for it in (499,999)]
        save('validation_records.json',records)
        wall=time.perf_counter()-STARTED;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        decision='M1 RESOURCE-INCONCLUSIVE' if 'TimeoutError' in error or 'MemoryError' in error else 'M1 INCONCLUSIVE'
        save('test_resource_audit.json',dict(decision=decision,principal_started=False,error=error,
            physical_native_invocations=PREFLIGHT_CALLS,experiment_wall_seconds=wall,peak_process_rss_bytes=rss))
        report(records,decision,[],wall,rss,False)
    else:run()
    manifest=json.loads((OUT/'input_manifest.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in manifest['input_sha256'].items())
    save('source_manifest.json',{p:sha256(ROOT/p) for p in ('r3/m1_v2.py','r3/m1_v2_compare.py')})
    print('DELIVERY COMPLETE',flush=True)

if __name__=='__main__':{'freeze':freeze,'execute':execute}[sys.argv[1]]()
