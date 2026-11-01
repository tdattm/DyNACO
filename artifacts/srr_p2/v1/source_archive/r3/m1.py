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
from r3.replay import reconstruct
from r3.diagnostics import canonical_tour,length

OUT=ROOT/'artifacts/r3/m1_additional_k32/v1'
M0=ROOT/'artifacts/r3/m0_srr_replay'
SOURCE=ROOT/'artifacts/r3/experiments/stage_a_v1_01'

def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def freeze():
    assert git('branch','--show-current')=='feature/r3-budget-aware-candidate-graph'
    assert git('rev-parse','HEAD')=='35c33701291b77d9300d91921f649be17688f91a'
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=dict(version=1,question='Exactly one additional unchanged K32 SRR call from pass1 final route/checklist',
        instances=list(range(8)),seeds=[31415,27182],iterations=[499,999],cases=32,
        control='Original single K32 SRR pass, exact saved refined ant0 and M0 route/checklist/delta match',
        intervention='Original identical routine once more; cursor reset, final checklist retained',
        robust_improvement='float64 baseline length - second-pass length > max(1e-8,1e-6*baseline length)',
        signal='All32 correct and at least one robust case in at least4 of8 instances',
        unit='Independent saved instance; phases within seed, seeds within instance; exploratory only',
        resources=dict(cpu_workers=1,wall_seconds=300,rss_bytes=1000000000,
                       processed_anchors_per_two_pass_path=2000,candidate_evaluations_per_two_pass_path=128000),
        call_accounting='Two sequential calls per treatment trajectory. Required logging verification repeats pass2 from identical pass1 state, not from pass2 output. Three physical invocations per case, two-pass path bounds; all verification time/RSS included in resource audit. No third sequential pass.',
        timing='perf_counter around unlogged native pass2 call only; includes pybind copies/conversion, excludes logging/independent checks/serialization',
        residual='M0 offline scanner, all anchors on final pass2 frozen tour; robust float64 threshold; K32 membership',
        failure='Stop on first correctness/resource failure; retain failing case and mark all remaining NOT_RUN; no replacements',
        restrictions=['No new instances/ACO/model/confirmation data','No solver or M0 edits','No other variants','No push'],
        historical=dict(stage_a='NO-GO',secondary='INCONCLUSIVE',m0='VALIDATED'))
    save('protocol.json',protocol)
    # Inventory all tracked files before M1: preservation covers R1, source, checkpoint and historical outputs.
    tracked=git('ls-files').splitlines()
    inputs={name:sha256(ROOT/name) for name in tracked if (ROOT/name).is_file()}
    inputs['r3/m1.py']=sha256(ROOT/'r3/m1.py')
    manifest=dict(head=git('rev-parse','HEAD'),input_sha256=inputs,
        protocol_sha256=sha256(OUT/'protocol.json'),binary_sha256=sha256(m0_native.__file__),
        binary_path=m0_native.__file__,freeze_time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    original=json.loads((M0/'manifest.json').read_text())
    assert manifest['binary_sha256']==original['binary_sha256']
    assert all(sha256(ROOT/p)==h for p,h in original['input_sha256'].items())
    save('input_manifest.json',manifest)
    print('FROZEN',manifest['protocol_sha256'],flush=True)

def run():
    start=time.perf_counter()
    manifest=json.loads((OUT/'input_manifest.json').read_text())
    assert sha256(OUT/'protocol.json')==manifest['protocol_sha256']
    assert all(sha256(ROOT/p)==h for p,h in manifest['input_sha256'].items())
    assert sha256(m0_native.__file__)==manifest['binary_sha256']
    original=json.loads((M0/'manifest.json').read_text())
    assert manifest['binary_sha256']==original['binary_sha256']
    # Enforce a stricter virtual-memory ceiling as well as measuring peak RSS.
    resource.setrlimit(resource.RLIMIT_AS,(1000000000,1000000000))
    def timeout(*_):raise TimeoutError('300s experiment wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    records=[];failure=None;begun=0;physical_calls=0
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
                assert p1==old,'pass1 differs from validated M0 events/route/checklist/delta'
                assert np.array_equal(p1['route'],ref)
                if len(saved_ref)==1001:assert np.array_equal(np.r_[p1['route'],p1['route'][0]],saved_ref)
                checks1=validate_trace(data[i],raw,cl,nn,p1)
                tick=time.perf_counter()
                p2=m0_native.run(data[i],p1['route'],p1['checklist'],nn,False);physical_calls+=1
                elapsed=time.perf_counter()-tick
                logged=m0_native.run(data[i],p1['route'],p1['checklist'],nn,True);physical_calls+=1
                save(key+'_trace.json',logged) # complete compact JSON, no event truncation
                assert all(p2[k]==logged[k] for k in ('route','checklist','total_change_f32'))
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
        virtual_memory_enforced_bytes=1000000000,cpu_workers=1,physical_native_invocations=physical_calls,
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
        'second_pass_unlogged_seconds','extra_eval_ratio','offline_final_robust_k32']
    instances=[]
    if correct:
        for i in range(8):
            row=dict(instance=i,robust_cases=sum(r['robust_improvement'] for r in records if r['instance']==i))
            for k in metrics:
                row[k]=statistics.mean(statistics.mean(r[k] for r in records if r['instance']==i and r['seed']==s)
                                       for s in (31415,27182))
            instances.append(row)
    dispersion={k:dict(mean=statistics.mean(r[k] for r in instances),median=statistics.median(r[k] for r in instances),
                      minimum=min(r[k] for r in instances),maximum=max(r[k] for r in instances),
                      sample_sd=statistics.stdev(r[k] for r in instances)) for k in metrics} if instances else {}
    save('instance_summary.json',dict(instances=instances,between_instance_dispersion=dispersion,
        nesting='Two phases averaged within seed, two seeds averaged within independent instance',
        decision=decision,signal_instances=signal_instances))
    table='\n'.join(f"| {r['instance']} | {r['robust_cases']} | {r['improvement']:.8f} | {r['relative_improvement']:.8f} | {r['pass1_evaluations']:.2f} | {r['pass2_evaluations']:.2f} | {r['pass2_accepted']:.2f} | {r['offline_final_robust_k32']:.2f} |" for r in instances)
    recommendation=('Design a new independently registered equal-compute controlled study.' if decision=='M1 EXPLORATORY SIGNAL'
                    else 'Stop the SRR investigation on these records.' if decision=='M1 NO SIGNAL'
                    else 'Stop further SRR experiments until the recorded technical/resource failure is resolved.')
    text=f'''# M1 one additional unchanged K32 SRR pass

**{decision}**. Robust improvements occurred in {len(signal_instances)}/8 instances
({signal_instances}); the frozen criterion requires at least four and all 32 cases correct.
Stage A remains NO-GO; secondary review remains INCONCLUSIVE; M0 remains VALIDATED.
This is an exploratory mechanics result on previously observed instances, not a scientific deployment gate.

## Protocol and execution

The protocol and its SHA256 were written before any M1 native call. Input/source
hashes cover all previously tracked repository files, including all M0 evidence,
baseline, R1, checkpoint and historical reports. The M0 binary and source hashes
were checked before reuse; no compilation or change to native methods was made.
All 32 original raw routes/checklists were reconstructed, then pass1 was compared
to the saved refined array (including closure) and full M0 event object, final
checklist and accumulated float32 delta. Pass2 starts only from that exact final
route/checklist and invokes the same K32 routine. No data, model or ACO generation,
confirmation-data access, alternative candidate policy or third sequential pass occurred.

Two sequential calls define the treatment path. The required logging-invariance
check repeats pass2 from the same pass1 state, making three physical invocations
per case. This verification duplicate never uses pass2 output as its input.
This accounting was explicitly frozen in the protocol; trajectory anchor and
evaluation limits apply to the two-pass path, while all verification work counts
toward experiment wall time and memory. This resolves the simultaneous requests
for two treatment passes and both logging modes without an extra trajectory pass.

The M0 independent validator checks every evaluation/selection, accepted move,
float32 gain, float64 edge/cost delta, permutation and checklist evolution.
The unchanged logged C++ path checks inverse positions after each accepted move.
Exact route, final checklist and accumulated float32 delta match across pass2
logging modes. Per-case validation files retain all failures or NOT_RUN states;
the run stops on the first failure. No case is silently dropped.

## Paired descriptive results

| Instance | Robust cases /4 | Mean absolute gain | Mean relative gain | Pass1 evaluations | Pass2 evaluations | Pass2 accepted | Offline K32 residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

`paired_metrics.csv` retains every saved case. `instance_summary.json` averages
phases within seed, then seeds within independent instance, and reports mean,
median, range and sample SD across eight instance summaries. All counts are
events or frozen-tour opportunities as named, not independent TSP samples.
Pass2 native accepted moves may include gains below the robust threshold.
Residual K32 opportunities are canonical float64 improving moves from the offline
final-tour scanner, using max(1e-8,1e-6*final tour length). They are not accepted
native moves, and they do not guarantee improvement under another search path.

Experiment wall time including checks/serialization/scanning: {wall:.6f}s.
Peak process RSS: {rss} bytes (Linux process lifetime ru_maxrss; conservative).
One CPU worker was used; a stricter 1GB virtual-memory ceiling was enforced.
Every two-pass path is checked against 2N anchors and 4NK evaluations.
Per-case unlogged pass2 elapsed time excludes logging and artifact serialization,
but includes Python/native boundary copies and result conversion. It is a single
small timing observation, not a reliable speed benchmark. Audit and hash scans
preceding the run and report/manifest writing following it are outside this timer.

## Interpretation and recommendation

Resetting the cursor revisits all nodes in the final checklist on an evolved
tour. This intervention cannot isolate suppressed requeues, pruning, candidate
membership, neural guidance, or any original-paper claim. These selected ant0
records share histories within instance; refined ant0 is not best-of-100 source.
No solver-level improvement or Stage B authorization follows.

**Recommendation: {recommendation}**
If a new study is designed, register new independent TSP instances, paired
solver seeds, original K32 control and an equal-compute revisit arm; fix a
shared evaluation/time budget and anytime/final best-of-100 quality metrics
before data. Measure actual native evaluations, accepted moves, runtime and
memory, and define instance-level uncertainty and prospective GO/NO-GO criteria.
Keep the reserved Stage A confirmation dataset sealed. No such study is
implemented or run here.
'''
    with (OUT/'M1_RESEARCH_REPORT.md').open('x') as f:f.write(text)
    with (OUT/'HANDOFF_SUPPLEMENT.md').open('x') as f:f.write(f'# M1 handoff\n\n{decision}. {len(signal_instances)}/8 instances meet the robust-case condition.\n\n{recommendation}\n\nStage A NO-GO; secondary INCONCLUSIVE; M0 VALIDATED. No Stage B or solver-level experiment authorized. No push.\n')

if __name__=='__main__':
    {'freeze':freeze,'run':run}[sys.argv[1]]()
