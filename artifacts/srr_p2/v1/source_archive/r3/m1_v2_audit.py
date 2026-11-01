"""Independent delivery audit of saved results only; no native calls."""
import csv
import json
import statistics
import time
import numpy as np
from r3.common import ROOT,sha256
from r3.diagnostics import length

OUT=ROOT/'artifacts/r3/m1_additional_k32/v2'

def save(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main():
    start=time.perf_counter()
    manifest=json.loads((OUT/'input_manifest.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in manifest['input_sha256'].items())
    assert sha256(OUT/'protocol.json')==manifest['protocol_sha256']
    assert sha256(manifest['binary_path'])==manifest['binary_sha256']
    v1=json.loads((OUT.parent/'v1/protocol.json').read_text())
    v2=json.loads((OUT/'protocol.json').read_text())
    for k,v in v1.items():
        if k!='version':assert v2[k]==v,(k,'scientific field changed')
    pre=json.loads((OUT/'preflight_verdict.json').read_text())
    tests=json.loads((OUT/'comparator_tests.json').read_text())
    audit=json.loads((OUT/'test_resource_audit.json').read_text())
    records=json.loads((OUT/'validation_records.json').read_text())
    assert tests['passed'] and tests['tests']==10 and tests['native_invocations']==0
    assert pre['passed'] and pre['cases_passed']==32 and pre['second_pass_invocations']==0
    assert len(records)==32 and all(r['status']=='PASS' for r in records)
    assert audit['correctness_pass'] and audit['resource_pass'] and audit['physical_native_invocations']==128
    data=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy')
    instances=json.loads((OUT/'instance_summary.json').read_text())['instances']
    for r in records:
        key=r['case'];p1=json.loads((OUT/(key+'_pass1.json')).read_text());p2=json.loads((OUT/(key+'_trace.json')).read_text())
        a=length(data[r['instance']],p1['route']);b=length(data[r['instance']],p2['route'])
        assert a==r['baseline_length_f64'] and b==r['second_length_f64']
        assert a-b==r['improvement'] and ((a-b)>max(1e-8,1e-6*a))==r['robust_improvement']
        assert r['pass2_accepted']==sum(e['selected_gain_f32']>0 for e in p2['events'])
        assert r['pass2_evaluations']==sum(len(e['evaluations']) for e in p2['events'])
        assert r['pass1_processed']+r['pass2_processed']<=2000
        assert r['pass1_evaluations']+r['pass2_evaluations']<=128000
    for item in instances:
        rows=[r for r in records if r['instance']==item['instance']]
        for k in ('improvement','relative_improvement','extra_eval_ratio','pass2_accepted','offline_final_robust_k32'):
            assert abs(item[k]-statistics.mean(r[k] for r in rows))<1e-12
    with (OUT/'paired_metrics.csv').open() as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==32 and {r['case'] for r in csvrows}=={r['case'] for r in records}
    result=dict(status='PASS',rehashed_historical_inputs=len(manifest['input_sha256']),native_calls_added=0,
        unchanged_v1_scientific_fields=True,case_lengths_independently_recomputed=32,
        robust_cases=sum(r['robust_improvement'] for r in records),
        signal_instances=len(audit['signal_instances']),pass2_accepted_moves=sum(r['pass2_accepted'] for r in records),
        extra_evaluations=sum(r['pass2_evaluations'] for r in records),
        instance_mean_relative_improvement=statistics.mean(r['relative_improvement'] for r in instances),
        instance_mean_extra_evaluation_ratio=statistics.mean(r['extra_eval_ratio'] for r in instances),
        instance_mean_offline_k32_residual=statistics.mean(r['offline_final_robust_k32'] for r in instances),
        delivery_audit_seconds=time.perf_counter()-start,decision=audit['decision'])
    save('delivery_audit.json',result)
    save('delivery_source_manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'r3').glob('m1_v2*.py')})
    save('output_manifest.json',{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
