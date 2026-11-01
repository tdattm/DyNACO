"""Read-only diagnosis of the stopped M1 harness; no native replay calls."""
import json
from r3.common import ROOT,sha256

OUT=ROOT/'artifacts/r3/m1_additional_k32/v1'
M0=ROOT/'artifacts/r3/m0_srr_replay'

def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')

def main():
    audit=json.loads((OUT/'test_resource_audit.json').read_text())
    records=json.loads((OUT/'validation_records.json').read_text())
    inputs=json.loads((OUT/'input_manifest.json').read_text())
    assert audit['decision']=='M1 INCONCLUSIVE' and audit['physical_native_invocations']==1
    assert [r['status'] for r in records].count('FAILED')==1
    assert [r['status'] for r in records].count('NOT_RUN')==31
    assert not list(OUT.glob('*_trace.json'))
    assert all(sha256(ROOT/p)==h for p,h in inputs['input_sha256'].items())
    old=json.loads((M0/'i0_s31415_t499_events.json').read_text())
    patches=[e['route_updates'] for e in old['events'] if e.get('route_updates')]
    assert patches and isinstance(patches[0][0],list)
    cpp=(M0/'replayer.cpp').read_text()
    assert 'updates.append(py::make_tuple(z,route[z]))' in cpp
    # This verifies the representation issue without invoking native SRR.
    assert dict(route_updates=[(1,2)]) != json.loads(json.dumps(dict(route_updates=[(1,2)])))
    save('failure_diagnosis.json',dict(status='HARNESS_COMPARISON_ERROR',native_calls_added=0,
        issue='Native route_updates contain tuples; JSON round-trip contains lists. Direct whole-dict equality is not representation invariant.',
        evidence=['M0 replayer.cpp uses py::make_tuple for route_updates',
                  'Saved first-case M0 log has nonempty list-valued updates',
                  'Python tuple/list equality differs after JSON round-trip'],
        actual_pass1_output_persisted=False,
        limitation='Failed assertion preceded explicit saved-tour and checklist/delta checks; actual numerical equality was not established in this run.',
        action='No patch/retry of frozen run; pass2 never called; 31 cases NOT_RUN',
        original_inputs_rehashed=len(inputs['input_sha256']),original_inputs_unchanged=True))
    # Preserve the first auto-generated narrative, whose success-path wording
    # is inappropriate after early termination. Publish an explicit correction.
    for name in ('M1_RESEARCH_REPORT.md','HANDOFF_SUPPLEMENT.md'):
        draft=OUT/(name+'.initial_generated')
        assert not draft.exists()
        (OUT/name).rename(draft)
    report=f'''# M1 one additional K32 pass: stopped technical run

**M1 INCONCLUSIVE.** One control invocation was attempted on
`i0_s31415_t499`. Its harness comparison failed. Pass2 was never executed;
zero cases passed the full correctness gate and 31 cases are NOT_RUN.
There is no measured second-pass effect and no assessed mechanics signal.
An empty signal-instance list must not be interpreted as 0/8 negative results.

## Frozen scope and stopping rule

Protocol SHA256: `{inputs['protocol_sha256']}`. The protocol and source/input
inventory were persisted before any M1 call. The experiment used the validated
M0 binary without rebuilding or editing M0 source or outputs. The first saved
raw tour and initial checklist were reconstructed before the native call.
Execution stopped immediately on the failed assertion, as registered.
No correction, retry, replacement case, alternative variant, second-pass call,
new ACO trajectory or confirmation-data access followed.

## Failure and evidence limits

The assertion compared the entire freshly returned native object directly
against the deserialized M0 JSON object. Native route-update records are Python
tuples (the unchanged C++ code uses py::make_tuple); saved JSON represents
those tuples as lists. Python whole-object equality therefore rejects this
representation difference even when the numeric values could be identical.
The saved M0 log for this case has nonempty route updates. A read-only schema
review and tuple/JSON demonstration establish this harness defect without
additional native calls; see `failure_diagnosis.json`.

The returned pass1 object was not persisted before the assertion. Explicit
saved-array, final-checklist and accumulated-delta checks occur later in the
frozen harness and were not reached. Therefore this M1 run neither confirms
nor refutes numerical replay equality. M0's separately validated 64/64 result
remains unchanged. The failed predicate is retained verbatim with traceback;
the frozen M1 source has not been repaired. A later attempt would require a
new versioned protocol and a representation-safe control comparison before
execution, retaining this failed attempt in the evidence chain.

## Records and resources

`validation_records.json` and `paired_metrics.csv` retain all 32 planned case
identities, with one FAILED and 31 NOT_RUN statuses. There are no pass2 traces,
accepted-move metrics, paired improvements, residual scans or instance effect
estimates because the control gate stopped execution. `instance_summary.json`
accordingly contains no numerical instance summaries. Missing metrics have
not been converted to zero or used to apply the signal criterion.

Elapsed experiment wall time: {audit['experiment_wall_seconds']:.6f} seconds.
Peak process RSS: {audit['peak_process_rss_bytes']} bytes, measured as Linux
process-lifetime ru_maxrss *1024. One CPU worker was configured and a stricter
1,000,000,000-byte virtual-memory ceiling was enforced. One physical native
invocation occurred; no treatment two-pass path completed. Resources were
within limits. Timing covers input checks and the failed call/handling, not a
successful pass2 benchmark. All pre-existing tracked inputs were rehashed and
unchanged. Output/source integrity is pinned separately after this review.

The frozen protocol interprets the two-call limit as two sequential calls per
treatment path; its required logging verification would repeat pass2 from the
same pass1 state, making three physical invocations if reached. This accounting
was disclosed before execution, and no such verification was reached here.

## Decisions and recommendation

Stage A remains **NO-GO**. Secondary review remains **INCONCLUSIVE**. M0 remains
**VALIDATED**. This run supplies no evidence about additional-pass benefit,
suppressed requeues, candidate limitations, pruning, neural guidance or final
best-of-100 solver performance. No Stage B or solver-level study is authorized.

**Recommendation: stop the SRR investigation at this failed M1 run.** The
technical failure must be resolved under a separately versioned registration
before any further mechanics execution. This result does not justify starting
an independently registered solver-level equal-compute study. Such a study
would still require credible mechanics evidence and new independent data.

The initial auto-generated report and handoff are preserved with the suffix
`.initial_generated` for transparency. Their unconditional success-path wording
was inappropriate after early termination; this report and its handoff are
the authoritative descriptions of the stopped run.
'''
    with (OUT/'M1_RESEARCH_REPORT.md').open('x') as f:f.write(report)
    with (OUT/'HANDOFF_SUPPLEMENT.md').open('x') as f:f.write('''# M1 handoff

M1 INCONCLUSIVE: one control call failed a tuple-versus-JSON-list whole-object
comparison; no second pass was run. One FAILED and 31 NOT_RUN records retained.
No paired effect or signal criterion was assessed. No frozen source was patched
and no native retry was made. See M1_RESEARCH_REPORT.md and failure_diagnosis.json.

Recommendation: stop at this failed M1 run; require a new versioned protocol
and corrected representation-safe checks before further mechanics execution.
Stage A NO-GO; secondary INCONCLUSIVE; M0 VALIDATED. No Stage B, solver-level
experiment or push is authorized by this delivery.
''')
    save('source_manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in
        (ROOT/'r3/m1.py',ROOT/'r3/m1_failure_review.py')})
    save('output_manifest.json',{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()})
    print('Read-only failure review complete. No native SRR calls performed.')

if __name__=='__main__':main()
