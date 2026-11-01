# M1 one additional K32 pass: stopped technical run

**M1 INCONCLUSIVE.** One control invocation was attempted on
`i0_s31415_t499`. Its harness comparison failed. Pass2 was never executed;
zero cases passed the full correctness gate and 31 cases are NOT_RUN.
There is no measured second-pass effect and no assessed mechanics signal.
An empty signal-instance list must not be interpreted as 0/8 negative results.

## Frozen scope and stopping rule

Protocol SHA256: `aecb4747ed0611cb5343b7e07a160665e1bb7ade7b43f04648c0f5b7cd94840f`. The protocol and source/input
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

Elapsed experiment wall time: 5.247849 seconds.
Peak process RSS: 144273408 bytes, measured as Linux
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
