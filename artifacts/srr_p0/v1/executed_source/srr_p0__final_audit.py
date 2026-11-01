"""Independent saved-evidence and source-boundary audit; no solver calls."""
import json,statistics,hashlib
from pathlib import Path
import numpy as np
from r3.common import ROOT,sha256,git
OUT=ROOT/'artifacts/srr_p0/v1'

def main():
    allowed={'faco.py','src/binding.cpp','src/mfaco_train.cpp','src/mfaco_train.h'}
    old=json.loads((OUT/'historical_hashes.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in old.items() if p not in allowed)
    # Method bodies affecting arithmetic/mutation must remain byte-for-byte.
    baseline=Path('/mnt/d/GitRepository/DyNACO-r3')
    def method(text,name):
        start=text.index(name);brace=text.index('{',start);end=brace+1;depth=1
        while depth:
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        return text[start:end]
    original=(baseline/'src/mfaco_train.cpp').read_text();current=(ROOT/'src/mfaco_train.cpp').read_text()
    for name in ('MFACO_TSP::flip_route_section(', 'MFACO_TSP::get_succ(', 'MFACO_TSP::get_pred(',
                 'MFACO_TSP::select_next_node(', 'MFACO_TSP::relocate_node(', 'MFACO_TSP::build_nn_lists(',
                 'MFACO_TSP::compute_probmat(', 'MFACO_TSP::update_pheromone('):
        assert method(original,name)==method(current,name),name
    cases=json.loads((OUT/'saved_case_gate.json').read_text());assert len(cases)==32
    folder=OUT/'verification_v3'
    gate=json.loads((folder/'correctness_gate.json').read_text());assert gate['status']=='PASS'
    recorded=json.loads((OUT/'m15_events.json').read_text())
    assert sum(e['empty_patch'] for e in recorded)==396 and len(recorded)==536
    assert sum(e['robust_reduction'] for e in recorded)==140
    # Independently compare the archived worker arrays, not just gate flags.
    with np.load(folder/'frozen_baseline.npz') as a,np.load(folder/'disabled.npz') as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    with np.load(folder/'enabled.npz') as a,np.load(folder/'enabled_measured.npz') as b,np.load(folder/'enabled_repeat.npz') as c:
        assert all(np.array_equal(a[k],b[k]) and np.array_equal(a[k],c[k]) for k in a.files)
    resources={tag:json.loads((folder/(tag+'.json')).read_text()) for tag in
               ('frozen_baseline','disabled','enabled','enabled_measured','enabled_repeat')}
    assert all(r['peak_rss_bytes']<3*1024**3 and r['peak_cuda_allocated']<512*1024**2 for r in resources.values())
    stats=json.loads((OUT/'rng_consumption.json').read_text());assert len(stats)==3
    audit=dict(status='PASS',decision='M2-P0 READY FOR PROTOCOL REVIEW',
        historical_files_unchanged=len(old)-len(allowed),production_changes=sorted(allowed),
        source_semantics_methods_verified=9,saved_cases_exact=32,disabled_array_archive_exact=True,
        measured_array_archive_exact=True,resources=resources,
        prior_harness_failures='Two worker API-name errors preserved in original and verification_v2 outputs. Correct binding is _update_pheromone_from_flat. Neither was a measured regression mismatch.',
        principal_results_collected=False)
    (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2))
    report=f'''# M2-P0 integration and readiness

**M2-P0 READY FOR PROTOCOL REVIEW**. This permits protocol review only.
M2 principal, M2 scientific GO and R3 Stage B are not authorized or executed.

M1.5 independently confirms536 accepted events:396 empty patches,140 nonempty
patches and140 robust float64 reductions. All396 no-ops have structurally
degenerate endpoints and tiny positive native float32 gains; their actual
float64 length changes are zero.73 extend the observed checklist despite no
route change. Event-level endpoint/gain/insertion records remain available;
this descriptive audit does not assign a causal contribution to final quality.

## Localized implementation

Branch codex/srr-two-pass-p0, separate D:/GitRepository/DyNACO-srr-p0 worktree,
based on e3eae0055e61789e75a445d61dfa75238ec7ef27. Original and R3 checkouts
and all old artifacts are preserved. The original checkout's existing untracked
.r1-env remains untouched. Four production files change: Python wrapper adds
explicit extra_srr_pass/measure_srr flags default false; native binding offers
configure_srr and metrics; solver adds a common scheduler and passive atomic
counts. Both fast and traced non-NLS paths use that same scheduler: original
call, then exactly one extra call only when enabled. Original final route,
positions and checklist are reused; cursor resets inside the unchanged call.
Enabled NLS or non-K32 configurations are rejected rather than silently changing
their algorithms. NLS behavior remains unchanged with the feature disabled.

No native 2-opt correction is made. Float32 distance/gain expressions, strict
ties, pruning, flip logic and checklist duplicate suppression retain original
semantics. Instrumentation adds counters and, only when enabled, route-copy
mutation detection; atomics avoid races in normal parallel sampling. Counters
count native calls/evaluations/accepted events/actual mutations separately.
Measurement time includes this overhead. A test-only macro exposes direct
saved-route scheduling with inverse-position validation; standard builds do
not expose that replay interface. Revert the four-file diff to remove integration.

## Executed correctness and pilot

32/32 saved raw/initial-checklist inputs reproduce exact saved pass1 and pass2
route/checklist arrays through the shared native scheduling interface. The K32
graph is copied exactly from saved native lists. Pass/candidate-evaluation,
accepted and mutation counts match M0/M1 logs; instrumentation on/off agrees.
These test-only inputs supply states that ordinary sampling bindings do not
return; ordinary end-to-end checklist equality is not claimed independently.

Frozen-baseline versus disabled integration compares every persisted cost/tour,
source/best, pheromone, heuristic, candidates, raw traced arrays, log probabilities,
saved guidance input, actual neural graph tensors and frozen-checkpoint CUDA
inference output. Each sampling mode uses its own frozen RNG contract. Normal
parallel equals traced parallel. Serial traced is compared to its own baseline,
since its RNG initialization is different by design. Enabled versus disabled
consumption is additionally checked by synchronizing source/pheromone/candidate
states after a sample without reseeding; subsequent samples match on three paths.

Only after saved-state and disabled gates passed, the enabled development smoke
pilot used one saved instance,10 ants and3 iterations on each of three paths.
Instrumentation and deterministic repeat match all archived arrays. This is a
small functional pilot, not an end-to-end quality estimate or a performance
benchmark. Correctness checks include permutation/closure and independent
float64 cost with2e-5 relative tolerance. Resources are listed below; GPU runs
are sequential, CPU threads2, worker timeout90s, RSS cap3GiB and CUDA cap512MiB.

{json.dumps(resources,indent=2)}

Two preliminary worker attempts used wrong Python names for the existing
pheromone-update API. Their complete failure logs and saved-case gates remain
preserved. A separate verification_v3 namespace uses the exact binding name;
no numerical regression was bypassed or tolerance relaxed. Native arithmetic
and policy were not changed in response to any pilot quality observation.

## Limits and next step

The regression and pilot are bounded development evidence. They do not prove
equivalence for every possible input, full192-run feasibility or scientific
benefit. Atomic/mutation telemetry imposes overhead; the draft protocol calls
for review of the final principal measurement mode. Equal-iteration and
equal-wall-time outcomes are distinct; actual counts/times must be reported.
M2_DRAFT_PROTOCOL.md specifies new independent data, paired seeds, fixed
checkpoint, budget definitions, instance-level uncertainty, failure handling
and prospective decision criteria. No principal dataset was generated.

Recommendation: review and independently register an equal-compute scheduling
study only after resolving the stated measurement/feasibility prerequisites.
Stage A NO-GO; secondary INCONCLUSIVE; M0 VALIDATED; M1v1 technical INCONCLUSIVE;
M1v2 EXPLORATORY SIGNAL remain unchanged. No candidate expansion, training,
targeted reactivation, M2 principal or Stage B occurred. No push or merge.
'''
    (OUT/'M2_P0_REPORT.md').write_text(report)
    (OUT/'HANDOFF.md').write_text('M2-P0 READY FOR PROTOCOL REVIEW only. See M15_REPORT.md, M2_P0_REPORT.md, independent_audit.json and M2_DRAFT_PROTOCOL.md. Historical decisions unchanged; principal not run; no push. Review measurement overhead and full-budget feasibility before independent registration.\n')
    paths=[p for p in (ROOT/'srr_p0').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    paths += [ROOT/p for p in allowed]
    (OUT/'source_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha256(p) for p in paths},indent=2))
    binaries=[* (ROOT/'.srr-build/native').glob('*.so'),*(ROOT/'.srr-build/test').glob('*.so'),
              *Path('/mnt/d/GitRepository/DyNACO-r3/.r3-build/native').glob('faco_opt*.so')]
    (OUT/'binary_manifest.json').write_text(json.dumps({str(p):sha256(p) for p in binaries},indent=2))
    (OUT/'output_manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='output_manifest.json'},indent=2))
    print(audit['decision'])
if __name__=='__main__':main()
