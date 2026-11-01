# M2-P0 integration and readiness

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

{
  "frozen_baseline": {
    "status": "PASS",
    "seconds": 57.87640699899998,
    "peak_rss_bytes": 1369866240,
    "ants": 10,
    "iterations": 3,
    "paths": [
      "normal parallel",
      "traced parallel",
      "traced serial"
    ],
    "extra": false,
    "measure": false,
    "binary": "/mnt/d/GitRepository/DyNACO-r3/.r3-build/native/faco_opt.cpython-313-x86_64-linux-gnu.so",
    "peak_cuda_allocated": 128539136
  },
  "disabled": {
    "status": "PASS",
    "seconds": 43.04986928299999,
    "peak_rss_bytes": 1349898240,
    "ants": 10,
    "iterations": 3,
    "paths": [
      "normal parallel",
      "traced parallel",
      "traced serial"
    ],
    "extra": false,
    "measure": false,
    "binary": "/mnt/d/GitRepository/DyNACO-srr-p0/.srr-build/test/faco_opt.cpython-313-x86_64-linux-gnu.so",
    "peak_cuda_allocated": 128539136
  },
  "enabled": {
    "status": "PASS",
    "seconds": 35.96991673499997,
    "peak_rss_bytes": 1348804608,
    "ants": 10,
    "iterations": 3,
    "paths": [
      "normal parallel",
      "traced parallel",
      "traced serial"
    ],
    "extra": true,
    "measure": false,
    "binary": "/mnt/d/GitRepository/DyNACO-srr-p0/.srr-build/test/faco_opt.cpython-313-x86_64-linux-gnu.so",
    "peak_cuda_allocated": 128539136
  },
  "enabled_measured": {
    "status": "PASS",
    "seconds": 34.568396500000006,
    "peak_rss_bytes": 1352364032,
    "ants": 10,
    "iterations": 3,
    "paths": [
      "normal parallel",
      "traced parallel",
      "traced serial"
    ],
    "extra": true,
    "measure": true,
    "binary": "/mnt/d/GitRepository/DyNACO-srr-p0/.srr-build/test/faco_opt.cpython-313-x86_64-linux-gnu.so",
    "peak_cuda_allocated": 128539136
  },
  "enabled_repeat": {
    "status": "PASS",
    "seconds": 36.92622408799997,
    "peak_rss_bytes": 1350049792,
    "ants": 10,
    "iterations": 3,
    "paths": [
      "normal parallel",
      "traced parallel",
      "traced serial"
    ],
    "extra": true,
    "measure": false,
    "binary": "/mnt/d/GitRepository/DyNACO-srr-p0/.srr-build/test/faco_opt.cpython-313-x86_64-linux-gnu.so",
    "peak_cuda_allocated": 128539136
  }
}

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
