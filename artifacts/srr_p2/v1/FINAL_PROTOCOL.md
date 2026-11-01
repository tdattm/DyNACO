# Revised M2 prospective protocol — FOR REVIEW, NOT EXECUTED

This draft supersedes no historical result. Approval and a final immutable
registration are required before principal generation or access. No principal
instances were created or inspected by P1. P1 READY is technical/resource
readiness, not scientific GO or R3 Stage B authorization.

## Data, code and deterministic order

After approval only: NumPy Generator(PCG64(202612070201)).random((16,1000,2),
dtype=np.float32), one C-contiguous draw, no rejection/rescaling/replacement.
Pin NumPy/Python versions in source_environment_manifest.json; persist .npy
bytes and SHA256 before any solver run. Use16 independent instances and paired
execution seeds31415,27182,16180. Reserved Stage A confirmation stays sealed.
No principal data is present here. Checkpoint SHA256:
33f7da301e49f74d5b66de2a12c7bd95f247de8c217343c3410bb716a6f22358. Freeze source_environment_manifest.json and binary hashes
from P1 with the final registration; no unapproved source/data change thereafter.

Deterministic order: budget modes iterations then wall; instance0..15; seeds
in the listed order; arm order(0,1) if(instance+seed_index) even, otherwise(1,0).
192 total runs. Before each run utils.set_seed(execution_seed), construct and
discard the explicitly seeded first solver, construct active solver, then set
native seed int(np.random.randint(0,2**63-1)) exactly as utils.infer_instance.
Do not replace this original duplicate-construction/native-seed contract.

## Arms, guidance and budgets

Original K32 versus exactly one additional unchanged K32 SRR call. Both use
100 ants, backup32, min_new_edges12, rho0.1, alpha1,p_best0.05, local search and
checklist extension on, smooth MMAS on, normalized heuristic on, NLS off,
fixed_steps0, dynamic graph guidance, frozen original checkpoint, CUDA float32.
Both use lightweight call/evaluation counters; detailed route-copy telemetry
is off. srr_metrics fields2/3 are unmeasured, not zero accepted-event claims.
Candidate ordering, RNG, neural architecture, pheromone and original numerical
no-op acceptance stay unchanged. Two CPU threads, sequential GPU execution.

Guidance refresh every100 completed iterations; within each block factor
1-mini/99, gamma1,min_gamma0. Use actual native global-best route, not only
the last ant or selected source. P1 runner.py and runner_reference_gate.json
demonstrate original checkpoint/seed/guidance/update semantics on development.

Equal iterations: exactly1000 completed iterations(H10,mini_H100), report
actual time/evaluations. Equal wall time:4.0s search budget, variable iterations
and continuing100-iteration guidance blocks. Stop before starting an iteration
if budget expired; finish an already started iteration. Primary quality uses
only the last completed update timestamp<=4.0s. Persist all attempted times,
global-best tours and overshoot separately; never credit over-budget results.
No eligible update makes that run technically/resource inconclusive, not zero
improvement. Do not claim equal iterations and equal wall time simultaneously.

Model import/CUDA initialization/checkpoint load occur once and are separately
reported. Per-run setup includes seeding, both solver constructions and
coordinate normalization and is excluded equally from search budgets. One
100-iteration development warmup per arm, unmeasured, before principal runs;
report warmup as setup, never principal evidence. Search starts before first
graph forward and includes inference, transfers, sampling, update, CUDA sync,
checkpoint extraction and lightweight counters. Timestamp completed updates
after synchronization; recording overhead is charged through the running
clock. Final serialization and independent float64 evaluation occur afterward.
Compilation is outside search and setup timing, archived in build logs.

## Outcomes, aggregation and decision rules

Co-primary metrics: float64 final global-best length in equal-iteration mode
and float64 last-in-budget global-best length in equal-time mode. Pair relative
gain(control-treatment)/control within instance/seed/mode, average three seeds
within instance. The independent unit is the16 TSP instances. Secondary: actual
native calls/gain evaluations, runtime/setup/memory, completed iterations,
overshoot and anytime global-best at0,0.5,...4s using only earlier completed
updates (initial route at0). Equal-iteration runs additionally retain all1000
completed-update checkpoints. Native accepted/no-op counts are separate offline
diagnostics, not principal light-counter observations.

Report full per-instance values, mean/median/SD/range and10000 paired instance
bootstrap95% percentile intervals, PCG64 seed202612070202, common resample map
across modes. analysis.py implements strict complete-grid aggregation and
rejects missing/duplicate records; its unit fixture has no TSP data.

Draft scientific GO requires BOTH modes: median instance gain>=0.1%, mean
bootstrap lower bound>0 and>=12/16 positive instance means. Otherwise NO-GO
for a complete valid study. Any missing/invalid case, nonfinite value,
unapproved hash drift or correctness failure is INCONCLUSIVE; stop, preserve
all records, no exclusion/replacement or retry with changed settings.
Resource failure is RESOURCE-INCONCLUSIVE. No optional stopping, policy tuning
or selective reporting. Both modes are co-primary; no scientific GO from
equal-iteration advantage alone.

0.1% is a proposed practical relevance threshold, not a validated effect size.
Under an ideal independent continuous symmetric-null sign model,>=12/16 has
one-sided tail2517/65536~3.84%; ties are nonpositive. This is not a formal
combined test calibration: the median/sign/bootstrap conditions are correlated,
bootstrap n16 is small, and the two modes are paired. Report these limitations;
the conjunction supplies a conservative screening rule, not a proof of error
control or generality. Do not treat seeds, moves or iterations as independent.

## Feasibility and resource stops

Retain4.0s: P1 development arms had usable in-budget checkpoints; no quality
outcome was used to select this budget. P1 timing/overhead variation is in
timing_memory_audit.json. Conservative measured-max projection for192 runs:
750.1s including estimated setup/startup; doubling search/setup gives
1459.2s versus2700s aggregate cap (margin1.85x).
This extrapolates ONE observed instance/seed, not guaranteed new-data runtime.
No data shrink or time-budget changes after principal inspection.

Caps: RTX3050,1GiB allocated CUDA,3GiB peak RSS, two CPU threads, one GPU job,
60s search per iteration-budget run,120s setup per run,45min aggregate including
setup,warmup,search and output handling. GPU allocator fraction.24 is enabled;
check allocated/reserved bytes and process RSS at each completed iteration.
Keep outer process watchdogs for setup/aggregate limits. Pin launch environment
and verify source/binary/data hashes before execution. These resource gates
must be wired into the approved192-run orchestration; P1 executes no such job.

The existing runner accepts provided coordinates only; it cannot generate
principal data. Final approval must seal the separate dataset manifest and
192-run dispatch schedule and review orchestration watchdog wiring. This is a
registration prerequisite, not permission to generate data or execute now.

# P2 final registration annex (supersedes draft launch prerequisites only)

This file freezes the retained P1 scientific design for review/registration.
No principal data generation or execution is authorized by this package.
Scientific criteria, data specification, seeds, 192 identities, parameter
settings and budgets above remain unchanged. The principal content hash is
intentionally absent until the second sealing moment below.

1. Protocol freeze precedes data generation. registration.json pins the source
implementation commit, exact source/checkpoint/native binary hashes, dependency
versions, this protocol and dispatch.json. registration.sha256 is the external
root of trust; archive it with the reviewed Git commit. No modification after
freeze without independent review and a new registration before principal data.
2. After explicit launch approval only, externally generate the one PCG64 draw,
save NPY and generator/environment provenance, compute its SHA256, and seal the
manifest before any solver call. No P2 development command contains a generator.
A separate execution approval binds both registration and dataset seal hashes.
No solver run is permitted with missing, malformed or changed sealed data.

Equal-time inference concerns the instrumented implementations. P1 mean paired
measurement overhead was approximately 19.9% control and 32.2% treatment on one
development instance/seed. Do not subtract overhead, extrapolate uninstrumented
quality, or apply post-hoc timing corrections. P2 additionally persists compact
completed-iteration crash journals equally in both arms; their recording cost
is charged prospectively to search. They contain global best states, not native
accepted-event or route-mutation diagnostics. Detailed telemetry remains off.

Search timestamps remain exactly the P1 synchronized completed-update convention.
The timestamp precedes extracting/recording that iteration's state; its recording
work remains on the same running clock and may make the next iteration ineligible.
The completed solver state itself must have timestamp <=4.0s to receive credit.
Post-search checks and serialization count against the aggregate cap, not search.
The final attempted over-budget iteration is retained, never credited. If none
is eligible, stop INCONCLUSIVE. The 100000-iteration loop safety ceiling cannot
silently truncate a wall run: reaching it before 4s is a technical failure.
Initial global best is persisted for anytime checkpoints at time zero. All tours
are implicit closed cycles; the repeated closing node is omitted from arrays.

One persistent sequential worker loads the model once. Two CPU threads, one GPU
workload, unmeasured development warmup100 iterations per arm. Warmup start/end
and durations are explicitly archived. Per-run setup excludes search; parent
watchdog120s bounds initialization/setup/output phases; search cap60s; RSS3GiB
and allocated CUDA1GiB; aggregate2700s includes preflight/setup/warmup/search/
serialization/analysis. CUDA allocator fraction .24 is a preventative allocator
limit, not a replacement for the explicit1GiB observation check. Reserved CUDA
is reported separately and is not an allocated-memory cap. Background contention
is a resource concern, never permission to replace cases or change budgets.

INCONCLUSIVE covers correctness, hashes/environment/data drift, missing/duplicate
records or worker failure without demonstrated resource cause. RESOURCE-INCONCLUSIVE
covers parent deadline/RSS stops, CUDA OOM/allocated cap and search timeouts.
A complete valid192-run grid alone permits scientific GO/NO-GO calculation using
the unchanged co-primary criteria above. No partial scientific aggregation,
selective exclusion, replacement, retries, principal tuning or early GO.
A hard stop preserves all prior output plus the current journal prefix; unknown
in-flight work remains explicitly unknown and cannot support a complete result.

The source commit pinned in registration is the immutable implementation commit.
The subsequent packaging commit contains registration and audit artifacts; this
avoids an impossible self-referential Git/content hash. All executed source bytes
are individually pinned, so later unapproved edits are rejected independently
of commit ancestry. This package changes no production source from reviewed P1.

Historical decisions remain Stage A NO-GO, secondary INCONCLUSIVE, M0 VALIDATED,
M1v1 INCONCLUSIVE, M1v2 EXPLORATORY SIGNAL, M1.5 536/396/140 accepted/no-op/robust
mutation events, P0 READY FOR PROTOCOL REVIEW, P1 READY FOR FINAL PROTOCOL FREEZE.
These decisions do not authorize R3 Stage B or any adaptive candidate expansion.