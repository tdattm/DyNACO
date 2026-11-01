# R3 final end-to-end — protocol v1

This is a separately user-authorized final study based on reference commit
e3eae0055e61789e75a445d61dfa75238ec7ef27. Historical Stage A remains NO-GO;
its old Stage B gate is neither satisfied nor reinterpreted. No two-pass SRR.

## Frozen question and solver

Does deterministic state-dependent allocation of additional SRR candidates
improve global-best TSP tour quality at equal wall time, beyond both original
K32 and uniform K40? N=1000; 100 ants; original sampling K32 and backup32;
min_new_edges=12, rho=.1, alpha=1, p_best=.05; smooth MMAS, extended checklist,
normalized heuristic, fixed_steps=0, NLS disabled. Original dynamic six edge
features, original frozen checkpoint and strict CUDA loading; no training.
Original duplicate constructor and NumPy-derived active native seed are retained.
Execution seeds 31415 and 27182 reset independently for each arm. Two CPU/OpenMP
threads; TF32 off, deterministic algorithms on, CUBLAS_WORKSPACE_CONFIG=:4096:8.

Every ant has exactly one original SRR pass. Only its candidate enumeration
changes. A: 32 per node (32000 directed slots). B: 40 per node (40000).
C: 750 nodes at 32, 250 at 64 (40000). Native nn_list followed by backup_list
is the authoritative ordered prefix; no new KD query or sort. GNN, pheromone,
heuristics and sampling dimensions remain Nx32. Preserve float32 distance/gain,
strict early breaks and ties, max_diff>0, reversal and checklist insertion.

## Adaptive policy

Before zero-based iterations 0,100,200,... take the current native global-best
tour, including initialized best at zero. In float64 on original float32 coords:
score(u)=max(d(u,pred(u)),d(u,succ(u)))/max(mean(d(u,K32(u))),1e-12).
Descending score, exact ties ascending node index; select exactly 250 nodes.
No RNG draw, oracle, future state, model call, or outcome-based tuning. Formula
matches Stage A geometry, but this new policy uses current global-best and
100-iteration epochs. Local scale is recomputed each epoch and charged to C.

## Development and data seal

Before generating principal coordinates: compare full 1000-iteration ARM A
with unchanged original binary/utils on existing development data; require
exact ant routes/costs, source/best, sparse pheromone, neighbor/heuristic state,
priors and seed. Check normal and parallel-traced paths, setter RNG neutrality,
small collinear/duplicate cases, candidate validity/prefix/slots, one SRR call
per ant, finite float64 tour lengths and resource feasibility in all arms.
Do not alter numerical no-op semantics. Detailed diagnostic replay is untimed.

Generate exactly once with NumPy Generator(PCG64(202612070301)).random(
(16,1000,2),dtype=np.float32). Store C-contiguous .npy, full and per-instance
SHA256, versions, timestamp, source commit/hash map, native binary/checkpoint
hash, environment and hardware. Seal before first run; never regenerate.
Do not generate/inspect reserved historical confirmation seed20261104.

## Complete grid and schedule

16 instances x seeds (31415,27182) x arms A/B/C x modes =192 runs. Run equal
iterations first, wall time second; ascending instance and listed seed order.
For p=2*instance+seed_index rotate ABC, BCA, CAB by p mod3. No data-driven reorder.
Equal iterations: exactly1000. Equal wall:8.0s. Neural forward once every100
iterations; original prior anneals 1 to0 with factor1-(iteration mod100)/99.
If wall mode completes more than1000 iterations, repeat the identical100-step
block schedule; no iteration ceiling. This is fixed before principal outcomes.

## Timing and official outcome

Exclude imports/checkpoint loading equally. Record setup, development warmup,
serialization and total separately. Setup includes original constructors,
normalization, native-seed assignment, shared instrumentation initialization,
initial valid route/candidate validation; it excludes policy-specific allocation.
Warm each arm on existing development-only coordinates, never principal data.
perf_counter search clock starts before policy setter/score and first graph.
Include allocation, graph/neural, transfers, sampling/SRR, update, CUDA sync,
online counters and checkpoint copying/recording. Same instrumentation all arms.
Counters use thread-local slots; no Python callback in native ant loops.
Before starting each iteration check deadline. Complete any started iteration,
retain overshoot, synchronize CUDA, charge checkpoint recording, timestamp.
No over-budget improvement enters primary outcome. Keep initial route at t=0
and every global-best change, and every completed-update timestamp. Choose the
latest global-best record with timestamp<=8, not the best overshoot. At least
one completed update<=8 is required. Anytime points 0,1,2,4,6,8 use the same rule.
Native global-best selection remains float32; independent float64 quality can
occasionally increase due to roundoff. Do not replace it by a float64-selected best.
Independent NumPy float64 cyclic length on valid permutations is official.

## Paired analysis and frozen decision

Per instance/seed/mode gain_C_X=(L_X-L_C)/L_X. Mean over two seeds within
each of16 independent instances. Primary decision uses wall mode only.
For BOTH C-A and C-B require: median instance gain>=.001; >=12/16 positive
instance means; lower bound of two-sided95% percentile bootstrap CI of mean>0.
10000 instance-level resamples, size16 with replacement, PCG64(202612070302).
Generate the10000x16 index matrix once and reuse for both comparisons (and
secondary iteration summaries). NumPy percentile [2.5,97.5], linear method.
Complete valid192 grid and both primary comparisons pass => scientific GO;
otherwise complete valid grid => scientific NO-GO. Secondary equal-iteration
findings never change primary criteria. These correlated practical screening
criteria do not claim calibrated familywise error or arbitrary-distribution power.

Report all individual instances, means/medians, CI, signs, extrema, completed
iterations, runtime/iteration, native candidate visits (including early-break
boundary), actual gain evaluations (after pruning), calls, expanded ranks,
accepted native moves, slots/unique undirected edges, selection/score changes.
Accepted float32 counts do not prove robust float64 improvement. Untimed
development replay verifies per-move float64 effects separately, with robustness
threshold max(1e-8,1e-6*pre-move tour length). No causal credit from counters alone.

## Prospective failure/resource policy

One solver/GPU worker, sequential identities, supervisor holds an exclusive
lock. Each attempt in a new directory with stdout/stderr, started/completed
atomic manifests, hashes and identity. Never silently reuse corrupt outputs.
Before every run verify dataset, protocol, scientific source and binary hashes.
Disk reserve1GiB on D: (far above measured compact output growth); process RSS
ceiling4GiB, torch allocated ceiling3.5GiB. Supervisor monitors free space/RSS
and progress; watchdog300s per identity including process startup. No aggregate
45-minute cap or obsolete M2 admission bound. No storage stress tests/deletions.
Record nvidia-smi utilization/memory/temperature outside timed loop each run.
No subjective exclusion based on OS performance or solution quality.

At most one retry per identity, only for external interruption, watchdog, process
termination, CUDA out-of-memory or I/O infrastructure failure; preserve attempts.
No retry for quality, assertion/permutation/finite-state/scientific validation
failure. No outcome inspection drives a retry. Resume only hash-valid completed
runs in schedule. No policy/data/seed/budget/threshold edits after start. If full
comparison cannot be completed defensibly: NO-GO FOR FURTHER INVESTMENT IN THIS
ATTEMPT, SCIENTIFIC CONCLUSION NOT ESTABLISHED; never algorithmic inferiority.

## Provenance and final reporting

Commit frozen implementation/protocol before data, then commit seal manifests
before execution. Preserve exact executed source commit and byte hashes, binary
and checkpoint. Keep raw records compact and archive all192 identities. Publish
Vietnamese academic report, complete analysis/figures and investment decision.
Historical Stage A, M1/M2 and numerical no-op evidence are separate experiments.
