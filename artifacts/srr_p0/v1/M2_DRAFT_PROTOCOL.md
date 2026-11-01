# Draft M2 prospective protocol — NOT REGISTERED, NOT EXECUTED

Scientific question: does exactly one additional unchanged K32 SRR call improve
end-to-end DyNACO best-of-100 tour quality under matched computational budget?
This is an SRR scheduling study. Stage A stays NO-GO; all historical decisions
retain their meanings. M1 ant0 evidence supplies no scientific GO by itself.

## Registration and data separation

After external protocol review, freeze this document, code/binary SHA256,
NumPy/Python/CUDA/compiler versions and all settings before generating data.
No principal data or results are collected in M2-P0. Principal data will be
16 independent uniform Euclidean TSP1K instances from
`Generator(PCG64(202612070201)).random((16,1000,2), dtype=np.float32)` in one call,
C-contiguous, no rescaling or rejection. Persist exact .npy bytes and hashes.
Paired solver execution seeds: 31415, 27182, 16180. Each instance, seed, budget
mode and arm is run once; no seed search, replacements or selective exclusion.
The reserved Stage A confirmation set is excluded and stays sealed. P0 uses
only already persisted Stage A development coordinates and guidance, plus the
existing checkpoint. P0 quality is never used to tune these criteria.

## Fixed arms and settings

Control: original single K32 SRR. Treatment: original first call followed once
by the unchanged second call with final checklist retained. K32/backup32,
100 ants, min_new_edges12, rho0.1, alpha1, p_best0.05, local search/extension on,
smooth MMAS on, normalized heuristic on, NLS off, checkpoint
`tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt`
(SHA256 `33f7da301e49f74d5b66de2a12c7bd95f247de8c217343c3410bb716a6f22358`).
Keep original guidance architecture, graph construction, RNG and annealing
semantics. State solver-native seed derivation explicitly in the future runner
and validate paired identical initial states before any principal execution.
No architecture training, candidate changes, no-op repair or targeted requeue.

## Separate budget evaluations

1. Equal iterations: 1000 completed iterations, H10/mini_H100, same ants,
   paired seeds, coordinates, checkpoint, guidance refresh/annealing schedule.
   Report actual search wall time and native evaluations separately; time is
   not claimed equal.
2. Equal wall time: 4.0 seconds search budget per arm with iterations allowed
   to differ. Refresh guidance every100 completed iterations and use the same
   within-block annealing as the equal-iteration arm. Before starting each
   iteration, synchronize CUDA and check remaining budget; complete an iteration
   already begun. Report actual overshoot, completed iterations and quality both
   at the last fully completed iteration at or before4.0s and at termination.
   Primary wall-time outcome uses only the former checkpoint; never credit an
   over-budget iteration. No returned result is interpolated from a future tour.

Setup includes checkpoint load, model/device initialization, candidate build,
solver initialization and deterministic seed setup; measure/report separately
and exclude equally from search budgets. Search starts immediately before the
first graph/guidance computation and includes inference, CPU sampling,
pheromone/source update, CPU/GPU transfer, synchronization and measurement.
Stop timer after synchronized final update. One fixed development warmup before
all principal runs, excluded equally and logged. Counter instrumentation runs
identically in both arms and is charged to wall time; report its P0 overhead
without using it to adjust budgets. Do not run expensive route-copy mutation
telemetry in principal search; register the lighter evaluation/call-only
measurement mode and demonstrate disabled/measurement invariance before review.
That lighter mode is a future prerequisite, not implemented by this draft.

## Measurement and analysis

Primary outcome for each budget mode: independent float64 final global-best
length, derived from the actual closed tour. Pair relative difference
`(control-treatment)/control`, average three seeds within each independent
instance, then analyze the16 instance summaries. Co-primary scientific claim
requires both budget comparisons to pass; no success claim based solely on the
equal-iteration arm. Secondary: anytime best-of-100 trajectories at0,0.5,...4s
(last completed state at each time), final source quality, total iterations,
accepted events versus actual mutations, native gain evaluations/calls, wall
time, setup time, memory, no-op counts from separate diagnostic replays only.
Persist raw tours, times, seeds, source/best and measurement validity indicators.

Report mean, median, sample SD, full range, per-instance signs and fixed-seed
10000-resample instance bootstrap95% percentile intervals (PCG64 seed202612070202).
Resample paired instance summaries, never moves, seeds or phases independently.
No tuning from principal results, optional stopping or extra variants.

## Technical, resource and scientific decisions

Technical preflight: exact flag-off regression including RNG, neural inputs,
source/best/pheromone; enabled saved-state equality; passive measurement;
permutation/closure and float64-native cost tolerance2e-5 relative. Any mismatch
blocks principal start. During principal: any invalid tour/nonfinite output,
missing case, source/binary hash drift or unavailable required measurement
makes the full result INCONCLUSIVE; retain all data and no replacements.

Resource caps: RTX3050 only, one sequential GPU workload, two CPU threads,
1GB peak CUDA allocation,3GB peak process RSS,45min total search/setup wall time,
192 planned runs (16*3*2 arms*2 modes). Log peak allocated/reserved CUDA bytes,
Linux peak RSS and per-run monotonic wall time. Resource failure is
RESOURCE-INCONCLUSIVE; stop without shrinking data or changing settings.
Before principal registration, demonstrate feasibility on fixed development
inputs at the complete planned per-run settings; P0's small pilot alone does
not prove192-run feasibility.

Proposed prospective GO: both budget modes have median paired instance
improvement >=0.1%, positive mean-bootstrap lower bound and >=12/16 positive
instances, with no technical/resource failure. Otherwise NO-GO when complete
and valid, INCONCLUSIVE when required evidence is incomplete. These are draft
criteria awaiting review, not retroactive M1 thresholds or deployment approval.

Only a separately approved frozen registration may authorize execution. This
draft, P0 READY or M1 signal does not authorize M2 principal or R3 Stage B.
