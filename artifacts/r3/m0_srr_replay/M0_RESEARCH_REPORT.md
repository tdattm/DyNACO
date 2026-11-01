# M0 exact K32 SRR replay and execution trace

**M0 VALIDATED**: 64/64 saved ant0 pairs pass the exact replay gate.
Historical Stage A remains **NO-GO**; the secondary evidence decision remains
**INCONCLUSIVE**. No M1, candidate expansion, ACO trajectory, model operation,
new instance generation or reserved confirmation-data access was performed.

## Exactness and scope

Starting branch: `feature/r3-budget-aware-candidate-graph`, HEAD
`390f3e46186113ffd5ed714343fc93ea09119b0a`. The R3 worktree was clean before editing.
The original checkout is `D:/GitRepository/DyNACO` on `final`, with its existing
untracked `.r1-env/` preserved; this work is in `D:/GitRepository/DyNACO-r3`.
Only new M0 files are delivered. See `repository_safety.json` for the status inventory.

`manifest.json` pins baseline source, build flags, generated replay source,
binary, saved coordinates, native K32 lists, raw/refined/source arrays, trace
inputs, frozen protocol and historical decisions. The complete 440-file
Stage A ledger was rehashed. Inputs were rehashed again after validation.
The executable is a standalone test-only C++17 pybind module; it does not
include, construct or link the production solver. The C++ method bodies for
the uninstrumented path are extracted literally from unchanged baseline source.
The logged path applies the explicit hook substitutions saved in
`instrumentation_replacements.json`. Both paths compile together with GCC13,
`-O3 -std=c++17`, as in the baseline, without fast-math flags. No rounding mode
or CPU floating-point environment is changed.

All 64 raw tours and initial checklists were independently reconstructed from
source-before plus saved relocation decisions, without random draws.
Each replay final route matches the saved refined ant0 array exactly, with
closure checked when the saved array has N+1 elements. No cyclic normalization,
rotation or reversal is used to obtain matches. The logged and literal paths
match route, final checklist and accumulated float32 delta. Python independently
checks every evaluated gain in native order, each selection update, pruning
boundary, accepted route mutation, permutation, checklist insertion, and
accepted edge delta against a float64 full-tour cost difference. C++ additionally
checks position inverses after each logged accepted move. Independent final
float64 costs match those of the saved refined arrays in all cases.
`test_results.json` and `validation_summary.json` retain every case, including
empty mismatch-position lists. There are zero dropped or failed cases.

## Semantic dependencies from source audit

- Minimal state: exact float32 coordinates, N=1000, K=32, ordered native K32
  lists, raw route array, inverse positions, ordered initial checklist and
  `extend_ls=true`. No pheromone, network, RNG, prior or accumulated ACO cost
  is required. Saved configuration has local search enabled and L=0.
- `src/mfaco_train.h:338`: coordinate subtraction, squared sum, square root,
  and returned distance are float32. Gain expressions remain left associative
  in float32; no robust threshold is introduced into native acceptance.
- `src/mfaco_train.cpp:358`: KD-tree candidate construction uses double
  coordinates promoted from the saved floats, exact Euclidean distance and
  repeated nearest queries with deletion. `src/kd_tree.h:460` updates on `<=`,
  so equal-distance identity follows traversal/bucket order rather than a
  node-ID stable sort. Replay uses persisted list identity/order, never a
  newly sorted or regenerated list. Thus tie ordering is inherited exactly.
- Traced sampling initializes the checklist with start_node, then appends
  current, chosen and chosen's pre-relocation predecessor for each new source
  edge, in that order, with whole-list de-duplication. Source-edge membership
  is determined against the original source positions. Reconstruction checks
  saved booleans, raw route order and checklist order exactly.
- `two_opt_nn` visits checklist entries using a monotonically increasing
  cursor. It reads successor/predecessor from current positions, evaluates
  outgoing orientation first, incoming second, and scans ranks 0 through 31
  in each direction. Invalid neighbor terminates that direction. A neighbor
  is evaluated only while the old incident-edge distance is strictly greater
  than the neighbor distance; equality terminates scanning of the whole suffix.
- max_diff starts at zero; only strict `diff > max_diff` updates selection.
  Equal gains retain the earlier candidate/orientation. One best positive move
  is accepted per processed anchor, after both scans; there is no epsilon.
- `flip_route_section` sorts endpoint positions and reverses [first,last)
  when that segment is no longer than its complement (ties use this branch).
  The other branch uses the original circular swap schedule and position
  updates verbatim, including its first==0 arithmetic. It is not replaced by
  a canonical 2-opt reversal. The independent validator uses the same literal
  schedule and checks the resulting changed indices against the logged patch.
- After acceptance, endpoints best_move[0..3] are considered in order.
  Membership is tested against the entire checklist, including processed
  entries. An already processed endpoint therefore cannot be re-enqueued
  in this call. Subsequent anchors see the mutated route and updated positions.
  Accumulated total_change is decremented by selected max_diff in float32.

## Observed execution versus offline opportunities

Every `*_events.json` contains the exact processing order, each evaluated
neighbor/rank/orientation/gain and selection update, distance-pruning boundary,
final selected endpoints, changed route indices, and endpoint insertion or
duplicate attempt with its already-processed status. Events are bounded by N
checklist entries, at most 2K evaluations and two pruning boundaries per entry,
one accepted move and four insertion attempts per entry. Route patches have
at most N entries. No event sampling or truncation is applied. Final tour and
checklist are retained in each log; binary output is not needed to read them.

The following quantities must be read separately:

| Quantity | Meaning and limitation |
|---|---|
| Candidate unavailable | At an observed anchor, N-1-K directed neighbor identities are absent from its fixed K32 list. The manifest and graph establish this structural exclusion; those neighbors were never evaluated. This is not a count of improving moves. |
| Available but not reached | K32 rows of nodes never reached by the actual checklist remain unscanned. `unreached_nodes` counts such anchors; it does not freeze an earlier tour or assign gains to them. |
| Native pruning | The boundary neighbor and remaining suffix are skipped in that orientation on the actual current tour. `pruned_neighbor_slots` includes the boundary and repeated directional slots, not unique moves. |
| Evaluated but not chosen | Actual gain computations other than the final chosen neighbor at that anchor, including nonpositive gains and positive candidates superseded later. These are evaluation events, not unique moves. |
| Accepted move | One observed best positive float32 2-opt move and its exact mutation. Each is independently validated against full-tour float64 costs. |
| Processed node rejected for requeue | An observed extension attempt for an endpoint whose anchor was already processed; whole-list membership prevents insertion. Counts are attempts, and can repeat a node. |
| Offline final residual | Robust improving canonical 2-opt moves scanned only after replay passes. Float64 gain must exceed max(1e-8,1e-6*final length). These are opportunities on the final frozen tour, with no claim of native evaluation or acceptance. |

Mean counts per saved phase (zero-based iteration):

| Iteration | Processed | Evaluations | Pruned slots | Evaluated unchosen | Accepted | Processed requeue attempts | Offline K32 residual | Offline off-K32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 66.94 | 743.25 | 3540.75 | 705.69 | 37.56 | 78.31 | 337.81 | 70.31 |
| 249 | 24.56 | 167.44 | 1404.56 | 153.25 | 14.19 | 31.56 | 5.81 | 0.00 |
| 499 | 62.31 | 579.31 | 3408.69 | 543.81 | 35.50 | 73.69 | 14.81 | 2.25 |
| 999 | 69.56 | 734.81 | 3717.19 | 696.25 | 38.56 | 80.12 | 14.12 | 0.38 |

Middle/late phase means within seed, then the two seeds within each instance:

| Instance | Processed | Evaluations | Pruned slots | Evaluated unchosen | Accepted | Processed requeue attempts | Offline K32 residual | Offline off-K32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 69.25 | 710.00 | 3722.00 | 668.50 | 41.50 | 87.00 | 13.50 | 0.00 |
| 1 | 55.00 | 570.00 | 2950.00 | 539.50 | 30.50 | 64.25 | 14.00 | 0.25 |
| 2 | 59.50 | 581.75 | 3226.25 | 550.75 | 31.00 | 64.50 | 11.25 | 0.00 |
| 3 | 68.25 | 657.75 | 3710.25 | 619.50 | 38.25 | 80.75 | 21.00 | 9.00 |
| 4 | 73.00 | 711.25 | 3960.75 | 673.50 | 37.75 | 72.75 | 17.50 | 0.00 |
| 5 | 58.25 | 627.75 | 3100.25 | 594.75 | 33.00 | 66.75 | 7.50 | 0.00 |
| 6 | 71.25 | 688.25 | 3871.75 | 647.00 | 41.25 | 90.75 | 19.25 | 1.25 |
| 7 | 73.00 | 709.75 | 3962.25 | 666.75 | 43.00 | 88.50 | 11.75 | 0.00 |

`descriptive_summary.json` includes median, minimum and maximum by phase and
instance, plus dispersion across the eight independent instance means.
Seeds and phases are nested within instance. These 64 records are selected
ant0 examples from eight saved TSP instances, not 64 independent instances;
they share histories within a trajectory. Refined ant0 is not the selected
best-of-100 source. No population inference or passed scientific gate follows.
Processed requeue attempts and final residuals establish observable mechanics,
but cannot prove that a particular skipped or unchosen move would improve
the final trajectory. The tour at the skipped event generally differs from
the final scanner's tour. Candidate, pruning and scheduling effects interact.

## Smallest proposed controlled M1 (not executed)

Prospectively freeze one mechanics question: does one additional call of the
unchanged K32 routine, starting from M0's validated final route and final
checklist, find robust additional gains on saved middle/late ant0 examples?
Compare the validated single-call endpoint with exactly one second call;
keep candidates, distances, pruning, tie order, extension and move selection
unchanged. Instrument both calls. This resets only the processing cursor at a
call boundary; it tests a bounded revisit/scheduling intervention, not candidate
expansion or an alternative production solver. It does not isolate every
possible mechanism of an individual missed move.

Use the existing 32 middle/late records as an explicitly exploratory paired
mechanics diagnostic, with phase/seed nested in the eight saved instances.
Cap at two total calls, at most 2N processed anchors and 4NK gain evaluations
per input, one CPU worker, 5 minutes wall time, 1 GB RSS. Freeze a descriptive
signal criterion before M1: additional float64 reduction > max(1e-8,1e-6*baseline
final length) in at least one record in at least four of eight instances,
with all permutations valid and native logged/unlogged invariance intact.
Report evaluation cost and any worsening case. Failure of correctness stops
M1; absence of the signal is a negative mechanics finding. This is not a
scientific GO criterion or authorization to run M1. Any claim about best-of-100
solver quality would require a separate registered equal-compute experiment
on new independent instances; the reserved Stage A confirmation set stays sealed.

The delivered result is **M0 VALIDATED**. Historical decisions retain
their existing meanings and hashes. M0 establishes replay and observed events;
the proposed M1 remains unimplemented and unexecuted.
