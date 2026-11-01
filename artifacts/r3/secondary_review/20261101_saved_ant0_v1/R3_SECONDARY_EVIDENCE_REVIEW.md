# R3 secondary evidence review: saved ant0 tours only

**Research decision: INCONCLUSIVE.** The existing records show both a pre-SRR
candidate opportunity and residual local-search opportunities, but they do not
identify which mechanism could improve the actual solver. The frozen Stage A
decision remains **NO-GO**; G0/G4 true and G1-G3 false. This review does not
authorize Stage B or change the source tour gate.

## Provenance and evidence chain

Branch `feature/r3-budget-aware-candidate-graph`, starting HEAD `f48175de9656f6507a2aa6e52c4df583af92e043`.
The input manifest hashes 319 specific files; the
original Stage A ledger's 440 files
were rehashed. Historical gate SHA256 `5bc9e603e9853d8fb13ab1d0a1ae54bd9b6db83f8b534bdbc25d51c39a7056d6`;
historical report SHA256 `767c481425a99f0bf503a63cc7fef0bd27c68b80dc437bca61f1da05ab5a94a8`. The frozen
protocol, source, native diagnostic, saved coordinates/candidates, tours,
traces and checklists are pinned by SHA256 in `input_manifest.json`.
No instance, solver trajectory, model or confirmation data was generated.

`r3/analysis.py:73-82` calls `evaluate(..., True, complete, resources)`:
G0's technical-validity flag was asserted after some internal checks, rather
than derived from a separate machine-readable checklist of every G0 premise.
`r3/audit.py` later rehashed files, regenerated the dataset, checked tours,
reapplied 5407 selected gains and reaggregated the gate. This is strong
independent support for integrity and the stated source-tour metric. It does
not validate every one of the 498500 gains per snapshot, prove actual SRR
checklist extension/evaluation/acceptance, or certify local optimality.
The binding returns empty `touched_list`; reconstructed raw tours validate
relocation and the initial checklist, not subsequent SRR decisions.
These are evidence-chain limits, not a retrospective edit to G0 or its gate.

## Paired ant0 reduction and residual moves

Each raw/refined pair is the saved tour of the *same ant0* in one iteration.
Length is independently recomputed in float64 on the saved float32 coordinates.
The refined ant0 tour is not the best-of-100 source selected after the iteration.

| Phase (zero-based iteration) | Pairs | Mean raw-to-refined length decrease | Mean raw off-K32 robust moves | Mean refined off-K32 | Mean refined K32-eligible residual | Mean raw K32 available without initial anchor | Mean raw initially reachable but pruned |
|---|---:|---:|---:|---:|---:|---:|---:|
| initial | 16 | 1.132693 | 108.19 | 70.31 | 337.81 | 365.19 | 6.06 |
| early | 16 | 0.286889 | 1.12 | 0.00 | 5.81 | 3.75 | 0.75 |
| middle | 16 | 1.001857 | 15.50 | 2.25 | 14.81 | 18.62 | 5.50 |
| late | 16 | 1.076969 | 23.00 | 0.38 | 14.12 | 22.12 | 5.88 |

Middle/late phases are nested within each execution seed, then two seed values
are averaged within each independent instance. The table below gives those
eight instance summaries, rather than treating 32 pairs as independent.

| Instance | Mean paired length decrease | Raw off-K32 | Refined off-K32 | Refined K32-eligible residual |
|---:|---:|---:|---:|---:|
| 0 | 1.064991 | 25.75 | 0.00 | 13.50 |
| 1 | 1.077310 | 29.50 | 0.25 | 14.00 |
| 2 | 0.816579 | 7.00 | 0.00 | 11.25 |
| 3 | 1.002035 | 24.75 | 9.00 | 21.00 |
| 4 | 1.034354 | 17.50 | 0.00 | 17.50 |
| 5 | 1.086470 | 7.75 | 0.00 | 7.50 |
| 6 | 1.120617 | 28.25 | 1.25 | 19.25 |
| 7 | 1.112949 | 13.50 | 0.00 | 11.75 |

Descriptive instance-level mean/median/range and 10000 fixed-seed bootstrap
percentile intervals are in `summary.json`. They describe this small sample;
the underlying tours and seeds share search histories. Mean robust off-K32
one-step gain mass in the middle/late phases, nested by instance, is:

| Instance | Raw off-K32 gain mass | Refined off-K32 gain mass |
|---:|---:|---:|
| 0 | 0.805550 | 0.000000 |
| 1 | 0.690687 | 0.002929 |
| 2 | 0.220835 | 0.000000 |
| 3 | 0.656879 | 0.139935 |
| 4 | 0.562739 | 0.000000 |
| 5 | 0.192807 | 0.000000 |
| 6 | 0.749784 | 0.032446 |
| 7 | 0.246383 | 0.000000 |

These overlapping one-step gains are offline opportunity measures, not
additive tour improvement. The per-pair policy CSV retains the decomposed
gain masses for both raw and refined tours.

## Equal-slot offline checklist comparison

For every pair, native K64 candidate identities and the *same stored raw-tour
edge allocation* are held fixed while evaluating raw and refined tours.
K32 is the base. Uniform K40, geometry concentration and each of ten random
concentrations use 8000 extra directed slots. Candidate eligibility uses both
directed 2-opt orientations, a canonical move pair and the baseline's float32
distance-ordered early break. Only initial-checklist anchors are counted.
After a tour changes, this static checklist is hypothetical and differs from
the dynamically extended native SRR checklist. Every policy's raw all-node
counts and gain mass were checked against saved Stage A records; base checklist
metrics were checked separately.

Incremental gain mass beyond K32 under the reconstructed initial checklist,
averaged within seed/phase and then instance:

| Instance | Raw geometry | Raw uniform | Raw random mean | Refined geometry | Refined uniform |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.241881 | 0.142834 | 0.081563 | 0.000000 | 0.000000 |
| 1 | 0.142063 | 0.050164 | 0.026572 | 0.000000 | 0.000000 |
| 2 | 0.203870 | 0.068244 | 0.056058 | 0.000000 | 0.000000 |
| 3 | 0.259671 | 0.208901 | 0.048105 | 0.000000 | 0.000000 |
| 4 | 0.175812 | 0.022409 | 0.032054 | 0.000000 | 0.000000 |
| 5 | 0.216211 | 0.103657 | 0.060690 | 0.000000 | 0.000000 |
| 6 | 0.243832 | 0.085266 | 0.047956 | 0.012046 | 0.007814 |
| 7 | 0.219041 | 0.080873 | 0.062123 | 0.000000 | 0.000000 |

The CSV contains all ten random allocations separately, directed slot counts,
and unique move coverage. These are **frozen-tour potential** metrics. They do
not show native evaluation, chosen moves, accepted moves or realized solver
improvement. Undefined ratios are left undefined; this secondary screen has
no retrospective GO threshold.

## Mechanism assessment and decision

The count/gain decomposition on each frozen tour is: unavailable candidate;
candidate available but no reconstructed initial-checklist anchor; initially
reachable but stopped by distance pruning; and initially reachable potential.
These are disjoint after OR over the four directed representations. The raw
checklist is exact at SRR entry, but later checklist extension and tour changes
are unobserved. `src/mfaco_train.cpp:964-1040` has a monotone checklist cursor
and de-duplicates against the whole vector, so an already processed node is
not re-enqueued in one call. The available records cannot count move losses
caused specifically by that behavior. A residual K32 move after refined ant0
may result from checklist scope, pruning at an earlier state, selection order,
tour evolution, or lack of revisit. Causal shares require an intervention.

**INCONCLUSIVE** is the research recommendation. Candidate expansion has
offline headroom before SRR, while K32 residual moves and static checklist
limits leave a distinct SRR hypothesis. The present ant0 traces do not
distinguish which change would help the best-of-100 source under equal compute.

For a separately registered study, a minimal falsifiable hypothesis is:
*on independent TSP-1K instances, adding rank33-64 candidates only to the
initial-checklist SRR scan at 8N directed slots improves final best-of-100
tour length beyond equal-slot uniform K40 and random concentrated controls,
at no more than 10% search-wall-time overhead*. Controls must include unchanged
K32, uniform K40 LS-only, ten seeded random concentrated LS-only, predefined
geometry LS-only, and a K32 checklist-revisit-only arm to isolate search scope.
Freeze all move thresholds, seeds, node-selection rule, trajectory sampling,
checkpoint, native trace semantics and runtime accounting before data. Use a
new independent eight-instance PCG64 TSP-1K set with two solver seeds; keep
the reserved Stage A confirmation set sealed until a later independently
approved confirmation protocol. Bound the study to 1000 iterations, 100 ants,
two CPU threads, sequential RTX3050 workloads, 90 minutes wall time and 4 GB
process RSS; report resource failure without replacement cases. Measure
paired instance-level final tour length, anytime quality, actual LS evaluations
and accepted moves via passive disabled-by-default instrumentation, candidate
slots, source/ant0 trajectories, wall time and memory. A prospective GO could
require a median paired final-length improvement >=0.1% versus *each* equal-slot
control in at least 6/8 instances, with no per-instance invalid tour and <=10%
median wall overhead; otherwise NO-GO or INCONCLUSIVE for invalid data. These
are proposed future criteria, not Stage A thresholds and not an experiment run
here. If the native instrumentation cannot preserve exact disabled-feature
behavior or expose actual checklist/evaluation events, stop before comparison.
