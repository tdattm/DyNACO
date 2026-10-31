# R3 Stage A report

**Decision: NO-GO. Stage B was not executed.**

All 32 primary late-phase source snapshots had zero robust improving moves
outside K32. The predefined selector, uniform K40 and all ten random controls
therefore recovered zero additional gain mass in the primary comparison;
their 0/0 ratios remain undefined. K32-eligible robust moves still remained
(means 9.25 at iteration 500 and 6.9375 at 1000). The measured limitation at
these checkpoints is therefore not missing K32 candidate membership. This
negative result applies to this frozen sample/configuration; it does not rule
out useful expansion during initialization or on other problem distributions.
The bounded pre-SRR ant0 sample does contain off-K32 opportunities (30/32 late
raw tours). This secondary observation is reported below; it cannot replace
the frozen primary source-tour gate or authorize Stage B.

## Research question and frozen design

Test whether native K32 omits material robust 2-opt moves on genuine search
states, and whether a predefined geometry/tour-excess allocation covers more
incremental gain than equal-directed-slot uniform and random controls. The
proposal supplies hypotheses, not measured results. Original paper results,
historical R1 evidence, new baseline verification and these diagnostics are
separate evidence categories.

Protocol: `r3/protocols/stage_a_v1.json`; SHA256
`1f9d4cc3e66c8bc22a4f792594c1adc3cbd1a906e34d39fd4e784d1e79bc84cc`. Frozen before principal data generation; no gate,
selector, phase boundary, dataset or sample-size changes after inspection.
Primary observable: source tour; global best is secondary and reported in raw
tables, never substituted into the gate. Completed iterations 500 and 1000 are
averaged within execution seed, then equally across the two seeds per instance.

## Baseline source and verification

Original SHA `cd16e2a68fd035f94a8916d41463904f31483d7f`. R3 working SHA at collection:
`4b8961744f597b75a623dda70f074471bdcfa51b`. Production solver, wrapper, GNN and inference
sources are unchanged. The isolated C++17/OpenMP build has the same SHA256 as
the earlier audited R1 rebuild; checkpoint SHA256:
`33f7da301e49f74d5b66de2a12c7bd95f247de8c217343c3410bb716a6f22358`. Strict model loading, CUDA, K32/K64 forward
shapes and native imports passed. Reduced-budget checks ran at N=100/200/500/1K;
the 1K 1000-iteration captured trajectory exactly matched unchanged
`utils.infer_instance`, and a same-seed repeat matched every ant cost.

The collection manifest also records the uncommitted read-only WSL Git
line-ending compatibility fix and the full source-byte hash map. Its working
commit alone is not the complete research-source identity. Archived Windows
text files showed representation-only diffs after byte-preservation attributes
were added; their original bytes and the initial manifests were retained.

Configuration: K32+backup32, 100 ants, H10/S100, 1000 ACO iterations, M12,
rho=.1, smooth pheromone bounds 1/K and 1, original SRR with extension,
normalized heuristic, dynamic features, original per-block linear prior
annealing from 1 to 0, no phased Mix injection and no NLS. Run-level seed reset
preserves original duplicate construction and NumPy-derived active native seed.
Two CPU/OpenMP threads; RTX3050 4GB via WSL. Exact replay uses strict CUDA and
TF32 disabled. These timings are not non-strict R1 benchmark timings or paper
reproduction on different hardware/datasets.

See [source audit](SOURCE_AUDIT.md) for all 15 implementation questions and
precise source lines. Important findings: directed storage; backup absent from
SRR/GNN; both scan orientations and float32 early breaks; once-per-distinct-node
checklist; greedy initialization without local optimality; incomplete snapshot
replay and stale normalized heuristic cache after native snapshot load.

## Data and provenance

Eight independent uniform float32 TSP1K instances, PCG64 SeedSequence([20261101,
1000]); IDs 0-7. Execution seeds 31415 and 27182. Dataset SHA256:
`8ec4e091f5f749ce15dcd236e47e8919601d7112cd0c3bac3c458ae002527789`. Every generated coordinate block has a saved
hash. Each native K32/K40/K48/K64 row was independently distance-ranked and
prefix-checked; K32 backups exactly matched native ranks33-64. R1 pilot datasets
were read as Git blobs, rehashed and independently regenerated. Confirmation
seed 20261104 remains reserved, ungenerated and uninspected.

All 16 principal runs completed: 16000 ACO iterations, 64 persisted solver states,
128 frozen source/best diagnoses and 64 bounded ant0 raw diagnoses. Native
perturbation traces reconstructed all 64 pre-SRR raw tours exactly, including
array order and original-source edge tests. No solver RNG was drawn for state
capture or allocation controls.

## Diagnostic definitions and numerical checks

Enumerate every nonadjacent cyclic tour-edge pair once: N(N-3)/2=498500 per
tour, excluding identical/adjacent/wraparound-adjacent edges. Float64 reference
uses unrounded Euclidean distances on exact native float32 coordinate values.
Robust improvement means gain > max(1e-8,1e-6*current length). Raw signed gains,
near-zero samples and marginal counts are retained. Distinct directed
representations (a,c),(c,a),(b,d),(d,b) are combined by OR per canonical move.
Membership coverage is separate from all-node potential evaluation under exact
native float32 ordered early-break prefixes.

Known crossed/optimal tours, collinear/duplicated/near-duplicated coordinates,
numerically marginal gains, invalid candidate/tour data, both scan orientations,
and exhaustive independent copied-tour reversal tests passed. Every selected
missed move and bounded robust sample was applied to a copied valid cyclic
permutation and independently recomputed in float64. A single-move oracle
improvement is feasibility evidence, not an improvement achieved by DyNACO;
independent move gains are never summed into an achievable tour improvement.

## Candidate coverage and phase dependence

Primary source-tour means across all 16 executions at each phase:

| Iteration | Full robust moves | Outside K32 membership | Largest missed gain | K32 all-node scan eligible |
|---|---|---|---|---|
| 0 | 471 | 90 | 0.19176095368365287 | 377.125 |
| 250 | 2.5 | 0 | 0.0 | 2.5 |
| 500 | 9.25 | 0 | 0.0 | 9.25 |
| 1000 | 6.9375 | 0 | 0.0 | 6.9375 |

Late snapshots with any membership miss: 0/32.
Per-K membership/scan counts, missed rates, gain-weighted coverage, increments,
largest missed gain, relative-gain and candidate-rank histograms, per-node missed
gain mass/concentration and all individual random results are retained in JSON
and `snapshot_policy_metrics.csv`. No denominator-zero coverage is fabricated.

## SRR versus graph limitation

Initial checklist membership and potential pruning-aware coverage are reported
on reconstructed raw tours. Actual native evolving SRR evaluation/acceptance
sequences are unavailable through the original API; booleans remain null.
Checklist extension and changing tours make initial membership insufficient to
infer dynamic execution. Empty `touched_list` is not used as checklist evidence.
The all-node frozen-tour reference therefore measures candidate headroom,
not scope-restricted native solver behavior or realized accepted moves.

The secondary raw ant0 inputs differ substantially from the chosen post-iteration
source tours. Across 16 runs at each bounded raw phase:

| Zero-based iteration before SRR | Raw tours with off-K32 moves | Mean full robust moves | Mean off-K32 moves | K32 all-node potential | Initial-checklist potential |
|---|---|---|---|---|---|
| 0 | 16/16 | 528.75 | 108.1875 | 416.1875 | 49.3125 |
| 249 | 3/16 | 29.625 | 1.125 | 28.375 | 24 |
| 499 | 15/16 | 103.1875 | 15.5 | 87.125 | 63.5625 |
| 999 | 15/16 | 110.9375 | 23 | 87.0625 | 59.9375 |

For raw iteration499, mean extra pruning-aware gain mass was geometry0.218014,
uniform0.098848, mean-random0.063257; at raw iteration999 it was geometry0.295668,
uniform0.122797, mean-random0.091084. These are offline sums over incompatible
one-step opportunities, not attainable total tour improvement or solver gains.
They are secondary descriptive quantities, not a replacement gate analysis or
a post-hoc selector redesign. The primary source is the best of 100 refined ants,
whereas this bounded raw sample is ant0, so their difference cannot be attributed
solely to SRR or interpreted as a paired expansion effect. This leaves a distinct
pre-SRR research question open, requiring a separate prospectively registered
study if the coordinator chooses to pursue it. No such study was started.

Secondary scope summary: `artifacts/r3/manifests/secondary_scope_summary_v1.json`;
calculated entirely from existing frozen raw records, without additional runs.

The paper's Appendix B.2 describes node reactivation and assumes a 2-opt optimal
incumbent. Actual baseline initialization and checklist processing do not meet
those assumptions. Residual K32-eligible moves cannot automatically be blamed
on the graph. Exact-distance ties also lack a global node-index tie-break:
constructed duplicates show cross-K prefix differences. Tests detect that case;
all actual principal lists passed exact prefix checks.

## Equal-edge-budget selector screen and gate

Base K32; pool ranks33-64. Uniform K40 adds eight slots per node; geometry and
each of ten random controls select 25% of nodes and add 32 slots each. All use
8000 extra directed slots, keeping unique undirected counts separate. Full K64
uses 32000 extra slots and is a higher-budget reference. Geometry score and
node-index ties were frozen. No oracle-selected allocator was used.

Each table row averages late-phase quantities within seed then across seeds;
ratios divide aggregate incremental pruning-aware gain masses. Undefined ratios
remain `None` and fail conservatively.

| Instance | Missed count | Verified best relative gain | Geometry extra gain mass | Uniform | Mean random | Geo/uniform | Geo/random | G3 instance |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | False |
| 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | False |
| 2 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | False |
| 3 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | False |
| 4 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | False |
| 5 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | False |
| 6 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | False |
| 7 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | False |

G0-G4: `{'G0': True, 'G1': False, 'G2': False, 'G3': False, 'G4': True}`. G1 passing instances 0/8;
G2 0/8; G3 0/8. Required: at least 6/8 for each,
G2 mean relative best missed gain >=1e-4, G3 both ratios >=1.10 and both median
ratios >=1.10. Medians: `{'ratio_vs_uniform': None, 'ratio_vs_random': None}`. Instance-level paired
descriptive bootstrap summaries (10000 resamples, fixed seed) are saved in
`gate.json`; no move/snapshot-level independence or significance is asserted.

## Runtime, memory, failures and limits

Principal collection/analysis to gate: 106.212s,
90-minute cap. Process RSS high-water: 1432006656 bytes;
peak PyTorch CUDA allocation: 127517696 bytes.
These are process-wide RSS and Torch allocated memory, not isolated scanner RSS
or total device/context memory. Native/graph-forward/search/extraction times,
KNN construction/audit, enumeration and classification, bounded sampling, edge
counts, gain verification and report generation are saved with their scopes.
No concurrent benchmark GPU workload ran during principal measurements.

Initial test failures and environment-path failure are retained: a Windows Git
worktree pointer needed explicit WSL Git paths; temporary solver owners caused
invalid borrowed NumPy views in research fixtures; the duplicate-coordinate
test exposed a real native tie-prefix limitation and was changed to require
detection. Production code and frozen scientific thresholds were unchanged.

A subsequent preflight source-change alarm was confirmed to be Windows/WSL
CRLF configuration, not changed source bytes: all ten initial audited hashes
matched, Windows baseline diff was empty, and explicit read-only WSL autocrlf
handling removed the false source diff. A temporary live Git index lock caused
two local add/commit attempts to fail; no lock was removed and no history was
rewritten. These failures occurred before principal data collection. See the
preserved attempt logs and failure records.

Post-collection audit passed: 440 finalized files rehashed, dataset regenerated
byte-for-byte, all 128 frozen-tour records and 64 raw records checked, 5407
selected moves independently reapplied, and the instance-level gate recomputed
from persisted results. This audit is additional validation, not new principal
data. The principal experiment directory remains immutable.

This study is 8 instances with two execution seeds, one frozen checkpoint,
source-tour late-phase aggregation and offline candidate/pruning semantics.
It does not reproduce paper gaps, certify local optimality, estimate adaptive
solver performance, or establish neural-guidance gains. Any future benefit
from LS expansion could be additional local search, requiring real rollouts and
equal-budget controls. All negative/near-zero outcomes remain in raw artifacts.

## Decision and next action

**NO-GO.** The exact frozen gate is in `gate.json`. Do not implement Stage B
unless all conditions pass. Stage B and confirmation were not run in this
execution. Preserve this result and bring the implementation/paper SRR mismatch
and the measured instance-level gate evidence to the main coordination chat.
