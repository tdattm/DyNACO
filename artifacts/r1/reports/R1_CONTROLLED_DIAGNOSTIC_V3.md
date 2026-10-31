# R1 Round 3 — controlled refresh-value diagnostics

**Recommendation: INCONCLUSIVE; stop without expansion.** Technical validation
and the 72-decision pilot passed. Refreshing sometimes improved tours, but
stagnation did not improve held-out-instance prediction over prior age plus
phase: primary-horizon MSE increased by about 1.54%, and none of four instance
folds improved. The frozen scientific continuation gate failed. No new
scheduler, extra instances or held-out confirmation campaign was launched.

## Scope and provenance

This round tests whether stagnation duration adds information about marginal
refresh value after accounting for cached-prior age and broad search phase.
It implements no scheduler, learned controller, training, architecture or
native algorithm change. All inference uses the frozen original TSP-1K model.

Local branch: `feature/r1-event-triggered-guidance`. Initial and current HEAD:
`27cf8b9223e60dae4ca8d5821d92533d1ecc1bdc`, which matched the pushed branch at
audit. The working tree was initially clean. Changes in this round are local
and uncommitted; no push, merge, reset or history rewrite was performed.

The original DyNACO base is cd16e2a; v1 sources/results were later packaged at
8ee944b, and the strict replay resolution was packaged at 27cf8b9. This round's
actual uncommitted source bytes are identified by SHA-256 in its run manifest;
HEAD alone does not identify the experiment source state.

Artifact root: `artifacts/r1/diagnostics/20261031_controlled_v3_01/`.
`preservation_manifest.json` hashes all previously tracked research artifacts.
`pilot/manifest.json` records source/checkpoint/data hashes, exact strict CUDA
settings, hyperparameters and GPU telemetry. Existing v1/v2 measurements and
reports were not rewritten. The existing isolated WSL CUDA environment was reused.

## Technical gate and implementation

The audit suite passed **26 existing tests** before modification. After the
small diagnostic extensions, **28 tests passed**, including exact neural/native
guidance inputs, all-ant cost/tour prefixes, incumbent/source/pheromone states,
full HOLD negative controls and cross-process deterministic graph inference.
Two additional CPU-only statistical-contract checks verify training-only
scaling, nested predictor design and independence from held-out target values.

The runner reuses `r1.replay_pilot` and `r1.v2_replay`, not a duplicate solver.
The older `v2_experiment` runner was reviewed but not chosen: the pilot runner
already contains the stronger full native-history controls and final-state
assertions needed here. Its protocol-controlled extension adds resource
projection, prior capture and decision metadata.

`branch(..., capture_priors=True)` copies the cached prior at the probe and the
new actual native prior after the single intervention forward. This is enabled
only for REFRESH arms. Arrays are immutable float32 snapshots in compressed
NPZ files; no extra model calls occur. A dedicated strict integration test
confirms capture-on/off trajectories, native input hashes, final states and
call counts are identical. Existing calls default to capture disabled.

Net, checkpoint, native C++, candidate construction, ACO sampling, SRR,
pheromone updates and incumbent semantics remain unchanged. Stagnation uses
the backend global incumbent with the existing absolute 1e-6 significant
improvement tolerance; it does not track iteration-best or source-tour cost.

Strict CUDA requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process startup,
`torch.use_deterministic_algorithms(True, warn_only=False)`, deterministic
cuDNN, benchmark disabled and TF32 disabled. It remains opt-in. No operator
fallback or tolerance relaxation was used. These timings are not comparable
to v1's default CUDA-kernel benchmark timings.

## Frozen experimental protocol

`pilot_protocol.json` was saved before pilot outcomes. Four **new independent
TSP-1K instances** use dataset seed **20261201**, PCG64
`SeedSequence([dataset_seed,N])`, generator version 1, uniform float32 points
in [0,1)^2. This differs from the v1 dataset seed 20261031 and prior paired
diagnostic seed 20261101. Raw coordinates, stable IDs and hashes are persisted.
Solver seeds: **31415, 27182, 16180**.

| Broad phase | Age 25 probe | Age 75 probe |
|---|---:|---:|
| Early | 125 | 175 |
| Middle | 425 | 475 |
| Late | 725 | 775 |

The reference context is 1,000 ACO iterations. Each replay executes only through
its probe plus the maximum 100-iteration continuation: 225/275/525/575/825/875
iterations, not a full 1,000-iteration benchmark. Horizons 25/50/100 are nested
observations of that one continuation.

At each of 4 × 3 × 6 = **72 decisions**, HOLD, HOLD_CONTROL and REFRESH rerun
fixed-S100 prehistory from the same seeds. Subsequent refreshes are frozen in
all arms for 100 iterations; REFRESH alone adds one live neural call at the
probe. There are 216 arm executions and 216 nested horizon records, but only
four independent problem instances and 72 interventions.

HOLD calls count the initial evaluation and are 2/2/5/5/8/8 at the six times;
REFRESH counts are 3/3/6/6/9/9. HOLD_CONTROL duplicates HOLD. The matched prefix
contains exact actual native prior bytes, all-ant costs and tours and checked
search states. Full negative controls must also match beyond the probe.

All runs use CUDA:0, two OpenMP/PyTorch threads, 100 ants, K=32, rho=.1,
min_new_edges=12, original local search/extension, smooth MMAS, normalized
heuristic, constant guidance and no phased injection. Method order is shuffled
by a separate fixed Random(9237) stream. No thresholds or rule were tuned.

Prior age is **not randomized**: ages 25 and 75 occur at different exact
iterations. Crossing broad phases removes the previous early/late age
confounding at the group level, but does not identify age independently of
iteration offset, search history, phase-age interactions or unmeasured state.

## Resource protocol

The first six decisions, covering all six times on instance 000/first seed,
form the runtime/VRAM pilot and remain in the diagnostic data. Native/model
warmup is performed first. They took 30.0 seconds; a 50% margin projected
540.1 seconds for all 72, within the frozen **900-second cap**. Observed
allocated GPU peak was 127,389,696 bytes (~121.5 MiB), below the separate 1 GiB
allocated-memory cap. Those values are allocator peaks, not total/reserved VRAM.

All GPU work is sequential. Each new decision reserves 1.5 times the longest
observed decision duration before it starts. Failure or cap exhaustion retains
raw arm outputs and completed observations. No additional instances may be
launched merely to favor an outcome.

## Measurements and analysis definitions

The primary outcome is **raw HOLD cost minus REFRESH cost**, positive favoring
REFRESH. Relative benefit is `100*(HOLD-REFRESH)/HOLD`; it is not an optimality
gap. Exact ties have zero native-cost difference; near ties have nonzero
absolute difference ≤1e-6. Remaining signs are labeled improvement/harm. Raw
values are always preserved; one float32 cost ULP is not automatically a
meaningful geometric improvement. A separate final-horizon check compares
effects from independently recomputed float64 Euclidean tour lengths.

Recent improvement rate is `(C(t-50)-C(t))/50`, using the shared global
incumbent prefix only, recorded in raw length/iteration and percent/iteration.
Prior changes use float64 **analysis** of the two actual float32 prior arrays:
mean absolute difference, RMS, maximum absolute difference, relative L2
`norm(new-old)/max(norm(old),1e-12)`, and RMS after removing each row's mean
change. These metrics are descriptive; a changed neural prior does not imply
an improved tour.

The predeclared primary horizon is 100; 25/50 are separate secondary outcomes,
not additional independent samples. Three least-squares descriptive predictors
are compared separately at each horizon:

1. Age alone (two-age linear basis, equivalent to an age indicator).
2. Age plus categorical broad phase.
3. Age plus phase plus `log1p(stagnation)`.

Cross-validation leaves **one complete independent instance out**, retaining
all its seeds/probes only for testing. Standardization uses training data only.
Reported MSE has squared raw-length units. Rank/condition diagnostics, every
held-out prediction, all instance folds and coefficients are retained. These
are small-sample descriptive prediction checks, not controller training or
significance tests. No p-values or unsupported uncertainty intervals are used.

Within each of the six age-phase cells, descriptive correlations examine
stagnation, recent improvement and prior-change RMS against benefit. A cell
has 12 repeated decisions but only four independent instances. Correlations
are not independent-observation causal estimates. The three nested predictors
are additive; six-cell descriptions must be consulted because interactions
and exact iteration effects are not fully removed by an additive model.

The single REFRESH-vs-HOLD intervention is a controlled comparison at its
verified shared prefix. The association between the observed state and that
effect is observational. Native internal RNG is not exposed; no exact
post-intervention random-number coupling is claimed. Full HOLD controls show
determinism in the unaltered continuation, not unrestricted coupling after
the two search paths diverge.

## Scientific coverage and expansion gate

The frozen exploratory screening gate requires all of:

- All 72 decisions valid, unchanged inputs/weights and uncorrupted records.
- Four independent pilot instances.
- At least four of six cells with stagnation range ≥25, at least three distinct
  durations and at least three instances showing across-seed variation.
- At least 18 non-tied primary outcomes and at least two non-ties in four cells.
- Model 3 reduces model-2 instance-blocked MSE by at least 10%, improves in at
  least three of four held-out instances, and has the same stagnation-coefficient
  sign in at least three folds.

These are frozen screening criteria, not statistical-significance thresholds.
Any failed prerequisite results in INCONCLUSIVE and no expansion. An eligible
result would still require a resource check for four additional exploration
instances (seed20261202) and four reserved independent confirmation instances
(seed20261203), with frozen analysis specification and no held-out tuning.

## Reproduction and files

Primary source changes are the optional prior capture in `r1/v2_replay.py` and
the opt-in controlled-v3/resource extension in `r1/replay_pilot.py`. Added tests
are `tests/test_r1_v3_capture.py` and `tests/test_r1_v3_metrics.py`. The artifact
root contains `analyze_v3.py` and two CPU-only analysis-contract tests, raw data,
protocol/manifests, test logs, statistical tables and figures.

Commands executed from repository PowerShell:

```powershell
git status --short
git rev-parse HEAD
git branch --show-current
git ls-remote --heads origin feature/r1-event-triggered-guidance
wsl -d Ubuntu -- env CUBLAS_WORKSPACE_CONFIG=:4096:8 bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests -q
wsl -d Ubuntu -- env CUBLAS_WORKSPACE_CONFIG=:4096:8 bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m r1.replay_pilot artifacts/r1/diagnostics/20261031_controlled_v3_01
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh artifacts/r1/diagnostics/20261031_controlled_v3_01/test_analysis_contract.py
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh artifacts/r1/diagnostics/20261031_controlled_v3_01/analyze_v3.py
```

The existing 26-test invocation preceded the 28-test invocation. All output
directories are unique; replaying these commands requires a new experiment ID.

## Actual run completion and independent audit

All **72 decisions / 216 arm runs** completed with zero failed or missing
decisions. They executed **118,800 native iterations** and **1,152 actual
principal GNN calls**. Warmup is separate. There are **216 nested horizon
records**, not 216 independent interventions or problem instances.

The independent persisted-data audit passed: every native-prior/all-ant prefix,
full HOLD control, final search state, call budget, candidate identity, monotonic
incumbent, tour permutation, recent-rate formula, effect join, prior hash,
protocol/source/data hash and historical artifact hash checked. Newly saved
coordinate instances differ from prior same-scale diagnostic instances.

The complete run took **565.4 seconds** including model/data initialization,
warmup, repeated prefixes, diagnostic capture and file persistence, excluding
Python imports and subsequent analysis. This exceeded the initial 540.1-second
projection but stayed below the 900-second cap. No further run was added to
compensate for a result. Maximum recorded allocated GPU memory was 127,389,696
bytes (~121.5 MiB); there was no OOM. Device snapshots showed about 257 MiB total
usage and temperature rising from 54°C before the principal work to 70°C after.
These snapshots are not total-VRAM peaks, and clocks/background load were not
locked.

## Outcomes: single REFRESH intervention versus HOLD

Positive raw benefit means shorter REFRESH tours. All cases, including harm
and ties, remain in the means and raw files.

| Horizon | Mean raw benefit | Mean relative benefit | Improved | Harmed | Exact ties | Near ties |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.00366362 | 0.015731% | 27 | 15 | 30 | 0 |
| 50 | 0.00749509 | 0.032162% | 33 | 13 | 26 | 0 |
| 100 | 0.00836876 | 0.035900% | 44 | 11 | 17 | 0 |

Each row contains 72 paired decisions on four independent instances. The
primary-horizon median benefit is only 0.00055122. Independent-instance means
at horizon100 are **0.006453, 0.011471, 0.004500 and 0.011051** raw length.
All four are positive, while one instance has a negative average at horizon25.
This supports a limited average effect of the tested intervention distribution;
it does not identify a useful state-triggering rule or establish significance.

The final-horizon float64 Euclidean check gives mean benefit **0.00836818**,
close to the native float32 mean. Maximum absolute discrepancy between a native
effect and its recomputed effect is **2.93e-5**. Two nonzero native effects
(+1.90735e-6 and -1.90735e-6) have exactly zero recomputed geometric effect.
Thus the frozen raw labels include two one-ULP effects that should not be
interpreted as substantive improvements/harms. No frozen labels or thresholds
were changed after observing this. Nested horizons remain correlated.

## Does stagnation add information after age and phase?

At the primary horizon, each fixed phase/age cell contains four independent
instances and 12 repeated instance/seed decisions:

| Probe | Phase / age | Stagnation min–median–max | Non-ties | Mean raw benefit |
|---:|---|---|---:|---:|
| 125 | early / 25 | 0–0–2 | 12 | 0.01797485 |
| 175 | early / 75 | 0–2.5–11 | 12 | 0.02916876 |
| 425 | middle / 25 | 0–50–188 | 7 | 0.00110547 |
| 475 | middle / 75 | 0–50.5–238 | 9 | 0.00179609 |
| 725 | late / 25 | 6–86–222 | 7 | 0.00000731 |
| 775 | late / 75 | 20–136–272 | 8 | 0.00016006 |

Larger benefits again occur in actively improving early states with very little
stagnation. Pooling those early states with later states would confound a
stagnation association with phase. The six-cell comparison does not show a
consistent within-cell stagnation relationship: primary-horizon descriptive
correlations of log1p(stagnation) with benefit are approximately
**+0.242, +0.154, -0.215, -0.047, +0.333, -0.362** at the six times. These are
small clustered descriptions, not significant or causal coefficients.

Leave-one-independent-instance-out predictive MSE (squared raw length):

| Horizon | Age only | Age + phase | Age + phase + stagnation |
|---:|---:|---:|---:|
| 25 | 0.000232966 | 0.000211286 | 0.000216546 |
| 50 | 0.000327309 | 0.000232214 | 0.000234747 |
| 100 | 0.000457164 | 0.000357283 | 0.000362785 |

Adding phase improves mean prediction relative to age alone. Adding stagnation
then makes mean MSE worse at **all three horizons**. At the primary horizon it
improves **0/4** held-out instance folds. The four stagnation coefficients per
log1p unit are -0.0000969, +0.0012138, +0.0018290 and +0.0011608. Three sharing
a sign is not sufficient when prediction does not improve. All design matrices
are full rank (2/4/5 columns), with maximum standardized condition number about
4.09; this negative incremental result is not a numerical solver failure.

The evidence is limited to the frozen additive linear predictors. It does not
rule out every possible nonlinear signal, nor justify searching many new
models on this pilot. Four correlated cross-validation training sets are not
four independent replications of a fitted scientific effect. No p-value or
statistical-significance claim is made.

## Recent improvement and prior-value changes

Recent 50-iteration rates and all prior-change metrics are retained per
decision. At horizon100, within-cell descriptive correlations of recent rate
with benefit range from about -0.628 to +0.415; prior-change RMS correlations
range from about -0.219 to +0.457. Their signs and magnitudes are inconsistent
across cells and can reflect instance/seed variation. They were not used to
tune the three frozen predictors or invent another rule.

| Probe | Mean prior-change RMS (logit units) | Mean relative prior L2 |
|---:|---:|---:|
| 125 | 2.0851 | 0.3094 |
| 175 | 2.7192 | 0.4036 |
| 425 | 0.5229 | 0.0778 |
| 475 | 0.8436 | 0.1253 |
| 725 | 0.5178 | 0.0768 |
| 775 | 0.7631 | 0.1132 |

Neural prior changes are larger early and at age75 in these broad cells, but
they also occur when refresh benefit is tiny or absent. Raw logit changes do
not directly measure the solver's normalized transition-probability change.
No claim that a large prior change causes improvement is supported here.

## Actual inference overhead and timing uncertainty

The extra live strict-mode forward averages **92.89 ms** per intervention;
probe-group means range from about 76.54 to 130.06 ms. These are synchronized
model-forward wall intervals, excluding separately recorded graph construction,
prior transfer, prefix reconstruction and NPZ analysis/persistence. Exactly
one additional forward is counted per intervention, not per nested horizon.

At horizon100 the mean instrumented REFRESH-minus-HOLD continuation time is
**+108.93 ms**, with individual differences ranging -629.48 to +763.04 ms.
Different continuation work, instrumentation, GPU/runtime variation and
unlocked laptop conditions affect these differences. Negative individual
differences are preserved, not interpreted as reliable speedups. Old-prior
capture adds a small CPU-transfer cost only to REFRESH diagnostics; full-native
history hashing also adds overhead. There is no ordinary live scheduler,
equal-call policy comparison, or wall-budget performance gain established.

These data are strict-mode instrumented diagnostics. No comparison with v1
default-mode wall-clock benchmarks is made, and no artificial inference time
was added to any arm.

## Scientific gate and decision

`scientific_coverage_gate.json` records **INCONCLUSIVE**:

- Technical validity: PASS; all 72 decisions and full controls validated.
- Independent pilot instances: four, as predeclared.
- Variation: four of six cells meet the frozen range/distinctness/instance
  variation screen; the two early cells have narrow stagnation ranges.
- Outcomes: 55 non-tied native primary outcomes, at least two in all six cells.
- Incremental prediction: FAIL; stagnation increases MSE and improves no
  held-out instance folds, rather than meeting the frozen ≥10%/≥3-fold rule.

The resource budget was not a reason to expand after this failed scientific
gate. **No 8–12-instance expansion or confirmation data were generated.** The
reserved seeds20261202/20261203 remain unused. No thresholds or predictor
specifications were changed to rescue the result.

**INCONCLUSIVE for useful stagnation predictiveness; no expansion and no new
scheduler justified.** The paired intervention can have positive value, but
this pilot provides no incremental predictive advantage from stagnation after
the stated coarse controls. A claim about state-triggered scheduling deserves
independent evidence beyond that fact. Stronger population conclusions would
require more independent graphs and better identification of state, exact
iteration and interaction effects; they are not supplied by this round.

## Artifact index and figures

All paths below are under
[`20261031_controlled_v3_01`](../diagnostics/20261031_controlled_v3_01/):

- `pilot_protocol.json`, `validation_gate.json`, `preservation_manifest.json`.
- `technical_tests.log`, `extended_tests.log`, `analysis_contract_tests.log`.
- `pilot/manifest.json`, `pilot/datasets/`, `pilot/runtime_pilot.json`,
  `pilot/completion.json`, `execution.log`.
- [All paired observations](../diagnostics/20261031_controlled_v3_01/pilot/paired_observations.csv),
  `pilot/controls.json`, `pilot/runs/`: all 216 arm JSONs and 72 prior-pair NPZs.
- [Instance summaries](../diagnostics/20261031_controlled_v3_01/independent_instance_summary.csv),
  `outcome_summary.csv`, `coverage_by_age_phase.csv`.
- [Prediction models](../diagnostics/20261031_controlled_v3_01/prediction_model_summary.csv),
  `instance_blocked_prediction_folds.csv`, `instance_blocked_predictions.csv`,
  `within_stratum_descriptive_associations.csv`.
- `final_length_precision_check.csv`, `artifact_audit.json`, `analysis.log`.
- [Age/phase-controlled scatter](../diagnostics/20261031_controlled_v3_01/stratified_stagnation_benefit.png)
  and [independent-instance effects](../diagnostics/20261031_controlled_v3_01/instance_benefit.png).

Both figures were generated exclusively from persisted measurements and
visually checked. The common vertical scale in the six-cell plot intentionally
shows how small middle/late effects are compared with early effects; raw tables
retain their numerical values. No illustrative curves or confidence bands are
substituted for data.
