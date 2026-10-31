# R1 Round 2 — paired replay blocker resolved

## Status

**Technical validation gate: PASS.** Strict CUDA inference resolved the
cached-guidance byte mismatch without changing the model, weights, graph,
native ACO or numerical dtype. Recorded-guidance replay was not needed.

The original 21 tests pass in the default environment. The extended suite
passes all **26 tests**, including exact native input priors, every ant's costs
and tours in the prefix, complete HOLD-vs-HOLD continuation, final search state,
actual forward-call counts and cross-process frozen-graph output identity.

The bounded pilot completed **16 decisions / 48 arm executions / 48 horizon
observations**, with all controls passing. Recommendation: **CONDITIONAL GO for
additional paired diagnostics only**. These results do not justify a new
state-aware scheduler or an end-to-end speedup claim.

## Git state and preservation

Branch `feature/r1-event-triggered-guidance`, HEAD
`8ee944b8f5929bbadea8d69a00beac12f89779e6`; changes are local and uncommitted.
No push, merge, reset, environment reinstall or native rebuild was performed.
All v1 experiment files passed their original SHA-256 checks again. Old v2
measurements and failed logs remain intact. Copies of the previous report,
handoff and failing test source are in the new output directory:

`artifacts/r1/diagnostics/20261031_replay_resolution_01/`.

The original base is cd16e2a. V1 experiment sources were identified by hashes
while uncommitted on that base; 8ee944b later packaged them. The previous
blocked v2 attempt and this resolution are further local source states, with
new source hashes recorded in the pilot manifest. None of those states should
be equated merely because HEAD stayed at the packaging commit.

## Original failure and exact hashed values

The preserved failed test is
`../diagnostics/20261031_v2_01/replay_tests.log`; it failed on
`cached_prior_sha256` in the HOLD-vs-HOLD prefix assertion. Its test source is
copied as `original_replay_test.py` in the new directory. It used persisted
TSP-100 instance 000, seed 31415 and probe iteration 25.

The old implementation hashed the cached prior at the branch point by copying
the CUDA tensor to CPU and immediately hashing its NumPy bytes. It retained a
hash string, not a mutable tensor reference. The new diagnosis traced a longer
S100 HOLD prefix on the same instance/seed through iteration 175 to compare
both refreshes at completed iterations 0 and 100. The very first refresh is
the same normalized graph construction as the original failing test.

Each refresh now has immutable NumPy snapshots in `default_02/` and `strict/`:
normalized graph x, edge_index, edge_attr, pheromone, source tour, output prior
reshaped to (100,32), and the actual NumPy prior passed to native sampling.
CUDA is synchronized before copying. Layout is canonical C-contiguous, bytes
are copied, and dtype stays float32 (int64/int32 graph/tour indices keep their
own dtype). No numerical rounding or tolerance is used for hashing. Metadata
includes completed iteration, call ordinal, eval state, full state_dict hash
including buffers, raw input coordinate hash, graph/state hashes, output
dtype/device/layout/stride and determinism settings. The recorded output
tensor shape is the flat model output (3200,); saved/native prior arrays are
(100,32). Both representations contain the same ordered float32 values.

## Earliest divergence and numerical magnitude

Before the initial model forward, both HOLD branches have identical coordinate,
graph x/edge_index/edge_attr, pheromone, source tour and model-state bytes. All
state_dict hashes remain unchanged after the forward; eval mode is active.
The **first divergence is the initial neural output**, before any ACO sampling.
The actual native prior carries the same differences as that output.

Default-mode replay comparisons over 3,200 float32 prior values:

| Refresh iteration | Unequal elements | Maximum absolute difference | RMS difference |
|---:|---:|---:|---:|
| 0 | 2655 | 4.9591064e-5 | 8.1870077e-6 |
| 100 | 2944 | 2.4795532e-5 | 4.0205297e-6 |

At iteration zero, maximum relative difference is 5.85654e-5, using
`abs(a-b)/max(abs(a),abs(b),float32.tiny)` elementwise. ULP-distance quantiles
0/50/90/99/100% are 0/2/6/16/768. Signed IEEE float32 values are mapped to an
ordered integer representation before computing distance; signed zero is
checked separately. No NaNs, infinities or signed-zero-only differences were
observed. SHA inequality therefore represented small rounding differences,
not a large or invalid neural output.

Immutable snapshots reproduced their recorded hashes in the independent audit.
Changing an original CUDA tensor after taking a snapshot leaves that snapshot
unchanged in a dedicated test. Mutable aliasing is not the explanation.

## Frozen-graph reproduction and root cause evidence

The actual initial graph and checkpoint were used in an isolated probe:

- Two repeated forwards on the same CUDA graph and two forwards on newly cloned,
  byte-identical CUDA graph tensors all differed from a reference CUDA forward.
  Maximum absolute output differences ranged from 3.05e-5 to 5.72e-5.
- Two repeated CPU forwards on the frozen graph were byte-identical. CPU and
  CUDA outputs were not required to match each other.
- Hooks found identical first-layer linear outputs. The first divergent module
  output was `emb_net.layers.0.v_bn.module`, downstream of aggregation.
- Calling the first layer's `global_mean_pool` twice on the **same fixed message
  tensor and destination indices** produced 1,399 differing values out of
  3,200, maximum absolute difference 2.98023e-8 and RMS 6.06995e-9.
- Inspection of the installed PyG implementation shows `global_mean_pool`
  routes to mean scatter, whose summation uses PyTorch `scatter_add_`.

This directly localizes an earliest nondeterministic operation to CUDA mean
aggregation on identical inputs. Those small differences propagate through the
12-layer network. It is not inferred merely from a final hash mismatch or from
the name of a suspected operator. The investigation does not claim a complete
kernel-level trace of every later rounding difference.

## Strict CUDA execution and replay contract

A separate process was started with `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before
PyTorch/CUDA initialization**. It enabled:

```python
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

PyTorch remains 2.7.1+cu126, CUDA runtime 12.6, PyG 2.7.0. No operator raised an
unsupported deterministic implementation error. Both replay refreshes, all
four frozen-graph repeats, and the isolated aggregation were byte-identical.
A second process/test loaded the persisted graph and reproduced the same exact
output bytes. The full old hash assertion is retained and now passes under
this explicit configuration; it was not replaced with a numerical tolerance.

Strict mode is opt-in in `replay_determinism.py`. The replay tests use a scoped
fixture restoring prior PyTorch flags, and the standalone pilot requests strict
mode explicitly. The v1 entry points/default timing protocol do not enable it.
Test commands must set the workspace variable before process startup; a missing
variable raises an explicit error rather than silently relaxing determinism.

Both counterfactual arms now regenerate their prehistory with live strict-mode
GNN evaluation, resetting the documented original Python/NumPy/Torch and native
seed semantics. The native seed draw and duplicate solver construction remain
as before. At every ACO iteration, the exact native prior bytes, all-ant cost
array and all-ant closed-tour array are hashed. Branch-point pheromone,
incumbent/source tours and costs, candidate/backup lists, heuristic, trail limits
and cached guidance must match. A complete HOLD-vs-HOLD continuation must match
all those per-iteration histories and final search-state hashes. Actual model
forward hooks independently verify no extra neural calls.

This is stronger than v1's best-cost trace agreement and stronger than the
previous branch-point hash check. It proves exact observed native inputs,
solutions and exposed states for tested replays. Native RNG state is still not
exposed. No complete internal RNG-state inspection or exact post-intervention
random-number coupling is claimed. Per-ant trajectories may consume/use random
draws differently after prior values change. Conclusions concern the defined
single intervention under shared initialization and verified identical prefix.

Recorded guidance playback was not implemented because strict live inference
satisfied the exact contract. The fallback is unnecessary for this environment.

## Tests and technical gate

| Check | Result |
|---|---|
| Original v1 suite, default environment/no workspace override | 21 passed, 1 deprecation warning, 20.61 s |
| Extended suite, strict configuration scoped to replay tests | 26 passed, 1 deprecation warning, 26.15 s |
| Identical frozen graph, repeated strict CUDA outputs | Exact |
| Persisted frozen graph, separate process | Exact |
| Original cached-guidance equality assertion | Retained and passing in strict mode |
| All-ant costs/tours and native prior history, shared prefix | Exact |
| Full HOLD negative control and final exposed states | Exact |
| Weights/buffers unchanged; actual model-call count | Passed |
| Observer/sample logging vs ordinary fixed prefix | Passed |

The original failing test/log remain preserved as historical evidence. New
tests cover immutable snapshots, numerical/signed-zero comparison, the stronger
native history, exact call counts, full controls and cross-process graph replay.
`validation_gate.json` was written with PASS **before** launching the pilot.

One preliminary diagnosis attempt stopped on a reporting bug: `np.finfo` was
applied to an integer graph array. That comparison helper was corrected to use
a floating denominator for integer comparisons. The failed attempt and its raw
snapshots remain under `default/`; the complete successful default diagnosis is
`default_02/`. This did not alter a model, prior or solver value.

## Bounded paired pilot

The plan was saved in `pilot_protocol.json` before observing pilot outcomes.
Fresh generator-v1 float32 uniform coordinates use PCG64
`SeedSequence([20261101,N])`, with **two independent instances per size** at
N=500/1000 and execution seeds **31415/27182**. Dataset files, IDs and SHA-256
are saved in `pilot/datasets/` and `pilot/manifest.json`.

Probe times **175 and 725**, horizons **25/50/100**. Prefix refreshes occur at
0,100,... strictly before the probe. Subsequent refreshes are frozen in all
arms; REFRESH makes exactly one additional live GNN call at the probe. HOLD and
HOLD_CONTROL retain their old prior. Actual calls are **2/3** for HOLD/REFRESH
at t175 and **8/9** at t725, counting the initial call. Arm order is shuffled by
a separate fixed RNG seed 9237.

All other controls use the frozen 1K checkpoint, K=32, 100 ants, rho=.1,
min_new_edges=12, smooth MMAS, normalized heuristic, original local search and
extension, two OpenMP/PyTorch threads, CUDA:0, constant guidance and no phased
injection. Both native backend and model were warmed up at each size.

The pilot completed all 16 decisions in **97.9 s** after Python imports,
including model/data initialization, warmup, repeated prefixes, hashing and
writing raw outputs, within its 240 s cap. The maximum recorded per-run
PyTorch allocated peak was **127,881,216 bytes (~122 MiB)**; this is not total
VRAM/context/reserved memory. No OOM or failed decision occurred. Post-pilot
GPU telemetry reported 265 MiB device usage and 64°C as a snapshot, not a peak.

## Pilot findings

Positive benefit means HOLD cost minus REFRESH cost, so positive favors REFRESH.
Percent benefit is `100*(HOLD-REFRESH)/HOLD`, not an optimality gap. Means below
use eight matched decisions per size/horizon, but those decisions contain only
**two independent problem instances**. HOLD_CONTROL is a repeatability check,
not an additional seed or problem instance.

| N | Horizon | Mean benefit, raw length | Mean benefit, percent | Better / worse / tied decisions |
|---:|---:|---:|---:|---|
| 500 | 25 | -0.00010324 | -0.000627% | 2 / 1 / 5 |
| 500 | 50 | -0.00312757 | -0.018943% | 1 / 3 / 4 |
| 500 | 100 | -0.00045323 | -0.002768% | 1 / 3 / 4 |
| 1000 | 25 | +0.01702166 | +0.072562% | 4 / 0 / 4 |
| 1000 | 50 | +0.01391077 | +0.059322% | 3 / 1 / 4 |
| 1000 | 100 | +0.01165223 | +0.049799% | 3 / 2 / 3 |

Ties use absolute raw difference ≤1e-6 as a descriptive category; exact raw
values remain available. At N=1000/h100, both instance-level means are positive
(0.009218 and 0.014087). At N=500/h100 the two instance means have opposite signs
(-0.002608 and +0.001701). No confidence interval or significance claim is made
with two independent instances per size.

Splitting by probe reveals an important limitation:

- At TSP-1K/t175, mean raw benefits are +0.034043/+0.027822/+0.023587 over
  25/50/100 iterations. Stagnation duration is only **1–3 iterations**, while
  cached prior age is 75. These are actively improving states, not prolonged
  stagnation.
- At TSP-1K/t725, benefits are 0/0/-0.000283. Prior age is 25 and stagnation
  ranges 1–122. One of four decisions worsens at h100; the others tie.
- All TSP-500/t725 decisions tie at every horizon. Its t175 outcomes are mixed.

Thus the pilot provides limited evidence that a refresh can have a marginal
effect at some states; it does **not** show that stagnation predicts that benefit.
Absolute search phase and cached-prior age are confounded at the two probe
times. In particular, these observations cannot justify making the original
stagnation rule more elaborate.

## Timing interpretation

The additional strict-mode synchronized forward costs average **64.7 ms** at
N=500 and **80.5 ms** at N=1000. These are actual model executions, separate
from graph construction and native continuation. At horizon 100, mean
instrumented suffix wall differences are +54.5 ms and +48.4 ms respectively;
noise and different continuation work mean these need not equal forward cost.
All raw per-decision timings are retained, including negative differences.

Full replay reconstruction cost is separately available in each run's setup,
wall, graph, forward, transfer, decision and sample/update fields. Hashing all
native priors/tours and recording branch snapshots add diagnostic overhead to
these live-inference runs. There are **no recorded-prior playback timings** and
no artificial added inference times. These instrumented strict-mode measurements
are neither ordinary DyNACO runtime nor comparable to v1 default-kernel timings.
There is no end-to-end wall-budget scheduler result in this task.

## Files, artifacts and reproduction

Changed existing R1 files: `engine.py` adds an optional post-sample observer;
`v2_replay.py` adds exact native prior/all-ant histories and final-state checks;
the previous replay test adds the scoped strict fixture and full-continuation
checks. Original hash assertions remain. `README.md` documents the new command.

New modules: `replay_determinism.py` (strict opt-in settings),
`replay_diagnose.py` (immutable per-refresh/frozen-graph diagnosis),
`replay_pilot.py` (bounded gated experiment). New tests:
`conftest.py` (scoped configuration), `test_r1_replay_resolution.py` (three
stronger tests). Changes from the preceding uncommitted v2 attempt are also
preserved. Native C++, `net.py`, `utils.py`, checkpoint weights and v1 results
are unchanged.

Within the new diagnostic directory:

- `default_02/diagnosis.json`, `strict/diagnosis.json`: quantitative comparisons
  and layer/aggregation/CPU controls, with raw per-refresh `.npz` snapshots.
- `hold_a/hold_b` trajectory JSONs in those directories: replay cost histories,
  branch states, refreshes and timings.
- `v1_default_tests.log`, `resolved_tests.log`, `validation_gate.json`.
- `pilot_protocol.json`, `pilot/manifest.json`, `pilot/completion.json`.
- `pilot/runs/`: all 48 raw arm records, including exact prior/solution hashes.
- `pilot/controls.json`, `pilot/paired_observations.csv`: 16 controls/48 horizons.
- `instance_summary.csv`, `pilot_summary.csv`: explicit raw/percent outcomes.
- `pilot_instance_benefit.png`: real instance-level results, averaging its two
  solver seeds and two probe times; no synthetic curve or uncertainty band.
- `analyze_results.py`, `artifact_audit.json`, `artifact_audit.log`: independent
  persisted-data audit of joins, exact prefixes/full controls, call counts,
  monotonic costs, tours, source/data hashes, and original v1 artifact hashes.

Executed commands (from repository PowerShell; output IDs are unique):

```powershell
git status --short
git rev-parse HEAD
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m r1.replay_diagnose --output artifacts/r1/diagnostics/20261031_replay_resolution_01/default_02
wsl -d Ubuntu -- env CUBLAS_WORKSPACE_CONFIG=:4096:8 bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m r1.replay_diagnose --strict --output artifacts/r1/diagnostics/20261031_replay_resolution_01/strict
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests/test_r1_fixed.py tests/test_r1_scheduler.py tests/test_r1_integration.py tests/test_r1_datasets.py tests/test_r1_metrics.py -q
wsl -d Ubuntu -- env CUBLAS_WORKSPACE_CONFIG=:4096:8 bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests -q
wsl -d Ubuntu -- env CUBLAS_WORKSPACE_CONFIG=:4096:8 bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m r1.replay_pilot artifacts/r1/diagnostics/20261031_replay_resolution_01
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh artifacts/r1/diagnostics/20261031_replay_resolution_01/analyze_results.py
```

Existing outputs are not rerun into the same directories. For a new run, use a
new root ID and protocol/gate files. Historical v2's BLOCKED gate is retained;
the new PASS gate applies only to this documented exact strict-mode contract.

## Next R1 experiment

**CONDITIONAL GO for further diagnostics; no scheduler development yet.**
Use fresh independent TSP-1K instances and predeclared probe times crossing both
prior ages (25/75) at matched early/late phases, with enough instances to assess
within-cell stagnation associations. Keep the exact negative control, include
uninformative and harmful interventions, and aggregate by problem instance.
Only after state predictiveness survives age/phase controls should an equal-call
comparison against fixed and state-independent irregular schedules be considered.
No such expansion, threshold tuning or new rule was initiated here.
