# R1 initial results — 31 October 2026

**Recommendation: NO-GO for scaling or adopting the current stagnation rule.**
The implementation and reproduction gates passed. The tested event policy did
not show a compelling overall advantage over fixed S=100/S=200. This is a
negative result for this exploratory rule and sample, not a disproof of all
event-triggered guidance. A small positive wall-budget observation warrants
verification before any broader claim.

## 1. Executive summary

- Original base: `cd16e2a68fd035f94a8916d41463904f31483d7f`; initially clean `final`.
- Working branch: `feature/r1-event-triggered-guidance`; HEAD remains the base.
  No local commits, pushes, merges, resets or rebases were performed.
- Original inference reproduced at N=100/200/500/1000 with the tracked TSP-1K
  checkpoint. The original CLI ran two TSP-100 instances, 100 ants, 100 iterations.
- Instrumented fixed-S matched original 100-iteration traces exactly with and
  without annealing. Same-seed baseline repeats were exact.
- All **21 tests passed**, none skipped. The independent artifact audit passed.
- **80 principal GPU runs**, **80,000 ACO iterations**, **1,612 GNN calls**,
  **zero failed principal runs**. Six genuine-data figures were generated.
- Two additional TSP-1K, 1,000-iteration repeats exactly reproduced traces,
  final tours and refresh schedules. These repeats are excluded from summaries.
- At N=1000, event mean final cost was 23.189062 versus 23.188910 for S=100,
  taking 2.017 s versus 1.837 s and 17.75 versus 10 neural calls. At a predefined
  one-second budget, event had the lowest observed mean cost among these five
  methods. Earlier budgets favored other fixed schedules. This timing-dependent
  observation has only four instance-seed pairs and no statistical claim.

## 2. Evidence and provenance

The authoritative experiment directory is
[`20261031_matched_v1`](../experiments/20261031_matched_v1/).

| Artifact | Purpose |
|---|---|
| [Environment manifest](../environment/environment_manifest.json) | Runtime, packages, CUDA, native/checkpoint hashes and successful smoke checks |
| [Windows/repository audit](../environment/repository_windows_audit.json) | Initial clean checkout, branch/remotes, Windows, Python inventory, WSL and hardware |
| [Dependency lock](../environment/requirements-lock.txt) | Exact installed package versions |
| [Native build log](../environment/build_attempt2.log) | C++17/OpenMP probe and separate native rebuild |
| [Baseline validation](../baseline/baseline_validation.json) | Every baseline trace, active seeds, tour checks and repeats |
| [Original CLI log](../baseline_gates_attempt1.log) | Effective checkpoint-overridden configuration and unmodified `test.py` smoke |
| [Dataset manifest](../datasets/dataset_manifest.json) | Saved coordinates, generation parameters, IDs and checksums |
| [Gate record](../validation_gates.json) | A–D validation evidence |
| [Configuration](../experiments/20261031_matched_v1/benchmark_config.json) | Full controlled protocol, checkpoint metadata, source hashes and ordering |
| [Pilot](../experiments/20261031_matched_v1/pilot.json) | Observed runtime/memory before deciding the workload |
| [Runs CSV](../experiments/20261031_matched_v1/runs.csv) | Every principal configuration/result, including failure fields |
| [Iterations CSV](../experiments/20261031_matched_v1/iterations.csv) | 80,000 per-iteration incumbent/time records |
| [Refresh CSV](../experiments/20261031_matched_v1/refresh_events.csv) | 1,612 actual forward calls and reasons |
| [Summary CSV](../experiments/20261031_matched_v1/summary.csv) / [JSON](../experiments/20261031_matched_v1/summary.json) | Counts, mean, median, sample SD and descriptive comparisons |
| [Paired differences](../experiments/20261031_matched_v1/paired_differences.csv) | Every matched instance/seed difference |
| [Call-budget matches](../experiments/20261031_matched_v1/gnn_budget_matches.csv) | Exact counts and limitations of nearest fixed schedules |
| [Wall-budget results](../experiments/20261031_matched_v1/wall_budget_results.csv) | Quality at predeclared elapsed times |
| [Principal repeats](../experiments/20261031_matched_v1/reproducibility.json) | Exact TSP-1K repeat checks |
| [Artifact audit](../experiments/20261031_matched_v1/artifact_audit.json) | Independent counts, hashes, seeds, candidate identity and trace checks |

Each principal run also has a full JSON with tour, iteration trace, effective
configuration, scheduler configuration and GPU telemetry. SHA-256 of checkpoint:
`33f7da301e49f74d5b66de2a12c7bd95f247de8c217343c3410bb716a6f22358`.

## 3. Environment and setup

Windows 11 Home Single Language, build 26300.9711; WSL 2.4.13.0, Ubuntu 24.04.2,
kernel 5.15.167.4. Intel i5-11320H, four physical/eight logical cores, about 16 GB
system RAM. WSL exposed 7.7 GiB RAM and 2 GiB swap. All search used **two OpenMP
threads**, and PyTorch/MKL/OpenBLAS were bounded to two threads.

Project-local CPython **3.13.18**, PyTorch **2.7.1+cu126**, CUDA runtime **12.6**,
PyG **2.7.0**, NumPy **2.5.4**, SciPy **1.18.1**, pybind11 **3.1.0**. GPU:
RTX 3050 Laptop, 4096 MiB, Windows driver 577.00. CUDA availability and tensor
matrix multiplication passed inside the selected Linux environment. No CPU
benchmark results are mixed into the GPU experiment.

No compiler or OpenMP runtime was initially available in WSL. Ubuntu GCC 13.3
and required development/runtime packages were downloaded and extracted under
`.r1-env/sysroot`; they were not installed system-wide. An initial stale
`linux-libc-dev` package URL returned 404; the published base version
6.8.0-31.31 was fetched instead. The first local link attempt lacked libc runtime
paths; local symlinks to the existing host libc resolved it. The C++17/OpenMP
probe reported `OpenMP=201511 threads=2`, and unchanged sources rebuilt into
`.r1-env/native`. No tracked binary/object was overwritten. Rebuilt extension
SHA-256: `8e1e67b69343956e4b1ce5f1ff667fd052b74132a7522d36346afc73fe926955`.

The environment is deliberately a minimal inference/test installation, not a
full `uv sync` of notebook/training dependencies. The user’s Windows Python,
drivers, WSL settings and system packages were not changed. No CUDA Toolkit,
Linux NVIDIA driver, Docker, administrator operation or restart was needed.

## 4. Baseline reproduction and validation gates

| Gate | Status | Evidence |
|---|---|---|
| A: native solver and frozen neural model execute | PASS | Separate rebuild, OpenMP probe, strict checkpoint load, finite (100,32) guidance, 100 valid sampled tours |
| B: valid reproducible original fixed-S | PASS | Ten diagnostic runs across four sizes, 100 iterations each; two exact repeats at TSP-100; original CLI smoke |
| C: instrumentation preserves fixed-S semantics | PASS | Two original-vs-instrumented regression tests; exact traces with annealing on/off and logging on/off |
| D: mandatory scheduler/integration verification | PASS | 21 passing tests, zero failed/skipped in final run |
| Post-experiment data consistency | PASS | 80 runs, 80,000 rows, 1,612 forward events; candidate/seed/hash consistency; two exact principal repeats |

Baseline used two persisted instances at every size, execution seed 1234,
K=32, 100 ants, H=2, mini_H=50, rho=0.1, min_new_edges=12, original local linear
annealing 1→0, full local search/extension, smooth MMAS and normalized heuristic.
All neural state keys loaded strictly. Every sampled ant tour was checked for
closure, permutation and independently recomputed Euclidean length. Maximum
absolute length discrepancies were approximately 4.4e-6, 9.0e-6, 2.0e-5 and
4.0e-5 at N=100, 200, 500 and 1000 respectively. The validation bound is
`2e-5 * max(1,length)` to accommodate float32 native accumulation versus float64
recomputation. No invalid or non-finite baseline solution was accepted.

### Recorded failures and corrections

The first diagnostic harness incorrectly expected N entries; the binding
returns N+1 with the start node repeated. The harness now verifies closure and
checks the first N entries. The failed diagnostic is retained as
`environment_manifest_prebuilt.json` and `diagnose_prebuilt.log`; it is not a
solver failure or a baseline compatibility patch.

The first full suite had 18 passes and one CPU/GPU precision failure. A fixed
elementwise absolute tolerance 2e-5 failed near a zero-valued logit. Independent
[float64 diagnostics](../environment/device_precision.json) found identical
coordinates/candidate indices, feature difference 3.58e-7, CPU/GPU maximum
output difference 6.48e-5, and CPU/GPU float32 errors against float64 of
6.31e-5/5.14e-5 on output scale 73.62. The corrected test checks both devices
against float64 using a normwise bound of 16 float32 epsilons times that scale;
it does not demand cross-device bit identity. Same-device regression and repeat
tests still demand exact traces/tours. Both test logs are retained. No production
model, neural weights, RNG or solver changes were used to resolve this test.

### RNG implications

`utils.infer_instance` constructs two solvers. Its explicit `seed` applies to
the discarded first solver. The active second solver receives one NumPy global
RNG draw. Therefore the function seed alone does **not** control the active
search: NumPy state, dataset order and prior calls matter. This behavior was
recorded before implementing R1 and is preserved by the measured engine.

Each benchmark run resets Python/NumPy/Torch to its recorded execution seed;
dataset generation uses an independent PCG64 stream. All compared methods in a
matched instance/seed block receive the exact same active native seed and
candidate graph. The original CLI uses one continuing NumPy stream across its
dataset, whereas diagnostic independent runs reset per run. Their different
dataset traversal seed contracts must not be conflated. No RNG fix was needed.
MFACO's parallel fast path generates ant seeds before entering OpenMP and uses
per-ant generators; exact reproducibility was observed with two threads on this
environment. This does not establish bit identity on arbitrary hardware,
thread counts, PyTorch builds or for the separate MMAS algorithms.

## 5. Implementation and changed files

Only `.gitignore` changed among pre-existing files; it ignores `.r1-env/`.
Original `utils.py`, `test.py`, `faco.py`, `net.py`, training and native sources
remain unchanged relative to the base.

| New files | Purpose |
|---|---|
| `r1/engine.py` | Isolated measured inference mirroring the original search/seed behavior; exact budgets, tour checks and synchronized timing |
| `r1/scheduler.py` | Fixed-S/event policies, significant global incumbent tracking and reasons |
| `r1/datasets.py` | Versioned deterministic PCG64 datasets, stable IDs and checksums; refuses incompatible overwrite |
| `r1/benchmark.py` | Warmup/pilot, bounded sequential matched experiment, manifests, raw records and repeat checks |
| `r1/metrics.py` | Paired statistics and right-continuous wall-budget measurements |
| `r1/report.py` | Summary CSV/JSON and six measured-data figures |
| `r1/__init__.py`, `r1/README.md` | Package and reproducible command/semantics documentation |
| `tests/test_r1_fixed.py`, `test_r1_scheduler.py`, `test_r1_integration.py` | Original regression, scheduler boundaries, real native/model integration and precision |
| `tests/test_r1_datasets.py`, `test_r1_metrics.py` | Dataset preservation, correct pairing and clock-budget sampling |
| `artifacts/r1/environment/` scripts | Local bootstrap/compiler/build/run, OpenMP probe, baseline diagnostics, precision investigation and independent result audit |
| `artifacts/r1/baseline/`, `datasets/`, `experiments/`, `reports/` | Preserved evidence, datasets, raw results, six plots and scientific handoff |

`GuidanceScheduler.decide(j,best)` runs before the next ACO iteration, with j
completed iterations. `observe(j+1,best)` runs after pheromone update. Initial
refresh at j=0 counts. No refresh occurs at or after the final budget. Partial
last blocks are truncated without extra iterations or GNN calls.

The event rule first checks age ≥200; otherwise age ≥25 plus stagnation ≥50.
Stagnation is measured from C++ **global `best_cost`**, including its initial
incumbent, not `source_cost`, an ant's iteration best, or Python sampled best.
Significant improvement means more than absolute 1e-6 below the last significant
incumbent; accumulated tiny improvements can eventually cross that threshold.
Refreshing does not reset stagnation. Long stagnation can therefore refresh
repeatedly every 25 iterations. Guidance is reused unchanged between calls.

Event mode requires constant guidance (`no_anneal`). Primary experiments disable
phased injection. Original-fixed local annealing is supported and regression
tested separately. No candidate admission, candidate size, SRR, architecture,
weights, training, adaptive annealing or native search behavior changed.

## 6. Experimental design and commands

Hypothesis: an event schedule improves quality versus computation relative to
fixed-S. Independent variable: refresh schedule. Outcomes: final/anytime tour
length, model calls, synchronized runtime and component overhead.

Each size has two uniform [0,1)^2 float32 instances, dataset seed 20261031,
PCG64 `SeedSequence([dataset_seed,N])`, stable IDs and saved `.npy` coordinates.
Solver seeds: **1234 and 5678**. Every method uses the same saved instances,
checkpoint, active seeds, K=32, 100 ants, rho=0.1, min_new_edges=12, local search,
extension, smooth MMAS, normalized heuristic, two threads and CUDA:0.

Five policies: fixed S=25/50/100/200 and event (min=25,max=200,stagnation=50),
**1,000 iterations each**. Four matched instance-seed pairs per method/size,
20 runs per size, 80 total. Method order is shuffled within blocks by a separate
`random.Random(314159)`. No threshold tuning occurred on these instances. This
is exploratory evidence; no separate independent held-out set is claimed.

Both native backend and model were warmed up for 10 iterations at every size.
Four 200-iteration pilots projected 199.7 s, with a 2x margin of 399.4 s below
the 600 s cap. The measured search phase, including persistence and two
principal repeats, took 131.5 s; through completion metadata, excluding imports
and figure generation, 137.5 s. Cold imports on the Windows mount were slow and
are excluded from search timing.

Important executed commands, from PowerShell in the repository:

```powershell
git status --short
git branch -vv
git rev-parse HEAD
git remote -v
git log -1 --format=fuller
git switch -c feature/r1-event-triggered-guidance
wsl --status
wsl --list --verbose
wsl --version
py -0p
nvidia-smi
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/bootstrap.sh
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/compiler.sh
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/build.sh
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh artifacts/r1/environment/baseline_gates.py
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests/test_r1_fixed.py -q
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh artifacts/r1/environment/device_precision.py
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests -q
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m r1.benchmark --output artifacts/r1/experiments/20261031_matched_v1 --max-search-seconds 600
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh artifacts/r1/environment/audit_results.py artifacts/r1/experiments/20261031_matched_v1
```

`baseline_gates.py` prints and executes the full original CLI command with an
explicit checkpoint/dataset, H=2, mini_H=50, `--run_model_anneal`, `--baseline
none`, `--no_baseline`, `--threads 2`, `--iter_log` and explicit size/ant/seed
overrides. The effective configuration is in its log; checkpoint H/mini_H and
size did not silently override these values. Future runs must use a **new**
output ID; the benchmark refuses existing directories.

## 7. Actual matched-iteration results

Mean over four instance-seed pairs. Lower cost and runtime are better. These
are raw Euclidean lengths, **not optimality gaps**. Median, SD, every individual
result and component timing are in the linked CSV/JSON.

| N | Method | Mean final cost | Mean search seconds | Mean GNN calls |
|---:|---|---:|---:|---:|
| 100 | fixed S=25 | 8.129684 | 1.423 | 40 |
| 100 | fixed S=50 | 8.127321 | 1.110 | 20 |
| 100 | fixed S=100 | 8.127321 | 0.974 | 10 |
| 100 | fixed S=200 | 8.129684 | 1.072 | 5 |
| 100 | event | 8.132047 | 1.387 | 35.50 |
| 200 | fixed S=25 | 10.744143 | 1.669 | 40 |
| 200 | fixed S=50 | 10.744177 | 1.475 | 20 |
| 200 | fixed S=100 | 10.744177 | 1.176 | 10 |
| 200 | fixed S=200 | 10.746121 | 1.136 | 5 |
| 200 | event | 10.746120 | 1.489 | 31.50 |
| 500 | fixed S=25 | 16.901385 | 1.979 | 40 |
| 500 | fixed S=50 | 16.878096 | 1.288 | 20 |
| 500 | fixed S=100 | 16.884894 | 1.376 | 10 |
| 500 | fixed S=200 | 16.873687 | 1.159 | 5 |
| 500 | event | 16.901187 | 1.516 | 18.25 |
| 1000 | fixed S=25 | 23.208785 | 2.090 | 40 |
| 1000 | fixed S=50 | 23.242323 | 2.044 | 20 |
| 1000 | fixed S=100 | 23.188910 | 1.837 | 10 |
| 1000 | fixed S=200 | 23.188728 | 1.867 | 5 |
| 1000 | event | 23.189062 | 2.017 | 17.75 |

Paired mean relative cost differences, `100*(event/fixed-1)`. Positive is worse
for event. These percentages compare methods, not optimality gaps.

| N | vs S25 | vs S50 | vs S100 | vs S200 |
|---:|---:|---:|---:|---:|
| 100 | +0.02893% | +0.05786% | +0.05786% | +0.02893% |
| 200 | +0.01813% | +0.01782% | +0.01782% | -0.00001% |
| 500 | +0.00115% | +0.13789% | +0.09694% | +0.16377% |
| 1000 | -0.08514% | -0.22847% | +0.00100% | +0.00151% |

Event mean runtime exceeded S100 by 0.413/0.313/0.140/0.180 seconds at the four
sizes. Scheduler decisions/observations took about 4.6–6.5 ms per event run;
the additional model calls are a larger cost. Mean event GNN time was
0.520/0.577/0.440/0.462 seconds, graph construction 0.053/0.042/0.027/0.031
seconds and guidance D2H transfer 0.083/0.088/0.089/0.114 seconds.

Peak PyTorch allocated memory was 36,306,944 bytes (**34.6 MiB**) at TSP-1K.
This excludes allocator reservations and CUDA context memory. Recorded
post-run GPU snapshots were about 135 MiB total device usage; these snapshots
are not a total-VRAM peak measurement. No OOM occurred. Later telemetry was
around 68–69°C; clocks, Windows workload and laptop power management were not
locked, so small timing differences remain uncertain.

### Comparable GNN-call budgets

Only two of 16 event runs exactly matched one of the tested fixed call counts.
At TSP-500 instance 001/seed 5678, both event and S50 made 20 calls; event cost
was **0.039358 higher**. At TSP-1K instance 000/seed 5678, both made 20 calls;
event cost was **0.071367 lower**. Five of 16 had a nearest control within 10%
of the event call count. The other eleven do not support a close-budget
comparison. No full equal-call-budget causal experiment was achieved; nearest
matching was selected by call count, never by favorable quality.

### Comparable wall-clock budgets

Budgets were predefined as 0.1/0.25/0.5/1/2/5 s. A budget is included only if
every run at that size reached it, avoiding extrapolation or arbitrary unequal
iteration truncation. Incumbents are right-continuous and initialized from the
same native initial tour at elapsed zero. All include the initial neural call;
all exclude initialization/warmup consistently. No 2 s or 5 s comparisons had
a common observation horizon in this experiment.

TSP-1K mean incumbent cost at eligible budgets:

| Budget | Event | S25 | S50 | S100 | S200 |
|---:|---:|---:|---:|---:|---:|
| 0.10 s | 24.085020 | 23.873757 | 24.369212 | 24.057000 | 24.116854 |
| 0.25 s | 23.759727 | 23.326611 | 23.830862 | 23.745172 | 23.757031 |
| 0.50 s | 23.611530 | 23.233067 | 23.320063 | 23.277436 | 23.620075 |
| 1.00 s | 23.192452 | 23.216389 | 23.282531 | 23.227194 | 23.204066 |

The favorable event observation at 1 s is retained, alongside its worse early
quality and longer full-run runtime. It could reflect policy timing, a small
number of trajectories, or measurement jitter. It is not sufficient to claim
a general improved quality–compute frontier.

## 8. Figures from raw measurements

All six figures were generated from saved successful run JSON/CSV; no
illustrative curves were substituted. Wall curves stop at the common observed
horizon. Raw instance/seed variability includes between-instance differences;
paired differences are the appropriate method-comparison statistics.

- [Incumbent vs iterations](../experiments/20261031_matched_v1/figures/anytime_iterations.png)
- [Incumbent vs wall time](../experiments/20261031_matched_v1/figures/anytime_wall_time.png)
- [Final quality vs GNN calls](../experiments/20261031_matched_v1/figures/quality_vs_gnn_calls.png)
- [Final quality vs runtime](../experiments/20261031_matched_v1/figures/quality_vs_wall_seconds.png)
- [Refresh timeline](../experiments/20261031_matched_v1/figures/refresh_timeline.png)
- [Instance/seed variability](../experiments/20261031_matched_v1/figures/instance_seed_variability.png)

## 9. Interpretation and limitations

**Reported paper findings:** not independently reassessed here. No reproduction
of a published numerical benchmark, optimality gap or claimed paper speedup is
asserted. The paper-style phased-injection-plus-annealing protocol and unguided
ACO comparison were not separately benchmarked in this round. They are distinct
from the original Model(anneal) smoke and the clean no-anneal scheduling study.

**Reproduced original behavior:** original pretrained inference executes, valid
tours/costs are produced, fixed-S and annealing semantics are preserved, and
selected same-device configurations repeat exactly.

**New R1 observations:** the initial stagnation rule often spends extra calls
after prolonged stagnation, particularly at smaller sizes. It saves calls
relative to S25 but generally uses more than S100/S200. Improvements versus
weaker controls at TSP-1K do not establish an advantage versus all relevant
fixed schedules. Its one-second TSP-1K observation is a limited positive signal,
not proof of usefulness.

**Unverified hypotheses:** stagnation might predict a useful refresh in other
rules/datasets; the present study does not test the refresh-versus-retain causal
mechanism. Solver snapshots do not expose complete RNG state. Paired
counterfactual branching was deferred rather than treating unequal RNG states
as equivalent. No invasive native changes were made for that optional test.

There are only two independent instances per size, with two seeds each; four
pairs are not four independent instances. No significance tests or uncertainty
intervals are justified here. The frozen 1K checkpoint is used cross-scale;
there are no verified optimal tours or reference gaps. No held-out generalization
claim, broad threshold search, training or learned controller is involved.

Timing uses explicit CUDA synchronization, avoiding asynchronous launch-time
claims, but synchronization/logging alters absolute throughput. Controls receive
the same measurement protocol and logging-on/off output identity was checked.
Host/device transfers inside feature construction/native wrappers are attributed
to those components rather than separately isolated. Peak memory is allocated
PyTorch memory, not full VRAM. CPU/GPU numerical consistency is approximate;
same-device reproducibility was observed only under the tested configuration.
Results do not establish performance at TSP-5K/10K or on other machines.

## 10. Go/no-go and next research round

**NO-GO for promoting or scaling the current rule unchanged.** Fixed S100/S200
offer equal or better average final quality with lower average search time
under the principal iteration protocol. The R1 hypothesis remains unresolved
beyond these conditions. Preserve this negative result and the 1 s positive
observation together; do not select only favorable controls or seeds.

The most valuable bounded next R1 experiment is a predeclared equal-call-budget
comparison on fresh TSP-1K instances, including fixed spacing near the measured
event call counts and repeated wall-time measurements around 0.5–1 s. Use a
separate exploration set if changing the rule to avoid repeated refresh during
one long stagnation episode; reserve held-out instances for the final test.
Do not initiate training, R3 or adaptive candidate graphs. No next round was
launched by this task.

## 11. Research handoff summary

R1 initial PoC complete; gates A/B/C/D pass. Branch
`feature/r1-event-triggered-guidance`, HEAD/base
`cd16e2a68fd035f94a8916d41463904f31483d7f`; uncommitted local additions, no push.
Original solver/model/native sources unchanged. Added isolated fixed/event
inference, deterministic datasets, synchronized measurement, 21 tests and raw
CSV/JSON plus six plots. Eighty matched GPU runs succeeded at N=100/200/500/1000,
1000 iterations, 100 ants, two instances × two seeds. Original and instrumented
traces agree; two principal TSP-1K repeats are exact. No compelling advantage
over S100/S200; small one-second TSP-1K signal remains unverified. Recommendation:
NO-GO for scaling this rule. Next step: fresh-instance equal-call-budget and
repeated wall-budget test, without extending the research scope.
