# R1: event-triggered neural guidance PoC

Round 2 exact replay now uses **opt-in strict CUDA**. Start the test process with
the workspace variable set before PyTorch/CUDA initialization:

```powershell
wsl -d Ubuntu -- env CUBLAS_WORKSPACE_CONFIG=:4096:8 bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests -q
```

Only tests requesting `strict_cuda_replay` enable deterministic algorithms;
their fixture restores the previous precision/backend flags. The original v1
benchmark path does not enable this configuration. Replay diagnostic timings
include strict kernels and hashing and must not be treated as v1 runtime.
See `artifacts/r1/reports/R1_REPLAY_RESOLUTION.md` for the cause, exact replay
contract, preserved failure evidence and bounded live-inference pilot.

The hypothesis is unverified. This package compares scheduling policies using
the frozen upstream TSP-1K model and unchanged MFACO search. The original
`test.py`, `utils.py`, `faco.py`, `net.py` and native sources remain available.

## Validated local environment

Windows 11 / WSL2 Ubuntu 24.04.2, isolated CPython 3.13 environment under
`.r1-env/`, CUDA-enabled PyTorch, and a separate rebuild under `.r1-env/native`.
No global Python packages, GPU drivers, WSL settings, or system packages changed.
Full versions are in `artifacts/r1/environment/requirements-lock.txt` and
`environment_manifest.json`. Only inference/test dependencies were installed;
the full notebook/training dependency set from `uv sync` was not installed.

Run these commands in PowerShell from the repository root:

```powershell
# Existing environment: imports, native extension and GPU are selected by run.sh.
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests -q
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m r1.benchmark --output artifacts/r1/experiments/NEW_UNIQUE_ID --max-search-seconds 600
```

The benchmark refuses an existing output directory. It persists pilots before
estimating the full cost, runs one GPU workload at a time, and retains failures.
It never substitutes CPU measurements into GPU comparisons. See
`artifacts/r1/environment/{bootstrap,compiler,build}.sh` for the actual setup:
local uv installation, extracted Ubuntu packages, a C++17/OpenMP probe, and an
out-of-tree native rebuild. Those scripts assume this repository's current path.
For a fresh exact environment, install the recorded requirements lock using
the PyTorch `https://download.pytorch.org/whl/cu126` extra package index; the
bootstrap's unpinned ancillary packages reproduce the setup procedure, not a
guarantee of future exact package versions. Preserve the recorded Ubuntu package
versions and Python 3.13.18 for an exact reconstruction.

## Scheduling and RNG semantics

`RunConfig` controls inference; `SchedulerConfig` selects `fixed_s` or
`event_stagnation`. `scheduler=None` runs instrumented fixed-S. The ordinary
upstream CLI remains unchanged and is the independent regression reference.

All scheduler iteration numbers count **completed** ACO iterations. A refresh
at iteration 50 happens before the 51st ACO iteration. The initial neural call
at 0 counts. Only iterations strictly below the requested budget can refresh.
Fixed-S handles a partial last block by truncation; original-fixed annealing
uses the original S-1 denominator even in that partial block.

The event rule refreshes initially, at a maximum age of 200, or after at least
25 iterations since refresh and 50 iterations since a significant global-best
improvement. Maximum age takes precedence. Stagnation uses C++ `best_cost`,
including its initialized incumbent; it never uses `source_cost` or the
iteration-best ant. A significant improvement is more than absolute 1e-6 below
the last significant incumbent. Smaller cumulative improvements can eventually
cross that threshold. Refreshing does **not** reset stagnation; prolonged
stagnation can therefore cause repeated refreshes every 25 iterations.

Event mode requires `no_anneal`. The primary experiment disables phased
injection. `original_fixed` preserves upstream local linear annealing for
fixed-S fidelity tests. No adaptive annealing or injection policy is inferred.

The engine deliberately preserves upstream duplicate solver construction. Its
first explicit solver seed is discarded; the active solver is seeded by one
draw from the NumPy global RNG. Each independent run resets Python/NumPy/Torch
with its recorded execution seed. Dataset generation uses an independent PCG64
stream and saved coordinates. This establishes a run-level seed contract
without changing upstream RNG semantics. It differs from traversing an entire
dataset under a single uninterrupted NumPy stream in `test.py`; do not equate
its `seed+i` argument with the active native seed.

## Measurement definitions

Search wall time starts immediately before the first scheduling decision and
includes initial guidance, every ACO iteration/update, synchronization, checks,
and logging. Solver initialization and normalized coordinates are separately
recorded as setup. Each size receives native/model warmup before pilots and
principal measurements. Cold imports, CUDA initialization, setup and warmup are
excluded from search wall time, with aggregate startup recorded separately.

All CUDA timing uses explicit synchronization. Graph timing includes internal
host-to-device feature transfers. `transfer_seconds` measures the guidance
device-to-host conversion on every ACO iteration, preserving the original path.
`sample_update_seconds` includes native sampling and pheromone updates plus
wrapper tensor transfers. These internal transfers are not separately attributed.
Scheduler timing includes decisions and incumbent observations. Component sums
need not equal wall time: finite checks, logging and synchronization boundaries
also cost time. GPU peak is PyTorch allocated bytes, not total context/reserved
VRAM; `nvidia-smi` telemetry is retained separately. Process RSS high-water marks
are process-wide, not isolated per-run memory peaks.

All reported tours are independently checked as permutations with Euclidean
float64 length recomputation; native costs accumulate in float32. The bound is
`2e-5 * max(1, recomputed_length)`. CPU/GPU neural consistency is checked against
a float64 reference with a normwise bound of 16 float32 epsilons times the
reference output infinity norm. This is separate from exact same-device trace
regression/reproducibility tests. The initial failed elementwise tolerance and
its precision diagnosis are preserved under `artifacts/r1/`.

## Files and evidence

- `engine.py`: inference, counters, timing, incumbent and tour validation.
- `scheduler.py`: small deterministic policy and input validation.
- `datasets.py`: persisted float32 coordinates, stable IDs and SHA-256.
- `benchmark.py`: sequential pilots, matched blocks, manifests, raw data and repeats.
- `metrics.py`: matched differences and right-continuous wall-budget sampling.
- `report.py`: tables and six figures drawn from persisted measurements.
- `tests/test_r1_*.py`: scheduler, actual checkpoint/backend regression, dataset,
  metrics, device precision and integration verification.
- `artifacts/r1/reports/R1_INITIAL_RESULTS.md`: scientific interpretation and handoff.

No optimum is fabricated. No optimality gaps or statistical significance are
reported. The refresh-versus-retain counterfactual is deferred because the
binding does not expose complete RNG state for equivalent paired branching.
