# R3: budget-aware candidate-graph hypothesis test

This research package leaves all original solver/model/inference sources
unchanged. It starts from exact baseline
`cd16e2a68fd035f94a8916d41463904f31483d7f`, independently of R1.
The frozen scientific gate must pass before any adaptive solver implementation.

The Windows-created worktree is `D:\GitRepository\DyNACO-r3`; the original
checkout and R1 environment remain under `D:\GitRepository\DyNACO`.
R3 reuses interpreter/compiler dependencies read-only and rebuilds native code
under `.r3-build/`, preserving upstream binaries. The shell scripts currently
document the audited local environment path; adjust that explicit dependency
path for a fresh machine rather than silently using another backend.

From PowerShell:

```powershell
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3/r3/build.sh
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3/r3/run.sh -m pytest /mnt/d/GitRepository/DyNACO-r3/tests -q
```

Run order for the recorded study was environment inventory, protocol freeze,
baseline verification, final scanner resource pilots, then principal Stage A.
Entry points are `r3.environment`, `r3.protocol`, `r3.baseline`, `r3.pilot`,
`r3.principal`. Each refuses existing output files/directories. Do not overwrite
finalized outputs to rerun: create a new explicit experiment ID and retain its
relation to the unchanged frozen protocol. Failed attempt logs are retained.

- `protocols/stage_a_v1.json`: all seeds, thresholds, scope, controls and budgets.
- `native/two_opt_diagnostic.cpp`: independent C++17 O(N²) streaming scanner,
  O(NK) lookup state and bounded detailed storage; no solver includes/calls.
- `diagnostics.py`: native candidate audits and independently applied reversal
  gain verification using NumPy float64.
- `state_capture.py`: original H/S/annealing loop verified against unchanged
  `utils.infer_instance`; captures copied state without solver RNG draws.
- `replay.py`: source-edge-aware native relocation/checklist reconstruction.
- `selectors.py`: uniform, ten random, and predefined geometry controls.
- `gate.py` / `analysis.py`: instance-first summaries, explicit undefined ratios,
  gate decision, CSV/JSON raw summaries, report and handoff.

Primary evidence is `artifacts/r3/reports/R3_STAGE_A_REPORT.md` and
`artifacts/r3/experiments/stage_a_v1_01/gate.json`. Source-line audit is
`artifacts/r3/reports/SOURCE_AUDIT.md`. Raw `.npy/.npz/.json/.csv`, hashes,
native build logs, environment versions and regression traces are preserved.

Strict CUDA determinism (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off) is an
explicit execution contract. Reported runtimes must not be compared directly
with non-strict or differently instrumented historical runs.

Candidate membership, frozen-tour pruning-aware potential evaluation, initial
checklist coverage, actual SRR evaluation, oracle one-step gain and actual
solver improvement are distinct. The original binding exposes no native SRR
evaluation/acceptance sequence, so those actual-execution fields remain null;
all-node offline metrics do not substitute for scope-restricted execution.

The confirmation set (seed 20261104) remains uninspected unless later stages
are authorized by their scientific gates. No checkpoint training, learned edge
allocator, production candidate change or automatic push is part of Stage A.
