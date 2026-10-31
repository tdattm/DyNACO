# Main Coordination handoff — R1

- **Status:** Initial PoC and controlled experiment complete; gates A/B/C/D PASS.
- **Branch:** `feature/r1-event-triggered-guidance`.
- **Base / HEAD:** `cd16e2a68fd035f94a8916d41463904f31483d7f`.
- **Git delivery:** Uncommitted local additions; no new commits or push.
- **Changes:** Isolated fixed/event inference, deterministic saved datasets,
  synchronized timing, actual incumbent signals, 21 automated tests, CSV/JSON
  experiment records and six measured-data plots. Original solver/model/native
  sources and checkpoint weights remain unchanged.
- **Verified:** Original CLI and checkpoint run on WSL2 CUDA; unchanged native
  sources rebuild with locally extracted GCC/OpenMP. Fixed-S traces agree
  exactly with upstream. All 21 tests pass. All 80 matched runs succeed at
  N=100/200/500/1000, two instances × two seeds × five policies, 1,000 ACO
  iterations and 100 ants. Two additional principal TSP-1K repeats are exact.
- **Results:** TSP-1K event cost/time/calls = 23.189062 / 2.017 s / 17.75 means;
  S100 = 23.188910 / 1.837 s / 10. No compelling benefit over S100/S200. At the
  predefined one-second TSP-1K budget, event has the lowest observed mean cost;
  this remains a small exploratory observation.
- **Limitations:** Only two independent instances per size; no significance or
  optimality-gap claims; cross-scale 1K checkpoint; laptop timing noise; only
  two exact call-budget matches; causal refresh-versus-retain branching deferred
  because full RNG state is not exposed. Initial harness/precision failures and
  their justified corrections are retained. No unresolved mandatory gate.
- **Recommendation:** NO-GO for scaling the current rule unchanged; the general
  R1 hypothesis is unresolved.
- **Next step:** Predeclared fresh-instance equal-call-budget and repeated
  0.5–1 s wall-budget comparison. No new research round has been launched.
- **Evidence:** `artifacts/r1/reports/R1_INITIAL_RESULTS.md` and
  `artifacts/r1/experiments/20261031_matched_v1/`.
