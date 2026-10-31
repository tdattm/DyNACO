# R1 Round 2 coordination handoff — replay resolved

**Latest status:** Exact replay validation PASS. The historical BLOCKED summary
below is superseded by this resolution; its original file was also copied to
`artifacts/r1/diagnostics/20261031_replay_resolution_01/` before editing.

- **Branch/HEAD:** `feature/r1-event-triggered-guidance`,
  `8ee944b8f5929bbadea8d69a00beac12f89779e6`; local changes uncommitted; no push/merge.
- **Cause:** Initial CUDA GNN outputs differed despite identical frozen graph
  and model state. Isolated first-layer mean aggregation reproduced differences;
  default initial prior max absolute difference 4.96e-5, RMS 8.19e-6.
- **Resolution:** Opt-in strict CUDA with workspace configured before process
  startup and `warn_only=False`. Exact repeated/cross-process outputs, actual
  native priors, all-ant cost/tour prefixes and full HOLD controls now pass.
  No recorded-guidance fallback, numerical tolerance relaxation, model or
  native algorithm changes were required.
- **Tests:** Original 21 pass in default mode; complete extended suite 26 pass.
- **Pilot:** 16 decisions, 48 live-inference arm runs, 48 horizon observations;
  TSP-500/1K, two fresh instances per size, seeds 31415/27182, probes175/725,
  horizons25/50/100. All controls pass; one extra neural call per intervention.
- **Findings:** At horizon100 mean HOLD-minus-REFRESH cost is -0.000453 at
  N500 and +0.011652 at N1000. Early N1000 states were actively improving;
  late probes mostly tied. Prior age and phase are confounded. Only two
  independent instances per size; no significance or state-predictiveness claim.
- **Timing:** Strict instrumented diagnostics, not v1 runtime. Marginal forward
  costs average 64.7/80.5 ms; no end-to-end scheduler benefit established.
- **Recommendation:** CONDITIONAL GO for age/phase-controlled paired
  diagnostics only. No new scheduler or expanded benchmark was launched.
- **Evidence:** `artifacts/r1/reports/R1_REPLAY_RESOLUTION.md` and
  `artifacts/r1/diagnostics/20261031_replay_resolution_01/`.

## Historical blocked handoff

- **Status:** BLOCKED at paired replay validation; Stage A/B complete.
- **Branch/HEAD:** `feature/r1-event-triggered-guidance`,
  `8ee944b8f5929bbadea8d69a00beac12f89779e6`; local uncommitted changes.
  No push, merge or history rewrite.
- **Provenance:** All seven v1 source hashes match packaging commit 8ee944b.
  V1 ran with those sources uncommitted on base cd16e2a. All v1 experiment
  artifact hashes remain unchanged.
- **Completed:** Retrospective descriptions of 1,612 refreshes, 568 episodes,
  and 4,664 eligible 25/50/100-iteration follow-ups; three real-data figures.
- **Observation:** Event policy median noninitial interval is 25 at every size;
  max calls in an uninterrupted stagnation episode are 38/31/19/18. At TSP-1K,
  44/71 event calls occur late. These findings are observational, not causal.
- **Software:** Original 21 tests pass; extended suite 22 pass/1 fail. Added
  read-only observer, exact replay checks and a proposed runner guarded by a
  BLOCKED technical gate. No native or neural model changes.
- **Blocker:** Same-seed HOLD replay costs and checked exposed search states
  match, but cached neural guidance byte hashes differ. Full negative control
  and intervention validity are not certified. No tolerance was relaxed.
- **Research:** Zero accepted causal decisions; no fresh diagnostic datasets,
  new scheduler, equal-call comparison or held-out benchmark executed.
- **Recommendation:** BLOCKED for v2 science; no additional rule complexity.
  V1 NO-GO for scaling its existing rule remains applicable.
- **Next safe action:** Diagnose exact deterministic GPU guidance replay;
  validate negative control and complete branch state before the predeclared
  TSP-500/1K pilot. Full internal native RNG equivalence is not claimed.
- **Evidence:** `artifacts/r1/reports/R1_DIAGNOSTIC_V2.md` and
  `artifacts/r1/diagnostics/20261031_v2_01/`.
