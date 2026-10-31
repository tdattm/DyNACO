# R3 research handoff

- Branch: `feature/r3-budget-aware-candidate-graph`; collection commit `4b8961744f597b75a623dda70f074471bdcfa51b`.
- Exact original base: `cd16e2a68fd035f94a8916d41463904f31483d7f`; R1 branch/history/artifacts unchanged.
- Work completed: isolated worktree/build, source/paper audit, frozen protocol,
  independent C++17 2-opt diagnostic, numerical/candidate/replay/regression tests,
  four-size baseline smokes, full 1K/1000-iteration exact original regression,
  same-seed repeat, all 16 principal runs and instance-level gate.
- Data: 8 TSP1K PCG64 instances seed20261101; solver seeds31415/27182;
  original K32,100 ants,H10,S100,M12,rho.1,smooth,normalized,extended SRR,
  original Model(anneal), frozen checkpoint. Strict CUDA, two CPU threads.
- Decision: **NO-GO**; G0-G4 `{'G0': True, 'G1': False, 'G2': False, 'G3': False, 'G4': True}`;
  G1/G2/G3 counts `{'G1': 0, 'G2': 0, 'G3': 0}`; ratios `{'ratio_vs_uniform': None, 'ratio_vs_random': None}`.
- Primary evidence: 0/32 late source snapshots had robust off-K32 moves; all
  equal-budget policies added zero gain mass. Residual improving moves were
  already K32-eligible. This does not rule out initialization/other-distribution
  effects and is not a universal rejection of candidate expansion.
- Secondary raw-ant0 scope: 30/32 late pre-SRR tours did have off-K32 moves,
  and the frozen geometry selector covered more offline gain mass on average.
  This is a different observable from the best-of-100 post-SRR source and cannot
  replace the gate or establish solver improvement. Any follow-up needs its own
  prospectively registered question; no automatic continuation was performed.
- Tests: 23/23 passed on the final scanner build; 18 focused checks passed after
  extra artifact fields. Post-collection audit rehashed 440 files, regenerated
  data exactly, checked 192 tour records, reapplied 5407 moves and matched the
  recorded gate. Failed development attempts remain available.
- Collection provenance: the recorded SHA is supplemented by its stored dirty
  diff and complete source hashes, including a read-only WSL Git compatibility
  fix. Final delivery commit is reported with the terminal summary.
- Offline oracle/coverage findings are not actual solver improvements. Actual
  dynamic SRR evaluation/acceptance trace is unavailable; initial checklist and
  all raw tours were reconstructed exactly. Source and global best are separate.
- Runtime to gate 106.212s; RSS high-water
  1432006656 bytes; Torch allocation peak
  127517696 bytes.
- Stage B: not executed. Confirmation seed20261104: reserved/uninspected.
- Main coordination action: inspect the frozen gate and source/paper mismatch;
  do not reinterpret expanded-LS oracle coverage as neural or end-to-end gain.
- Primary report: `artifacts/r3/reports/R3_STAGE_A_REPORT.md`.
- Raw/provenance/gate: `artifacts/r3/experiments/stage_a_v1_01/`.
- Protocol SHA256: `1f9d4cc3e66c8bc22a4f792594c1adc3cbd1a906e34d39fd4e784d1e79bc84cc`.
- Dataset SHA256: `8ec4e091f5f749ce15dcd236e47e8919601d7112cd0c3bac3c458ae002527789`.
- Checkpoint SHA256: `33f7da301e49f74d5b66de2a12c7bd95f247de8c217343c3410bb716a6f22358`.
- No push, merge, reset, stash, clean or R1 modification was performed.
