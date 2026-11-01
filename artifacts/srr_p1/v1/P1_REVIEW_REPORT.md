# M2-P1 technical and feasibility review

**M2-P1 READY FOR FINAL PROTOCOL FREEZE**, conditional on approval/sealing of
the final protocol, principal manifest and dispatch schedule. M2 principal was
not run and no principal coordinates were created or accessed.

Lightweight telemetry is backward compatible and opt-in. It counts only native
calls/evaluations with relaxed atomics; detailed mode keeps route-copy mutation
accounting. After removing only the new counter-condition disjunction, original
two_opt_nn body matches P0 text exactly. Scheduling, flip, selection and
pheromone methods are unchanged. Definitions and unmeasured slots are documented
in srr_p1/MEASUREMENT.md. Revert four production files to undo the integration
delta; all P0/history files are preserved in their original worktrees.

32 saved inputs match route/checklist/reference counts in unmeasured, detailed
and light modes, with no third pass or change to numerical no-ops. Disabled
arrays match frozen baseline; enabled-light matches P0's unmeasured archives,
including normal/traced paths, neural graph/input/output and deterministic RNG
behavior. The runner separately matches original utils.infer_instance on a
20-iteration100-ant gate, including discarded seeded construction, actual native
seed derivation, two guidance refreshes and annealing, cost history, source,
best and pheromone. Full-scale paired runs and repeat archives are exact on
tours/costs/state; timings appropriately differ. All12 full1000-iteration runs
and both4s runs completed within caps, after correctness gates and warmups.

Measured telemetry overhead by arm:
{
  "0": {
    "unmeasured": [
      2.293831358000034,
      2.332535507999978,
      2.6492755030000126
    ],
    "measured": [
      2.887554949000048,
      2.8715606340000477,
      2.935621868999988
    ],
    "paired_fraction": [
      0.2588348916450749,
      0.23108978369304833,
      0.10808478230207452
    ],
    "mean_fraction": 0.19933648588006592,
    "range_fraction": [
      0.10808478230207452,
      0.2588348916450749
    ],
    "sample_sd": 0.08023467335405315
  },
  "1": {
    "unmeasured": [
      2.563613750000002,
      2.4969250320000356,
      2.5044229619999783
    ],
    "measured": [
      3.16624594700005,
      3.246582246999992,
      3.5804961800000115
    ],
    "paired_fraction": [
      0.23507137024836422,
      0.3002321677233062,
      0.4296691231183667
    ],
    "mean_fraction": 0.32165755369667903,
    "range_fraction": [
      0.23507137024836422,
      0.4296691231183667
    ],
    "sample_sd": 0.09905229289922894
  }
}

This is material overhead, not zero-overhead measurement. Principal comparisons
would measure both arms identically; their speed/quality claim would apply to
that measured implementation. No algorithm policy was selected from pilot quality.
Unmeasured versus measured timing uses paired identical inputs, three repeats,
alternating order and per-arm100-iteration warmup. Variability and startup noise
limit inference from three repetitions on one seed.

Wall-time checkpoints: control1400 eligible of
1401 attempts; treatment1218 of
1218. Overshoot0.050733s and
0.000000s. Audit recomputes the credited float64 length
from the last eligible saved tour; over-budget output is not credited.
Actual calls/evaluations include all attempted work, even the over-budget last
iteration. The4s budget is technically usable and retained prospectively.

Checkpoint load in the main pilot: 0.2612s.
Peak pilot RSS:1378066432 bytes. Every run's allocated/reserved
CUDA memory and setup/search timing are persisted. startup_probe.json measures
import, CUDA initialization and checkpoint loading in a SEPARATE startup-only
process; it is not an exact decomposition of the main pilot's first startup.
Compilation happened before tests/pilots and is excluded from search timing.
Warmup durations were not individually persisted; overall process time includes
them. This measurement limit does not change the paired search comparisons.

The inherited build shell had CRLF and failed before compiling; a new P1 LF
build script preserves P0. The first pilot launch lacked r3_diagnostic on its
dependency path and failed at import before registration or replay; its log is
retained. The corrected environment changed no source, budget or scientific
design. No correctness mismatch was ignored or tolerance relaxed.

## Protocol gap analysis

Original utils.infer_instance constructs twice and draws its active native seed
from global NumPy state: runner preserves this, not a direct execution-seed
substitution. Guidance block horizon is100; wall runs may complete more blocks
than H10, but preserve the same refresh/within-block annealing. Original reference
uses unsynchronized time.time; the prospective runner uses synchronized monotonic
search checkpoints for credible equal-time accounting. Per-step recording has
cost and is charged equally. Initial construction, normalization, checkpoint
load and warmup stay outside search with separate reporting. Float64 global-best
tours, not selected source or final ant, determine outcomes. Complete-grid
instance aggregation is implemented, but principal dispatch/data sealing awaits
approval and must not be bypassed by the development-only pilot runner.

192-run estimate:750.1s; doubled-search/setup envelope:1459.2s,
45min cap margin:1.85x. Generalization across instance difficulty,
hardware thermals and background load is uncertain. Resource failure remains a
predefined inconclusive outcome, never grounds for replacing instances.

M2_REVISED_PROTOCOL.md explains co-primary rules,16-instance paired uncertainty,
practical0.1% threshold and limitations of the>=12/16 sign condition. These
criteria are proposed and not scientifically validated by development results.
Historical Stage A NO-GO, secondary INCONCLUSIVE, M0 VALIDATED, M1v1 technical
INCONCLUSIVE, M1v2 EXPLORATORY SIGNAL, M1.5 counts and P0 READY remain unchanged.
No native numerical repair, adaptive graph, NLS change, training, M2 principal
or Stage B occurred. No push/merge is authorized by this delivery.
