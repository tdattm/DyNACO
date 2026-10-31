# R1 Round 2 — original diagnostic blocker (now resolved)

**Current status:** The subsequent exact-replay resolution passed its technical
gate and completed a bounded 16-decision pilot. See
[R1_REPLAY_RESOLUTION.md](R1_REPLAY_RESOLUTION.md) for the diagnosis, strict CUDA
contract, 26 passing tests and mixed pilot findings. The text below preserves
the earlier blocked round as historical evidence. A verbatim copy before this
status note is saved in `../diagnostics/20261031_replay_resolution_01/`.

## Outcome and stopping decision

Stages A and B completed. Stage C's strict replay validation failed before the
planned TSP-500/1K diagnostic pilot. No causal research probes, new scheduling
policy, budget-controlled comparison or held-out benchmark were launched.
The evidence does not justify further rule complexity. The scientific decision
is **BLOCKED**, rather than a claim that state signals are useful or useless.

Exact prefix costs are insufficient to certify identical guidance history.
Two independent HOLD replays from the same seed had different cached neural
prior SHA-256 values at the branch point. The test retained its exact equality
requirement and failed. No tolerance relaxation, approximate pairing, native
change or new controller was used to bypass this result.

## Git and provenance

Local branch: `feature/r1-event-triggered-guidance`.
HEAD and verified remote branch: `8ee944b8f5929bbadea8d69a00beac12f89779e6`.
Original DyNACO base: `cd16e2a68fd035f94a8916d41463904f31483d7f`.
This round has uncommitted local changes; no commit, push, merge, reset or
history rewrite was performed.

V1 ran while HEAD still pointed to the original base and the R1 files were
uncommitted. Its seven recorded source hashes identify the actual experiment
source state. Commit 8ee944b later packaged those sources and the results.
Before v2 code changes, all seven working source hashes matched the experiment
manifest exactly. Historical blobs at 8ee944b also match all seven hashes.
V1's report statements about uncommitted work describe its experiment-time
state, not the later delivery state.

The independent v2 audit verified every saved v1 experiment artifact against
the hashes captured at the start of this round. No previous measurement was
overwritten. The already established WSL2 CUDA environment was reused without
installation, rebuild or baseline reconstruction.

## Stage A: audit and software validation

The original 21 tests passed before changes: `21 passed, 1 warning in 36.04s`.
After adding the observer and replay tests: **22 passed, 1 failed**, with the
same PyG deprecation warning. All original 21 tests remained passing; the new
observer-equivalence test passed. The failing test is
`test_replay_intervention_and_negative_control`, at `cached_prior_sha256`.

V1 counters were checked against each run's events and iteration records:
1,612 neural calls across 80 runs and 80,000 ACO iterations. Stagnation values
match reconstruction from the C++ global incumbent, using the existing
absolute 1e-6 significant-improvement tolerance. Source-tour cost is not used
as the stagnation signal. Initial calls count; no terminal refresh is added.

V1 search timing includes the initial guidance evaluation and excludes setup
and warmup. It synchronizes CUDA around measured operations. Graph construction
contains internal H2D transfers; guidance D2H is measured per iteration; native
sampling/update timing contains wrapper transfers. Timing components do not
sum exactly to wall time because checks, logging and synchronization also cost
time. Those definitions were preserved. V1's old ambiguous paired
`difference` column was not rewritten; v2 uses explicit `_raw` and `_percent`
names in every new comparative summary.

## Stage B: observational refresh diagnostics

Authoritative new output directory:
`artifacts/r1/diagnostics/20261031_v2_01/`.

There are 1,612 complete event descriptions, 568 run/episode records and 4,664
eligible event/horizon observations. Horizons are 25, 50 and 100 iterations;
observations are omitted when the recorded trajectory ends before the horizon.
Every observation retains its horizon and number of intervening refreshes.
Initial events are included and explicitly labeled. These repeated events and
overlapping windows are correlated; they are not independent instances.

Event-policy totals over four instance/seed runs at each size:

| N | Calls | Median noninitial interval | Median stagnation at call | Early / middle / late calls | Maximum calls in one stagnation episode |
|---:|---:|---:|---:|---|---:|
| 100 | 142 | 25 | 425 | 37 / 51 / 54 | 38 |
| 200 | 126 | 25 | 250 | 28 / 46 / 52 | 31 |
| 500 | 73 | 25 | 125 | 10 / 24 / 39 | 19 |
| 1000 | 71 | 25 | 125 | 8 / 19 / 44 | 18 |

Early means iterations 0–332, middle 333–666, late 667–999. Initial age is zero
and is excluded from interval quantiles. Noninitial event interval ranges are
25–113, 25–134, 25–200 and 25–200 respectively. The interquartile interval is
25–25 for N=100/200, 25–62 for N=500, and 25–61.5 for N=1000.

All policies together have 80 initial, 1,136 fixed-interval, 385 stagnation and
11 maximum-interval events. Fixed schedules have constant noninitial ages
25/50/100/200 by construction, with 40/20/10/5 calls per run. The event rule
puts 62.0% of its TSP-1K calls in late search, compared with 30% for S100.
Episode identity is the completed iteration of the last significant incumbent
improvement, and refresh does not re-arm or reset it. Calls spent in episodes
with more than one refresh total 134/115/53/55 at the four event-policy sizes.

These results confirm repeated spending during long stagnation and a strong
search-phase difference. They do not demonstrate that those calls are
unproductive: subsequent improvement might arise without refreshing, and
other refreshes may occur during each follow-up window. The initial greedy
incumbent also makes initial-call follow-up improvements much larger than
many late-call improvements. No causal comparison is inferred from these
descriptive means.

The raw observation CSV supports joint analysis by size, phase, prior age,
stagnation and horizon. It records both `subsequent_improvement_raw` (tour
length before minus after) and `subsequent_improvement_percent`
(`100*(before-after)/before`). Percentages are relative improvements from the
observed incumbent, not optimality gaps.

Predeclared diagnostic hypotheses, without a threshold sweep:

1. Marginal refresh benefit may vary with stagnation within comparable size,
   search phase and prior-age strata.
2. Repeated late refreshes during one long stagnation episode may yield little
   benefit, even when the stagnation threshold remains satisfied.
3. Any benefit may be too small to compensate for one additional GNN execution.

## Stage C: protocol and exact replay blocker

The predeclared protocol is saved as `protocol.json`. Planned diagnostic data:
fresh uniform float32 TSP-500 and TSP-1K coordinates, generator version 1,
PCG64 dataset seed **20261101**, four independent instances per size, solver
seeds **31415, 27182, 16180**. Probe iterations **125, 175, 425, 475, 725, 775**
cross three search phases with cached-prior ages 25 and 75. Horizons are
**25/50/100**. No such diagnostic dataset was generated or evaluated because
the prerequisite replay test failed. These are planned parameters, not results.

The intended prehistory is S100 with constant guidance. After the probe, both
arms freeze subsequent refreshes for 100 iterations; REFRESH alone evaluates
the GNN at the probe. HOLD retains its cached prior. HOLD_CONTROL reruns HOLD
from the same initial seed. Arm order is randomized. Each decision would require
three complete replays, and the planned four-decision pilot would precede a
resource-capped expansion. The cap is 900 seconds; no pilot/runtime projection
is claimed because the scientific experiment never started.

The small software validation case used existing TSP-100 instance 000,
execution seed 31415, probe t=25 and total 125 iterations. HOLD, HOLD_CONTROL
and REFRESH were executed as test fixtures, not independent scientific samples.
The HOLD prefixes had identical sampled-best and global-incumbent traces.
Exact checks passed for exposed incumbent/source costs, stagnation, prior age,
pheromone, incumbent tour, source tour, candidate graph, backup graph and
heuristic array before the comparison reached the cached prior. The
`cached_prior_sha256` assertion failed. The remainder of the negative-control
test was not reached, so complete HOLD-vs-HOLD continuation equivalence and
the REFRESH intervention are not certified.

The observed blocker is same-seed cached-guidance byte inequality. Floating
point GPU reduction nondeterminism is a plausible explanation, but the
responsible operation and numerical magnitude were not isolated. There is no
evidence here of a native algorithm change. The original cost-trace tests can
pass while small guidance differences leave the sampled trajectory unchanged;
they provide a weaker guarantee than the new branch-point requirement.

The native fast path generates per-ant RNG seeds before OpenMP work. With an
identical prefix and identical guidance, replay is a plausible coupling
mechanism. Native RNG state is not exposed, however, and equivalence cannot be
certified merely by hashing exposed snapshots. After interventions diverge,
candidate selection may consume random values differently. Matching initial
seeds would then supply common initial randomness without guaranteeing that
all subsequent random draws correspond to the same decisions. No full internal
RNG-state equivalence or unrestricted common-random-number coupling is claimed.

The next safe action is a bounded diagnosis of deterministic neural execution
on the same GPU, checking exact cached guidance and all branch states before
rerunning the negative control. This could require a separately validated
deterministic PyTorch configuration. It was not silently enabled in this round,
and numerical tolerances were not loosened to manufacture an accepted pair.

## Gate and answers to the research questions

| Question | Finding |
|---|---|
| Does refresh cause measurable marginal benefit over hold? | Unknown: no accepted causal pairs. |
| Does benefit depend on stagnation or cheap signals? | Unknown; v1 descriptions alone cannot establish predictiveness. |
| Can signal effects be separated from prior age/search phase? | Planned factorial probe times address these confounds, but the experiment did not run. |
| Does a state-aware rule beat fixed-S at equal calls? | Not evaluated in v2; Stage D was not authorized by the gate. |
| Is there end-to-end wall-clock benefit? | No new valid measurement; v1's one-second observation remains unverified. |
| Is it robust on held-out instances/seeds? | Not established; Stage E did not run. |
| Is further R1 rule complexity justified? | No. Resolve replay validity before further scientific development. |

The predeclared scientific continuation gate required technical validity,
coverage of both stagnation groups within multiple phase/age cells, and
consistent within-cell evidence across independent instances at both 50 and
100 iterations. Its first prerequisite failed. No new budgeted policy or random
schedule control was developed, and no held-out claims are made. Budget tests
for a new policy are therefore not applicable, rather than silently skipped
while claiming a completed scheduler comparison.

## Files, commands and artifacts

Existing file changed: `r1/engine.py` gains an optional read-only observer before
each scheduling decision. With no observer, the prior path remains available.
The new observer identity test passed. Observer work would count in wall time;
proposed suffix timing includes branch-state capture and is not a clean isolated
neural execution measurement. No suffix timings were accepted as research data.

New modules: `v2_retrospective.py` (v1 audit and records), `v2_replay.py`
(intervention and exact checks), `v2_experiment.py` (guarded proposed runner),
`v2_describe.py` (derived summaries, figures and artifact audit), plus
`tests/test_r1_v2_replay.py`. The proposed runner refuses execution unless a
separate technical gate record says PASS. Current `technical_gate.json` is
BLOCKED. The runner has not passed an end-to-end experiment test.

Executed commands, from the existing repository/environment:

```powershell
git status --short
git rev-parse HEAD
git branch --show-current
git ls-remote --heads origin feature/r1-event-triggered-guidance
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests -q
py -3.13 -m r1.v2_retrospective artifacts/r1/diagnostics/20261031_v2_01
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m pytest tests -q
py -3.13 -m r1.v2_retrospective artifacts/r1/diagnostics/20261031_v2_01/retrospective_complete_02
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO/artifacts/r1/environment/run.sh -m r1.v2_describe artifacts/r1/diagnostics/20261031_v2_01
git diff --check
```

An intermediate retrospective refresh attempted to compare the now-modified
v2 working engine against v1 hashes and correctly rejected it. Historical v1
blob checks are now used for retrospective source provenance. An artifact-audit
attempt also failed on Windows backslash paths under WSL; path normalization
fixed that portability issue. These were tooling failures, not v1 measurement
changes. Their logs are retained where captured; the successful complete audit
is `artifact_audit_attempt3.log`. No original log or measurement was rewritten.

Key artifacts relative to the diagnostic directory:

- `v1_audit.json`: initial source match, experiment/packaging provenance and
  hashes of all original v1 experiment artifacts.
- `initial_tests.log`: 21 original tests passing.
- `replay_tests.log`: 22 passing tests and the exact cached-prior failure.
- `protocol.json`, `technical_gate.json`: predeclared plan and stop decision.
- `retrospective_complete_02/v1_refresh_events_all.csv`: all 1,612 calls.
- `retrospective_complete_02/v1_episode_calls.csv`: 568 episode records.
- `retrospective_complete_02/v1_refresh_observations.csv`: 4,664 follow-ups.
- `retrospective_summary_all.csv`, `observational_followup_summary.csv`:
  explicit-unit descriptive summaries.
- `figures/observed_refresh_intervals.png`, `observed_refresh_phases.png`,
  `observational_followup.png`: three plots generated exclusively from v1 data.
- `artifact_audit.json`: independent observational consistency PASS, unchanged
  v1 hashes, source hashes, and zero accepted causal/held-out runs.

Plots are descriptive only. No plot presents invented refresh-versus-hold
benefits. The initial v1 audit and completed retrospective files are retained
in distinct locations rather than replacing earlier output.

## Recommendation

**BLOCKED: deterministic paired replay has not met the stated protocol.**
The retrospective results reinforce the need to understand repeated refresh
spending, but do not establish useful state prediction. Preserve v1's NO-GO
for scaling its existing rule. The most valuable next action is exact guidance
replay diagnosis, followed by the already predeclared bounded probe pilot only
if the negative control and branch-state checks pass.
