"""Independent audit and prospective review from saved development records only."""
import json,statistics,time,sys
from pathlib import Path
import numpy as np
from r3.common import ROOT,sha256,git,CHECKPOINT
from srr_p1.analysis import unit_check
OUT=ROOT/'artifacts/srr_p1/v1'
def save(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def main():
    assert not (OUT/'pilot_failure.json').exists(),'blocked pilot must be reviewed as failure'
    pilot=json.loads((OUT/'pilot_summary.json').read_text());assert pilot['status']=='PASS'
    assert json.loads((OUT/'correctness_gate.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'runner_reference_gate.json').read_text())['status']=='PASS'
    rows=pilot['runs'];assert len(rows)==14
    overhead={};measured={}
    for arm in (0,1):
        ratios=[];plain=[];light=[]
        for r in range(3):
            tag=f'iter_r{r}_arm{arm}'
            with np.load(OUT/(tag+'_light0.npz')) as a,np.load(OUT/(tag+'_light1.npz')) as b,np.load(OUT/(f'iter_r0_arm{arm}_light0.npz')) as reference:
                for k in ('costs','tours','source','best','pheromone'):
                    assert np.array_equal(a[k],b[k]) and np.array_equal(a[k],reference[k])
            p=next(x for x in rows if x['tag']==tag+'_light0');m=next(x for x in rows if x['tag']==tag+'_light1')
            assert p['iterations']==m['iterations']==1000 and m['calls']==100000*(arm+1)
            plain.append(p['search_seconds']);light.append(m['search_seconds']);ratios.append(m['search_seconds']/p['search_seconds']-1)
        overhead[arm]=dict(unmeasured=plain,measured=light,paired_fraction=ratios,
            mean_fraction=statistics.mean(ratios),range_fraction=[min(ratios),max(ratios)],sample_sd=statistics.stdev(ratios))
        measured[arm]=light
    walls=[r for r in rows if r['tag'].startswith('wall_')]
    for r in walls:
        with np.load(OUT/(r['tag']+'.npz')) as z:
            times=z['times'];idx=np.flatnonzero(times<=4.)
            assert len(idx)>0 and len(idx)==r['eligible_iterations'] and idx[-1]+1==r['primary_iteration']
            t=z['tours'][idx[-1]]
            x=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy')[0].astype(np.float64)
            length=float(np.linalg.norm(x[t]-x[np.roll(t,-1)],axis=1).sum())
            assert length==r['primary_length_f64']
    startup=json.loads((OUT/'startup_probe.json').read_text())
    search=48*max(measured[0])+48*max(measured[1])+96*(4+max(r['overshoot_seconds'] for r in walls))
    setup=192*max(r['setup_seconds'] for r in rows)
    startup_allowance=startup['import_seconds']+startup['cuda_initialization_seconds']+startup['checkpoint_load_after_cuda_seconds']+10
    estimate=search+setup+startup_allowance;conservative=2*(search+setup)+startup_allowance
    assert conservative<2700
    historical=json.loads((ROOT/'artifacts/srr_p0/v1/historical_hashes.json').read_text())
    allowed={'faco.py','src/mfaco_train.cpp','src/mfaco_train.h','src/binding.cpp'}
    assert all(sha256(ROOT/p)==h for p,h in historical.items() if p not in allowed)
    p0=json.loads((ROOT/'artifacts/srr_p0/v1/final_delivery_manifest.json').read_text())
    assert all(sha256(ROOT/'artifacts/srr_p0/v1'/p)==h for p,h in p0.items())
    baseline=Path('/mnt/d/GitRepository/DyNACO-srr-p0')
    def method(s,name):
        a=s.index(name);b=s.index('{',a)+1;n=1
        while n:n+=(s[b]=='{')-(s[b]=='}');b+=1
        return s[a:b]
    old=(baseline/'src/mfaco_train.cpp').read_text();current=(ROOT/'src/mfaco_train.cpp').read_text()
    assert method(old,'MFACO_TSP::two_opt_nn(')==method(current,'MFACO_TSP::two_opt_nn(').replace('measure_srr || lightweight_srr','measure_srr')
    for name in ('MFACO_TSP::flip_route_section(','MFACO_TSP::run_srr_schedule(','MFACO_TSP::select_next_node(','MFACO_TSP::update_pheromone('):assert method(old,name)==method(current,name)
    save('aggregation_unit_check.json',unit_check())
    summary=dict(decision='M2-P1 READY FOR FINAL PROTOCOL FREEZE',overhead=overhead,
        prospective_runs=192,estimated_seconds=estimate,conservative_2x_seconds=conservative,
        cap_seconds=2700,margin_ratio=2700/conservative,walls=walls,
        historical_hashes_preserved=True,principal_data_access=False,principal_execution=False,
        assessment='Conditional feasibility on one observed instance/seed; resource failure remains possible on independent data.')
    save('timing_memory_audit.json',summary)
    sources={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'srr_p1').glob('*') if p.is_file()}
    sources.update({p:sha256(ROOT/p) for p in allowed})
    save('source_environment_manifest.json',dict(base_commit='21ce14680c3b84ecd7f2e39223b6ca3c37674bd2',
        source_sha256=sources,checkpoint_sha256=sha256(CHECKPOINT),environment=startup,
        binary_sha256={str(p):sha256(p) for mode in ('native','test') for p in (ROOT/f'.srr-build/{mode}').glob('*.so')}))
    protocol=f'''# Revised M2 prospective protocol — FOR REVIEW, NOT EXECUTED

This draft supersedes no historical result. Approval and a final immutable
registration are required before principal generation or access. No principal
instances were created or inspected by P1. P1 READY is technical/resource
readiness, not scientific GO or R3 Stage B authorization.

## Data, code and deterministic order

After approval only: NumPy Generator(PCG64(202612070201)).random((16,1000,2),
dtype=np.float32), one C-contiguous draw, no rejection/rescaling/replacement.
Pin NumPy/Python versions in source_environment_manifest.json; persist .npy
bytes and SHA256 before any solver run. Use16 independent instances and paired
execution seeds31415,27182,16180. Reserved Stage A confirmation stays sealed.
No principal data is present here. Checkpoint SHA256:
{sha256(CHECKPOINT)}. Freeze source_environment_manifest.json and binary hashes
from P1 with the final registration; no unapproved source/data change thereafter.

Deterministic order: budget modes iterations then wall; instance0..15; seeds
in the listed order; arm order(0,1) if(instance+seed_index) even, otherwise(1,0).
192 total runs. Before each run utils.set_seed(execution_seed), construct and
discard the explicitly seeded first solver, construct active solver, then set
native seed int(np.random.randint(0,2**63-1)) exactly as utils.infer_instance.
Do not replace this original duplicate-construction/native-seed contract.

## Arms, guidance and budgets

Original K32 versus exactly one additional unchanged K32 SRR call. Both use
100 ants, backup32, min_new_edges12, rho0.1, alpha1,p_best0.05, local search and
checklist extension on, smooth MMAS on, normalized heuristic on, NLS off,
fixed_steps0, dynamic graph guidance, frozen original checkpoint, CUDA float32.
Both use lightweight call/evaluation counters; detailed route-copy telemetry
is off. srr_metrics fields2/3 are unmeasured, not zero accepted-event claims.
Candidate ordering, RNG, neural architecture, pheromone and original numerical
no-op acceptance stay unchanged. Two CPU threads, sequential GPU execution.

Guidance refresh every100 completed iterations; within each block factor
1-mini/99, gamma1,min_gamma0. Use actual native global-best route, not only
the last ant or selected source. P1 runner.py and runner_reference_gate.json
demonstrate original checkpoint/seed/guidance/update semantics on development.

Equal iterations: exactly1000 completed iterations(H10,mini_H100), report
actual time/evaluations. Equal wall time:4.0s search budget, variable iterations
and continuing100-iteration guidance blocks. Stop before starting an iteration
if budget expired; finish an already started iteration. Primary quality uses
only the last completed update timestamp<=4.0s. Persist all attempted times,
global-best tours and overshoot separately; never credit over-budget results.
No eligible update makes that run technically/resource inconclusive, not zero
improvement. Do not claim equal iterations and equal wall time simultaneously.

Model import/CUDA initialization/checkpoint load occur once and are separately
reported. Per-run setup includes seeding, both solver constructions and
coordinate normalization and is excluded equally from search budgets. One
100-iteration development warmup per arm, unmeasured, before principal runs;
report warmup as setup, never principal evidence. Search starts before first
graph forward and includes inference, transfers, sampling, update, CUDA sync,
checkpoint extraction and lightweight counters. Timestamp completed updates
after synchronization; recording overhead is charged through the running
clock. Final serialization and independent float64 evaluation occur afterward.
Compilation is outside search and setup timing, archived in build logs.

## Outcomes, aggregation and decision rules

Co-primary metrics: float64 final global-best length in equal-iteration mode
and float64 last-in-budget global-best length in equal-time mode. Pair relative
gain(control-treatment)/control within instance/seed/mode, average three seeds
within instance. The independent unit is the16 TSP instances. Secondary: actual
native calls/gain evaluations, runtime/setup/memory, completed iterations,
overshoot and anytime global-best at0,0.5,...4s using only earlier completed
updates (initial route at0). Equal-iteration runs additionally retain all1000
completed-update checkpoints. Native accepted/no-op counts are separate offline
diagnostics, not principal light-counter observations.

Report full per-instance values, mean/median/SD/range and10000 paired instance
bootstrap95% percentile intervals, PCG64 seed202612070202, common resample map
across modes. analysis.py implements strict complete-grid aggregation and
rejects missing/duplicate records; its unit fixture has no TSP data.

Draft scientific GO requires BOTH modes: median instance gain>=0.1%, mean
bootstrap lower bound>0 and>=12/16 positive instance means. Otherwise NO-GO
for a complete valid study. Any missing/invalid case, nonfinite value,
unapproved hash drift or correctness failure is INCONCLUSIVE; stop, preserve
all records, no exclusion/replacement or retry with changed settings.
Resource failure is RESOURCE-INCONCLUSIVE. No optional stopping, policy tuning
or selective reporting. Both modes are co-primary; no scientific GO from
equal-iteration advantage alone.

0.1% is a proposed practical relevance threshold, not a validated effect size.
Under an ideal independent continuous symmetric-null sign model,>=12/16 has
one-sided tail2517/65536~3.84%; ties are nonpositive. This is not a formal
combined test calibration: the median/sign/bootstrap conditions are correlated,
bootstrap n16 is small, and the two modes are paired. Report these limitations;
the conjunction supplies a conservative screening rule, not a proof of error
control or generality. Do not treat seeds, moves or iterations as independent.

## Feasibility and resource stops

Retain4.0s: P1 development arms had usable in-budget checkpoints; no quality
outcome was used to select this budget. P1 timing/overhead variation is in
timing_memory_audit.json. Conservative measured-max projection for192 runs:
{estimate:.1f}s including estimated setup/startup; doubling search/setup gives
{conservative:.1f}s versus2700s aggregate cap (margin{2700/conservative:.2f}x).
This extrapolates ONE observed instance/seed, not guaranteed new-data runtime.
No data shrink or time-budget changes after principal inspection.

Caps: RTX3050,1GiB allocated CUDA,3GiB peak RSS, two CPU threads, one GPU job,
60s search per iteration-budget run,120s setup per run,45min aggregate including
setup,warmup,search and output handling. GPU allocator fraction.24 is enabled;
check allocated/reserved bytes and process RSS at each completed iteration.
Keep outer process watchdogs for setup/aggregate limits. Pin launch environment
and verify source/binary/data hashes before execution. These resource gates
must be wired into the approved192-run orchestration; P1 executes no such job.

The existing runner accepts provided coordinates only; it cannot generate
principal data. Final approval must seal the separate dataset manifest and
192-run dispatch schedule and review orchestration watchdog wiring. This is a
registration prerequisite, not permission to generate data or execute now.
'''
    (OUT/'M2_REVISED_PROTOCOL.md').write_text(protocol)
    report=f'''# M2-P1 technical and feasibility review

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
{json.dumps(overhead,indent=2)}

This is material overhead, not zero-overhead measurement. Principal comparisons
would measure both arms identically; their speed/quality claim would apply to
that measured implementation. No algorithm policy was selected from pilot quality.
Unmeasured versus measured timing uses paired identical inputs, three repeats,
alternating order and per-arm100-iteration warmup. Variability and startup noise
limit inference from three repetitions on one seed.

Wall-time checkpoints: control{walls[0]['eligible_iterations']} eligible of
{walls[0]['iterations']} attempts; treatment{walls[1]['eligible_iterations']} of
{walls[1]['iterations']}. Overshoot{walls[0]['overshoot_seconds']:.6f}s and
{walls[1]['overshoot_seconds']:.6f}s. Audit recomputes the credited float64 length
from the last eligible saved tour; over-budget output is not credited.
Actual calls/evaluations include all attempted work, even the over-budget last
iteration. The4s budget is technically usable and retained prospectively.

Checkpoint load in the main pilot: {pilot['checkpoint_load_seconds']:.4f}s.
Peak pilot RSS:{pilot['peak_rss_bytes']} bytes. Every run's allocated/reserved
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

192-run estimate:{estimate:.1f}s; doubled-search/setup envelope:{conservative:.1f}s,
45min cap margin:{2700/conservative:.2f}x. Generalization across instance difficulty,
hardware thermals and background load is uncertain. Resource failure remains a
predefined inconclusive outcome, never grounds for replacing instances.

M2_REVISED_PROTOCOL.md explains co-primary rules,16-instance paired uncertainty,
practical0.1% threshold and limitations of the>=12/16 sign condition. These
criteria are proposed and not scientifically validated by development results.
Historical Stage A NO-GO, secondary INCONCLUSIVE, M0 VALIDATED, M1v1 technical
INCONCLUSIVE, M1v2 EXPLORATORY SIGNAL, M1.5 counts and P0 READY remain unchanged.
No native numerical repair, adaptive graph, NLS change, training, M2 principal
or Stage B occurred. No push/merge is authorized by this delivery.
'''
    (OUT/'P1_REVIEW_REPORT.md').write_text(report)
    (OUT/'HANDOFF.md').write_text(summary['decision']+'. Technical/resource review only; no principal authorization. Review M2_REVISED_PROTOCOL.md, timing_memory_audit.json and P1_REVIEW_REPORT.md. Freeze approved source/binary/dispatch/data manifests before principal generation. Historical evidence unchanged; no push.\n')
    save('independent_audit.json',dict(status='PASS',full_archive_repeat_invariance=True,last_in_budget_cost_verified=True,
        historical_evidence_preserved=True,source_semantics_verified=True,aggregation_unit_verified=True,
        principal_calls=0,decision=summary['decision']))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
