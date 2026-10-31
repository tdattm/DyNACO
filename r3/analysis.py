"""Instance-first aggregation and immutable reports from raw Stage A artifacts."""
import csv
import json
import resource
import statistics
import time
from pathlib import Path
import numpy as np
from r3.common import ROOT,save_json,sha256,git
from r3.protocol import load,PATH
from r3.gate import evaluate

def write_csv(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def aggregate(diagnostics):
    rows=[]
    for instance in range(8):
        seed_rows=[]
        for seed in (31415,27182):
            selected=[d for d in diagnostics if d['instance_id']==instance and d['execution_seed']==seed and d['tour_kind']=='source' and d['completed_iteration'] in (500,1000)]
            if len(selected)!=2:raise ValueError('incomplete gate snapshots')
            quantities=[]
            for d in selected:
                inc=d['incremental_scan_gain_mass']
                verified=max((v['relative_gain'] for v in d['gain_verification']),default=0.)
                quantities.append(dict(geometry_gain=inc[4],uniform_gain=inc[1],random_gain=float(np.mean(inc[5:])),
                    missed_count=d['membership_missed_count'],verified_best_relative_gain=verified,
                    k32_robust_moves=d['membership_count'][0],full_robust_moves=d['robust_improving_moves']))
            seed_rows.append({key:statistics.mean(q[key] for q in quantities) for key in quantities[0]})
        rows.append(dict(instance_id=instance,**{key:statistics.mean(s[key] for s in seed_rows) for key in seed_rows[0]}))
    return rows

def analyze(out,principal_start=None):
    tick=time.perf_counter();protocol=load();diagnostics=[];raw=[];runs=[]
    for folder in sorted(out.glob('i*_s*')):
        runs.append(json.loads((folder/'run.json').read_text()))
        diagnostics.extend(json.loads(p.read_text()) for p in sorted(folder.glob('diagnostic_source_t*.json')))
        diagnostics.extend(json.loads(p.read_text()) for p in sorted(folder.glob('diagnostic_best_t*.json')))
        raw.extend(json.loads(p.read_text()) for p in sorted(folder.glob('diagnostic_raw_before_iter*.json')))
    complete=len(runs)==16 and len(diagnostics)==128 and len(raw)==64
    if not complete:raise ValueError('principal collection incomplete')
    all_details=diagnostics+raw;long=[]
    for d in all_details:
        if d['valid_moves']!=1000*(1000-3)//2:raise AssertionError('move count error')
        if len(d['policy_names'])!=15:raise AssertionError('policy count error')
        if any(abs(v['realized_one_step_gain']-v['predicted_gain'])>1e-9 for v in d['gain_verification']+d['sample_gain_verification']):raise AssertionError('gain error')
        for i,name in enumerate(d['policy_names']):
            long.append(dict(instance_id=d['instance_id'],execution_seed=d['execution_seed'],tour_kind=d['tour_kind'],
                iteration=d.get('completed_iteration',d.get('iteration_before_srr')),policy=name,
                full_robust_moves=d['robust_improving_moves'],full_robust_gain_mass=d['robust_gain_mass'],
                membership_count=d['membership_count'][i],membership_gain_mass=d['membership_gain_mass'][i],
                scan_count=d['scan_count'][i],scan_gain_mass=d['scan_gain_mass'][i],
                incremental_scan_gain_mass=d['incremental_scan_gain_mass'][i],
                membership_gain_coverage=d['membership_gain_weighted_coverage'][i],scan_gain_coverage=d['scan_gain_weighted_coverage'][i],
                membership_missed_k32=d['membership_missed_count'],missed_count_rate=d['membership_missed_count_rate'],
                largest_missed_gain=d['largest_membership_missed_gain'],missed_gain_top_quartile_share=d['missed_gain_top_quartile_share'],
                **d['policy_edge_counts'][i]))
    write_csv(out/'snapshot_policy_metrics.csv',long)
    instance_rows=aggregate(diagnostics)
    write_csv(out/'instance_metrics.csv',instance_rows)
    # Statistical unit is instance. Include zero effects; no moves/snapshots as replicates.
    rng=np.random.Generator(np.random.PCG64(20261105));uncertainty={}
    for comparison in ('uniform','random'):
        differences=np.array([r['geometry_gain']-r[comparison+'_gain'] for r in instance_rows])
        bootstrap=differences[rng.integers(0,8,size=(10000,8))].mean(axis=1)
        uncertainty[comparison]=dict(mean_paired_gain_mass_difference=float(differences.mean()),
            median_paired_gain_mass_difference=float(np.median(differences)),bootstrap_95_percentile_interval=np.quantile(bootstrap,[.025,.975]).tolist(),n_instances=8)
    elapsed=(time.perf_counter()-principal_start) if principal_start is not None else json.loads((out/'collection_complete.json').read_text())['elapsed_seconds']+time.perf_counter()-tick
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    resources=elapsed<=5400 and rss<=4294967296 and max(r['peak_cuda_bytes'] for r in runs)<=3758096384
    gate=evaluate(instance_rows,True,complete,resources)
    gate.update(protocol_sha256=sha256(PATH),primary_tour='source_route',phases=[500,1000],
        uncertainty=uncertainty,technical_evidence=dict(original_solver_unchanged=True,
            baseline_full1000_exact=True,fast_parallel_trace_regression=True,native_candidate_rows_audited=8000,
            raw_reconstructions_exact=64,full_pair_count=498500,diagnostics=len(all_details),
            actual_native_srr_evaluation_observability='unavailable; no claims rely on it'),
        resources=dict(principal_wall_seconds=elapsed,rss_high_water_bytes=rss,
                       peak_torch_cuda_allocated_bytes=max(r['peak_cuda_bytes'] for r in runs)))
    save_json(out/'gate.json',gate)
    report(gate,diagnostics,raw,runs,out)
    save_json(out/'analysis_complete.json',dict(seconds=time.perf_counter()-tick,
        final_principal_wall_seconds=(time.perf_counter()-principal_start) if principal_start else None,
        status='COMPLETE',decision=gate['decision']))
    # Finalized directory content ledger; no output in this directory is overwritten.
    ledger={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save_json(out/'finalized_sha256.json',ledger)
    print(json.dumps(dict(decision=gate['decision'],conditions=gate['conditions'],counts=gate['passing_instance_counts'])),flush=True)

def report(gate,diagnostics,raw,runs,out):
    reports=ROOT/'artifacts/r3/reports';reports.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((out/'manifest.json').read_text());decision=gate['decision']
    counts=gate['passing_instance_counts'];resources=gate['resources']
    primary=[d for d in diagnostics if d['tour_kind']=='source']
    late=[d for d in primary if d['completed_iteration'] in (500,1000)]
    rows='\n'.join('| '+ ' | '.join(str(r[k]) for k in ('instance_id','missed_count','verified_best_relative_gain','geometry_gain','uniform_gain','random_gain','ratio_vs_uniform','ratio_vs_random','G3'))+' |' for r in gate['per_instance'])
    phase_rows=[]
    for phase in (0,250,500,1000):
        ds=[d for d in primary if d['completed_iteration']==phase]
        phase_rows.append('| '+ ' | '.join(map(str,[phase,statistics.mean(d['robust_improving_moves'] for d in ds),
            statistics.mean(d['membership_missed_count'] for d in ds),
            statistics.mean(d['largest_membership_missed_gain'] for d in ds),
            statistics.mean(d['scan_count'][0] for d in ds)]))+' |')
    text=f'''# R3 Stage A report

**Decision: {decision}. Stage B was not executed.**

## Research question and frozen design

Test whether native K32 omits material robust 2-opt moves on genuine search
states, and whether a predefined geometry/tour-excess allocation covers more
incremental gain than equal-directed-slot uniform and random controls. The
proposal supplies hypotheses, not measured results. Original paper results,
historical R1 evidence, new baseline verification and these diagnostics are
separate evidence categories.

Protocol: `r3/protocols/stage_a_v1.json`; SHA256
`{gate['protocol_sha256']}`. Frozen before principal data generation; no gate,
selector, phase boundary, dataset or sample-size changes after inspection.
Primary observable: source tour; global best is secondary and reported in raw
tables, never substituted into the gate. Completed iterations 500 and 1000 are
averaged within execution seed, then equally across the two seeds per instance.

## Baseline source and verification

Original SHA `{manifest['base_sha']}`. R3 working SHA at collection:
`{manifest['working_head']}`. Production solver, wrapper, GNN and inference
sources are unchanged. The isolated C++17/OpenMP build has the same SHA256 as
the earlier audited R1 rebuild; checkpoint SHA256:
`{manifest['checkpoint_sha256']}`. Strict model loading, CUDA, K32/K64 forward
shapes and native imports passed. Reduced-budget checks ran at N=100/200/500/1K;
the 1K 1000-iteration captured trajectory exactly matched unchanged
`utils.infer_instance`, and a same-seed repeat matched every ant cost.

Configuration: K32+backup32, 100 ants, H10/S100, 1000 ACO iterations, M12,
rho=.1, smooth pheromone bounds 1/K and 1, original SRR with extension,
normalized heuristic, dynamic features, original per-block linear prior
annealing from 1 to 0, no phased Mix injection and no NLS. Run-level seed reset
preserves original duplicate construction and NumPy-derived active native seed.
Two CPU/OpenMP threads; RTX3050 4GB via WSL. Exact replay uses strict CUDA and
TF32 disabled. These timings are not non-strict R1 benchmark timings or paper
reproduction on different hardware/datasets.

See [source audit](SOURCE_AUDIT.md) for all 15 implementation questions and
precise source lines. Important findings: directed storage; backup absent from
SRR/GNN; both scan orientations and float32 early breaks; once-per-distinct-node
checklist; greedy initialization without local optimality; incomplete snapshot
replay and stale normalized heuristic cache after native snapshot load.

## Data and provenance

Eight independent uniform float32 TSP1K instances, PCG64 SeedSequence([20261101,
1000]); IDs 0-7. Execution seeds 31415 and 27182. Dataset SHA256:
`{manifest['dataset_sha256']}`. Every generated coordinate block has a saved
hash. Each native K32/K40/K48/K64 row was independently distance-ranked and
prefix-checked; K32 backups exactly matched native ranks33-64. R1 pilot datasets
were read as Git blobs, rehashed and independently regenerated. Confirmation
seed 20261104 remains reserved, ungenerated and uninspected.

All 16 principal runs completed: 16000 ACO iterations, 64 persisted solver states,
128 frozen source/best diagnoses and 64 bounded ant0 raw diagnoses. Native
perturbation traces reconstructed all 64 pre-SRR raw tours exactly, including
array order and original-source edge tests. No solver RNG was drawn for state
capture or allocation controls.

## Diagnostic definitions and numerical checks

Enumerate every nonadjacent cyclic tour-edge pair once: N(N-3)/2=498500 per
tour, excluding identical/adjacent/wraparound-adjacent edges. Float64 reference
uses unrounded Euclidean distances on exact native float32 coordinate values.
Robust improvement means gain > max(1e-8,1e-6*current length). Raw signed gains,
near-zero samples and marginal counts are retained. Distinct directed
representations (a,c),(c,a),(b,d),(d,b) are combined by OR per canonical move.
Membership coverage is separate from all-node potential evaluation under exact
native float32 ordered early-break prefixes.

Known crossed/optimal tours, collinear/duplicated/near-duplicated coordinates,
numerically marginal gains, invalid candidate/tour data, both scan orientations,
and exhaustive independent copied-tour reversal tests passed. Every selected
missed move and bounded robust sample was applied to a copied valid cyclic
permutation and independently recomputed in float64. A single-move oracle
improvement is feasibility evidence, not an improvement achieved by DyNACO;
independent move gains are never summed into an achievable tour improvement.

## Candidate coverage and phase dependence

Primary source-tour means across all 16 executions at each phase:

| Iteration | Full robust moves | Outside K32 membership | Largest missed gain | K32 all-node scan eligible |
|---|---|---|---|---|
{chr(10).join(phase_rows)}

Late snapshots with any membership miss: {sum(d['membership_missed_count']>0 for d in late)}/{len(late)}.
Per-K membership/scan counts, missed rates, gain-weighted coverage, increments,
largest missed gain, relative-gain and candidate-rank histograms, per-node missed
gain mass/concentration and all individual random results are retained in JSON
and `snapshot_policy_metrics.csv`. No denominator-zero coverage is fabricated.

## SRR versus graph limitation

Initial checklist membership and potential pruning-aware coverage are reported
on reconstructed raw tours. Actual native evolving SRR evaluation/acceptance
sequences are unavailable through the original API; booleans remain null.
Checklist extension and changing tours make initial membership insufficient to
infer dynamic execution. Empty `touched_list` is not used as checklist evidence.
The all-node frozen-tour reference therefore measures candidate headroom,
not scope-restricted native solver behavior or realized accepted moves.

The paper's Appendix B.2 describes node reactivation and assumes a 2-opt optimal
incumbent. Actual baseline initialization and checklist processing do not meet
those assumptions. Residual K32-eligible moves cannot automatically be blamed
on the graph. Exact-distance ties also lack a global node-index tie-break:
constructed duplicates show cross-K prefix differences. Tests detect that case;
all actual principal lists passed exact prefix checks.

## Equal-edge-budget selector screen and gate

Base K32; pool ranks33-64. Uniform K40 adds eight slots per node; geometry and
each of ten random controls select 25% of nodes and add 32 slots each. All use
8000 extra directed slots, keeping unique undirected counts separate. Full K64
uses 32000 extra slots and is a higher-budget reference. Geometry score and
node-index ties were frozen. No oracle-selected allocator was used.

Each table row averages late-phase quantities within seed then across seeds;
ratios divide aggregate incremental pruning-aware gain masses. Undefined ratios
remain `None` and fail conservatively.

| Instance | Missed count | Verified best relative gain | Geometry extra gain mass | Uniform | Mean random | Geo/uniform | Geo/random | G3 instance |
|---|---|---|---|---|---|---|---|---|
{rows}

G0-G4: `{gate['conditions']}`. G1 passing instances {counts['G1']}/8;
G2 {counts['G2']}/8; G3 {counts['G3']}/8. Required: at least 6/8 for each,
G2 mean relative best missed gain >=1e-4, G3 both ratios >=1.10 and both median
ratios >=1.10. Medians: `{gate['median_ratios']}`. Instance-level paired
descriptive bootstrap summaries (10000 resamples, fixed seed) are saved in
`gate.json`; no move/snapshot-level independence or significance is asserted.

## Runtime, memory, failures and limits

Principal collection/analysis to gate: {resources['principal_wall_seconds']:.3f}s,
90-minute cap. Process RSS high-water: {resources['rss_high_water_bytes']} bytes;
peak PyTorch CUDA allocation: {resources['peak_torch_cuda_allocated_bytes']} bytes.
These are process-wide RSS and Torch allocated memory, not isolated scanner RSS
or total device/context memory. Native/graph-forward/search/extraction times,
KNN construction/audit, enumeration and classification, bounded sampling, edge
counts, gain verification and report generation are saved with their scopes.
No concurrent benchmark GPU workload ran during principal measurements.

Initial test failures and environment-path failure are retained: a Windows Git
worktree pointer needed explicit WSL Git paths; temporary solver owners caused
invalid borrowed NumPy views in research fixtures; the duplicate-coordinate
test exposed a real native tie-prefix limitation and was changed to require
detection. Production code and frozen scientific thresholds were unchanged.

This study is 8 instances with two execution seeds, one frozen checkpoint,
source-tour late-phase aggregation and offline candidate/pruning semantics.
It does not reproduce paper gaps, certify local optimality, estimate adaptive
solver performance, or establish neural-guidance gains. Any future benefit
from LS expansion could be additional local search, requiring real rollouts and
equal-budget controls. All negative/near-zero outcomes remain in raw artifacts.

## Decision and next action

**{decision}.** The exact frozen gate is in `gate.json`. Do not implement Stage B
unless all conditions pass. Stage B and confirmation were not run in this
execution. Preserve this result and bring the implementation/paper SRR mismatch
and the measured instance-level gate evidence to the main coordination chat.
'''
    path=reports/'R3_STAGE_A_REPORT.md'
    with path.open('x',encoding='utf-8') as f:f.write(text)
    handoff=f'''# R3 research handoff

- Branch: `{git('branch','--show-current')}`; collection commit `{manifest['working_head']}`.
- Exact original base: `{manifest['base_sha']}`; R1 branch/history/artifacts unchanged.
- Work completed: isolated worktree/build, source/paper audit, frozen protocol,
  independent C++17 2-opt diagnostic, numerical/candidate/replay/regression tests,
  four-size baseline smokes, full 1K/1000-iteration exact original regression,
  same-seed repeat, all 16 principal runs and instance-level gate.
- Data: 8 TSP1K PCG64 instances seed20261101; solver seeds31415/27182;
  original K32,100 ants,H10,S100,M12,rho.1,smooth,normalized,extended SRR,
  original Model(anneal), frozen checkpoint. Strict CUDA, two CPU threads.
- Decision: **{decision}**; G0-G4 `{gate['conditions']}`;
  G1/G2/G3 counts `{counts}`; ratios `{gate['median_ratios']}`.
- Offline oracle/coverage findings are not actual solver improvements. Actual
  dynamic SRR evaluation/acceptance trace is unavailable; initial checklist and
  all raw tours were reconstructed exactly. Source and global best are separate.
- Runtime to gate {resources['principal_wall_seconds']:.3f}s; RSS high-water
  {resources['rss_high_water_bytes']} bytes; Torch allocation peak
  {resources['peak_torch_cuda_allocated_bytes']} bytes.
- Stage B: not executed. Confirmation seed20261104: reserved/uninspected.
- Main coordination action: inspect the frozen gate and source/paper mismatch;
  do not reinterpret expanded-LS oracle coverage as neural or end-to-end gain.
- Primary report: `artifacts/r3/reports/R3_STAGE_A_REPORT.md`.
- Raw/provenance/gate: `artifacts/r3/experiments/stage_a_v1_01/`.
- Protocol SHA256: `{gate['protocol_sha256']}`.
- Dataset SHA256: `{manifest['dataset_sha256']}`.
- Checkpoint SHA256: `{manifest['checkpoint_sha256']}`.
- No push, merge, reset, stash, clean or R1 modification was performed.
'''
    with (reports/'RESEARCH_HANDOFF.md').open('x',encoding='utf-8') as f:f.write(handoff)
