"""Read-only paired analysis of persisted R3 Stage A ant0 tours."""
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import r3_diagnostic
from r3.common import ROOT, sha256, save_json, git
from r3.diagnostics import canonical_tour, length
from r3.replay import reconstruct

SOURCE = ROOT / 'artifacts/r3/experiments/stage_a_v1_01'
OUT = ROOT / 'artifacts/r3/secondary_review/20261101_saved_ant0_v1'
ITERATIONS = (0, 249, 499, 999)
PHASES = {0:'initial',249:'early',499:'middle',999:'late'}

def mean(xs): return float(statistics.mean(xs))
def median(xs): return float(statistics.median(xs))
def csv_write(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    t0=time.perf_counter()
    input_paths=[*ROOT.glob('r3/**/*.py'),*ROOT.glob('r3/**/*.cpp'),*ROOT.glob('src/*.cpp'),*ROOT.glob('src/*.h'),*ROOT.glob('*.py'),
        ROOT/'r3/protocols/stage_a_v1.json',ROOT/'artifacts/r3/reports/R3_STAGE_A_REPORT.md',
        ROOT/'artifacts/r3/reports/RESEARCH_HANDOFF.md',ROOT/'artifacts/r3/manifests/post_collection_audit_v1.json',
        ROOT/'r3/analysis.py',ROOT/'r3/gate.py',ROOT/'r3/audit.py',ROOT/'r3/replay.py',
        ROOT/'r3/diagnostics.py',ROOT/'r3/native/two_opt_diagnostic.cpp',ROOT/'src/mfaco_train.cpp',
        ROOT/'src/mfaco_train.h',ROOT/'src/binding.cpp',SOURCE/'gate.json',SOURCE/'manifest.json',
        SOURCE/'finalized_sha256.json',SOURCE/'datasets/tsp1000_seed20261101_float32.npy']
    for i in range(8):
        input_paths.append(SOURCE/f'native_candidates_i{i}.npz')
        for s in (31415,27182):
            d=SOURCE/f'i{i}_s{s}'
            input_paths.append(d/'run.json')
            for completed in (0,250,500,1000):input_paths.append(d/f'state_t{completed}.npz')
            for it in ITERATIONS:
                input_paths.extend((d/f'raw_before_iter{it}.npz',d/f'perturbation_iter{it}.json',d/f'diagnostic_raw_before_iter{it}.json'))
    inputs={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in input_paths}
    gate=json.loads((SOURCE/'gate.json').read_text())
    if gate['decision']!='NO-GO' or gate['stage_b_authorized'] or gate['passing_instance_counts']!={'G1':0,'G2':0,'G3':0}:
        raise AssertionError('historical gate differs')
    ledger=json.loads((SOURCE/'finalized_sha256.json').read_text())
    for name,digest in ledger.items():
        if sha256(SOURCE/name)!=digest:raise AssertionError(f'historical artifact changed: {name}')
    dataset=np.load(SOURCE/'datasets/tsp1000_seed20261101_float32.npy',allow_pickle=False)
    manifest=dict(version=1,scope='secondary descriptive analysis of 64 saved ant0 pairs; no new trajectories',
        source_branch=git('branch','--show-current'),source_head=git('rev-parse','HEAD'),
        historical_gate='NO-GO',historical_gate_sha256=sha256(SOURCE/'gate.json'),
        historical_report_sha256=sha256(ROOT/'artifacts/r3/reports/R3_STAGE_A_REPORT.md'),
        input_sha256=inputs,source_tree_hash=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        stage_a_ledger_entries_verified=len(ledger),diagnostic_binary=__import__('r3_diagnostic').__file__,
        diagnostic_binary_sha256=sha256(__import__('r3_diagnostic').__file__),
        metric_scope='all-node or fixed reconstructed initial-checklist frozen-tour potential; native SRR execution unobserved')
    save_json(OUT/'input_manifest.json',manifest)
    rows=[];checks=0;reconstructions=0
    for i,x in enumerate(dataset):
        with np.load(SOURCE/f'native_candidates_i{i}.npz') as c: nn=np.ascontiguousarray(c['k64'])
        for seed in (31415,27182):
            folder=SOURCE/f'i{i}_s{seed}'
            for it in ITERATIONS:
                with np.load(folder/f'raw_before_iter{it}.npz') as saved:
                    raw=canonical_tour(saved['raw'],1000);refined=canonical_tour(saved['refined'],1000)
                    before=canonical_tour(saved['source_before'],1000)
                    cl=np.ascontiguousarray(saved['initial_checklist'],dtype=np.int32)
                trace=json.loads((folder/f'perturbation_iter{it}.json').read_text())['trace']
                reconstructed,checklist,_=reconstruct(before,trace)
                if not np.array_equal(reconstructed,raw) or not np.array_equal(checklist,cl):
                    raise AssertionError('raw tour/checklist reconstruction mismatch')
                reconstructions+=1
                original=json.loads((folder/f'diagnostic_raw_before_iter{it}.json').read_text())
                widths=np.ascontiguousarray(original['per_node_widths'],dtype=np.int32)
                names=original['policy_names']
                assert len(names)==15 and np.array_equal(widths[1],np.full(1000,40))
                raw_length=length(x,raw);refined_length=length(x,refined)
                reduction=raw_length-refined_length
                if reduction < -2e-5*raw_length:raise AssertionError('native refined ant0 length increased materially')
                # The same saved edge allocations are evaluated on both tours.
                # Each policy is placed in row zero because Stage A only returned
                # checklist-specific aggregates for that policy row.
                for p,name in enumerate(names):
                    width=np.ascontiguousarray(widths[p:p+1])
                    raw_scan=r3_diagnostic.scan(x,raw,nn,width,cl,0)
                    refined_scan=r3_diagnostic.scan(x,refined,nn,width,cl,0)
                    if raw_scan['membership_count'][0]!=original['membership_count'][p] or raw_scan['scan_count'][0]!=original['scan_count'][p]:
                        raise AssertionError('independent policy reorder disagrees with saved all-node record')
                    if abs(raw_scan['membership_gain_mass'][0]-original['membership_gain_mass'][p])>1e-8 or abs(raw_scan['scan_gain_mass'][0]-original['scan_gain_mass'][p])>1e-8:
                        raise AssertionError('policy gain mass disagrees with saved all-node record')
                    if p==0 and (raw_scan['initial_checklist_membership_count']!=original['initial_checklist_membership_count'] or raw_scan['initial_checklist_scan_count']!=original['initial_checklist_scan_count']):
                        raise AssertionError('saved K32 checklist coverage mismatch')
                    for label,scan,tour_length in (('raw',raw_scan,raw_length),('refined',refined_scan,refined_length)):
                        robust=scan['robust_improving_moves'];member=scan['membership_count'][0]
                        anchor=scan['initial_checklist_membership_count'];potential=scan['initial_checklist_scan_count']
                        if not (0<=potential<=anchor<=member<=robust):raise AssertionError('nonmonotone coverage decomposition')
                        gm=scan['robust_gain_mass'];mg=scan['membership_gain_mass'][0]
                        ag=scan['initial_checklist_membership_gain_mass'];pg=scan['initial_checklist_scan_gain_mass']
                        if min(gm-mg,mg-ag,ag-pg,pg)<-1e-9:raise AssertionError('nonmonotone gain decomposition')
                        rows.append(dict(instance=i,seed=seed,iteration_before_srr=it,phase=PHASES[it],tour=label,policy=name,
                            raw_length=raw_length,refined_length=refined_length,paired_length_reduction=reduction,
                            paired_relative_reduction=reduction/raw_length,initial_checklist_size=len(cl),
                            robust_moves=robust,off_graph_moves=robust-member,no_initial_anchor_moves=member-anchor,
                            initially_reachable_but_pruned_moves=anchor-potential,initial_checklist_potential_moves=potential,
                            all_node_scan_potential_moves=scan['scan_count'][0],
                            robust_gain_mass=gm,off_graph_gain_mass=gm-mg,no_initial_anchor_gain_mass=mg-ag,
                            initially_reachable_but_pruned_gain_mass=ag-pg,initial_checklist_potential_gain_mass=pg,
                            all_node_scan_potential_gain_mass=scan['scan_gain_mass'][0],
                            extra_directed_slots=int(width.sum()-32000),
                            actual_native_evaluations='unavailable',actual_native_acceptances='unavailable'))
                    checks+=2
                print(f'paired saved ant0 i={i} seed={seed} iter={it}',flush=True)
    csv_write(OUT/'paired_policy_metrics.csv',rows)
    summary=summarize(rows)
    summary.update(historical_gate='NO-GO',historical_gate_unchanged=sha256(SOURCE/'gate.json')==manifest['historical_gate_sha256'],
        historical_report_unchanged=sha256(ROOT/'artifacts/r3/reports/R3_STAGE_A_REPORT.md')==manifest['historical_report_sha256'],
        raw_reconstructions_verified=reconstructions,compiled_scans_verified=checks,
        stage_a_files_rehashed=len(ledger),input_files_hashed=len(inputs),elapsed_seconds=time.perf_counter()-t0)
    save_json(OUT/'summary.json',summary)
    write_report(summary,manifest,OUT)
    return summary

def summarize(rows):
    base=[r for r in rows if r['policy']=='k32']
    policies=['k32','k40','geometry']+[f'random_{r:02}' for r in range(10)]
    # Nested: phase mean within each execution seed, then seed mean within instance.
    per_instance=[]
    for i in range(8):
        seed_rows=[]
        for s in (31415,27182):
            phases=[]
            for phase in ('middle','late'):
                rs=[r for r in base if r['instance']==i and r['seed']==s and r['phase']==phase]
                assert len(rs)==2
                raw=next(r for r in rs if r['tour']=='raw');ref=next(r for r in rs if r['tour']=='refined')
                phases.append(dict(length_reduction=raw['paired_length_reduction'],relative_reduction=raw['paired_relative_reduction'],
                    raw_off_graph=raw['off_graph_moves'],refined_off_graph=ref['off_graph_moves'],
                    raw_off_graph_gain=raw['off_graph_gain_mass'],refined_off_graph_gain=ref['off_graph_gain_mass'],
                    raw_k32_residual=raw['robust_moves']-raw['off_graph_moves'],
                    refined_k32_residual=ref['robust_moves']-ref['off_graph_moves'],
                    raw_no_initial_anchor=raw['no_initial_anchor_moves'],raw_pruned=raw['initially_reachable_but_pruned_moves'],
                    raw_initial_potential=raw['initial_checklist_potential_moves'],
                    refined_initial_potential=ref['initial_checklist_potential_moves']))
            seed_rows.append({key:mean(p[key] for p in phases) for key in phases[0]})
        per_instance.append(dict(instance=i,**{key:mean(s[key] for s in seed_rows) for key in seed_rows[0]}))
    # Supplemental policy comparison under the *same* stored initial checklist.
    controls=[]
    for i in range(8):
        d={'instance':i}
        for tour in ('raw','refined'):
            selected=[r for r in rows if r['instance']==i and r['phase'] in ('middle','late') and r['tour']==tour]
            by={(r['seed'],r['phase'],r['policy']):r for r in selected}
            def inc(name):
                return mean(by[(s,ph,name)]['initial_checklist_potential_gain_mass']-by[(s,ph,'k32')]['initial_checklist_potential_gain_mass'] for s in (31415,27182) for ph in ('middle','late'))
            d[tour+'_geometry_incremental_gain']=inc('geometry')
            d[tour+'_uniform_incremental_gain']=inc('k40')
            d[tour+'_random_mean_incremental_gain']=mean(inc(name) for name in policies[3:])
        controls.append(d)
    phase=[]
    for ph in ('initial','early','middle','late'):
        rs=[r for r in base if r['phase']==ph]
        raw=[r for r in rs if r['tour']=='raw'];ref=[r for r in rs if r['tour']=='refined']
        phase.append(dict(phase=ph,pairs=len(raw),mean_raw_length=mean(r['raw_length'] for r in raw),
            mean_refined_length=mean(r['refined_length'] for r in ref),
            mean_paired_length_reduction=mean(r['paired_length_reduction'] for r in raw),
            median_paired_length_reduction=median(r['paired_length_reduction'] for r in raw),
            mean_raw_off_graph=mean(r['off_graph_moves'] for r in raw),
            mean_refined_off_graph=mean(r['off_graph_moves'] for r in ref),
            mean_refined_k32_residual=mean(r['robust_moves']-r['off_graph_moves'] for r in ref),
            mean_raw_no_initial_anchor=mean(r['no_initial_anchor_moves'] for r in raw),
            mean_raw_pruned=mean(r['initially_reachable_but_pruned_moves'] for r in raw)))
    rng=np.random.Generator(np.random.PCG64(20261107));uncertainty={}
    for key in ('length_reduction','refined_off_graph','refined_k32_residual'):
        values=np.array([p[key] for p in per_instance]);boot=values[rng.integers(0,8,size=(10000,8))].mean(axis=1)
        uncertainty[key]=dict(mean=float(values.mean()),median=float(np.median(values)),range=[float(values.min()),float(values.max())],
            bootstrap_95_percentile_interval=np.quantile(boot,[.025,.975]).tolist(),primary_unit='instance',n=8)
    return dict(phase=phase,per_instance=per_instance,checklist_policy_per_instance=controls,descriptive_uncertainty=uncertainty,
        secondary_decision='INCONCLUSIVE',stage_b_authorized=False)

def write_report(summary,manifest,out):
    phases=summary['phase'];instances=summary['per_instance'];controls=summary['checklist_policy_per_instance']
    phase_table='\n'.join(f"| {p['phase']} | {p['pairs']} | {p['mean_paired_length_reduction']:.6f} | {p['mean_raw_off_graph']:.2f} | {p['mean_refined_off_graph']:.2f} | {p['mean_refined_k32_residual']:.2f} | {p['mean_raw_no_initial_anchor']:.2f} | {p['mean_raw_pruned']:.2f} |" for p in phases)
    instance_table='\n'.join(f"| {p['instance']} | {p['length_reduction']:.6f} | {p['raw_off_graph']:.2f} | {p['refined_off_graph']:.2f} | {p['refined_k32_residual']:.2f} |" for p in instances)
    control_table='\n'.join(f"| {c['instance']} | {c['raw_geometry_incremental_gain']:.6f} | {c['raw_uniform_incremental_gain']:.6f} | {c['raw_random_mean_incremental_gain']:.6f} | {c['refined_geometry_incremental_gain']:.6f} | {c['refined_uniform_incremental_gain']:.6f} |" for c in controls)
    doc=f'''# R3 secondary evidence review: saved ant0 tours only

**Research decision: INCONCLUSIVE.** The existing records show both a pre-SRR
candidate opportunity and residual local-search opportunities, but they do not
identify which mechanism could improve the actual solver. The frozen Stage A
decision remains **NO-GO**; G0/G4 true and G1-G3 false. This review does not
authorize Stage B or change the source tour gate.

## Provenance and evidence chain

Branch `{manifest['source_branch']}`, starting HEAD `{manifest['source_head']}`.
The input manifest hashes {len(manifest['input_sha256'])} specific files; the
original Stage A ledger's {manifest['stage_a_ledger_entries_verified']} files
were rehashed. Historical gate SHA256 `{manifest['historical_gate_sha256']}`;
historical report SHA256 `{manifest['historical_report_sha256']}`. The frozen
protocol, source, native diagnostic, saved coordinates/candidates, tours,
traces and checklists are pinned by SHA256 in `input_manifest.json`.
No instance, solver trajectory, model or confirmation data was generated.

`r3/analysis.py:73-82` calls `evaluate(..., True, complete, resources)`:
G0's technical-validity flag was asserted after some internal checks, rather
than derived from a separate machine-readable checklist of every G0 premise.
`r3/audit.py` later rehashed files, regenerated the dataset, checked tours,
reapplied 5407 selected gains and reaggregated the gate. This is strong
independent support for integrity and the stated source-tour metric. It does
not validate every one of the 498500 gains per snapshot, prove actual SRR
checklist extension/evaluation/acceptance, or certify local optimality.
The binding returns empty `touched_list`; reconstructed raw tours validate
relocation and the initial checklist, not subsequent SRR decisions.
These are evidence-chain limits, not a retrospective edit to G0 or its gate.

## Paired ant0 reduction and residual moves

Each raw/refined pair is the saved tour of the *same ant0* in one iteration.
Length is independently recomputed in float64 on the saved float32 coordinates.
The refined ant0 tour is not the best-of-100 source selected after the iteration.

| Phase (zero-based iteration) | Pairs | Mean raw-to-refined length decrease | Mean raw off-K32 robust moves | Mean refined off-K32 | Mean refined K32-eligible residual | Mean raw K32 available without initial anchor | Mean raw initially reachable but pruned |
|---|---:|---:|---:|---:|---:|---:|---:|
{phase_table}

Middle/late phases are nested within each execution seed, then two seed values
are averaged within each independent instance. The table below gives those
eight instance summaries, rather than treating 32 pairs as independent.

| Instance | Mean paired length decrease | Raw off-K32 | Refined off-K32 | Refined K32-eligible residual |
|---:|---:|---:|---:|---:|
{instance_table}

Descriptive instance-level mean/median/range and 10000 fixed-seed bootstrap
percentile intervals are in `summary.json`. They describe this small sample;
the underlying tours and seeds share search histories. One-step gain mass and
largest missed gains remain offline opportunity measures, not additive tour
improvement. The per-pair policy CSV retains robust missed gain mass for both
raw and refined tours.

## Equal-slot offline checklist comparison

For every pair, native K64 candidate identities and the *same stored raw-tour
edge allocation* are held fixed while evaluating raw and refined tours.
K32 is the base. Uniform K40, geometry concentration and each of ten random
concentrations use 8000 extra directed slots. Candidate eligibility uses both
directed 2-opt orientations, a canonical move pair and the baseline's float32
distance-ordered early break. Only initial-checklist anchors are counted.
After a tour changes, this static checklist is hypothetical and differs from
the dynamically extended native SRR checklist. Every policy's raw all-node
counts and gain mass were checked against saved Stage A records; base checklist
metrics were checked separately.

Incremental gain mass beyond K32 under the reconstructed initial checklist,
averaged within seed/phase and then instance:

| Instance | Raw geometry | Raw uniform | Raw random mean | Refined geometry | Refined uniform |
|---:|---:|---:|---:|---:|---:|
{control_table}

The CSV contains all ten random allocations separately, directed slot counts,
and unique move coverage. These are **frozen-tour potential** metrics. They do
not show native evaluation, chosen moves, accepted moves or realized solver
improvement. Undefined ratios are left undefined; this secondary screen has
no retrospective GO threshold.

## Mechanism assessment and decision

The count/gain decomposition on each frozen tour is: unavailable candidate;
candidate available but no reconstructed initial-checklist anchor; initially
reachable but stopped by distance pruning; and initially reachable potential.
These are disjoint after OR over the four directed representations. The raw
checklist is exact at SRR entry, but later checklist extension and tour changes
are unobserved. `src/mfaco_train.cpp:964-1040` has a monotone checklist cursor
and de-duplicates against the whole vector, so an already processed node is
not re-enqueued in one call. The available records cannot count move losses
caused specifically by that behavior. A residual K32 move after refined ant0
may result from checklist scope, pruning at an earlier state, selection order,
tour evolution, or lack of revisit. Causal shares require an intervention.

**INCONCLUSIVE** is the research recommendation. Candidate expansion has
offline headroom before SRR, while K32 residual moves and static checklist
limits leave a distinct SRR hypothesis. The present ant0 traces do not
distinguish which change would help the best-of-100 source under equal compute.

For a separately registered study, a minimal falsifiable hypothesis is:
*on independent TSP-1K instances, adding rank33-64 candidates only to the
initial-checklist SRR scan at 8N directed slots improves final best-of-100
tour length beyond equal-slot uniform K40 and random concentrated controls,
at no more than 10% search-wall-time overhead*. Controls must include unchanged
K32, uniform K40 LS-only, ten seeded random concentrated LS-only, predefined
geometry LS-only, and a K32 checklist-revisit-only arm to isolate search scope.
Freeze all move thresholds, seeds, node-selection rule, trajectory sampling,
checkpoint, native trace semantics and runtime accounting before data. Use a
new independent eight-instance PCG64 TSP-1K set with two solver seeds; keep
the reserved Stage A confirmation set sealed until a later independently
approved confirmation protocol. Bound the study to 1000 iterations, 100 ants,
two CPU threads, sequential RTX3050 workloads, 90 minutes wall time and 4 GB
process RSS; report resource failure without replacement cases. Measure
paired instance-level final tour length, anytime quality, actual LS evaluations
and accepted moves via passive disabled-by-default instrumentation, candidate
slots, source/ant0 trajectories, wall time and memory. A prospective GO could
require a median paired final-length improvement >=0.1% versus *each* equal-slot
control in at least 6/8 instances, with no per-instance invalid tour and <=10%
median wall overhead; otherwise NO-GO or INCONCLUSIVE for invalid data. These
are proposed future criteria, not Stage A thresholds and not an experiment run
here. If the native instrumentation cannot preserve exact disabled-feature
behavior or expose actual checklist/evaluation events, stop before comparison.
'''
    (out/'R3_SECONDARY_EVIDENCE_REVIEW.md').write_text(doc,encoding='utf-8')
    handoff=f'''# Secondary R3 handoff supplement

- Historical Stage A: **NO-GO**, unchanged; gate SHA256 `{manifest['historical_gate_sha256']}`.
- New review: **INCONCLUSIVE**, descriptive saved ant0 pairs only; no Stage B authorization.
- Raw/refined pair records: 64; full policy rows: {len(summary['phase'])*16*2*15}; 8 independent instance summaries.
- Source/input hashes: `input_manifest.json`; raw policy rows: `paired_policy_metrics.csv`; machine summary: `summary.json`.
- Main coordination action: review candidate versus SRR hypotheses as separate future experiments; register controls and decision criteria before any new trajectory.
- Original Stage A report and artifacts remain unchanged; no push performed.
'''
    (out/'HANDOFF_SUPPLEMENT.md').write_text(handoff,encoding='utf-8')

if __name__=='__main__': print(json.dumps(main(),indent=2)[:1200])
