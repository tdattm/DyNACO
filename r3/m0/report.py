"""Aggregate already validated M0 event records; no replay or counterfactual."""
import json
import statistics
from r3.common import ROOT, sha256

OUT=ROOT/'artifacts/r3/m0_srr_replay'

def main():
    summary=json.loads((OUT/'validation_summary.json').read_text())
    rows=summary['cases'];passed=summary['passed']==64
    keys=['processed_nodes','unreached_nodes','candidate_evaluations','pruned_neighbor_slots',
          'evaluated_not_final_chosen','accepted_moves','processed_node_reinsertion_attempts']
    def stats(items):
        return {k:dict(mean=statistics.mean(r[k] for r in items),
                       median=statistics.median(r[k] for r in items),
                       minimum=min(r[k] for r in items),maximum=max(r[k] for r in items)) for k in keys}
    aggregates={}
    if passed:
        residual={r['case']:r for r in json.loads((OUT/'offline_final_residuals.json').read_text())}
        for r in rows:
            r['residual_k32_offline']=residual[r['case']]['k32_residual']
            r['residual_off_k32_offline']=residual[r['case']]['off_k32']
        keys+=['residual_k32_offline','residual_off_k32_offline']
        aggregates['by_phase']={str(it):stats([r for r in rows if r['iteration']==it]) for it in (0,249,499,999)}
        aggregates['by_instance']={str(i):stats([r for r in rows if r['instance']==i]) for i in range(8)}
        # Equal weight for phases within seed, seeds within instance (balanced design).
        aggregates['middle_late_instance_means']={str(i):{k:statistics.mean(
            statistics.mean(r[k] for r in rows if r['instance']==i and r['seed']==seed and r['iteration'] in (499,999))
            for seed in (31415,27182)) for k in keys} for i in range(8)}
        aggregates['middle_late_between_instance_dispersion']={k:dict(
            mean=statistics.mean(d[k] for d in aggregates['middle_late_instance_means'].values()),
            median=statistics.median(d[k] for d in aggregates['middle_late_instance_means'].values()),
            minimum=min(d[k] for d in aggregates['middle_late_instance_means'].values()),
            maximum=max(d[k] for d in aggregates['middle_late_instance_means'].values())) for k in keys}
    (OUT/'descriptive_summary.json').write_text(json.dumps(aggregates,indent=2)+'\n')
    phase_table=''
    instance_table=''
    if passed:
        cols=['processed_nodes','candidate_evaluations','pruned_neighbor_slots','evaluated_not_final_chosen',
              'accepted_moves','processed_node_reinsertion_attempts','residual_k32_offline','residual_off_k32_offline']
        phase_table='\n'.join('| '+it+' | '+' | '.join(f"{v[k]['mean']:.2f}" for k in cols)+' |' for it,v in aggregates['by_phase'].items())
        instance_table='\n'.join('| '+i+' | '+' | '.join(f'{v[k]:.2f}' for k in cols)+' |' for i,v in aggregates['middle_late_instance_means'].items())
    report=f'''# M0 exact K32 SRR replay and execution trace

**{summary['decision']}**: {summary['passed']}/64 saved ant0 pairs pass the exact replay gate.
Historical Stage A remains **NO-GO**; the secondary evidence decision remains
**INCONCLUSIVE**. No M1, candidate expansion, ACO trajectory, model operation,
new instance generation or reserved confirmation-data access was performed.

## Exactness and scope

Starting branch: `feature/r3-budget-aware-candidate-graph`, HEAD
`390f3e46186113ffd5ed714343fc93ea09119b0a`. The R3 worktree was clean before editing.
The original checkout is `D:/GitRepository/DyNACO` on `final`, with its existing
untracked `.r1-env/` preserved; this work is in `D:/GitRepository/DyNACO-r3`.
Only new M0 files are delivered. See `repository_safety.json` for the status inventory.

`manifest.json` pins baseline source, build flags, generated replay source,
binary, saved coordinates, native K32 lists, raw/refined/source arrays, trace
inputs, frozen protocol and historical decisions. The complete 440-file
Stage A ledger was rehashed. Inputs were rehashed again after validation.
The executable is a standalone test-only C++17 pybind module; it does not
include, construct or link the production solver. The C++ method bodies for
the uninstrumented path are extracted literally from unchanged baseline source.
The logged path applies the explicit hook substitutions saved in
`instrumentation_replacements.json`. Both paths compile together with GCC13,
`-O3 -std=c++17`, as in the baseline, without fast-math flags. No rounding mode
or CPU floating-point environment is changed.

All 64 raw tours and initial checklists were independently reconstructed from
source-before plus saved relocation decisions, without random draws.
Each replay final route matches the saved refined ant0 array exactly, with
closure checked when the saved array has N+1 elements. No cyclic normalization,
rotation or reversal is used to obtain matches. The logged and literal paths
match route, final checklist and accumulated float32 delta. Python independently
checks every evaluated gain in native order, each selection update, pruning
boundary, accepted route mutation, permutation, checklist insertion, and
accepted edge delta against a float64 full-tour cost difference. C++ additionally
checks position inverses after each logged accepted move. Independent final
float64 costs match those of the saved refined arrays in all cases.
`test_results.json` and `validation_summary.json` retain every case, including
empty mismatch-position lists. There are zero dropped or failed cases.

## Semantic dependencies from source audit

- Minimal state: exact float32 coordinates, N=1000, K=32, ordered native K32
  lists, raw route array, inverse positions, ordered initial checklist and
  `extend_ls=true`. No pheromone, network, RNG, prior or accumulated ACO cost
  is required. Saved configuration has local search enabled and L=0.
- `src/mfaco_train.h:338`: coordinate subtraction, squared sum, square root,
  and returned distance are float32. Gain expressions remain left associative
  in float32; no robust threshold is introduced into native acceptance.
- `src/mfaco_train.cpp:358`: KD-tree candidate construction uses double
  coordinates promoted from the saved floats, exact Euclidean distance and
  repeated nearest queries with deletion. `src/kd_tree.h:460` updates on `<=`,
  so equal-distance identity follows traversal/bucket order rather than a
  node-ID stable sort. Replay uses persisted list identity/order, never a
  newly sorted or regenerated list. Thus tie ordering is inherited exactly.
- Traced sampling initializes the checklist with start_node, then appends
  current, chosen and chosen's pre-relocation predecessor for each new source
  edge, in that order, with whole-list de-duplication. Source-edge membership
  is determined against the original source positions. Reconstruction checks
  saved booleans, raw route order and checklist order exactly.
- `two_opt_nn` visits checklist entries using a monotonically increasing
  cursor. It reads successor/predecessor from current positions, evaluates
  outgoing orientation first, incoming second, and scans ranks 0 through 31
  in each direction. Invalid neighbor terminates that direction. A neighbor
  is evaluated only while the old incident-edge distance is strictly greater
  than the neighbor distance; equality terminates scanning of the whole suffix.
- max_diff starts at zero; only strict `diff > max_diff` updates selection.
  Equal gains retain the earlier candidate/orientation. One best positive move
  is accepted per processed anchor, after both scans; there is no epsilon.
- `flip_route_section` sorts endpoint positions and reverses [first,last)
  when that segment is no longer than its complement (ties use this branch).
  The other branch uses the original circular swap schedule and position
  updates verbatim, including its first==0 arithmetic. It is not replaced by
  a canonical 2-opt reversal. The independent validator uses the same literal
  schedule and checks the resulting changed indices against the logged patch.
- After acceptance, endpoints best_move[0..3] are considered in order.
  Membership is tested against the entire checklist, including processed
  entries. An already processed endpoint therefore cannot be re-enqueued
  in this call. Subsequent anchors see the mutated route and updated positions.
  Accumulated total_change is decremented by selected max_diff in float32.

## Observed execution versus offline opportunities

Every `*_events.json` contains the exact processing order, each evaluated
neighbor/rank/orientation/gain and selection update, distance-pruning boundary,
final selected endpoints, changed route indices, and endpoint insertion or
duplicate attempt with its already-processed status. Events are bounded by N
checklist entries, at most 2K evaluations and two pruning boundaries per entry,
one accepted move and four insertion attempts per entry. Route patches have
at most N entries. No event sampling or truncation is applied. Final tour and
checklist are retained in each log; binary output is not needed to read them.

The following quantities must be read separately:

| Quantity | Meaning and limitation |
|---|---|
| Candidate unavailable | At an observed anchor, N-1-K directed neighbor identities are absent from its fixed K32 list. The manifest and graph establish this structural exclusion; those neighbors were never evaluated. This is not a count of improving moves. |
| Available but not reached | K32 rows of nodes never reached by the actual checklist remain unscanned. `unreached_nodes` counts such anchors; it does not freeze an earlier tour or assign gains to them. |
| Native pruning | The boundary neighbor and remaining suffix are skipped in that orientation on the actual current tour. `pruned_neighbor_slots` includes the boundary and repeated directional slots, not unique moves. |
| Evaluated but not chosen | Actual gain computations other than the final chosen neighbor at that anchor, including nonpositive gains and positive candidates superseded later. These are evaluation events, not unique moves. |
| Accepted move | One observed best positive float32 2-opt move and its exact mutation. Each is independently validated against full-tour float64 costs. |
| Processed node rejected for requeue | An observed extension attempt for an endpoint whose anchor was already processed; whole-list membership prevents insertion. Counts are attempts, and can repeat a node. |
| Offline final residual | Robust improving canonical 2-opt moves scanned only after replay passes. Float64 gain must exceed max(1e-8,1e-6*final length). These are opportunities on the final frozen tour, with no claim of native evaluation or acceptance. |

Mean counts per saved phase (zero-based iteration):

| Iteration | Processed | Evaluations | Pruned slots | Evaluated unchosen | Accepted | Processed requeue attempts | Offline K32 residual | Offline off-K32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{phase_table}

Middle/late phase means within seed, then the two seeds within each instance:

| Instance | Processed | Evaluations | Pruned slots | Evaluated unchosen | Accepted | Processed requeue attempts | Offline K32 residual | Offline off-K32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{instance_table}

`descriptive_summary.json` includes median, minimum and maximum by phase and
instance, plus dispersion across the eight independent instance means.
Seeds and phases are nested within instance. These 64 records are selected
ant0 examples from eight saved TSP instances, not 64 independent instances;
they share histories within a trajectory. Refined ant0 is not the selected
best-of-100 source. No population inference or passed scientific gate follows.
Processed requeue attempts and final residuals establish observable mechanics,
but cannot prove that a particular skipped or unchosen move would improve
the final trajectory. The tour at the skipped event generally differs from
the final scanner's tour. Candidate, pruning and scheduling effects interact.

## Smallest proposed controlled M1 (not executed)

Prospectively freeze one mechanics question: does one additional call of the
unchanged K32 routine, starting from M0's validated final route and final
checklist, find robust additional gains on saved middle/late ant0 examples?
Compare the validated single-call endpoint with exactly one second call;
keep candidates, distances, pruning, tie order, extension and move selection
unchanged. Instrument both calls. This resets only the processing cursor at a
call boundary; it tests a bounded revisit/scheduling intervention, not candidate
expansion or an alternative production solver. It does not isolate every
possible mechanism of an individual missed move.

Use the existing 32 middle/late records as an explicitly exploratory paired
mechanics diagnostic, with phase/seed nested in the eight saved instances.
Cap at two total calls, at most 2N processed anchors and 4NK gain evaluations
per input, one CPU worker, 5 minutes wall time, 1 GB RSS. Freeze a descriptive
signal criterion before M1: additional float64 reduction > max(1e-8,1e-6*baseline
final length) in at least one record in at least four of eight instances,
with all permutations valid and native logged/unlogged invariance intact.
Report evaluation cost and any worsening case. Failure of correctness stops
M1; absence of the signal is a negative mechanics finding. This is not a
scientific GO criterion or authorization to run M1. Any claim about best-of-100
solver quality would require a separate registered equal-compute experiment
on new independent instances; the reserved Stage A confirmation set stays sealed.

The delivered result is **{summary['decision']}**. Historical decisions retain
their existing meanings and hashes. M0 establishes replay and observed events;
the proposed M1 remains unimplemented and unexecuted.
'''
    (OUT/'M0_RESEARCH_REPORT.md').write_text(report)
    (OUT/'HANDOFF_SUPPLEMENT.md').write_text(f'''# M0 handoff supplement

- Result: {summary['decision']}, {summary['passed']}/64 exact array matches; no dropped cases.
- Historical Stage A: NO-GO. Secondary review: INCONCLUSIVE.
- See validation_summary.json, test_results.json, descriptive_summary.json and bounded per-pair event logs.
- M1 proposal is a second unchanged K32 call for a bounded exploratory revisit diagnostic; not implemented or run.
- No push is authorized by this M0 delivery.
''')
    # Inventory is captured with Windows Git, which owns these worktrees.
    # WSL Git interprets their Windows admin pointers and attributes differently.
    assert (OUT/'repository_safety.json').is_file()
    # Final source inventory includes this reporting-only script created after validation.
    final=dict(validation_manifest_sha256=sha256(OUT/'manifest.json'),
               m0_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'r3/m0').glob('*') if p.is_file()},
               output_sha256={p.name:sha256(p) for p in OUT.iterdir() if p.is_file() and p.name!='final_output_manifest.json'})
    (OUT/'final_output_manifest.json').write_text(json.dumps(final,indent=2)+'\n')
    print(summary['decision'],summary['passed'],'/64; report and aggregation complete')

if __name__=='__main__':main()
