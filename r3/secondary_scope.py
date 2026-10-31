"""Descriptive pre-SRR scope summary; cannot override the frozen source-tour gate."""
import json
import statistics
from r3.common import ROOT,save_json

def main():
    directory=ROOT/'artifacts/r3/experiments/stage_a_v1_01';rows=[]
    for iteration in (0,249,499,999):
        records=[json.loads(p.read_text()) for p in directory.glob(f'i*_s*/diagnostic_raw_before_iter{iteration}.json')]
        assert len(records)==16
        rows.append(dict(iteration_before_srr=iteration,records=16,
            records_with_off_k32_moves=sum(r['membership_missed_count']>0 for r in records),
            mean_full_robust_moves=statistics.mean(r['robust_improving_moves'] for r in records),
            mean_off_k32_moves=statistics.mean(r['membership_missed_count'] for r in records),
            mean_k32_membership_moves=statistics.mean(r['membership_count'][0] for r in records),
            mean_k32_all_node_potential_moves=statistics.mean(r['scan_count'][0] for r in records),
            mean_initial_checklist_membership_moves=statistics.mean(r['initial_checklist_membership_count'] for r in records),
            mean_initial_checklist_potential_moves=statistics.mean(r['initial_checklist_scan_count'] for r in records),
            mean_geometry_incremental_gain_mass=statistics.mean(r['incremental_scan_gain_mass'][4] for r in records),
            mean_uniform_incremental_gain_mass=statistics.mean(r['incremental_scan_gain_mass'][1] for r in records),
            mean_random_incremental_gain_mass=statistics.mean(statistics.mean(r['incremental_scan_gain_mass'][5:]) for r in records)))
    output=dict(scope='secondary descriptive pre-SRR ant0; not the primary frozen gate',
                gate_unchanged='NO-GO',rows=rows)
    save_json(ROOT/'artifacts/r3/manifests/secondary_scope_summary_v1.json',output)
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
