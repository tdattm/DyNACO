"""Independent consistency check of the saved secondary output and inputs."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from r3.common import ROOT, sha256, save_json

OUT=ROOT/'artifacts/r3/secondary_review/20261101_saved_ant0_v1'
SOURCE=ROOT/'artifacts/r3/experiments/stage_a_v1_01'

def main():
    manifest=json.loads((OUT/'input_manifest.json').read_text())
    for name,digest in manifest['input_sha256'].items():
        assert sha256(ROOT/name)==digest,name
    assert sha256(SOURCE/'gate.json')==manifest['historical_gate_sha256']
    assert sha256(ROOT/'artifacts/r3/reports/R3_STAGE_A_REPORT.md')==manifest['historical_report_sha256']
    historical=json.loads((SOURCE/'gate.json').read_text())
    assert historical['decision']=='NO-GO' and not historical['stage_b_authorized']
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['secondary_decision']=='INCONCLUSIVE' and not summary['stage_b_authorized']
    with (OUT/'paired_policy_metrics.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==64*2*15
    groups=defaultdict(list)
    for r in rows:
        key=(int(r['instance']),int(r['seed']),int(r['iteration_before_srr']))
        groups[key].append(r)
        robust=int(r['robust_moves'])
        counts=[int(r[k]) for k in ('off_graph_moves','no_initial_anchor_moves','initially_reachable_but_pruned_moves','initial_checklist_potential_moves')]
        assert sum(counts)==robust
        gains=[float(r[k]) for k in ('off_graph_gain_mass','no_initial_anchor_gain_mass','initially_reachable_but_pruned_gain_mass','initial_checklist_potential_gain_mass')]
        assert abs(sum(gains)-float(r['robust_gain_mass']))<1e-8
        assert int(r['extra_directed_slots'])==(32000 if r['policy']=='k64' else 0 if r['policy']=='k32' else 8000 if r['policy'] not in ('k48',) else 16000)
        assert r['actual_native_evaluations']=='unavailable' and r['actual_native_acceptances']=='unavailable'
    assert len(groups)==64
    for group in groups.values():
        assert len(group)==30
        raw=[r for r in group if r['tour']=='raw'];ref=[r for r in group if r['tour']=='refined']
        assert len(raw)==len(ref)==15
        assert {r['policy'] for r in raw}=={r['policy'] for r in ref}
        for r in group:
            assert abs(float(r['raw_length'])-float(raw[0]['raw_length']))<1e-12
            assert abs(float(r['refined_length'])-float(ref[0]['refined_length']))<1e-12
            assert abs(float(r['paired_length_reduction'])-(float(r['raw_length'])-float(r['refined_length'])))<1e-10
            assert int(r['initial_checklist_size'])==int(raw[0]['initial_checklist_size'])
    phase_map={0:'initial',249:'early',499:'middle',999:'late'}
    for record in summary['phase']:
        phase=record['phase'];iteration=next(k for k,v in phase_map.items() if v==phase)
        raw=[next(r for r in g if r['tour']=='raw' and r['policy']=='k32') for k,g in groups.items() if k[2]==iteration]
        refined=[next(r for r in g if r['tour']=='refined' and r['policy']=='k32') for k,g in groups.items() if k[2]==iteration]
        assert len(raw)==len(refined)==16
        comparisons={'mean_paired_length_reduction':statistics.mean(float(r['paired_length_reduction']) for r in raw),
            'mean_raw_off_graph':statistics.mean(int(r['off_graph_moves']) for r in raw),
            'mean_refined_off_graph':statistics.mean(int(r['off_graph_moves']) for r in refined),
            'mean_refined_k32_residual':statistics.mean(int(r['robust_moves'])-int(r['off_graph_moves']) for r in refined)}
        for key,value in comparisons.items():assert abs(record[key]-value)<1e-10,(phase,key)
    for i in range(8):
        observations=[]
        for seed in (31415,27182):
            for it in (499,999):
                raw=next(r for r in groups[(i,seed,it)] if r['tour']=='raw' and r['policy']=='k32')
                refined=next(r for r in groups[(i,seed,it)] if r['tour']=='refined' and r['policy']=='k32')
                observations.append((float(raw['paired_length_reduction']),int(raw['off_graph_moves']),int(refined['off_graph_moves']),
                    int(refined['robust_moves'])-int(refined['off_graph_moves'])))
        item=summary['per_instance'][i]
        for index,key in enumerate(('length_reduction','raw_off_graph','refined_off_graph','refined_k32_residual')):
            assert abs(item[key]-statistics.mean(v[index] for v in observations))<1e-10
    audit=dict(status='PASS',historical_gate='NO-GO',historical_gate_unchanged=True,
        original_input_files_rehashed=len(manifest['input_sha256']),paired_groups=len(groups),policy_rows=len(rows),
        count_and_gain_decompositions_checked=len(rows),phase_summaries_recomputed=4,
        instance_summaries_recomputed=8,secondary_source_sha256=sha256(ROOT/'r3/secondary_existing.py'),
        audit_source_sha256=sha256(ROOT/'r3/secondary_audit.py'),
        output_sha256={p.name:sha256(p) for p in OUT.iterdir()
                       if p.is_file() and p.suffix not in ('.log',)
                       and not p.name.startswith('audit_results')})
    save_json(OUT/'audit_results_v3.json',audit)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
