"""Independent artifact consistency check after the experiment (no neural calls)."""
import csv
import hashlib
import json
from pathlib import Path
import math
import statistics
import sys

root = Path(__file__).resolve().parents[3]
out = root / sys.argv[1]
config = json.loads((out/'benchmark_config.json').read_text())
rows = list(csv.DictReader((out/'runs.csv').open()))
iterations = list(csv.DictReader((out/'iterations.csv').open()))
events = list(csv.DictReader((out/'refresh_events.csv').open()))
summary = json.loads((out/'summary.json').read_text())
assert len(rows) == 80 and all(r['status']=='ok' for r in rows)
assert len({r['run_id'] for r in rows}) == 80
assert len(iterations) == 80000
assert sum(int(r['gnn_calls']) for r in rows) == len(events)
assert len({r['checkpoint_sha256'] for r in rows}) == 1
for entry in json.loads((out/'dataset_manifest.json').read_text()):
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
source_checks = {}
for name, digest in config['source_sha256'].items():
    source_checks[name] = hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
assert all(source_checks.values())
for r in rows:
    run_id = r['run_id']
    data = json.loads((out/'runs'/f'{run_id}.json').read_text())
    it = [v for v in iterations if v['run_id']==run_id]
    ev = [v for v in events if v['run_id']==run_id]
    assert [int(v['iteration']) for v in it] == list(range(1,1001))
    assert int(ev[0]['iteration']) == 0 and all(int(v['iteration']) < 1000 for v in ev)
    assert len(ev)==data['gnn_calls']==len(data['refresh_events'])
    assert data['iterations_executed']==data['pheromone_updates']==1000
    best = [data['initial_cost']] + [float(v['best_so_far']) for v in it]
    assert all(b <= a for a,b in zip(best,best[1:]))
    assert data['final_cost']==best[-1]
    assert sorted(data['tour'])==list(range(data['n']))
    for key in ('wall_seconds','setup_seconds','gnn_seconds','graph_seconds','transfer_seconds','decision_seconds','sample_update_seconds'):
        assert math.isfinite(data[key]) and data[key] >= 0
    if r['method'].startswith('fixed_s'):
        interval = int(r['method'].split('fixed_s')[1])
        assert [int(v['iteration']) for v in ev]==list(range(0,1000,interval))
    peers = [p for p in rows if (p['n'],p['instance_id'],p['seed'])==(r['n'],r['instance_id'],r['seed'])]
    assert len({p['candidate_hash'] for p in peers})==1
    assert len({p['instance_seed'] for p in peers})==1
for s in summary['summary']:
    matching = [float(r['final_cost']) for r in rows if int(r['n'])==s['n'] and r['method']==s['method']]
    assert len(matching)==s['count']==4
    assert statistics.mean(matching)==s['final_cost_mean']
repeat = json.loads((out/'reproducibility.json').read_text())
assert len(repeat)==2 and all(r['exact_trace'] and r['exact_tour'] and r['exact_refresh_events'] for r in repeat)
figures = sorted((out/'figures').glob('*.png'))
assert len(figures)==6 and all(p.stat().st_size > 10000 for p in figures)
result = dict(status='PASS', runs=len(rows), iterations=len(iterations), refresh_events=len(events), failed_runs=0,
              source_hashes_match=source_checks, principal_repeats_exact=True, figures=[p.name for p in figures])
(out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
