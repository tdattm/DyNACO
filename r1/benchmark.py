"""Sequential bounded pilots and matched-iteration experiments; no training.

python -m r1.benchmark --output artifacts/r1/experiments/<new-ID>
"""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import random
import resource
import subprocess
import time
import traceback

import numpy as np
import torch
from net import Net
from r1.datasets import ensure_dataset, sha256
from r1.engine import RunConfig, run, weight_hash
from r1.scheduler import GuidanceScheduler, SchedulerConfig

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def telemetry():
    p = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,memory.used,temperature.gpu,power.draw,utilization.gpu', '--format=csv'], capture_output=True, text=True, timeout=10)
    return p.stdout.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--sizes', type=int, nargs='+', default=[100, 200, 500, 1000])
    p.add_argument('--instances', type=int, default=2)
    p.add_argument('--seeds', type=int, nargs='+', default=[1234, 5678])
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--max-search-seconds', type=float, default=600.)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'runs').mkdir()
    begin = time.perf_counter()
    assert torch.cuda.is_available() and args.device.startswith('cuda'), 'GPU experiment only; no implicit CPU fallback'
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = Net(feats=2, edge_feats=6, logit_net=True).to(args.device).eval()
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    initial_weights = weight_hash(model)
    methods = [('fixed_s25', 'fixed_s', 25), ('fixed_s50', 'fixed_s', 50),
               ('fixed_s100', 'fixed_s', 100), ('fixed_s200', 'fixed_s', 200),
               ('event_stagnation', 'event_stagnation', 100)]
    manifest = dict(schema_version=1, base_commit='cd16e2a68fd035f94a8916d41463904f31483d7f',
                    current_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    arguments=vars(args), checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT),
                    checkpoint_metadata={k: v for k, v in ckpt.items() if k not in ('model_state_dict', 'optimizer_state_dict')},
                    weights_sha256=initial_weights, environment_manifest='../../environment/environment_manifest.json',
                    source_sha256={str(f.relative_to(ROOT)): sha256(f) for f in sorted((ROOT/'r1').glob('*.py'))},
                    device=args.device, threads=2, ants=100, k=32, rho=.1, min_new_edges=12,
                    annealing='no_anneal', phased_injection=False,
                    scheduler=dict(min_interval=25, max_interval=200, stagnation_threshold=50, improvement_tolerance=1e-6),
                    methods=methods, dataset_seed=20261031, solver_seeds=args.seeds,
                    method_order='random.Random(314159), shuffled separately within each matched size/instance/seed block',
                    time_budget_protocol='best-so-far at predefined .1,.25,.5,1,2,5 seconds only when every run at that N reaches the budget; search time excludes setup/warmup',
                    timing='synchronized perf_counter; graph includes internal H2D, transfer is guidance D2H per iteration; native sample/update includes wrapper synchronization; decision includes observe; residual includes synchronization/logging/checks',
                    gpu_before=telemetry())
    write_json(out/'benchmark_config.json', manifest)
    datasets = {}
    entries = []
    for n in args.sizes:
        coordinates, entry = ensure_dataset(ROOT/'artifacts/r1/datasets', n, args.instances)
        datasets[n] = coordinates
        entries.append(entry)
    write_json(out/'dataset_manifest.json', entries)
    # Warm up both graph/model and native backend at every evaluated scale.
    warmups, pilots = [], []
    for n in args.sizes:
        r = run(datasets[n][0], model, RunConfig(total_iterations=10, fixed_s=5, device=args.device))
        warmups.append(dict(n=n, wall_seconds=r['wall_seconds'], setup_seconds=r['setup_seconds'], peak_gpu_bytes=r['peak_gpu_bytes']))
        r = run(datasets[n][0], model, RunConfig(total_iterations=200, fixed_s=100, device=args.device))
        write_json(out/f'pilot_n{n}.json', r)
        pilots.append(dict(n=n, wall_seconds=r['wall_seconds'], setup_seconds=r['setup_seconds'], peak_gpu_bytes=r['peak_gpu_bytes'], iterations=200, gnn_calls=r['gnn_calls']))
        print(f'pilot N={n}: {r["wall_seconds"]:.3f}s, GPU allocated peak={r["peak_gpu_bytes"]/2**20:.1f}MiB', flush=True)
    predicted = sum(x['wall_seconds'] * args.iterations / 200 * args.instances * len(args.seeds) * len(methods) for x in pilots)
    # Extra guidance calls at S25 can exceed the S100 pilot, so retain 2x margin.
    write_json(out/'pilot.json', dict(warmups=warmups, pilots=pilots, projected_search_seconds=predicted, conservative_projection_seconds=2*predicted, setup_and_warmup_elapsed_seconds=time.perf_counter()-begin))
    if 2*predicted > args.max_search_seconds:
        raise RuntimeError(f'Bounded pilot predicts too much work: conservative {2*predicted:.1f}s > {args.max_search_seconds}s; no principal experiments launched')
    fields = ['run_id', 'status', 'method', 'n', 'instance_id', 'seed', 'dataset_sha256', 'checkpoint_sha256',
              'iterations_requested', 'iterations_executed', 'gnn_calls', 'final_cost', 'wall_seconds', 'setup_seconds',
              'graph_seconds', 'gnn_seconds', 'transfer_seconds', 'decision_seconds', 'sample_update_seconds',
              'peak_gpu_bytes', 'instance_seed', 'candidate_hash', 'error']
    iter_fields = ['run_id', 'iteration', 'best_so_far', 'sampled_best_so_far', 'iteration_best', 'elapsed_seconds', 'gnn_calls']
    event_fields = ['run_id', 'iteration', 'reason', 'global_best', 'stagnation', 'graph_seconds', 'gnn_seconds']
    order_rng = random.Random(314159)
    search_start = time.perf_counter()
    with (out/'runs.csv').open('x', newline='') as rf, (out/'iterations.csv').open('x', newline='') as itf, (out/'refresh_events.csv').open('x', newline='') as ef:
        rw, iw, ew = csv.DictWriter(rf, fields), csv.DictWriter(itf, iter_fields), csv.DictWriter(ef, event_fields)
        rw.writeheader(); iw.writeheader(); ew.writeheader()
        for n, entry in zip(args.sizes, entries):
            for index, coords in enumerate(datasets[n]):
                for seed in args.seeds:
                    order = list(methods)
                    order_rng.shuffle(order)
                    for method, mode, interval in order:
                        run_id = f'n{n}_i{index}_seed{seed}_{method}'
                        row = dict(run_id=run_id, method=method, n=n, instance_id=entry['instance_ids'][index], seed=seed,
                                   dataset_sha256=entry['sha256'], checkpoint_sha256=manifest['checkpoint_sha256'], iterations_requested=args.iterations)
                        try:
                            if time.perf_counter()-search_start > args.max_search_seconds:
                                raise RuntimeError('Total search-time cap reached; remaining configuration retained as failed')
                            cfg = RunConfig(total_iterations=args.iterations, fixed_s=interval, seed=seed, device=args.device)
                            sc = SchedulerConfig(mode=mode, total_iterations=args.iterations, fixed_s=interval)
                            r = run(coords, model, cfg, scheduler=GuidanceScheduler(sc))
                            assert weight_hash(model) == initial_weights
                            assert r['gnn_calls'] == len(r['refresh_events'])
                            assert r['iterations_executed'] == r['pheromone_updates'] == args.iterations
                            assert all(b['best_so_far'] <= a['best_so_far'] for a, b in zip(r['iterations'], r['iterations'][1:]))
                            assert all(np.isfinite(v) and v >= 0 for k, v in r.items() if k.endswith('_seconds'))
                            write_json(out/'runs'/f'{run_id}.json', dict(**row, status='ok', scheduler=asdict(sc), gpu_after=telemetry(), process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **r))
                            iw.writerows(dict(run_id=run_id, **record) for record in r['iterations'])
                            ew.writerows(dict(run_id=run_id, **record) for record in r['refresh_events'])
                            row.update({key: r[key] for key in fields if key in r})
                            row['status'] = 'ok'
                            print(f'{run_id}: cost={r["final_cost"]:.7f}, wall={r["wall_seconds"]:.3f}s, calls={r["gnn_calls"]}', flush=True)
                        except Exception:
                            row.update(status='failed', error=traceback.format_exc())
                            write_json(out/'runs'/f'{run_id}_failed.json', row)
                            print(f'FAILED {run_id}: {row["error"]}', flush=True)
                        rw.writerow(row)
                        rf.flush(); itf.flush(); ef.flush()
    # Re-check two principal-scale configurations, outside statistical samples.
    reproducibility = []
    n = max(args.sizes)
    seed = args.seeds[0]
    for method, mode, interval in (methods[2], methods[4]):
        run_id = f'n{n}_i0_seed{seed}_{method}'
        original_file = out/'runs'/f'{run_id}.json'
        if not original_file.exists():
            continue
        original = json.loads(original_file.read_text())
        cfg = RunConfig(total_iterations=args.iterations, fixed_s=interval, seed=seed, device=args.device)
        sc = SchedulerConfig(mode=mode, total_iterations=args.iterations, fixed_s=interval)
        repeated = run(datasets[n][0], model, cfg, scheduler=GuidanceScheduler(sc))
        write_json(out/f'reproducibility_{run_id}.json', repeated)
        reproducibility.append(dict(run_id=run_id, exact_trace=original['trace']==repeated['trace'], exact_tour=original['tour']==repeated['tour'], exact_refresh_events=[(e['iteration'],e['reason']) for e in original['refresh_events']]==[(e['iteration'],e['reason']) for e in repeated['refresh_events']], max_trace_difference=float(np.max(np.abs(np.asarray(original['trace'])-repeated['trace'])))))
    write_json(out/'reproducibility.json', reproducibility)
    write_json(out/'completion.json', dict(search_phase_seconds=time.perf_counter()-search_start, total_after_import_seconds=time.perf_counter()-begin, gpu_after=telemetry(), weight_identity=weight_hash(model)==initial_weights))
    from r1.report import generate
    generate(out)


if __name__ == '__main__':
    main()
