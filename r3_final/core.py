import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import utils
from r3.common import ROOT, CHECKPOINT, sha256, git, strict_cuda, model
from r3.state_capture import solver

OUT = ROOT / 'artifacts/r3_final_end_to_end/v1'
ARMS = ('A', 'B', 'C')
MODES = ('iterations', 'wall')
SEEDS = (31415, 27182)
COUNTERS = ('calls', 'candidates', 'expanded_candidates', 'gains', 'expanded_gains', 'accepted', 'expanded_accepted')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def length(x, tour):
    tour = np.asarray(tour)
    if len(tour) == len(x) + 1:
        if tour[-1] != tour[0]:
            raise AssertionError('unclosed tour')
        tour = tour[:-1]
    if not np.array_equal(np.sort(tour), np.arange(len(x))):
        raise AssertionError('invalid permutation')
    y = np.asarray(x, dtype=np.float64)[tour]
    value = float(np.linalg.norm(y - np.roll(y, -1, axis=0), axis=1).sum(dtype=np.float64))
    if not np.isfinite(value):
        raise AssertionError('nonfinite length')
    return value


def allocation(x, tour, nn32):
    # Deliberately compute local scales every epoch; all cost is charged to C.
    x = np.asarray(x, dtype=np.float64)
    scale = np.linalg.norm(x[:, None] - x[nn32], axis=2).mean(axis=1)
    succ = np.empty(len(x), dtype=np.int32)
    pred = np.empty_like(succ)
    succ[tour] = np.roll(tour, -1)
    pred[tour] = np.roll(tour, 1)
    score = np.maximum(np.linalg.norm(x-x[succ], axis=1), np.linalg.norm(x-x[pred], axis=1)) / np.maximum(scale, 1e-12)
    selected = np.lexsort((np.arange(len(x)), -score))[:len(x)//4]
    widths = np.full(len(x), 32, dtype=np.int32)
    widths[selected] = 64
    return widths, score


def identities():
    for mode in MODES:
        for i in range(16):
            for si, seed in enumerate(SEEDS):
                rotation = (2*i+si) % 3
                for arm in ARMS[rotation:] + ARMS[:rotation]:
                    yield dict(mode=mode, instance=i, seed=seed, arm=arm,
                               run_id=f'{mode}_i{i:02d}_s{seed}_{arm}')


def run(x, net, seed, arm, mode, iterations=1000, seconds=8., observer=None, traced=False, counted=True):
    begin = time.perf_counter()
    utils.set_seed(seed)
    # Exactly the original utils.infer_instance duplicate-construction/seed contract.
    discarded = solver(x)
    discarded.seed_rng(seed)
    del discarded
    cmin = x.min(axis=0)
    scale = (x.max(axis=0)-cmin).max()
    norm = (x-cmin)/(scale if scale >= 1e-6 else 1.)
    aco = solver(x)
    native_seed = int(np.random.randint(0, 2**63-1))
    aco.seed_rng(native_seed)
    aco._cpp.enable_ls_counts(counted)
    initial = aco.best_route.copy()
    nn64 = np.ascontiguousarray(np.concatenate((aco.nn_list, aco.backup_list), axis=1))
    if (nn64 < 0).any() or (nn64 >= len(x)).any() or (nn64 == np.arange(len(x))[:, None]).any():
        raise AssertionError('invalid native candidates')
    initial_source_hash = array_hash(aco.source_route)
    initial_pheromone_hash = array_hash(aco.pheromone_sparse_np)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    setup = time.perf_counter()-begin
    # No policy-specific work is hidden in setup. Timestamp includes checkpoint copies.
    routes = [initial]
    route_iterations = [0]
    route_times = [0.]
    times = [0.]
    costs = [aco.best_cost]
    allocations = []
    width_history = []
    score_history = []
    prior = None
    last_native = aco.best_cost
    allocation_seconds = 0.
    neural_seconds = 0.
    widths = np.full(len(x), 32, dtype=np.int32)
    completed = 0
    start = time.perf_counter()
    with torch.no_grad():
        while mode == 'wall' or completed < iterations:
            if mode == 'wall' and time.perf_counter()-start >= seconds:
                break
            # Iteration has started. Graph/policy preparation belongs to it, even on overshoot.
            if completed % 100 == 0:
                tick = time.perf_counter()
                previous = widths.copy()
                if arm == 'C':
                    widths, scores = allocation(x, aco.best_route, aco.nn_list)
                    aco._cpp.set_ls_widths(widths.tolist())
                elif completed == 0:
                    widths.fill(32 if arm == 'A' else 40)
                    aco._cpp.set_ls_widths(widths.tolist())
                    scores = np.empty(0, dtype=np.float64)
                else:
                    scores = np.empty(0, dtype=np.float64)
                width_history.append(widths.copy())
                if arm == 'C':
                    score_history.append(scores.copy())
                allocations.append(dict(iteration=completed, changed_nodes=int(np.count_nonzero(previous != widths)),
                                        slots=int(widths.sum()), expanded_nodes=int(np.count_nonzero(widths == 64))))
                allocation_seconds += time.perf_counter()-tick
                tick = time.perf_counter()
                graph = utils.build_pyg_data_tsp(aco, norm, 'cuda:0', dynamic=True,
                                                ablation_pheromone=False, ablation_incumbent=False)
                prior = net(graph).view(aco.n, aco.k)
                torch.cuda.synchronize()
                neural_seconds += time.perf_counter()-tick
            current = prior * (1. - (completed % 100)/99.)
            output = aco.sample(require_prob=traced, prior=current.cpu().numpy(), parallel_traced=True)
            ant_costs, flats = output[:2]
            idx = int(ant_costs.argmin().item())
            aco._update_pheromone_from_flat(flats[idx], float(ant_costs[idx]))
            completed += 1
            torch.cuda.synchronize()
            # Only copy a route when native global best changes. Same policy in every arm.
            changed = aco.best_cost != last_native
            if changed:
                routes.append(aco.best_route.copy())
                route_iterations.append(completed)
                last_native = aco.best_cost
            costs.append(last_native)
            # Charge online list/route recording before the completion timestamp.
            times.append(0.)
            if changed:
                route_times.append(0.)
            stamp = time.perf_counter()-start
            times[-1] = stamp
            if changed:
                route_times[-1] = stamp
            if observer:
                observer(completed, aco, output, current)
            if mode == 'wall' and stamp > seconds:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    counters = aco._cpp.get_ls_counts()
    peak_cuda = torch.cuda.max_memory_allocated()
    # All independent float64 checks, undirected counts, serialization are outside search.
    routes = np.asarray(routes, dtype=np.int32)
    route_lengths = np.array([length(x, r) for r in routes])
    route_times = np.asarray(route_times)
    times = np.asarray(times)
    eligible = np.flatnonzero(route_times <= seconds) if mode == 'wall' else np.arange(len(routes))
    primary_index = int(eligible[-1])
    eligible_updates = int(np.count_nonzero((times > 0) & (times <= seconds))) if mode == 'wall' else completed
    if mode == 'wall' and eligible_updates == 0:
        raise RuntimeError('no completed update within wall budget')
    initial_len = route_lengths[0]
    final_length = length(x, aco.best_route)
    source_length = length(x, aco.source_route)
    if abs(final_length-aco.best_cost) > 2e-5*final_length or abs(source_length-aco.source_cost) > 2e-5*source_length:
        raise AssertionError('native cost inconsistent')
    if counted and counters['calls'] != 100 * completed:
        raise AssertionError('not exactly one SRR call per ant')
    widths_np = np.asarray(width_history, dtype=np.int32)
    unique = []
    for w in widths_np:
        u = np.repeat(np.arange(len(x)), w)
        v = nn64[np.arange(64)[None, :] < w[:, None]]
        unique.append(int(len(np.unique(np.minimum(u, v)*len(x)+np.maximum(u, v)))))
    summary = dict(arm=arm, mode=mode, seed=seed, native_seed=native_seed,
                   official_length=float(route_lengths[primary_index]), initial_length=float(initial_len),
                   primary_route_index=primary_index, completed_iterations=completed,
                   eligible_completed_iterations=eligible_updates,
                   overshoot_iterations=completed-eligible_updates,
                   last_completed_before_budget=float(times[eligible_updates]),
                   search_seconds=elapsed, setup_seconds=setup, allocation_seconds=allocation_seconds,
                   graph_neural_seconds=neural_seconds, counters=counters,
                   rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   peak_cuda_bytes=peak_cuda, initial_route_sha256=array_hash(initial),
                   initial_source_sha256=initial_source_hash, initial_pheromone_sha256=initial_pheromone_hash,
                   native_candidates_sha256=array_hash(nn64),
                   slots=int(widths.sum()), unique_undirected_edges=unique,
                   allocations=allocations, final_overshoot_length=float(final_length))
    arrays = dict(routes=routes, route_lengths=route_lengths, route_iterations=np.asarray(route_iterations),
                  route_times=route_times, update_times=times, native_best=np.asarray(costs),
                  allocation_widths=widths_np, allocation_scores=np.asarray(score_history), native_nn64=nn64,
                  final_source=aco.source_route.copy(), final_best=aco.best_route.copy())
    return summary, arrays, aco
