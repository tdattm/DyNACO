"""Measured TSP inference mirroring upstream utils.infer_instance.

No training, native search changes or candidate-graph changes. Timing uses
synchronized perf_counter intervals. Logging never invokes extra neural calls.
"""
from dataclasses import dataclass, asdict
import hashlib
import time

import numpy as np
import torch
import faco
import utils


@dataclass(frozen=True)
class RunConfig:
    total_iterations: int = 1000
    fixed_s: int = 100
    seed: int = 1234
    device: str = 'cuda:0'
    threads: int = 2
    n_ants: int = 100
    k: int = 32
    rho: float = .1
    min_new_edges: int = 12
    annealing_mode: str = 'no_anneal'
    gamma: float = 1.
    min_gamma: float = 0.


def weight_hash(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def validate_tour(coords, tour, cost):
    tour = np.asarray(tour)
    n = len(coords)
    if len(tour) == n + 1:
        assert tour[0] == tour[-1]
        tour = tour[:-1]
    assert np.array_equal(np.sort(tour), np.arange(n)), 'Invalid TSP permutation'
    xy = np.asarray(coords, dtype=np.float64)[tour]
    length = float(np.linalg.norm(xy - np.roll(xy, -1, axis=0), axis=1).sum())
    assert np.isfinite(cost) and abs(length - cost) <= 2e-5 * max(1., length)
    return length


def run(coords, model, config=RunConfig(), *, logging=True, scheduler=None, observer=None, sample_observer=None):
    """One run; setup is separate from timed search, including RNG reset.

    Preserve upstream duplicate construction and NumPy-derived active seed.
    Explicit seed on the first, discarded solver is intentionally preserved.
    scheduler=None is the fixed-S reference, before enabling event policies.
    """
    c = config
    if c.total_iterations <= 0 or c.fixed_s <= 0 or c.threads <= 0:
        raise ValueError('Positive iteration, interval and thread counts required')
    if c.annealing_mode not in ('no_anneal', 'original_fixed'):
        raise ValueError('Unsupported annealing mode')
    if scheduler is not None:
        sc = scheduler.config
        if sc.total_iterations != c.total_iterations or sc.annealing_mode != c.annealing_mode or sc.fixed_s != c.fixed_s:
            raise ValueError('Engine and scheduler configurations differ')
    utils.set_seed(c.seed)
    torch.set_num_threads(c.threads)
    faco.set_faco_cpp_threads(c.threads)
    model.eval()
    coords = torch.as_tensor(np.asarray(coords, dtype=np.float32)).cpu()
    setup_start = time.perf_counter()
    kwargs = dict(coords=coords, n_ants=c.n_ants, cand_list_size=min(c.k, len(coords)-2),
                  backup_list_size=min(c.k, len(coords)-2), disable_heuristic=False,
                  use_local_search=True, decay=c.rho, device=c.device,
                  enable_torch_sync=True, smooth_mmas=True, min_new_edges=c.min_new_edges,
                  extend_ls=True, normalized_heuristic=True, fixed_steps=0)
    discarded = faco.MFACO_TSP(**kwargs)
    discarded.seed_rng(c.seed)
    low, high = coords.min(dim=0)[0], coords.max(dim=0)[0]
    scale = (high - low).max()
    if scale < 1e-6:
        scale = 1.
    norm_coords = (coords - low) / scale
    solver = faco.MFACO_TSP(**kwargs)
    del discarded
    instance_seed = int(np.random.randint(0, 2**63 - 1))
    solver.seed_rng(instance_seed)
    initial_cost = solver.best_cost
    nn_before = solver.nn_list.copy()
    backup_before = solver.backup_list.copy()
    candidate_hash = hashlib.sha256(nn_before.tobytes() + backup_before.tobytes()).hexdigest()
    sync(c.device)
    setup_seconds = time.perf_counter() - setup_start
    if str(c.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(c.device)
    totals = dict(graph_seconds=0., gnn_seconds=0., transfer_seconds=0., decision_seconds=0., sample_update_seconds=0.)
    iterations, events, cost_trace = [], [], []
    prior = None
    best_seen = float('inf')
    calls = 0
    updates = 0
    refresh_at = 0
    sync(c.device)
    start = time.perf_counter()
    with torch.no_grad():
        for iteration in range(c.total_iterations):
            if observer is not None:
                observer(iteration, solver, prior, time.perf_counter() - start)
            decision_start = time.perf_counter()
            reason = ('initial' if iteration == 0 else 'fixed_interval') if iteration % c.fixed_s == 0 else None
            if scheduler is not None:
                reason = scheduler.decide(iteration, solver.best_cost)
            totals['decision_seconds'] += time.perf_counter() - decision_start
            if reason is not None:
                refresh_at = iteration
                sync(c.device)
                tick = time.perf_counter()
                graph = utils.build_pyg_data_tsp(solver, norm_coords, c.device, dynamic=True)
                sync(c.device)
                graph_time = time.perf_counter() - tick
                tick = time.perf_counter()
                prior = model(graph).view(solver.n, solver.k)
                sync(c.device)
                gnn_time = time.perf_counter() - tick
                # Finite checks are outside timed forward component, within wall time.
                assert prior.shape == (solver.n, solver.k) and bool(torch.isfinite(prior).all())
                calls += 1
                totals['graph_seconds'] += graph_time
                totals['gnn_seconds'] += gnn_time
                if logging:
                    events.append(dict(iteration=iteration, reason=reason, global_best=solver.best_cost,
                                       stagnation=iteration - scheduler.last_improvement if scheduler is not None else None,
                                       graph_seconds=graph_time, gnn_seconds=gnn_time))
            current_prior = prior
            if c.annealing_mode == 'original_fixed' and prior is not None:
                local = iteration - refresh_at
                ratio = local / (c.fixed_s - 1) if c.fixed_s > 1 else 0.
                current_prior = prior * (c.gamma * (1. - ratio) + c.min_gamma * ratio)
            sync(c.device)
            tick = time.perf_counter()
            prior_arg = current_prior.cpu().numpy() if current_prior is not None else None
            sync(c.device)
            totals['transfer_seconds'] += time.perf_counter() - tick
            tick = time.perf_counter()
            costs, tours, *_ = solver.sample(require_prob=False, prior=prior_arg, parallel_traced=True)
            if sample_observer is not None:
                sample_observer(iteration, costs, tours, prior_arg)
            best_idx = int(costs.argmin().item())
            iteration_best = float(costs[best_idx].item())
            assert np.isfinite(iteration_best)
            best_seen = min(best_seen, iteration_best)
            solver._update_pheromone_from_flat(tours[best_idx], iteration_best)
            updates += 1
            sync(c.device)
            totals['sample_update_seconds'] += time.perf_counter() - tick
            if scheduler is not None:
                decision_start = time.perf_counter()
                scheduler.observe(iteration + 1, solver.best_cost)
                totals['decision_seconds'] += time.perf_counter() - decision_start
            cost_trace.append(best_seen)
            if logging:
                iterations.append(dict(iteration=iteration + 1, best_so_far=solver.best_cost,
                                       sampled_best_so_far=best_seen, iteration_best=iteration_best,
                                       elapsed_seconds=time.perf_counter() - start, gnn_calls=calls))
    sync(c.device)
    wall_seconds = time.perf_counter() - start
    tour = solver.best_route.copy()
    recomputed = validate_tour(coords.numpy(), tour, solver.best_cost)
    assert np.array_equal(nn_before, solver.nn_list) and np.array_equal(backup_before, solver.backup_list)
    assert len(cost_trace) == c.total_iterations
    return dict(config=asdict(c), final_cost=solver.best_cost, sampled_best=best_seen,
                initial_cost=initial_cost, recomputed_length=recomputed, tour=tour.tolist(),
                trace=cost_trace, iterations=iterations, refresh_events=events,
                gnn_calls=calls, iterations_executed=len(cost_trace), pheromone_updates=updates,
                candidate_hash=candidate_hash, instance_seed=instance_seed,
                setup_seconds=setup_seconds, wall_seconds=wall_seconds,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(c.device) if str(c.device).startswith('cuda') else 0,
                **totals)
