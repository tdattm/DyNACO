from dataclasses import replace
import numpy as np
import torch
import faco
from r1.engine import RunConfig, run, weight_hash
from r1.scheduler import GuidanceScheduler, SchedulerConfig
from test_r1_fixed import model, ROOT


def test_fixed_scheduler_disabled_equivalence_and_counters(model, monkeypatch):
    coords = np.load(ROOT / 'artifacts/r1/datasets/tsp100_seed20261031_v1.npy')[1]
    c = RunConfig(total_iterations=107, fixed_s=25)
    plain = run(coords, model, c)
    updates, forwards = [], []
    update = faco.MFACO_TSP.update_pheromone

    def counted(self, *args):
        updates.append(1)
        return update(self, *args)

    monkeypatch.setattr(faco.MFACO_TSP, 'update_pheromone', counted)
    hook = model.register_forward_hook(lambda *args: forwards.append(1))
    try:
        measured = run(coords, model, c, scheduler=GuidanceScheduler(SchedulerConfig(total_iterations=107, fixed_s=25)))
    finally:
        hook.remove()
    assert measured['trace'] == plain['trace'] and measured['tour'] == plain['tour']
    assert len(updates) == measured['pheromone_updates'] == 107
    assert len(forwards) == measured['gnn_calls'] == len(measured['refresh_events']) == 5


def test_event_native_reproducibility_and_weight_identity(model):
    coords = np.load(ROOT / 'artifacts/r1/datasets/tsp200_seed20261031_v1.npy')[0]
    c = RunConfig(total_iterations=157, fixed_s=50)
    sc = SchedulerConfig(mode='event_stagnation', total_iterations=157, fixed_s=50)
    before = weight_hash(model)
    first = run(coords, model, c, scheduler=GuidanceScheduler(sc))
    repeat = run(coords, model, c, scheduler=GuidanceScheduler(sc), logging=False)
    assert first['trace'] == repeat['trace'] and first['tour'] == repeat['tour']
    assert first['candidate_hash'] == repeat['candidate_hash']
    assert first['gnn_calls'] == repeat['gnn_calls'] == len(first['refresh_events'])
    assert first['iterations_executed'] == first['pheromone_updates'] == len(first['iterations']) == 157
    costs = [first['initial_cost']] + [row['best_so_far'] for row in first['iterations']]
    assert all(b <= a for a, b in zip(costs, costs[1:]))
    assert first['final_cost'] == min(costs)
    assert weight_hash(model) == before


def test_cpu_gpu_device_consistency(model):
    import copy
    import utils
    coords = np.load(ROOT / 'artifacts/r1/datasets/tsp100_seed20261031_v1.npy')[0]
    # Compare features/output numerically; stochastic solver traces need not match across devices.
    outputs, graphs = [], []
    for device in ('cpu', 'cuda:0'):
        solver = faco.MFACO_TSP(coords, n_ants=100, device=device)
        with torch.no_grad():
            graph = utils.build_pyg_data_tsp(solver, coords, device)
            outputs.append(model.to(device)(graph).cpu().numpy())
            graphs.append(graph.cpu())
        result = run(coords, model, RunConfig(total_iterations=10, fixed_s=5, device=device))
        assert np.isfinite(result['final_cost']) and result['gnn_calls'] == 2
    assert torch.equal(graphs[0].edge_index, graphs[1].edge_index)
    assert torch.equal(graphs[0].x, graphs[1].x)
    assert float((graphs[0].edge_attr-graphs[1].edge_attr).abs().max()) <= 4*np.finfo(np.float32).eps
    reference = copy.deepcopy(model).cpu().double()
    graph64 = graphs[0].clone()
    graph64.x, graph64.edge_attr = graph64.x.double(), graph64.edge_attr.double()
    with torch.no_grad():
        expected = reference(graph64).numpy()
    # Normwise float32 accuracy for a 12-layer residual network. Near-zero
    # logits make elementwise relative error unstable. See device_precision.json:
    # both devices differ from float64 by ~5-6e-5 on output scale ~74.
    bound = 16*np.finfo(np.float32).eps * max(1., np.max(np.abs(expected)))
    assert np.max(np.abs(outputs[0]-outputs[1])) <= bound
    assert all(np.max(np.abs(output-expected)) <= bound for output in outputs)
