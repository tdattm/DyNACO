"""Validate the untouched utils.infer_instance path before implementing R1."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import faco
import utils
from net import Net

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/r1/baseline'
DATA = ROOT / 'artifacts/r1/datasets'
CKPT = ROOT / 'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt'


class CheckedSolver(faco.MFACO_TSP):
    created = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.samples = 0
        self.updates = 0
        self.seeds = []
        self.max_error = 0.
        self.__class__.created.append(self)

    def seed_rng(self, seed):
        self.seeds.append(int(seed))
        return super().seed_rng(seed)

    def sample(self, **kwargs):
        result = super().sample(**kwargs)
        costs, tours = result[:2]
        costs, tours = np.asarray(costs), np.asarray(tours)
        assert np.isfinite(costs).all()
        assert tours.shape == (self.n_ants, self.n + 1)
        assert np.array_equal(tours[:, -1], tours[:, 0])
        tours = tours[:, :-1]
        assert np.array_equal(np.sort(tours, axis=1), np.broadcast_to(np.arange(self.n), tours.shape))
        xy = self._coords_np[tours].astype(np.float64)
        lengths = np.linalg.norm(xy - np.roll(xy, -1, axis=1), axis=2).sum(axis=1)
        error = float(np.max(np.abs(lengths - costs)))
        self.max_error = max(self.max_error, error)
        # Float32 native distance accumulation vs float64 independent recomputation.
        assert error <= 2e-5 * max(1., float(lengths.max())), (error, lengths.max())
        prior = kwargs['prior']
        assert prior is not None and prior.shape == (self.n, self.k) and np.isfinite(prior).all()
        self.samples += 1
        return result

    def update_pheromone(self, tour, cost):
        super().update_pheromone(tour, cost)
        self.updates += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    opts = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    faco.set_faco_cpp_threads(2)
    result = {'schema_version': 1, 'gate_b': 'BLOCKED', 'source': 'unmodified base utils.infer_instance, faco.py, net.py and native sources', 'device': opts.device, 'threads': 2, 'execution_seed': 1234, 'dataset_seed': 20261031, 'runs': []}
    try:
        ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
        state = ckpt['model_state_dict']
        model = Net(feats=state['emb_net.v_lin0.weight'].shape[1], edge_feats=state['emb_net.e_lin0.weight'].shape[1], logit_net=not ckpt.get('config', {}).get('no_logit_net', False)).to(opts.device).eval()
        result['checkpoint_load'] = str(model.load_state_dict(state, strict=True))
        result['checkpoint_metadata'] = {k: v for k, v in ckpt.items() if k not in ('model_state_dict', 'optimizer_state_dict')}
        config = dict(H=2, mini_H=50, disable_heuristic=False, no_local_search=False, rho=.1, device=opts.device, no_smooth_mmas=False, min_new_edges=12, no_extend_ls=False, no_normalized_heuristic=False, L=0, iter_log=True, iter_print=False, no_anneal=False, gamma=1., min_gamma=0., timed=True)
        result['effective_configuration'] = config
        datasets = []
        for n in (100, 200, 500, 1000):
            # Independent size-specific PCG64 streams; shared persisted data for every run.
            path = DATA / f'tsp{n}_seed20261031_v1.npy'
            if path.exists():
                coordinates = np.load(path, allow_pickle=False)
            else:
                coordinates = np.random.default_rng(np.random.SeedSequence([20261031, n])).random((2, n, 2), dtype=np.float32)
                np.save(path, coordinates, allow_pickle=False)
                torch.save(torch.from_numpy(coordinates), path.with_suffix('.pt'))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            datasets.append({'n': n, 'instances': 2, 'path': str(path.relative_to(ROOT)), 'sha256': digest, 'dtype': 'float32', 'distribution': 'uniform [0,1)^2', 'rng': 'numpy PCG64 SeedSequence([20261031,n])', 'version': 1, 'instance_ids': [f'tsp{n}-20261031-v1-{i:03}' for i in range(2)], 'reference_optimum': None})
            for i, coords in enumerate(coordinates):
                traces = []
                for repeat in range(2 if n == 100 else 1):
                    utils.set_seed(1234)
                    CheckedSolver.created = []
                    calls = []
                    handle = model.register_forward_hook(lambda mod, args, out: calls.append({'shape': list(out.shape), 'finite': bool(torch.isfinite(out).all())}))
                    start = time.perf_counter()
                    try:
                        _, best, _, extra = utils.infer_instance('tsp', CheckedSolver, utils.build_pyg_data_tsp, model, torch.from_numpy(coords), 32, 100, True, SimpleNamespace(**config), seed=1234 + i)
                    finally:
                        handle.remove()
                    torch.cuda.synchronize() if opts.device.startswith('cuda') else None
                    solver = CheckedSolver.created[-1]
                    trace = extra['iter_stats']
                    assert len(CheckedSolver.created) == 2
                    assert solver.samples == solver.updates == len(trace) == 100
                    assert len(calls) == 2 and all(c['finite'] for c in calls)
                    assert all(trace[j]['best'] <= trace[j-1]['best'] for j in range(1, len(trace)))
                    record = {'n': n, 'instance': i, 'repeat': repeat, 'dataset_sha256': digest, 'best': best, 'backend_global_best': solver.best_cost, 'seconds_including_validation': time.perf_counter() - start, 'iterations': solver.samples, 'updates': solver.updates, 'forward_calls': len(calls), 'max_length_error': solver.max_error, 'seeds_in_constructed_solvers': [s.seeds for s in CheckedSolver.created], 'trace': trace}
                    result['runs'].append(record)
                    traces.append([row['best'] for row in trace])
                    print(f'n={n} instance={i} repeat={repeat} best={best:.9f} seconds={record["seconds_including_validation"]:.3f}', flush=True)
                if len(traces) == 2:
                    delta = float(np.max(np.abs(np.asarray(traces[0]) - traces[1])))
                    result.setdefault('reproducibility', []).append({'instance': i, 'exact_trace': traces[0] == traces[1], 'max_absolute_cost_difference': delta})
                    assert delta <= 1e-6, f'Baseline reproducibility failed: max trace difference {delta}'
        result['gate_b'] = 'PASS'
        (DATA / 'dataset_manifest.json').write_text(json.dumps({'schema_version': 1, 'datasets': datasets}, indent=2) + '\n')
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        (OUT / 'baseline_validation.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    if result['gate_b'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
