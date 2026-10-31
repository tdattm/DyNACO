"""Gate C: actual checkpoint and backend regression, no event scheduler yet."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import faco
import utils
from net import Net
from r1.engine import RunConfig, run, weight_hash

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def model():
    checkpoint = torch.load(ROOT / 'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt', map_location='cpu', weights_only=False)
    m = Net(feats=2, edge_feats=6, logit_net=True).to('cuda:0').eval()
    m.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return m


@pytest.mark.parametrize('anneal', [False, True])
def test_original_regression_and_logging(model, anneal):
    coords = np.load(ROOT / 'artifacts/r1/datasets/tsp100_seed20261031_v1.npy')[0]
    args = SimpleNamespace(H=2, mini_H=50, disable_heuristic=False, no_local_search=False,
                           rho=.1, device='cuda:0', no_smooth_mmas=False, min_new_edges=12,
                           no_extend_ls=False, no_normalized_heuristic=False, L=0, iter_log=True,
                           iter_print=False, no_anneal=not anneal, gamma=1., min_gamma=0., timed=True)
    utils.set_seed(1234)
    faco.set_faco_cpp_threads(2)
    torch.set_num_threads(2)
    before = weight_hash(model)
    _, best, _, extra = utils.infer_instance('tsp', faco.MFACO_TSP, utils.build_pyg_data_tsp,
                                           model, torch.from_numpy(coords), 32, 100, True, args, seed=1234)
    config = RunConfig(total_iterations=100, fixed_s=50, annealing_mode='original_fixed' if anneal else 'no_anneal')
    measured = run(coords, model, config)
    silent = run(coords, model, config, logging=False)
    assert measured['trace'] == [row['best'] for row in extra['iter_stats']]
    assert measured['sampled_best'] == best
    assert measured['trace'] == silent['trace']
    assert measured['tour'] == silent['tour']
    assert measured['gnn_calls'] == silent['gnn_calls'] == len(measured['refresh_events']) == 2
    assert measured['iterations_executed'] == measured['pheromone_updates'] == len(measured['iterations']) == 100
    assert silent['iterations'] == silent['refresh_events'] == []
    assert weight_hash(model) == before
    assert measured['candidate_hash'] == silent['candidate_hash']
    assert all(row['best_so_far'] <= measured['initial_cost'] for row in measured['iterations'])
    assert all(np.isfinite(measured[key]) and measured[key] >= 0 for key in measured if key.endswith('_seconds'))
