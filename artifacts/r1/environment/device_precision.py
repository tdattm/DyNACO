"""Diagnose CPU/GPU float32 discrepancy independently against float64."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import faco
import utils
from net import Net

root = Path(__file__).resolve().parents[3]
torch.set_num_threads(2)
faco.set_faco_cpp_threads(2)
ckpt = torch.load(root/'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt', map_location='cpu', weights_only=False)
model = Net(feats=2, edge_feats=6, logit_net=True).eval()
model.load_state_dict(ckpt['model_state_dict'], strict=True)
coords = np.load(root/'artifacts/r1/datasets/tsp100_seed20261031_v1.npy')[0]
graphs, outputs = [], []
with torch.no_grad():
    for device in ('cpu', 'cuda:0'):
        solver = faco.MFACO_TSP(coords, n_ants=100, device=device)
        graph = utils.build_pyg_data_tsp(solver, coords, device)
        outputs.append(model.to(device)(graph).cpu().numpy())
        graphs.append(graph.cpu())
    reference = copy.deepcopy(model).cpu().double()
    graph64 = graphs[0].clone()
    graph64.x = graph64.x.double()
    graph64.edge_attr = graph64.edge_attr.double()
    expected = reference(graph64).numpy()
scale = max(1., float(np.max(np.abs(expected))))
result = dict(edge_index_identical=torch.equal(graphs[0].edge_index, graphs[1].edge_index),
              coordinates_identical=torch.equal(graphs[0].x, graphs[1].x),
              max_feature_difference=float((graphs[0].edge_attr-graphs[1].edge_attr).abs().max()),
              reference_max_absolute=scale,
              max_cpu_gpu_absolute=float(np.max(np.abs(outputs[0]-outputs[1]))),
              cpu_vs_float64_max_absolute=float(np.max(np.abs(outputs[0]-expected))),
              gpu_vs_float64_max_absolute=float(np.max(np.abs(outputs[1]-expected))),
              normalized_cpu_gpu_max=float(np.max(np.abs(outputs[0]-outputs[1]))/scale),
              tolerance_policy='max absolute error / max(1, float64 output infinity norm) <= 16*float32 epsilon; elementwise relative error unstable near zero logits',
              normalized_tolerance=float(16*np.finfo(np.float32).eps))
(root/'artifacts/r1/environment/device_precision.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
