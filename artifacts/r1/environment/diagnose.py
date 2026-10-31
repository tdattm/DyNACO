"""Stage A diagnostics; does not modify the model, solver or inference path."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/r1/environment'
CHECKPOINT = ROOT / 'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt'


def command(*args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {'command': list(args), 'error': 'timeout after 30s; Windows Git audit is authoritative for working-tree status'}
    return {'command': list(args), 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def main():
    result = {
        'schema_version': 1, 'base_commit': 'cd16e2a68fd035f94a8916d41463904f31483d7f',
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'git': {name: command('git', *args) for name, args in {
            'head': ['rev-parse', 'HEAD'], 'branch': ['branch', '--show-current'],
            'status': ['status', '--short'], 'remotes': ['remote', '-v']}.items()},
        'hardware': command('/usr/lib/wsl/lib/nvidia-smi'),
        'memory': command('free', '-b'), 'os_release': Path('/etc/os-release').read_text(),
        'compiler': command(str(ROOT / '.r1-env/sysroot/usr/bin/x86_64-linux-gnu-g++-13'), '--version'),
        'openmp': command(str(ROOT / '.r1-env/openmp_probe')),
        'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'environment_variables': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LD_LIBRARY_PATH']},
        'gate_a': 'BLOCKED',
    }
    try:
        print('Loading NumPy, PyTorch, PyG and native wrapper', flush=True)
        import numpy as np
        import torch
        import torch_geometric
        import faco_opt
        import faco
        from net import Net
        from utils import build_pyg_data_tsp
        print('Imports complete; testing CUDA and checkpoint', flush=True)
        result['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
        result['extension'] = {'path': faco_opt.__file__, 'sha256': hashlib.sha256(Path(faco_opt.__file__).read_bytes()).hexdigest(), 'ldd': command('ldd', faco_opt.__file__), 'provenance': 'isolated rebuild of unchanged upstream source' if '.r1-env/native' in faco_opt.__file__ else 'tracked upstream Linux CPython 3.13 artifact; not rebuilt'}
        result['torch'] = {'version': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
        assert torch.cuda.is_available(), 'GPU unavailable in isolated WSL environment'
        torch.set_num_threads(2)
        faco.set_faco_cpp_threads(2)
        a = torch.arange(16, device='cuda:0', dtype=torch.float32).reshape(4, 4)
        torch.cuda.synchronize()
        assert torch.isfinite(a @ a).all()
        result['torch']['device'] = torch.cuda.get_device_name(0)
        ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        result['checkpoint_metadata'] = {k: v for k, v in ckpt.items() if k not in ('model_state_dict', 'optimizer_state_dict')}
        state = ckpt['model_state_dict']
        model = Net(feats=state['emb_net.v_lin0.weight'].shape[1], edge_feats=state['emb_net.e_lin0.weight'].shape[1], logit_net=not ckpt.get('config', {}).get('no_logit_net', False)).to('cuda:0').eval()
        result['model_load'] = str(model.load_state_dict(state, strict=True))
        coords = np.random.default_rng(20261031).random((100, 2), dtype=np.float32)
        solver = faco.MFACO_TSP(torch.from_numpy(coords), n_ants=100, cand_list_size=32, backup_list_size=32, device='cuda:0')
        solver.seed_rng(1234)
        with torch.no_grad():
            graph = build_pyg_data_tsp(solver, coords, 'cuda:0')
            guidance = model(graph).view(solver.n, solver.k)
            assert guidance.shape == (100, 32) and torch.isfinite(guidance).all()
            costs, tours, *_ = solver.sample(prior=guidance.cpu().numpy())
        values = np.asarray(costs)
        errors = []
        for cost, tour in zip(values, tours):
            tour = np.asarray(tour)
            assert tour.shape == (101,) and tour[-1] == tour[0]
            tour = tour[:-1]  # Native binding returns a closed N+1 tour.
            assert np.array_equal(np.sort(tour), np.arange(100))
            xy = coords[tour].astype(np.float64)
            length = np.linalg.norm(xy - np.roll(xy, -1, axis=0), axis=1).sum()
            errors.append(abs(length - float(cost)))
        assert np.isfinite(values).all() and max(errors) < 1e-4
        solver.update_pheromone(tours[int(values.argmin())], float(values.min()))
        torch.cuda.synchronize()
        result['smoke'] = {'ants': 100, 'n': 100, 'k': 32, 'iterations': 1, 'threads': 2, 'guidance_shape': list(guidance.shape), 'finite': True, 'all_tours_valid': True, 'max_length_error': max(errors), 'best_cost': float(values.min()), 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
        result['gate_a'] = 'PASS'
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        (OUT / 'environment_manifest.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
        print(json.dumps(result, indent=2, default=str))
    if result['gate_a'] == 'BLOCKED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
