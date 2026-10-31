"""Sequential gates in one process to avoid repeated cold imports on DrvFS."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
print('Starting rebuilt-backend environment validation', flush=True)
runpy.run_path(str(root / 'artifacts/r1/environment/diagnose.py'), run_name='__main__')
print('Starting original fixed-S baseline validation', flush=True)
runpy.run_path(str(root / 'artifacts/r1/environment/validate_baseline.py'), run_name='__main__')
print('Starting unmodified test.py CLI smoke test', flush=True)
sys.argv = ['test.py', '--problem', 'tsp', '--n_node', '100', '--k_sparse', '32',
            '--checkpoint', 'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt',
            '--dataset', 'artifacts/r1/datasets/tsp100_seed20261031_v1.pt',
            '--val_size', '2', '--n_ants', '100', '--H', '2', '--mini_H', '50',
            '--rho', '0.1', '--min_new_edges', '12', '--device', 'cuda:0',
            '--threads', '2', '--seed', '1234', '--baseline', 'none', '--no_baseline',
            '--run_model_anneal', '--iter_log', '--timed']
print('CLI:', ' '.join(sys.argv), flush=True)
runpy.run_path(str(root / 'test.py'), run_name='__main__')
