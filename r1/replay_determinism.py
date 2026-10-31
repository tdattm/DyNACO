"""Opt-in strict CUDA configuration for diagnostics, never v1 defaults."""
import os
import torch

def settings():
    return dict(workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),torch=torch.__version__,cuda=torch.version.cuda)

def strict_cuda():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before starting this diagnostic process')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
