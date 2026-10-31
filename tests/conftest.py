"""Strict replay is explicitly scoped; original v1 tests retain their defaults."""
import pytest
import torch
from r1.replay_determinism import settings,strict_cuda

@pytest.fixture
def strict_cuda_replay():
    original=settings()
    strict_cuda()
    yield
    torch.use_deterministic_algorithms(original['deterministic'],warn_only=original['warn_only'])
    torch.backends.cudnn.benchmark=original['cudnn_benchmark']
    torch.backends.cudnn.deterministic=original['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32=original['cudnn_tf32']
    torch.backends.cuda.matmul.allow_tf32=original['matmul_tf32']
    torch.set_float32_matmul_precision(original['matmul_precision'])
