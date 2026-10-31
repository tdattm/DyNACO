import numpy as np
import pytest
from r1.datasets import ensure_dataset


def test_dataset_identity_and_no_overwrite(tmp_path):
    a, first = ensure_dataset(tmp_path, 200, 2, 17)
    b, second = ensure_dataset(tmp_path, 200, 2, 17)
    assert np.array_equal(a, b) and first == second
    assert len(set(first['instance_ids'])) == 2
    assert a.dtype == np.float32 and np.all((a >= 0) & (a < 1))
    other, _ = ensure_dataset(tmp_path, 200, 2, 18)
    assert not np.array_equal(a, other)
    with pytest.raises(ValueError):
        ensure_dataset(tmp_path, 200, 3, 17)
