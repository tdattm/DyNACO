"""Version 1 persisted, size-specific PCG64 Euclidean datasets."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ensure_dataset(directory, n, count=2, seed=20261031):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'tsp{n}_seed{seed}_v1.npy'
    expected = np.random.default_rng(np.random.SeedSequence([seed, n])).random((count, n, 2), dtype=np.float32)
    if path.exists():
        coordinates = np.load(path, allow_pickle=False)
        if not np.array_equal(coordinates, expected):
            raise ValueError(f'Existing dataset differs from requested parameters: {path}')
    else:
        np.save(path, expected, allow_pickle=False)
    manifest = dict(version=1, n=n, count=count, seed=seed, path=str(path), sha256=sha256(path),
                    rng='PCG64 SeedSequence([dataset_seed,n])', distribution='Uniform [0,1)^2',
                    dtype='float32', reference_optimum=None,
                    instance_ids=[f'tsp{n}-{seed}-v1-{i:03}' for i in range(count)])
    return expected, manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='artifacts/r1/datasets')
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    entries = [ensure_dataset(args.directory, n, args.count, args.seed)[1] for n in (100, 200, 500, 1000)]
    target = Path(args.directory) / f'dataset_manifest_seed{args.seed}_count{args.count}.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(entries, indent=2) + '\n')
