"""Independent PCG64 generation, audited against R1's algorithm-independent utility."""
import numpy as np
from r3.common import save_json,sha256

def generate(directory,n=1000,count=8,seed=20261101):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'tsp{n}_seed{seed}_float32.npy'
    if path.exists():raise FileExistsError(path)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,n])))
    data=rng.random((count,n,2),dtype=np.float32)
    with path.open('xb') as f:np.save(f,data,allow_pickle=False)
    save_json(directory/f'tsp{n}_seed{seed}_manifest.json',dict(n=n,count=count,seed=seed,
        bit_generator='PCG64',seed_sequence=[seed,n],numpy_version=np.__version__,
        generation='Generator.random((count,n,2), dtype=float32)',distribution='uniform [0,1)^2',
        dtype=str(data.dtype),sha256=sha256(path),instance_ids=list(range(count)),
        coordinate_sha256=[__import__('hashlib').sha256(x.tobytes(order='C')).hexdigest() for x in data],
        reference_optimum=None))
    return data
