"""Independent startup timing only; no solver calls or TSP data."""
import time
start=time.perf_counter()
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import sys,json,platform,resource
import numpy as np
import torch
from r3.common import ROOT,strict_cuda,model,sha256
imports=time.perf_counter()-start
begin=time.perf_counter();strict_cuda();torch.cuda.init();torch.cuda.synchronize();cuda=time.perf_counter()-begin
begin=time.perf_counter();net,_,_=model();torch.cuda.synchronize();load=time.perf_counter()-begin
out=dict(import_seconds=imports,cuda_initialization_seconds=cuda,checkpoint_load_after_cuda_seconds=load,
    note='Separate startup-only process after main pilot; not a decomposition measured in the pilot process',
    python=sys.version,platform=platform.platform(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(),cpu_threads=torch.get_num_threads(),
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(ROOT/'artifacts/srr_p1/v1/startup_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
