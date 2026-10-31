import importlib.metadata
import os
import platform
import subprocess
import sys
import torch
import faco_opt
from r3.common import ROOT,BASE,CHECKPOINT,sha256,save_json,strict_cuda,model,git

def main():
    strict_cuda();net,ckpt,loaded=model()
    sources={p:sha256(ROOT/p) for p in ('faco.py','utils.py','net.py','test.py','experiment.py','src/mfaco_train.cpp','src/mfaco_train.h','src/binding.cpp','src/kd_tree.h','src/setup.py')}
    def probe(cmd):
        p=subprocess.run(cmd,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    env=dict(base_sha=BASE,head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        python=sys.version,executable=sys.executable,platform=platform.platform(),source_sha256=sources,
        native_path=faco_opt.__file__,native_sha256=sha256(faco_opt.__file__),
        native_ldd=probe(['ldd',faco_opt.__file__]),
        gpu=probe(['/usr/lib/wsl/lib/nvidia-smi']),cpu=probe(['lscpu']),memory=probe(['free','-b']),
        openmp=probe(['/mnt/d/GitRepository/DyNACO/.r1-env/openmp_probe']),
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        checkpoint_path=str(CHECKPOINT),checkpoint_sha256=sha256(CHECKPOINT),
        checkpoint_source=f'tracked checkpoint at original baseline {BASE}',checkpoint_metadata={k:str(v) if k!='config' else v for k,v in ckpt.items() if k not in ('model_state_dict','optimizer_state_dict')},
        checkpoint_load=loaded,torch=dict(version=torch.__version__,cuda_runtime=torch.version.cuda,
            device=torch.cuda.get_device_name(),cuda_available=torch.cuda.is_available(),strict=True),
        environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','LD_LIBRARY_PATH','PYTHONPATH')},
        build_identity='R3 isolated rebuild of unchanged baseline sources; build logs and hashes retained')
    save_json(ROOT/'artifacts/r3/environment/environment_manifest.json',env)
    print(dict(cuda=True,checkpoint=env['checkpoint_sha256'],native=env['native_sha256'],loaded=loaded))
if __name__=='__main__':main()
