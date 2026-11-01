"""Freeze identities and existing artifacts only. Never generates coordinates."""
import sys,platform,subprocess,shutil,importlib.metadata
from pathlib import Path
from srr_p2.core import ROOT,sha,save,schedule,read

def main():
    import torch,numpy as np
    from r3.common import git
    out=ROOT/'artifacts/srr_p2/v1'
    if read(out/'final_validation/tests.json')['status']!='PASS':raise ValueError('tests prerequisite')
    binary=out/'native';binary.mkdir(exist_ok=False)
    for src in (Path('/mnt/d/GitRepository/DyNACO-srr-p1/.srr-build/native/faco_opt.cpython-313-x86_64-linux-gnu.so'),Path('/mnt/d/GitRepository/DyNACO-r3/.r3-build/native/r3_diagnostic.cpython-313-x86_64-linux-gnu.so')):
        shutil.copyfile(src,binary/src.name)
    prod=str((binary/'faco_opt.cpython-313-x86_64-linux-gnu.so').relative_to(ROOT))
    if sha(ROOT/prod)!='f6b52878fd2e0312c7857ea7c938ca9a40be709b1a5f632b951d9bf150be5bc3':raise ValueError('P1 production binary mismatch')
    save(out/'dispatch.json',schedule())
    # Include all tracked Python/native source, not any data under sealed namespaces.
    paths=[p for p in git('ls-files').splitlines() if Path(p).suffix in ('.py','.cpp','.h','.sh')]
    paths += [prod,str((binary/'r3_diagnostic.cpython-313-x86_64-linux-gnu.so').relative_to(ROOT)),
        'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt',
        'artifacts/srr_p2/v1/FINAL_PROTOCOL.md','artifacts/srr_p2/v1/dispatch.json',
        'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy']
    compiler='/mnt/d/GitRepository/DyNACO/.r1-env/sysroot/usr/bin/x86_64-linux-gnu-g++-13'
    env=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),platform=platform.platform(),compiler=subprocess.check_output([compiler,'--version'],text=True))
    save(out/'environment_packages.json',dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())))
    for p in paths:
        if Path(p).suffix in ('.py','.cpp','.h','.sh'):
            dst=out/'source_archive'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dst)
    save(out/'registration.json',dict(schema=1,status='FROZEN_PROTOCOL_AWAITING_EXPLICIT_PRINCIPAL_APPROVAL',source_commit=git('rev-parse','HEAD'),reviewed_p1_commit='ac24f9323ca0ac99037f5790378e5fdd18fc3029',
        environment=env,files={p:sha(ROOT/p) for p in sorted(set(paths))},schedule='artifacts/srr_p2/v1/dispatch.json',binary_dir=str(binary.relative_to(ROOT)),production_binary=prod,
        diagnostic_binary=str(binary/'r3_diagnostic.cpython-313-x86_64-linux-gnu.so'),development_file='artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy',dataset_content_sha256=None,
        dataset_sealing='After explicit approval: externally generate once exactly as protocol, save NPY and hash manifest, before first solver run. No generator in P2 execution commands.'))
    (out/'registration.sha256').open('x').write(sha(out/'registration.json')+'\n')
if __name__=='__main__':main()
