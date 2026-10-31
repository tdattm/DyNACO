import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='cd16e2a68fd035f94a8916d41463904f31483d7f'
CHECKPOINT=ROOT/'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt'

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def git_command(*args):
    # Worktree was created by Windows Git. Pass its admin directory explicitly in
    # WSL; do not rewrite the shared .git pointer or touch R1 metadata.
    pointer=(ROOT/'.git').read_text().strip().removeprefix('gitdir: ')
    if str(ROOT).startswith('/mnt/') and len(pointer)>2 and pointer[1]==':':
        pointer='/mnt/'+pointer[0].lower()+pointer[2:]
    return ['git','--git-dir='+pointer,'--work-tree='+str(ROOT),*args]
def git(*args):return subprocess.check_output(git_command(*args),cwd=ROOT,text=True).strip()
def strict_cuda():
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision('highest')
    if not torch.cuda.is_available():raise RuntimeError('CUDA baseline required; no CPU substitution')

def model():
    import torch
    from net import Net
    if sha256(CHECKPOINT)!='33f7da301e49f74d5b66de2a12c7bd95f247de8c217343c3410bb716a6f22358':raise AssertionError('checkpoint identity mismatch')
    ckpt=torch.load(CHECKPOINT,map_location='cuda:0',weights_only=False)
    net=Net(feats=2,edge_feats=6,logit_net=True).to('cuda:0')
    result=net.load_state_dict(ckpt['model_state_dict'],strict=True)
    net.eval()
    return net,ckpt,str(result)
