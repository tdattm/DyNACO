"""Read-only repository/source provenance and reference-document extraction."""
import hashlib
import json
import subprocess
from pathlib import Path

BASE = 'cd16e2a68fd035f94a8916d41463904f31483d7f'
R1 = '41d9a15b5f311d85118592550c7bdeaf71b3fb9e'
ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def command(args, cwd=ROOT):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

def main():
    out = ROOT / 'artifacts/r3/manifests'
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'repository_initial.json'
    if target.exists():
        raise FileExistsError(target)
    commands = [['git', *args] for args in [
        ['status','--porcelain=v1'], ['branch','--show-current'], ['rev-parse','HEAD'],
        ['branch','-avv'], ['remote','-v'], ['worktree','list'], ['show','--no-patch','--format=fuller',BASE]]]
    sources = ['faco.py','utils.py','net.py','test.py','experiment.py','src/mfaco_train.cpp',
               'src/mfaco_train.h','src/binding.cpp','src/kd_tree.h','src/setup.py']
    manifest = dict(base_sha=BASE, r1_inspected_sha=R1, repository=[command(c) for c in commands],
                    original_checkout=[command(c,ROOT.parent/'DyNACO') for c in commands],
                    safety={'original_untracked':['.r1-env/'],
                            'original_ignored':['artifacts/','r1/','tests/','output/','.pytest_cache/','__pycache__/'],
                            'safe_action':'Separate new worktree; no switch, reset, stash, clean, or R1 writes.',
                            'initial_before_worktree':'final at exact BASE, porcelain ?? .r1-env/ only; no tracked changes'},
                    source_hashes={s:sha256(ROOT/s) for s in sources})
    target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    refs = ROOT/'artifacts/r3/references'
    refs.mkdir(parents=True,exist_ok=True)
    r1paths=['r1/README.md','r1/datasets.py','artifacts/r1/environment/run.sh',
             'artifacts/r1/environment/build.sh','artifacts/r1/environment/compiler.sh',
             'artifacts/r1/environment/environment_manifest.json','artifacts/r1/baseline/baseline_validation.json',
             'artifacts/r1/reports/R1_INITIAL_RESULTS.md','artifacts/r1/reports/R1_REPLAY_RESOLUTION.md',
             'artifacts/r1/datasets/dataset_manifest.json']
    for name in r1paths:
        p=subprocess.run(['git','show',f'{R1}:{name}'],cwd=ROOT,capture_output=True,check=True)
        (refs/('r1_'+name.replace('/','__'))).write_bytes(p.stdout)
    request=Path('C:/Users/TIEN DAT/.codex/attachments/44d9b9ff-9be7-4ca5-b4d9-a8297d644822/Pasted text.txt')
    (refs/'user_request.txt').write_bytes(request.read_bytes())
    from pypdf import PdfReader
    docs=[]
    for name in ['De_xuat_nghien_cuu_mo_rong_DyNACO_R1_R3.pdf','3770855.3817893.pdf']:
        source=Path('C:/Users/TIEN DAT/Downloads')/name
        reader=PdfReader(source)
        (refs/(source.stem+'.txt')).write_text('\n\n'.join(f'PAGE {i+1}\n{p.extract_text()}' for i,p in enumerate(reader.pages)),encoding='utf-8')
        docs.append(dict(path=str(source),sha256=sha256(source),pages=len(reader.pages)))
    (refs/'documents.json').write_text(json.dumps(docs,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'repository_manifest':str(target),'documents':docs},indent=2))

if __name__=='__main__': main()
