"""Source invariance and saved-ant0 native SRR gates only; no ACO trajectory."""
import os,sys,ast,json,hashlib,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from r3.common import git
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
    with (OUT/name).open('x') as f:json.dump(v,f,indent=2)
class StripGuards(ast.NodeTransformer):
    def visit_ImportFrom(self,n):return None if n.module=='m2_safety.guard' else n
    def visit_Assign(self,n):return None if any(isinstance(t,ast.Name) and t.id=='safety_fd' for t in n.targets) else self.generic_visit(n)
    def visit_Expr(self,n):return None if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='require' else self.generic_visit(n)
    def visit_keyword(self,n):
        if n.arg=='pass_fds' and isinstance(n.value,ast.Tuple):n.value.elts=[e for e in n.value.elts if not (isinstance(e,ast.Name) and e.id=='safety_fd')]
        return self.generic_visit(n)
def main():
    changes={}
    for p in ('srr_p2/execute.py','srr_p2/worker.py'):
        old=git('show','136a1ed:'+p);new=(ROOT/p).read_text()
        assert ast.dump(ast.parse(old))==ast.dump(StripGuards().visit(ast.parse(new))),p
        changes[p]=dict(semantic_delta='Guard import/checks and sealed-fd inheritance only',new_sha256=sha(ROOT/p))
    reg=json.loads((ROOT/'artifacts/srr_p2/v1/registration.json').read_text())
    drift=[p for p,h in reg['files'].items() if sha(ROOT/p)!=h]
    assert sorted(drift)==sorted(changes)
    (OUT/'source_delta.patch').write_text(git('diff','136a1ed','--','srr_p2/execute.py','srr_p2/worker.py')+'\n')
    for p in changes:changes[p]['old_sha256']=reg['files'][p]
    save('SOURCE_INVARIANCE.json',dict(status='PASS',changed_pinned_files=changes,other_registered_hashes_unchanged=len(reg['files'])-2))
    binary=Path('/mnt/d/GitRepository/DyNACO-srr-p1/.srr-build/test/faco_opt.cpython-313-x86_64-linux-gnu.so')
    assert sha(binary)=='172c02ccd8e54926f21b87d3e113dc85882ea69eeee304f28f4166ed89411b32'
    sys.path.insert(0,str(binary.parent))
    # Run the existing saved-state gate unchanged up to its first output write.
    # Exclude its later sampling worker/performance portions by construction.
    source=(ROOT/'srr_p1/gates.py').read_text().split("    (OUT/'saved_state_gates.json')")[0]+'    return cases\n'
    namespace={'__name__':'saved_state_gate_without_historical_writes'};exec(compile(source,str(ROOT/'srr_p1/gates.py'),'exec'),namespace)
    cases=namespace['main']();assert len(cases)==32
    save('SAVED_STATE_REGRESSION.json',dict(status='PASS',cases=cases,arms=2,modes=3,exact_route_checklist_and_counters=True,test_binary_sha256=sha(binary),new_ACO_trajectories=0,principal_data_accessed=False))
    save('environment.json',dict(python=sys.version,platform=platform.platform(),unshare=subprocess.check_output(['unshare','--version'],text=True),UID=os.getuid(),pidfd_supported=hasattr(os,'pidfd_open'),method='WSL2 unprivileged user/PID/mount namespaces; PR_SET_PDEATHSIG; sealed memfd',native_production_binary_sha256=reg['files'][reg['production_binary']]))
    print('PASS guard-only AST invariance; 32 saved-state SRR cases, no sampling')
if __name__=='__main__':main()
