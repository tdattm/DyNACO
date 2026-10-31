import itertools
import numpy as np
import pytest
import r3_diagnostic
from r3.diagnostics import apply_move,length,canonical_tour,verify_moves

def scan_small(x,t,width=None,cl=None):
    x=np.array(x,dtype=np.float32);t=np.array(t,dtype=np.int32);n=len(t)
    nn=np.array([sorted((v for v in range(n) if v!=u),key=lambda v:(np.linalg.norm(x[u].astype(float)-x[v]),v)) for u in range(n)],dtype=np.int32)
    w=np.full((1,n),n-1 if width is None else width,dtype=np.int32)
    return r3_diagnostic.scan(x,t,nn,w,np.arange(n,dtype=np.int32) if cl is None else np.array(cl,dtype=np.int32),1000)

@pytest.mark.parametrize('x,t,expected',[
    ([[0,0],[1,0],[1,1],[0,1]],[0,2,1,3],1),
    ([[0,0],[1,0],[1,1],[0,1]],[0,1,2,3],0),
    ([[0,0],[1,0],[2,0],[3,0]],[0,1,2,3],0),
    ([[0,0],[0,0],[1,1],[1,0]],[0,1,2,3],0),
    ([[0,0],[1e-10,0],[1,1],[1,0]],[0,1,2,3],0),
])
def test_constructed_examples(x,t,expected):
    out=scan_small(x,t)
    assert out['robust_improving_moves']==expected
    assert out['valid_moves']==len(t)*(len(t)-3)//2
    verify_moves(np.array(x,dtype=np.float32),np.array(t),out['sample_robust_moves'])

def test_exhaustive_independent_application():
    x=np.random.default_rng(7).random((8,2),dtype=np.float32);t=np.random.default_rng(9).permutation(8)
    out=scan_small(x,t);expected=[];before=length(x,t)
    for i,j in itertools.combinations(range(8),2):
        if j-i<2 or (i==0 and j==7):continue
        gain=before-length(x,apply_move(t,i,j))
        if gain>out['robust_threshold']:expected.append((i,j,gain))
    assert out['robust_improving_moves']==len(expected)
    assert out['robust_gain_mass']==pytest.approx(sum(g for _,_,g in expected),abs=1e-12)
    assert out['valid_moves']==sum(out[k] for k in ('robust_improving_moves','positive_marginal_moves','zero_gain_moves','negative_gain_moves'))

def test_marginal_is_not_robust():
    # Slightly crossed narrow rectangle: positive raw gain below robust threshold.
    out=scan_small([[0,0],[1,0],[1,1e-5],[0,1e-5]],[0,2,1,3])
    assert out['robust_improving_moves']==0
    assert out['positive_marginal_moves']>0

def test_invalid_moves_and_tours():
    for pair in [(0,0),(0,1),(0,3),(-1,2),(2,1)]:
        with pytest.raises(ValueError):apply_move(np.arange(4),*pair)
    with pytest.raises(ValueError):canonical_tour([0,1,1,3],4)
    with pytest.raises(ValueError):canonical_tour([0,1,2,3,1],4)

def test_all_orientations_and_pruning():
    x=np.random.default_rng(98).random((12,2),dtype=np.float32);tour=np.random.default_rng(12).permutation(12)
    nn=np.array([sorted((v for v in range(12) if v!=u),key=lambda v:np.linalg.norm(x[u].astype(float)-x[v])) for u in range(12)],dtype=np.int32)
    widths=np.full((1,12),3,dtype=np.int32)
    out=r3_diagnostic.scan(x,tour,nn,widths,np.arange(12,dtype=np.int32),1000)
    def df(u,v):
        z=x[v]-x[u];return np.sqrt(z[0]*z[0]+z[1]*z[1])
    for rec in out['sample_robust_moves']:
        i,j=rec['i'],rec['j'];a,b,c,d=tour[i],tour[(i+1)%12],tour[j],tour[(j+1)%12]
        reps=[(a,c,b),(c,a,d),(b,d,a),(d,b,c)]
        member=any(v in nn[u,:3] for u,v,_ in reps)
        available=False
        for u,v,adjacent in reps:
            for neighbor in nn[u,:3]:
                if not df(u,adjacent)>df(u,neighbor):break
                if neighbor==v:available=True
        assert rec['candidate_membership'][0]==member
        assert rec['potentially_evaluated_all_nodes'][0]==available
