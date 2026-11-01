import numpy as np
import pytest
import torch
import r3_diagnostic
from r3.state_capture import solver
from r3.diagnostics import audit_candidates
from r3.selectors import policies,edge_counts
from r3.common import strict_cuda,model
import utils

@pytest.fixture(scope='module')
def coords():return np.random.default_rng(20261031).random((100,2),dtype=np.float32)

def test_native_rank_and_backup_prefixes(coords):
    strict_cuda();lists={k:solver(coords,n_ants=3,k=k) for k in (32,40,48,64)}
    audit=audit_candidates(coords,{k:a.nn_list.copy() for k,a in lists.items()})
    assert audit['rows_checked']==100
    assert np.array_equal(lists[32].backup_list,lists[64].nn_list[:,32:64])

def test_native_tie_prefix_discrepancy_is_detected():
    x=np.array([[0,0],[0,0],[1,0],[-1,0],[0,1],[0,-1]]*12,dtype=np.float32)
    owners={k:solver(x,n_ants=1,k=k) for k in (32,40,48,64)}
    lists={k:a.nn_list.copy() for k,a in owners.items()}
    # Native KD-tree delete/undelete traversal can reorder exact ties across K.
    # Do not silently substitute an independently tie-broken list for native K32.
    with pytest.raises(AssertionError,match='native K prefix mismatch'):
        audit_candidates(x,lists)

def test_policy_budget_and_no_global_rng_leak(coords):
    owner=solver(coords,n_ants=1,k=64);nn=owner.nn_list.copy();tour=np.arange(100,dtype=np.int32)
    np.random.seed(76);before=np.random.get_state()
    names,w,s=policies(coords,tour,nn,122);after=np.random.get_state()
    assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    for row in w[4:]:assert (row-32).sum()==800
    assert (w[1]-32).sum()==800
    assert np.array_equal(w,policies(coords,tour,nn,122)[1])
    counts=edge_counts(nn,w)
    assert all(c['extra_directed_slots']==800 for c in counts[4:])
    assert counts[1]['extra_directed_slots']==800

def test_invalid_candidates_rejected(coords):
    owner=solver(coords,n_ants=1,k=64);nn=owner.nn_list.copy();tour=np.arange(100,dtype=np.int32);w=np.full((1,100),32,dtype=np.int32)
    for bad in (-1,100,0):
        changed=nn.copy();changed[0,0]=bad
        with pytest.raises(ValueError):r3_diagnostic.scan(coords,tour,changed,w,tour)
    for kind in ('duplicate','ordering','width'):
        changed=nn.copy();widths=w.copy()
        if kind=='duplicate':changed[0,1]=changed[0,0]
        if kind=='ordering':changed[0,[0,10]]=changed[0,[10,0]]
        if kind=='width':widths[0,0]=65
        with pytest.raises(ValueError):r3_diagnostic.scan(coords,tour,changed,widths,tour)

def test_k64_checkpoint_tensor_compatibility(coords):
    net,_,_=model()
    with torch.no_grad():
        for k in (32,40,48,64):
            a=solver(coords,n_ants=1,k=k);p=net(utils.build_pyg_data_tsp(a,coords,'cuda:0'))
            assert p.shape==(100*k,) and torch.isfinite(p).all()

def test_k_boundaries(coords):
    a=solver(coords,n_ants=1,k=64);assert a.k==64
    with pytest.raises(RuntimeError):solver(coords,n_ants=1,k=65)
