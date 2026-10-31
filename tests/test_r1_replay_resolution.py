import numpy as np
import pytest
import torch
from test_r1_fixed import model,ROOT
from r1.v2_replay import branch,assert_prefix,observations
from r1.replay_diagnose import array,compare
from r1.engine import weight_hash
from r1.engine import run,RunConfig

def test_snapshot_immutable_and_numerical_comparison():
    x=torch.arange(10,dtype=torch.float32,device='cuda')
    saved=array(x); x.add_(1)
    assert np.array_equal(saved,np.arange(10,dtype=np.float32))
    same=compare(saved,saved.copy()); assert same['byte_equal'] and same['max_absolute']==0
    zeros=compare(np.array([0.],np.float32),np.array([-0.],np.float32))
    assert not zeros['byte_equal'] and zeros['unequal_elements']==0 and zeros['signed_zero_changes']==1

@pytest.mark.usefixtures('strict_cuda_replay')
def test_stronger_native_history_and_actual_call_counts(model):
    coords=np.load(ROOT/'artifacts/r1/datasets/tsp1000_seed20261031_v1.npy')[0]
    before=weight_hash(model); calls=[]
    handle=model.register_forward_hook(lambda *args:calls.append(1))
    try:
        a=branch(coords,model,27182,175,False)
        b=branch(coords,model,27182,175,False)
        c=branch(coords,model,27182,175,True)
    finally: handle.remove()
    assert_prefix(a,b,275); assert_prefix(a,c,175)
    assert a['final_search_state_hashes']==b['final_search_state_hashes']
    assert len(calls)==a['gnn_calls']+b['gnn_calls']+c['gnn_calls']==7
    rows=observations(c,a,175)
    assert len(rows)==3 and all(r['additional_gnn_calls']==1 for r in rows)
    assert a['iterations_executed']==c['iterations_executed']==275
    assert weight_hash(model)==before
    plain=run(coords,model,RunConfig(total_iterations=175,seed=27182))
    assert plain['trace']==a['trace'][:175]

@pytest.mark.usefixtures('strict_cuda_replay')
def test_persisted_graph_exact_across_processes(model):
    from torch_geometric.data import Data
    archive=np.load(ROOT/'artifacts/r1/diagnostics/20261031_replay_resolution_01/strict/hold_a_refresh0.npz')
    graph=Data(**{key:torch.from_numpy(archive[key].copy()).cuda() for key in ('x','edge_index','edge_attr')})
    with torch.inference_mode(): output=array(model(graph).view(100,32))
    assert output.tobytes()==archive['prior'].tobytes()
