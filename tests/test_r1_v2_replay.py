import numpy as np
import pytest
from test_r1_fixed import model,ROOT
from r1.engine import RunConfig,run,weight_hash
from r1.v2_replay import branch,assert_prefix,observations

@pytest.mark.usefixtures('strict_cuda_replay')
def test_replay_intervention_and_negative_control(model):
    coords=np.load(ROOT/'artifacts/r1/datasets/tsp100_seed20261031_v1.npy')[0]
    before=weight_hash(model)
    hold=branch(coords,model,31415,25,False)
    control=branch(coords,model,31415,25,False)
    refresh=branch(coords,model,31415,25,True)
    assert_prefix(hold,control,25); assert_prefix(refresh,hold,25)
    assert hold['trace']==control['trace'] and hold['tour']==control['tour']
    assert_prefix(hold,control,125)
    assert hold['final_search_state_hashes']==control['final_search_state_hashes']
    assert hold['gnn_calls']==control['gnn_calls']==1 and refresh['gnn_calls']==2
    rows=observations(refresh,hold,25)
    assert [r['horizon_iterations'] for r in rows]==[25,50,100]
    assert refresh['iterations_executed']==hold['iterations_executed']==125
    assert weight_hash(model)==before

def test_observer_does_not_change_fixed_results(model):
    coords=np.load(ROOT/'artifacts/r1/datasets/tsp100_seed20261031_v1.npy')[0]
    a=run(coords,model,RunConfig(total_iterations=107))
    seen=[]
    b=run(coords,model,RunConfig(total_iterations=107),observer=lambda i,s,p,e:seen.append(i))
    assert seen==list(range(107)) and a['trace']==b['trace'] and a['tour']==b['tour']
