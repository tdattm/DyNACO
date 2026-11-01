import numpy as np
import faco
import utils
from r3.common import strict_cuda,model
from r3.state_capture import run,args

def test_original_inference_exact_regression():
    strict_cuda();net,_,_=model();x=np.random.default_rng(333).random((100,2),dtype=np.float32)
    observed=[];instances=[]
    class Observe(faco.MFACO_TSP):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);instances.append(self)
        def sample(self,*a,**kw):
            o=super().sample(*a,**kw);observed.append(o[0].copy());return o
    utils.set_seed(31415)
    utils.infer_instance('tsp',Observe,utils.build_pyg_data_tsp,net,x,32,100,True,args(2,25),
                         collect_metrics=False,seed=31415)
    actual=run(x,net,31415,H=2,mini_H=25,trace_iterations=(0,24,49))
    assert np.array_equal(actual['costs_history'],observed)
    for field in ('best_route','source_route','pheromone_sparse_np','heuristic_sparse_np','nn_list'):
        assert np.array_equal(getattr(actual['aco'],field),getattr(instances[-1],field))
    assert actual['aco'].best_cost==instances[-1].best_cost

def test_same_seed_exact_repeat():
    strict_cuda();net,_,_=model();x=np.random.default_rng(433).random((100,2),dtype=np.float32)
    a=run(x,net,27182,H=2,mini_H=10);b=run(x,net,27182,H=2,mini_H=10)
    assert a['native_seed']==b['native_seed']
    assert np.array_equal(a['costs_history'],b['costs_history'])
    assert np.array_equal(a['aco'].pheromone_sparse_np,b['aco'].pheromone_sparse_np)
