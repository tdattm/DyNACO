import numpy as np
import pytest
from test_r1_fixed import model,ROOT
from r1.v2_replay import branch,digest,assert_prefix

@pytest.mark.usefixtures('strict_cuda_replay')
def test_prior_capture_preserves_search_and_call_count(model):
    coords=np.load(ROOT/'artifacts/r1/datasets/tsp100_seed20261031_v1.npy')[0]
    calls=[]; hook=model.register_forward_hook(lambda *args:calls.append(1))
    try:
        a=branch(coords,model,27182,25,True)
        b=branch(coords,model,27182,25,True,capture_priors=True)
    finally: hook.remove()
    assert_prefix(a,b,125)
    assert a['tour']==b['tour'] and a['final_search_state_hashes']==b['final_search_state_hashes']
    assert len(calls)==4 and a['gnn_calls']==b['gnn_calls']==2
    pair=b['intervention_priors']
    assert digest(pair['old'])==b['branch_state']['cached_prior_sha256']
    assert digest(pair['new'])==b['native_guidance_hashes'][25]
    assert pair['old'].dtype==pair['new'].dtype==np.float32
