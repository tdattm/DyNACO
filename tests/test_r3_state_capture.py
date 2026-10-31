import numpy as np
import pytest
from r3.state_capture import solver,snapshot
from r3.common import strict_cuda
from r3.replay import reconstruct
from r3.diagnostics import canonical_tour

def test_traced_parallel_preserves_fast_rng_and_state():
    strict_cuda();x=np.random.default_rng(12).random((100,2),dtype=np.float32)
    a,b=solver(x,n_ants=10),solver(x,n_ants=10);a.seed_rng(91);b.seed_rng(91)
    for i in range(30):
        before=b.source_route.copy();fast=a.sample(require_prob=False,parallel_traced=True)
        traced=b.sample(require_prob=True,parallel_traced=True)
        assert np.array_equal(fast[0],traced[0]);assert np.array_equal(fast[1],traced[1])
        for ant,trace in enumerate(traced[4].to_trace_list()):
            raw,cl,steps=reconstruct(before,trace)
            assert np.array_equal(raw,canonical_tour(traced[6][ant],len(x)))
            assert cl[0]==trace['start_node'] and len(cl)==len(set(cl))
        idx=int(fast[0].argmin())
        for s,out in ((a,fast),(b,traced)):s.update_pheromone(out[1][idx],float(out[0][idx]))
        assert np.array_equal(a.pheromone_sparse_np,b.pheromone_sparse_np)
        assert np.array_equal(a.source_route,b.source_route)

def test_capture_is_rng_passive_and_owns_data():
    x=np.random.default_rng(4).random((100,2),dtype=np.float32)
    a,b=solver(x,n_ants=10),solver(x,n_ants=10);a.seed_rng(82);b.seed_rng(82)
    state,_=snapshot(a,x,0);saved=state['source_route'].copy()
    outa=a.sample();outb=b.sample()
    assert np.array_equal(outa[0],outb[0]);assert np.array_equal(outa[1],outb[1])
    a.update_pheromone(outa[1][0],float(outa[0][0]))
    assert np.array_equal(state['source_route'],saved)

def test_snapshot_load_rebuilds_raw_heuristic_but_stales_cache():
    x=np.random.default_rng(4).random((100,2),dtype=np.float32);a=solver(x,n_ants=2)
    normalized=a.heuristic_sparse_np.copy();cached=a.h_sparse_torch.cpu().numpy().copy()
    assert np.allclose(normalized.sum(axis=1),1.)
    a._cpp.load_snapshot(a.pheromone_sparse_np.copy(),a.source_route.copy(),a.source_cost,
        a.best_route.copy(),a.best_cost,a.tau_min,a.tau_max,a.nn_list.copy(),a.backup_list.copy())
    assert not np.allclose(a.heuristic_sparse_np,normalized)
    assert np.array_equal(a.h_sparse_torch.cpu().numpy(),cached)
    # R3 never uses this incomplete API for principal trajectory branching.
