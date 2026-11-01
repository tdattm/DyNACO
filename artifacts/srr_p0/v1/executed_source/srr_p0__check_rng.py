"""Controlled RNG-consumption check using saved development coordinates only."""
import sys,json
import numpy as np
from r3.common import ROOT
sys.path.insert(0,str(ROOT/'.srr-build/native'))
import faco_opt
faco_opt.set_num_threads(2)
x=np.load(ROOT/'artifacts/r3/experiments/stage_a_v1_01/datasets/tsp1000_seed20261101_float32.npy')[0]
records=[]
for traced,parallel in ((False,True),(True,True),(True,False)):
    a,b=[faco_opt.MFACO_TSP(x,10,32,32,12,.1,1.,.05,True,False,True,True) for _ in range(2)]
    a.seed_rng(91);b.seed_rng(91);b.configure_srr(True,False)
    first=[s.sample(require_prob=traced,parallel_traced=parallel) for s in (a,b)]
    if traced:assert np.array_equal(first[0][6],first[1][6])
    # Synchronize every source/pheromone/candidate state without reseeding.
    b.load_snapshot(a.pheromone_sparse_np.copy(),a.source_route.copy(),a.source_cost,
        a.best_route.copy(),a.best_cost,a.tau_min,a.tau_max,a.nn_list.copy(),a.backup_list.copy())
    b.configure_srr(False,False)
    second=[s.sample(require_prob=traced,parallel_traced=parallel) for s in (a,b)]
    assert np.array_equal(second[0][0],second[1][0]) and np.array_equal(second[0][1],second[1][1])
    records.append(dict(traced=traced,parallel=parallel,next_sample_exact_after_state_sync_without_reseeding=True))
(ROOT/'artifacts/srr_p0/v1/rng_consumption.json').write_text(json.dumps(records,indent=2))
print('PASS: enabled intervention preserves RNG consumption on three controlled paths')
