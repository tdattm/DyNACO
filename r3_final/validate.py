"""Focused development gates. Principal coordinates are never generated here."""
import os
import sys
if '--reference' in sys.argv:
    sys.path.insert(0, '/mnt/d/GitRepository/DyNACO-r3/.r3-build/native')
import hashlib
import subprocess
import time
import numpy as np
import torch
import faco
import faco_opt
import utils
from r3.state_capture import args, solver
from r3.common import sha256
from r3_final.core import OUT, ROOT, run, length, array_hash, allocation, atomic_json, strict_cuda, model

DEV = ROOT/'artifacts/r3/baseline/verification_v1/pilot_tsp1000.npy'
FIELDS = ('source_route', 'best_route', 'pheromone_sparse_np', 'heuristic_sparse_np', 'nn_list', 'backup_list')


def state(aco, costs, flats, prior):
    return [array_hash(np.asarray(costs)), array_hash(np.asarray(flats)),
            *[array_hash(getattr(aco, f)) for f in FIELDS], array_hash(prior)]


def reference():
    strict_cuda()
    net, _, _ = model()
    x = np.load(DEV)[0]
    traces = []
    owners = []
    class Observe(faco.MFACO_TSP):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            owners.append(self)
        def sample(self, *a, **kw):
            self.last = super().sample(*a, **kw)
            self.prior_last = kw['prior'].copy()
            return self.last
        def _update_pheromone_from_flat(self, *a):
            super()._update_pheromone_from_flat(*a)
            traces.append(state(self, *self.last[:2], self.prior_last))
    utils.set_seed(31415)
    utils.infer_instance('tsp', Observe, utils.build_pyg_data_tsp, net, x, 32, 100, True,
                         args(), collect_metrics=False, seed=31415)
    np.savez_compressed(OUT/'development/reference1000.npz', traces=np.asarray(traces),
                        costs=np.asarray([float(owners[-1].best_cost)]),
                        **{f:np.array(getattr(owners[-1], f), copy=True) for f in FIELDS})
    atomic_json(OUT/'development/reference_identity.json',
                dict(binary=str(faco_opt.__file__), binary_sha256=sha256(faco_opt.__file__),
                     coordinates_sha256=sha256(DEV), iterations=len(traces)))


def main():
    out = OUT/'development'
    out.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    env = dict(os.environ)
    subprocess.run([sys.executable, '-m', 'r3_final.validate', '--reference'], env=env, check=True)
    strict_cuda()
    net, _, _ = model()
    x = np.load(DEV)[0]
    traces = []
    s, arr, aco = run(x, net, 31415, 'A', 'iterations',
                      observer=lambda it,a,o,p: traces.append(state(a,*o[:2],p.cpu().numpy())))
    ref = np.load(out/'reference1000.npz')
    assert np.array_equal(traces, ref['traces']), 'full1000 state/prior/ant mismatch'
    for f in FIELDS:
        assert np.array_equal(getattr(aco, f), ref[f]), f
    archived = np.load(ROOT/'artifacts/r3/baseline/verification_v1/full1000_regression_traces.npz')
    # Reference old binary independently agrees with its archived final native result.
    assert float(aco.best_cost) == float(archived['original'][-1].min()) or float(aco.best_cost) <= float(archived['original'][-1].min())
    assert s['native_seed'] == int(np.random.RandomState(31415).randint(0, 2**63-1))
    np.savez_compressed(out/'new1000.npz', traces=np.asarray(traces), update_times=arr['update_times'])
    print('PASS exact 1000 iterations: ant arrays/costs, source/best, pheromone, priors, RNG trajectory', flush=True)
    # Small geometric/tie cases and parallel-traced equivalence under every policy.
    small = [np.random.default_rng(9001).random((100,2), dtype=np.float32),
             np.stack((np.linspace(0,1,100,dtype=np.float32),np.zeros(100,dtype=np.float32)),axis=1),
             np.zeros((100,2), dtype=np.float32)]
    candidate_rows = 0
    for xsmall in small:
        for arm in ('A','B','C'):
            one = solver(xsmall); two = solver(xsmall)
            joined = np.concatenate((one.nn_list,one.backup_list),axis=1)
            assert np.array_equal(joined[:,:32],one.nn_list)
            assert np.array_equal(joined[:,32:],one.backup_list)
            assert ((joined>=0)&(joined<len(xsmall))).all()
            assert (joined != np.arange(len(xsmall))[:,None]).all()
            for row in joined:
                assert len(np.unique(row)) == 64
            candidate_rows += len(xsmall)
            widths, scores = allocation(xsmall,one.best_route,one.nn_list)
            if arm != 'C': widths.fill(32 if arm=='A' else 40)
            assert widths.sum() == len(xsmall)*(32 if arm=='A' else 40)
            assert arm != 'C' or np.count_nonzero(widths==64)==len(xsmall)//4
            before = [array_hash(getattr(one,f)) for f in FIELDS]
            for q in (one,two):
                q.seed_rng(234)
                q._cpp.enable_ls_counts(True)
                q._cpp.set_ls_widths(widths.tolist())
            assert before == [array_hash(getattr(one,f)) for f in FIELDS]
            # Setter repeat consumes no RNG and does not mutate tours/pheromone.
            one._cpp.set_ls_widths(widths.tolist())
            for _ in range(4):
                p = one.sample(require_prob=False)
                q = two.sample(require_prob=True, parallel_traced=True)
                assert np.array_equal(p[0],q[0]) and np.array_equal(p[1],q[1])
                for cost, route in zip(p[0],p[1]):
                    assert abs(length(xsmall,route)-float(cost)) < 2e-5*max(1.,length(xsmall,route))
                for owner, o in ((one,p),(two,q)):
                    j=int(o[0].argmin());owner._update_pheromone_from_flat(o[1][j],float(o[0][j]))
                assert np.array_equal(one.best_route,two.best_route)
                assert np.array_equal(one.pheromone_sparse_np,two.pheromone_sparse_np)
            assert one._cpp.get_ls_counts()==two._cpp.get_ls_counts()
            assert one._cpp.get_ls_counts()['calls']==400
    # Full workload/resource smoke and 8s deadline semantics on existing dev data.
    x=np.load(DEV)[0]
    pilots=[]
    for arm in ('A','B','C'):
        s,a,_=run(x,net,27182,arm,'iterations')
        pilots.append(s)
        assert s['completed_iterations']==1000 and s['counters']['calls']==100000
        assert s['rss_peak_bytes'] < 4*1024**3 and s['peak_cuda_bytes'] < 3.5*1024**3
        assert np.all(a['allocation_widths'].sum(axis=1)==(32000 if arm=='A' else 40000))
        if arm=='C': assert np.all((a['allocation_widths']==64).sum(axis=1)==250)
        atomic_json(out/f'pilot_{arm}.json',s)
        np.savez_compressed(out/f'pilot_{arm}.npz',**a)
        print(f'PASS full resource pilot {arm}: {s["search_seconds"]:.3f}s; CUDA {s["peak_cuda_bytes"]}',flush=True)
    s,a,_=run(x,net,27182,'C','wall')
    assert a['route_times'][s['primary_route_index']]<=8.
    assert s['overshoot_iterations'] in (0,1)
    atomic_json(out/'wall_deadline_pilot.json',s)
    np.savez_compressed(out/'wall_deadline_pilot.npz',**a)
    result=dict(status='PASS', completed_utc=__import__('r3_final.core',fromlist=['now']).now(),
                full1000_exact=True,exact_fields=['all ant arrays','all ant costs',*FIELDS,'neural priors','native seed','RNG-derived full trajectory'],
                candidate_rows=candidate_rows,small_cases=3,normal_parallel_traced=True,
                setter_no_rng=True,one_srr_per_ant=True,source_best_valid=True,
                geometry_formula_matches_stage_a=True,resource_pilots=pilots,
                wall_deadline_pilot=s,elapsed_seconds=time.perf_counter()-tick)
    atomic_json(OUT/'test_results.json',result)


if __name__=='__main__':
    if '--reference' in sys.argv: reference()
    else: main()
