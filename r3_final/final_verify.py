"""Verify the final compiled binary, including candidate introspection, before seal."""
import json
import numpy as np
import faco_opt
from r3_final.core import *
from r3_final.validate import state,FIELDS,DEV


def main():
    strict_cuda();net,_,_=model();x=np.load(DEV)[0]
    observed=[]
    result,arrays,owner=run(x,net,31415,'A','iterations',observer=lambda it,a,o,p:observed.append(state(a,*o[:2],p.cpu().numpy())))
    reference=np.load(OUT/'development/reference1000.npz')
    assert np.array_equal(observed,reference['traces'])
    for f in FIELDS:assert np.array_equal(getattr(owner,f),reference[f])
    for arm in ARMS:
        q=solver(x);base=q.nn_list.copy();backup=q.backup_list.copy()
        widths,score=allocation(x,q.best_route,q.nn_list)
        if arm!='C':widths.fill(32 if arm=='A' else 40)
        q._cpp.set_ls_widths(widths.tolist())
        st=q._cpp.get_ls_state()
        assert np.array_equal(st['widths'],widths)
        if arm!='A':
            actual=np.array(st['candidates']).reshape(1000,64)
            assert np.array_equal(actual[:,:32],base)
            assert np.array_equal(actual[:,32:],backup)
            for u in range(1000):
                assert np.array_equal(actual[u,:40],np.r_[base[u],backup[u,:8]])
        assert np.array_equal(q.nn_list,base) and np.array_equal(q.backup_list,backup)
        assert q.pheromone_sparse_np.shape==(1000,32)
        assert q.h_sparse_torch.shape==(1000,32) and q.nn_torch.shape==(1000,32)
        # Invalid setter fails before mutation.
        for bad in ([32]*999,[33]*1000):
            try:q._cpp.set_ls_widths(bad)
            except ValueError:pass
            else:raise AssertionError('invalid width accepted')
        assert np.array_equal(q._cpp.get_ls_state()['widths'],widths)
    # Exact score ties are resolved by node index.
    xx=np.zeros((1000,2),np.float32);nn=np.tile(np.arange(32),(1000,1))
    w,score=allocation(xx,np.arange(1000),nn)
    assert np.array_equal(np.flatnonzero(w==64),np.arange(250))
    from r3_final.analyze import decision
    idx=np.random.Generator(np.random.PCG64(202612070302)).integers(0,16,(10000,16))
    assert decision(np.full(16,.002),idx)['passes']
    assert not decision(np.zeros(16),idx)['passes']
    assert not decision(np.full(16,.0005),idx)['passes']
    assert len(list(identities()))==192 and len(set(i['run_id'] for i in identities()))==192
    atomic_json(OUT/'development/final_binary_verification.json',dict(status='PASS',native_binary_sha256=sha256(faco_opt.__file__),
                full1000_exact=True,native_candidate_prefixes_exact=True,construction_dimensions_unchanged=True,
                invalid_setter_atomic=True,exact_tie_break=True,decision_boundary_checks=True,utc=now()))
    print('PASS final binary: exact1000, native buffer prefixes, isolation, tie-break and decision checks',flush=True)


if __name__=='__main__':main()
