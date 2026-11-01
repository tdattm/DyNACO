"""Paired instance aggregation, no data loading or experiment execution."""
import statistics
import numpy as np

def aggregate(records,expected_instances,seeds=(31415,27182,16180)):
    keyed={(r['mode'],r['instance'],r['seed'],r['arm']):r for r in records}
    assert len(keyed)==len(records),'duplicate run'
    expected={(mode,i,s,arm) for mode in ('iterations','wall') for i in expected_instances for s in seeds for arm in (0,1)}
    assert set(keyed)==expected,'missing/extra runs; no exclusions'
    paired={}
    for mode in ('iterations','wall'):
        values=[]
        for i in expected_instances:
            deltas=[]
            for s in seeds:
                a=keyed[(mode,i,s,0)]['length_f64'];b=keyed[(mode,i,s,1)]['length_f64']
                assert np.isfinite(a) and np.isfinite(b) and a>0 and b>0
                deltas.append((a-b)/a)
            values.append(statistics.mean(deltas))
        paired[mode]=np.asarray(values)
    # One common instance resampling map retains paired modes and arms.
    rng=np.random.Generator(np.random.PCG64(202612070202))
    indices=rng.integers(0,len(expected_instances),size=(10000,len(expected_instances)))
    return {mode:dict(instance_values=v.tolist(),mean=float(v.mean()),median=float(np.median(v)),
        minimum=float(v.min()),maximum=float(v.max()),sample_sd=float(v.std(ddof=1)),positive_instances=int((v>0).sum()),
        bootstrap95=np.quantile(v[indices].mean(axis=1),[.025,.975]).tolist()) for mode,v in paired.items()}

def unit_check():
    # Numerical table fixture only: no TSP coordinates or principal records.
    rows=[dict(mode=m,instance=i,seed=s,arm=a,length_f64=100.-a*(1 if i=='a' else 2))
          for m in ('iterations','wall') for i in ('a','b') for s in (1,2) for a in (0,1)]
    out=aggregate(rows,('a','b'),(1,2))
    assert out['wall']['instance_values']==[.01,.02]
    try:aggregate(rows[:-1],('a','b'),(1,2))
    except AssertionError:pass
    else:raise AssertionError('missing row was not rejected')
    return dict(status='PASS',fixture='numerical table only; no coordinates or principal data')
