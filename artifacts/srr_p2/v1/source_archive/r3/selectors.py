"""Frozen, non-oracle directed-slot policies."""
import numpy as np

def policies(coords,tour,nn64,allocation_seed):
    n=len(tour)
    if n%4: raise ValueError('N must be divisible by four for exact 8N budget')
    x=np.asarray(coords,dtype=np.float64)
    d=np.linalg.norm(x[:,None,:]-x[nn64[:,:32]],axis=2).mean(axis=1)
    succ=np.empty(n,dtype=int);pred=np.empty(n,dtype=int)
    succ[tour]=np.roll(tour,-1);pred[tour]=np.roll(tour,1)
    excess=np.maximum(np.linalg.norm(x-x[succ],axis=1),np.linalg.norm(x-x[pred],axis=1))
    score=excess/np.maximum(d,1e-12)
    selected=np.lexsort((np.arange(n),-score))[:n//4]
    rows=[np.full(n,k,dtype=np.int32) for k in (32,40,48,64)]
    names=['k32','k40','k48','k64','geometry']
    geo=np.full(n,32,dtype=np.int32);geo[selected]=64;rows.append(geo)
    for r in range(10):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([allocation_seed,r])))
        row=np.full(n,32,dtype=np.int32);row[rng.choice(n,n//4,replace=False)]=64
        rows.append(row);names.append(f'random_{r:02}')
    return names,np.ascontiguousarray(rows),score

def edge_counts(nn,widths):
    n=len(nn);rows=[]
    base={(min(u,int(v)),max(u,int(v))) for u in range(n) for v in nn[u,:32]}
    for w in widths:
        edges={(min(u,int(v)),max(u,int(v))) for u in range(n) for v in nn[u,:w[u]]}
        extra={(min(u,int(v)),max(u,int(v))) for u in range(n) for v in nn[u,32:w[u]]}
        rows.append(dict(directed_slots=int(w.sum()),extra_directed_slots=int(w.sum()-32*n),
                         unique_undirected_edges=len(edges),unique_undirected_extra_edges=len(extra),
                         new_unique_undirected_edges_beyond_base=len(edges-base)))
    return rows
