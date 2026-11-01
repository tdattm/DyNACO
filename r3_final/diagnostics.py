"""Untimed literal float32 SRR replay on development-only ant0 states."""
import numpy as np
from r3.replay import reconstruct
from r3_final.core import *


def replay(x, raw, checklist, nn64, widths):
    route=raw.copy();check=list(checklist);cursor=0;events=[]
    visits=gains=expanded=0
    def d(a,b):
        delta=x[b]-x[a]
        return np.sqrt(delta[0]*delta[0]+delta[1]*delta[1])
    while cursor<len(check):
        a=check[cursor];cursor+=1
        pos=np.argsort(route);an=int(route[(pos[a]+1)%len(x)]);ap=int(route[(pos[a]-1)%len(x)])
        best=np.float32(0);move=None;rank_best=-1
        for direction in (0,1):
            edge=d(a,an) if direction==0 else d(ap,a)
            for rank in range(int(widths[a])):
                b=int(nn64[a,rank]);visits+=1;expanded+=rank>=32
                ab=d(a,b)
                if not edge>ab:break
                gains+=1
                other=int(route[(pos[b]+(1 if direction==0 else -1))%len(x)])
                diff=((edge+d(b,other))-ab)-d(an,other) if direction==0 else ((edge+d(other,b))-ab)-d(ap,other)
                if diff>best:
                    best=diff;move=[an,other,a,b] if direction==0 else [a,b,ap,other];rank_best=rank
        if best>0:
            before=length(x,route);original=route.copy()
            first,last=sorted((int(pos[move[0]]),int(pos[move[1]])))
            if last-first<=len(x)-(last-first):route[first:last]=route[first:last][::-1]
            else:
                left=last%len(x);right=first-1 if first>0 else len(x)-1;i=0;j=len(x)-left+right+1
                while i<j:
                    route[left],route[right]=route[right],route[left]
                    left=(left+1)%len(x);right=(right-1)%len(x);i+=1;j-=1
            gain=before-length(x,route)
            events.append(dict(anchor=int(a),rank=rank_best+1,gain_f32=float(best),gain_f64=gain,
                               robust=bool(gain>max(1e-8,1e-6*before)),route_changed=not np.array_equal(original,route)))
            for u in move:
                if u not in check:check.append(u)
    return route,dict(visits=visits,gain_evaluations=gains,expanded_visits=expanded,events=events,
                      input_length=length(x,raw),output_length=length(x,route))


def main():
    strict_cuda();net,_,_=model()
    x=np.load(ROOT/'artifacts/r3/baseline/verification_v1/pilot_tsp1000.npy')[0]
    records=[]
    for arm in ARMS:
        init=solver(x);source=init.source_route.copy();nn64=np.concatenate((init.nn_list,init.backup_list),axis=1)
        widths=np.full(1000,32 if arm!='B' else 40,dtype=np.int32)
        if arm=='C':widths,_=allocation(x,init.best_route,init.nn_list)
        def observe(it,aco,output,prior):
            nonlocal source,widths
            if it in (1,500,1000):
                trace=output[4].to_trace_list()[0]
                raw,check,_=reconstruct(source,trace)
                assert np.array_equal(raw,np.asarray(output[6][0])[:-1])
                refined,record=replay(x,raw,check,nn64,widths)
                assert np.array_equal(refined,np.asarray(output[1][0])[:-1]),'literal SRR replay mismatch'
                record.update(arm=arm,completed_iteration=it)
                records.append(record)
            source=aco.source_route.copy()
            if arm=='C' and it%100==0:widths,_=allocation(x,aco.best_route,aco.nn_list)
        run(x,net,31415,arm,'iterations',traced=True,observer=observe)
    events=[e for r in records for e in r['events']]
    summary=dict(scope='development-only ant0 at iterations 0,499,999; not principal causal evidence',
                 exact_native_replays=len(records),accepted_native=len(events),
                 robust=sum(e['robust'] for e in events),positive_f64=sum(e['gain_f64']>0 for e in events),
                 zero_f64=sum(e['gain_f64']==0 for e in events),negative_f64=sum(e['gain_f64']<0 for e in events),
                 unchanged_route=sum(not e['route_changed'] for e in events),
                 expanded_accepted=sum(e['rank']>32 for e in events),records=records)
    atomic_json(OUT/'development/untimed_srr_diagnostics.json',summary)
    print('PASS exact native expanded SRR replay and independent per-move float64 diagnostics',flush=True)


if __name__=='__main__':main()
