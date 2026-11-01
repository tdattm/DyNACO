"""Reconstruct raw relocation trajectory without RNG or solver changes."""
import numpy as np
from r3.diagnostics import canonical_tour

def reconstruct(source,trace):
    route=list(map(int,source));n=len(route);positions={u:i for i,u in enumerate(source)}
    check=[int(trace['start_node'])];steps=[]
    for current,chosen,new_native in zip(trace['curr_nodes'],trace['chosen_nodes'],trace['is_new_edge']):
        current=int(current);chosen=int(chosen)
        diff=abs(positions[current]-positions[chosen]);new=diff not in (1,n-1)
        if new!=bool(new_native):raise AssertionError('source-tour edge semantics mismatch')
        index=route.index(chosen);pred=route[(index-1)%n]
        if new:
            for u in (current,chosen,pred):
                if u not in check:check.append(u)
        before=route.copy()
        # Match native array order, including the target-after-node case.
        target_pos=route.index(current);node_pos=route.index(chosen)
        if route[(target_pos+1)%n]!=chosen:
            route.pop(node_pos)
            if target_pos<node_pos:route.insert(target_pos+1,chosen)
            else:route.insert(target_pos,chosen)
        steps.append(dict(current=current,chosen=chosen,is_new_source_edge=new,chosen_predecessor=pred,
                          route_changed=before!=route))
    return canonical_tour(route,n),check,steps
