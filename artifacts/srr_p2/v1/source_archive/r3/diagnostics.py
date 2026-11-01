"""Float64 independent gain verification and authoritative native candidate audit."""
import time
import numpy as np
import r3_diagnostic
from r3.selectors import policies,edge_counts

def canonical_tour(tour,n):
    t=np.asarray(tour,dtype=np.int32)
    if len(t)==n+1:
        if t[0]!=t[-1]:raise ValueError('incorrect closing node')
        t=t[:-1]
    if len(t)!=n or not np.array_equal(np.sort(t),np.arange(n)):raise ValueError('invalid permutation')
    return np.ascontiguousarray(t)

def length(coords,tour):
    t=canonical_tour(tour,len(coords));x=np.asarray(coords,dtype=np.float64)[t]
    return float(np.linalg.norm(x-np.roll(x,-1,axis=0),axis=1).sum(dtype=np.float64))

def apply_move(tour,i,j):
    n=len(tour)
    if not(0<=i<j<n) or j-i<2 or (i==0 and j==n-1):raise ValueError('invalid edge pair')
    t=np.array(tour,copy=True);t[i+1:j+1]=t[i+1:j+1][::-1]
    return canonical_tour(t,n)

def verify_moves(coords,tour,records):
    before=length(coords,tour);out=[]
    for record in records:
        start=time.perf_counter();changed=apply_move(tour,record['i'],record['j'])
        after=length(coords,changed);observed=before-after;expected=record['raw_gain_f64']
        if abs(observed-expected)>1e-11*max(1,before):raise AssertionError('independent gain validation failed')
        out.append(dict(i=record['i'],j=record['j'],predicted_gain=expected,realized_one_step_gain=observed,
                        relative_gain=observed/before,seconds=time.perf_counter()-start,permutation_valid=True))
    return out

def audit_candidates(coords,native_lists):
    start=time.perf_counter();x=np.asarray(coords,dtype=np.float64);n=len(x)
    # Only O(N) reference distance storage per row, never retain an NxN matrix.
    ties=0
    for u in range(n):
        distances=np.linalg.norm(x-x[u],axis=1);distances[u]=np.inf
        reference=np.argsort(distances,kind='stable')[:64]
        actual=native_lists[64][u]
        if not np.array_equal(reference,actual):
            if not np.array_equal(distances[reference],distances[actual]):raise AssertionError(f'Native ranking discrepancy row {u}')
            ties+=1 # native KD traversal, not node index, resolves exact ties.
        for k in (32,40,48):
            if not np.array_equal(native_lists[k][u],actual[:k]):raise AssertionError('native K prefix mismatch')
    return dict(rows_checked=n,tied_rows_with_different_order=ties,seconds=time.perf_counter()-start)

def diagnose(coords,tour,nn64,allocation_seed,checklist=None,cap=32):
    tour=canonical_tour(tour,len(coords));names,widths,score=policies(coords,tour,nn64,allocation_seed)
    cl=np.arange(len(tour),dtype=np.int32) if checklist is None else np.asarray(checklist,dtype=np.int32)
    start=time.perf_counter();out=r3_diagnostic.scan(coords,tour,nn64,widths,cl,cap)
    out['scan_wall_seconds']=time.perf_counter()-start
    tick=time.perf_counter()
    out['policy_names']=names;out['policy_edge_counts']=edge_counts(nn64,widths)
    out['edge_count_seconds']=time.perf_counter()-tick
    out['allocation_seed']=allocation_seed;out['checklist_scope']='all_nodes' if checklist is None else 'reconstructed_initial'
    out['node_scores']=score.tolist();out['per_node_widths']=widths.tolist()
    gain=out['robust_gain_mass'];count=out['robust_improving_moves']
    out['membership_missed_count']=count-out['membership_count'][0]
    out['membership_missed_count_rate']=(count-out['membership_count'][0])/count if count else None
    out['membership_gain_weighted_coverage']=[v/gain if gain else None for v in out['membership_gain_mass']]
    out['scan_gain_weighted_coverage']=[v/gain if gain else None for v in out['scan_gain_mass']]
    out['incremental_scan_gain_mass']=[v-out['scan_gain_mass'][0] for v in out['scan_gain_mass']]
    out['incremental_scan_move_count']=[v-out['scan_count'][0] for v in out['scan_count']]
    tick=time.perf_counter()
    out['gain_verification']=verify_moves(coords,tour,out['top_membership_missed_moves'])
    out['sample_gain_verification']=verify_moves(coords,tour,out['sample_robust_moves'])
    out['gain_verification_seconds']=time.perf_counter()-tick
    out['new_membership_moves_each_k_increment']=[out['membership_count'][i]-out['membership_count'][i-1] for i in (1,2,3)]
    out['new_scan_moves_each_k_increment']=[out['scan_count'][i]-out['scan_count'][i-1] for i in (1,2,3)]
    x=np.asarray(coords,dtype=np.float64)
    for record in out['top_membership_missed_moves']+out['sample_robust_moves']:
        i,j=record['i'],record['j'];a,b,c,d=map(int,(tour[i],tour[(i+1)%len(tour)],tour[j],tour[(j+1)%len(tour)]))
        record['removed_edges']=[[a,b],[c,d]]
        record['directed_representations']=[[a,c],[c,a],[b,d],[d,b]]
        record['candidate_representation_distances_f64']=[float(np.linalg.norm(x[u]-x[v])) for u,v in record['directed_representations']]
    masses=np.sort(np.asarray(out['node_missed_gain_mass']))[::-1];total=masses.sum()
    out['missed_gain_top_quartile_share']=float(masses[:len(tour)//4].sum()/total) if total else None
    return out
