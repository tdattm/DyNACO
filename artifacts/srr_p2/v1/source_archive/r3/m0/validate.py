"""M0: saved-input replay only. No solver construction, RNG, or new instances."""
import csv
import json
import shutil
import sys
import traceback
from pathlib import Path
import numpy as np
from r3.common import ROOT, sha256, git
from r3.replay import reconstruct
from r3.diagnostics import canonical_tour, length

OUT=ROOT/'artifacts/r3/m0_srr_replay'
SOURCE=ROOT/'artifacts/r3/experiments/stage_a_v1_01'
sys.path.insert(0,str(ROOT/'.r3-build/m0'))
import m0_native

def save(name,obj):
    with (OUT/name).open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def df(x,a,b):
    d=x[b]-x[a]
    return np.sqrt(d[0]*d[0]+d[1]*d[1])

def validate_trace(x,raw,cl,nn,result):
    route=raw.copy(); checklist=list(map(int,cl));processed=set();moves=0
    total=np.float32(0);eval_count=pruned=reinserts=unchosen=0
    for cursor,event in enumerate(result['events']):
        a=checklist[cursor];assert event['cursor']==cursor and event['node']==a
        assert event['checklist_size_before']==len(checklist)
        processed.add(a);pos=np.argsort(route);an=int(route[(pos[a]+1)%len(route)]);ap=int(route[(pos[a]-1)%len(route)])
        best=np.float32(0);move=[-1]*4;expected=[];breaks=[]
        for direction in (0,1):
            edge=df(x,a,an) if direction==0 else df(x,ap,a)
            for rank,b in enumerate(nn[a]):
                b=int(b);ab=df(x,a,b)
                if not edge>ab:
                    breaks.append((direction,rank,b));break
                other=int(route[(pos[b]+(1 if direction==0 else -1))%len(route)])
                gain=((edge+df(x,b,other))-ab)-df(x,an,other) if direction==0 else ((edge+df(x,other,b))-ab)-df(x,ap,other)
                chosen=bool(gain>best)
                expected.append((direction,rank,b,float(gain),chosen))
                if chosen:
                    best=gain;move=[an,other,a,b] if direction==0 else [a,b,ap,other]
        actual=[(e['direction'],e['rank'],e['b'],e['gain_f32'],e['selected_at_this_point']) for e in event['evaluations']]
        assert actual==expected,'evaluation/rounding/selection mismatch'
        assert [(e['direction'],e['rank'],e['b']) for e in event['prunes']]==breaks
        assert event['selected_gain_f32']==float(best) and event['selected_move']==move
        eval_count+=len(actual);pruned+=sum(32-r for _,r,_ in breaks)
        unchosen+=len(actual)-(1 if best>0 else 0)
        inserts=[]
        if best>0:
            moves+=1;before=route.copy();cost_before=length(x,route)
            first,last=sorted((int(pos[move[0]]),int(pos[move[1]])))
            if last-first<=len(route)-(last-first):
                route[first:last]=route[first:last][::-1]
            else:
                # Literal native wrap swap schedule, including first==0 behavior.
                left=last%len(route);right=first-1 if first>0 else len(route)-1
                i=0;j=len(route)-left+right+1
                while i<j:
                    route[left],route[right]=route[right],route[left]
                    left=(left+1)%len(route);right=(right-1)%len(route);i+=1;j-=1
            updates=[(i,int(route[i])) for i in range(len(route)) if before[i]!=route[i]]
            assert updates==[tuple(u) for u in event['route_updates']]
            assert np.array_equal(np.sort(route),np.arange(len(route)))
            # Check gain in float64 against the literal removed/added edges.
            u,v,c,d=move;xx=x.astype(np.float64)
            dist=lambda p,q:float(np.linalg.norm(xx[p]-xx[q]))
            edge_gain=dist(c,u)+dist(d,v)-dist(c,d)-dist(u,v)
            observed=cost_before-length(x,route)
            assert abs(observed-edge_gain)<1e-10,'accepted move edge/cost mismatch'
            assert abs(observed-float(best))<2e-6,'accepted native gain mismatch'
            total=np.float32(total-best)
            for node in move:
                added=node not in checklist
                inserts.append(dict(node=node,already_processed=node in processed,inserted=added))
                if node in processed:reinserts+=1
                if added:checklist.append(node)
        assert inserts==event['insertions']
        assert event['checklist_size_after']==len(checklist)
    assert len(result['events'])==len(checklist)<=len(route)
    assert checklist==result['checklist'] and np.array_equal(route,result['route'])
    assert result['total_change_f32']==float(total)
    return dict(processed_nodes=len(processed),unreached_nodes=len(route)-len(processed),
                candidate_evaluations=eval_count,pruned_neighbor_slots=pruned,
                evaluated_not_final_chosen=unchosen,accepted_moves=moves,
                processed_node_reinsertion_attempts=reinserts,
                candidate_unavailable_directed_anchor_pairs=len(processed)*(len(route)-1-32))

def main():
    paths=[*ROOT.glob('r3/m0/*'),ROOT/'src/mfaco_train.cpp',ROOT/'src/mfaco_train.h',ROOT/'src/kd_tree.h',
           ROOT/'src/setup.py',ROOT/'r3/replay.py',ROOT/'r3/diagnostics.py',ROOT/'r3/native/two_opt_diagnostic.cpp',
           OUT/'replayer.cpp',OUT/'instrumentation_replacements.json',SOURCE/'gate.json',SOURCE/'finalized_sha256.json',
           SOURCE/'datasets/tsp1000_seed20261101_float32.npy',ROOT/'r3/protocols/stage_a_v1.json',
           ROOT/'artifacts/r3/reports/R3_STAGE_A_REPORT.md',
           ROOT/'artifacts/r3/secondary_review/20261101_saved_ant0_v1/summary.json',
           ROOT/'artifacts/r3/secondary_review/20261101_saved_ant0_v1/R3_SECONDARY_EVIDENCE_REVIEW.md']
    for i in range(8):
        paths.append(SOURCE/f'native_candidates_i{i}.npz')
        for seed in (31415,27182):
            folder=SOURCE/f'i{i}_s{seed}';paths.append(folder/'run.json')
            for it in (0,249,499,999):paths.extend([folder/f'raw_before_iter{it}.npz',folder/f'perturbation_iter{it}.json'])
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in paths if p.is_file()}
    ledger=json.loads((SOURCE/'finalized_sha256.json').read_text())
    assert all(sha256(SOURCE/p)==h for p,h in ledger.items())
    assert json.loads((SOURCE/'gate.json').read_text())['decision']=='NO-GO'
    secondary=ROOT/'artifacts/r3/secondary_review/20261101_saved_ant0_v1/summary.json'
    assert json.loads(secondary.read_text())['secondary_decision']=='INCONCLUSIVE'
    save('manifest.json',dict(head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        input_sha256=hashes,stage_a_ledger_verified=len(ledger),binary_sha256=sha256(m0_native.__file__),
        binary_path=m0_native.__file__,scope='64 saved ant0 pairs only; no RNG',compiler_flags=['-O3','-std=c++17']))
    dataset=np.load(SOURCE/'datasets/tsp1000_seed20261101_float32.npy',allow_pickle=False)
    records=[]
    for i,x in enumerate(dataset):
        with np.load(SOURCE/f'native_candidates_i{i}.npz') as c:nn=np.ascontiguousarray(c['k32'],dtype=np.int32)
        for seed in (31415,27182):
            folder=SOURCE/f'i{i}_s{seed}'
            config=json.loads((folder/'run.json').read_text())['config'];assert not config['no_extend_ls'] and config['L']==0
            for it in (0,249,499,999):
                key=f'i{i}_s{seed}_t{it}';record=dict(case=key,instance=i,seed=seed,iteration=it,pass_gate=False)
                try:
                    with np.load(folder/f'raw_before_iter{it}.npz') as z:
                        raw=canonical_tour(z['raw'],1000);ref=canonical_tour(z['refined'],1000)
                        source=canonical_tour(z['source_before'],1000);cl=z['initial_checklist'].astype(np.int32)
                        saved_ref=z['refined'].copy()
                    trace=json.loads((folder/f'perturbation_iter{it}.json').read_text())['trace']
                    reconstructed,recl,_=reconstruct(source,trace)
                    assert np.array_equal(raw,reconstructed) and np.array_equal(cl,recl)
                    record['raw_and_initial_checklist_exact']=True
                    plain=m0_native.run(x,raw.tolist(),cl.tolist(),nn,False)
                    logged=m0_native.run(x,raw.tolist(),cl.tolist(),nn,True)
                    save(key+'_events.json',logged)
                    record.update(exact_array_match=np.array_equal(plain['route'],ref),
                        closed_array_match=np.array_equal(np.r_[plain['route'],plain['route'][0]],saved_ref) if len(saved_ref)==1001 else np.array_equal(plain['route'],saved_ref),
                        logging_invariance=all(plain[k]==logged[k] for k in ('route','checklist','total_change_f32')),
                        mismatch_positions=np.flatnonzero(np.asarray(plain['route'])!=ref).tolist(),
                        final_cost_f64=length(x,plain['route']),saved_refined_cost_f64=length(x,ref))
                    record.update(validate_trace(x,raw,cl,nn,logged))
                    record['all_move_and_checklist_validations']=True
                    record['pass_gate']=bool(record['exact_array_match'] and record['closed_array_match'] and record['logging_invariance'])
                except Exception:
                    record['error']=traceback.format_exc()
                records.append(record);print(key,record['pass_gate'],record.get('error',''),flush=True)
    passed=sum(r['pass_gate'] for r in records)
    summary=dict(decision='M0 VALIDATED' if passed==64 else 'M0 INCONCLUSIVE/FAILED',passed=passed,total=64,
                 stage_a='NO-GO',secondary='INCONCLUSIVE',cases=records)
    save('validation_summary.json',summary)
    # Offline metrics are permitted only after the exact replay gate passes.
    if passed==64:
        import r3_diagnostic
        residual=[]
        for r in records:
            i=r['instance'];folder=SOURCE/f"i{i}_s{r['seed']}"
            with np.load(folder/f"raw_before_iter{r['iteration']}.npz") as z:ref=canonical_tour(z['refined'],1000)
            with np.load(SOURCE/f'native_candidates_i{i}.npz') as c:nn=np.ascontiguousarray(c['k64'])
            scan=r3_diagnostic.scan(dataset[i],ref,nn,np.full((1,1000),32,dtype=np.int32),np.arange(1000,dtype=np.int32),0)
            residual.append(dict(case=r['case'],robust_moves=scan['robust_improving_moves'],
                off_k32=scan['robust_improving_moves']-scan['membership_count'][0],
                k32_residual=scan['membership_count'][0],offline_only=True))
        save('offline_final_residuals.json',residual)
    assert all(sha256(ROOT/p)==h for p,h in hashes.items()),'input mutated'
    save('test_results.json',dict(all_input_hashes_unchanged=True,passed=passed,total=64,
         required_gate_passed=passed==64,selected_cases_dropped=0,new_trajectories=0,
         per_case_checks=['raw reconstruction','initial checklist exact','final exact array including closure',
         'logging invariance','float32 evaluation and selection','literal route mutation',
         'accepted edge gain vs independent float64 cost','permutation and positions','checklist evolution']))
    print(summary['decision'],passed,'/64')

if __name__=='__main__':main()
