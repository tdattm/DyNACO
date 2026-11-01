"""Post-collection integrity audit from persisted bytes, no solver rollout."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from r3.common import ROOT,sha256,save_json,CHECKPOINT
from r3.diagnostics import canonical_tour,length,apply_move
from r3.gate import evaluate
from r3.analysis import aggregate
from r3.protocol import load,PATH

def main():
    protocol=load();out=ROOT/'artifacts/r3/experiments/stage_a_v1_01'
    ledger=json.loads((out/'finalized_sha256.json').read_text());checked=0
    for name,digest in ledger.items():
        assert sha256(out/name)==digest,name
        checked+=1
    initial=json.loads((ROOT/'artifacts/r3/manifests/repository_initial.json').read_text())
    for name,digest in initial['source_hashes'].items():assert sha256(ROOT/name)==digest,name
    assert sha256(CHECKPOINT)==protocol['checkpoint_sha256']
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['protocol_sha256']==sha256(PATH)
    data=np.load(out/'datasets/tsp1000_seed20261101_float32.npy',allow_pickle=False)
    regen=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261101,1000]))).random((8,1000,2),dtype=np.float32)
    assert np.array_equal(data,regen)
    stream=io.BytesIO();np.save(stream,regen,allow_pickle=False)
    assert hashlib.sha256(stream.getvalue()).hexdigest()==manifest['dataset_sha256']
    diagnostics=[];gain_checks=0;raw_checks=0;state_checks=0
    for instance in range(8):
        x=data[instance]
        with np.load(out/f'native_candidates_i{instance}.npz') as candidates:
            nn={k:candidates[f'k{k}'].copy() for k in (32,40,48,64)}
        for k in (32,40,48):assert np.array_equal(nn[k],nn[64][:,:k])
        for seed in (31415,27182):
            folder=out/f'i{instance}_s{seed}';run=json.loads((folder/'run.json').read_text())
            assert run['native_seed']==int(np.random.RandomState(seed).randint(0,2**63-1))
            with np.load(folder/'trajectory.npz') as trajectory:
                assert trajectory['ant_costs'].shape==(1000,100)
                assert np.all(np.isfinite(trajectory['ant_costs']))
                assert np.all(np.diff(trajectory['global_best_cost'])<=0)
            for iteration in (0,250,500,1000):
                with np.load(folder/f'state_t{iteration}.npz') as state:
                    assert np.array_equal(state['coordinates'],x)
                    assert np.array_equal(state['nn_list'],nn[32])
                    assert np.array_equal(state['backup_list'],nn[64][:,32:64])
                    assert np.allclose(state['heuristic_sparse_np'].sum(axis=1),1,rtol=2e-6)
                    for kind in ('source','best'):
                        tour=canonical_tour(state[kind+'_route'],1000);before=length(x,tour)
                        assert abs(before-float(state[kind+'_length_f64']))<1e-11
                        assert abs(before-float(state[kind+'_cost_native']))<=2e-5*before
                        diag=json.loads((folder/f'diagnostic_{kind}_t{iteration}.json').read_text());diagnostics.append(diag)
                        gain_checks+=audit_diagnostic(diag,x,tour,nn[32])
                        state_checks+=1
            for iteration in (0,249,499,999):
                trace=json.loads((folder/f'perturbation_iter{iteration}.json').read_text())
                assert trace['raw_exact_match'] and trace['actual_native_acceptances'] is None
                with np.load(folder/f'raw_before_iter{iteration}.npz') as state:
                    diag=json.loads((folder/f'diagnostic_raw_before_iter{iteration}.json').read_text())
                    gain_checks+=audit_diagnostic(diag,x,state['raw'],nn[32]);raw_checks+=1
    assert len(diagnostics)==128 and state_checks==128 and raw_checks==64
    recorded=json.loads((out/'gate.json').read_text())
    independently_aggregated=aggregate(diagnostics)
    calculated=evaluate(independently_aggregated,True,True,True)
    for key in ('decision','conditions','per_instance','median_ratios','passing_instance_counts'):
        assert calculated[key]==recorded[key],key
    result=dict(status='PASS',finalized_files_rehashed=checked,production_source_bytes_unchanged=True,
        dataset_regenerated_exact=True,checkpoint_hash_valid=True,protocol_hash_valid=True,
        frozen_tour_records_checked=state_checks,raw_records_checked=raw_checks,
        independently_reapplied_moves=gain_checks,instance_gate_reaggregation_matches=True,
        decision=recorded['decision'],notes='No reserved confirmation data was generated or read.')
    save_json(ROOT/'artifacts/r3/manifests/post_collection_audit_v1.json',result)
    print(json.dumps(result,indent=2))

def audit_diagnostic(diag,x,tour,nn32):
    assert diag['valid_moves']==498500
    assert diag['valid_moves']==sum(diag[k] for k in ('robust_improving_moves','positive_marginal_moves','zero_gain_moves','negative_gain_moves'))
    assert diag['membership_count'][:4]==sorted(diag['membership_count'][:4])
    assert diag['scan_count'][:4]==sorted(diag['scan_count'][:4])
    assert all(a<=b for a,b in zip(diag['scan_count'],diag['membership_count']))
    assert diag['membership_missed_count']==diag['robust_improving_moves']-diag['membership_count'][0]
    before=length(x,tour);count=0
    for rec in diag['top_membership_missed_moves']+diag['sample_robust_moves']:
        changed=apply_move(tour,rec['i'],rec['j']);observed=before-length(x,changed)
        assert abs(observed-rec['raw_gain_f64'])<1e-10
        assert observed>max(1e-8,1e-6*before)
        assert rec['actually_accepted_native'] is None and rec['actually_evaluated_native'] is None
        count+=1
    widths=np.asarray(diag['per_node_widths'])
    assert np.all(widths[1]==40)
    for row in widths[4:]:assert int((row-32).sum())==8000 and int((row==64).sum())==250
    # Independently compute the predefined, non-oracle selector from saved data.
    successor=np.empty(1000,dtype=int);predecessor=np.empty(1000,dtype=int)
    successor[tour]=np.roll(tour,-1);predecessor[tour]=np.roll(tour,1)
    scores=np.asarray(diag['node_scores'])
    x64=x.astype(np.float64)
    delta=x64[:,None,:]-x64[nn32]
    denominator=np.hypot(delta[:,:,0],delta[:,:,1]).mean(axis=1)
    pred_delta=x64-x64[predecessor];succ_delta=x64-x64[successor]
    numerator=np.maximum(np.hypot(pred_delta[:,0],pred_delta[:,1]),np.hypot(succ_delta[:,0],succ_delta[:,1]))
    recomputed=numerator/np.maximum(denominator,1e-12)
    assert np.allclose(scores,recomputed,rtol=1e-12,atol=1e-12)
    expected=np.lexsort((np.arange(1000),-scores))[:250]
    assert np.array_equal(np.flatnonzero(widths[4]==64),np.sort(expected))
    for record in diag['policy_edge_counts'][4:]:assert record['extra_directed_slots']==8000
    return count

if __name__=='__main__':main()
