"""Immutable, canonical, full-refresh replay diagnostics and frozen-graph probes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import torch
import faco
import utils
from net import Net
from r1.engine import weight_hash,sync
from r1.v2_replay import branch
from r1.replay_determinism import settings,strict_cuda

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'checkpoints/tsp_n1000_k32_ants100_H10_miniH100_rho0.1_mne12_ppo_lr5e-06_best.pt'

def array(t):
    if torch.is_tensor(t):
        if t.is_cuda: sync(t.device)
        t=t.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(t).copy()

def sha(a): return hashlib.sha256(array(a).tobytes()).hexdigest()

def compare(a,b):
    a=array(a); b=array(b); x=a.astype(np.float64); y=b.astype(np.float64)
    tiny=np.finfo(a.dtype).tiny if np.issubdtype(a.dtype,np.floating) else np.finfo(np.float64).tiny
    delta=np.abs(x-y); denominator=np.maximum(np.maximum(np.abs(x),np.abs(y)),tiny)
    result=dict(shape=list(a.shape),dtype=str(a.dtype),byte_equal=a.tobytes()==b.tobytes(),unequal_elements=int(np.count_nonzero(a!=b)),
        max_absolute=float(delta.max()),max_relative=float((delta/denominator).max()),rms=float(np.sqrt(np.mean(delta**2))),
        relative_denominator='max(abs(a),abs(b),dtype.tiny)',nan_count=int(np.isnan(x).sum()+np.isnan(y).sum()),
        infinity_count=int(np.isinf(x).sum()+np.isinf(y).sum()),signed_zero_changes=int(np.count_nonzero((a==0)&(b==0)&(np.signbit(a)!=np.signbit(b)))))
    if a.dtype==b.dtype==np.float32:
        def ordered(v):
            bits=v.view(np.uint32).astype(np.int64)
            return np.where(bits & 0x80000000,0x80000000-(bits & 0x7fffffff),0x80000000+bits)
        ulp=np.abs(ordered(a)-ordered(b))
        result['ulp_quantiles_0_50_90_99_100']=np.quantile(ulp,[0,.5,.9,.99,1]).tolist()
    return result

def capture(coords,model,out,label,t=175):
    records=[]; snapshots=[]; current={}; iteration=0
    original_build=utils.build_pyg_data_tsp; original_sample=faco.MFACO_TSP.sample
    def build(solver,norm,device,**kwargs):
        graph=original_build(solver,norm,device,**kwargs)
        current.clear(); current.update(iteration=iteration,ordinal=len(records)+1,model_eval=not model.training,
            state_dict_sha256=weight_hash(model),coordinate_sha256=sha(coords),settings=settings())
        snapshot=dict(x=array(graph.x),edge_index=array(graph.edge_index),edge_attr=array(graph.edge_attr),
            pheromone=array(solver.pheromone_sparse_np),source_tour=array(solver.source_route))
        snapshots.append(snapshot)
        return graph
    def hook(module,args,output):
        snapshot=snapshots[-1]; snapshot['prior']=array(output.view(len(coords),32))
        current.update(hashes={k:sha(v) for k,v in snapshot.items()},prior_dtype=str(output.dtype),prior_shape=list(output.shape),
            prior_stride=list(output.stride()),prior_layout=str(output.layout),prior_device=str(output.device),state_dict_after_sha256=weight_hash(model))
        records.append(copy.deepcopy(current))
    def sample(solver,**kwargs):
        nonlocal iteration
        passed=array(kwargs['prior'])
        ordinal=len(records)-1
        if 'native_prior' not in snapshots[ordinal]:
            snapshots[ordinal]['native_prior']=passed
            records[ordinal]['native_prior_sha256']=sha(passed)
        assert sha(passed)==sha(snapshots[ordinal]['prior'])
        result=original_sample(solver,**kwargs); iteration+=1
        return result
    utils.build_pyg_data_tsp=build; faco.MFACO_TSP.sample=sample
    handle=model.register_forward_hook(hook)
    try: result=branch(coords,model,31415,t,False)
    finally:
        handle.remove(); utils.build_pyg_data_tsp=original_build; faco.MFACO_TSP.sample=original_sample
    for i,snapshot in enumerate(snapshots): np.savez(out/f'{label}_refresh{i}.npz',**snapshot)
    (out/f'{label}_refreshes.json').write_text(json.dumps(records,indent=2))
    (out/f'{label}_trajectory.json').write_text(json.dumps(result,indent=2))
    return records,snapshots

def forward_probes(model,snapshot):
    from torch_geometric.data import Data
    graph=Data(**{k:torch.from_numpy(snapshot[k].copy()).cuda() for k in ('x','edge_index','edge_attr')})
    layers=[]
    def forward(m,g):
        outputs={}; handles=[]
        for name,module in m.named_modules():
            if name and ('.layers.' in name or name in ('emb_net.v_lin0','emb_net.e_lin0')):
                def hook(mod,args,out,name=name):
                    outputs[name]=[array(v) for v in (out if isinstance(out,tuple) else (out,))]
                handles.append(module.register_forward_hook(hook))
        with torch.inference_mode(): value=array(m(g))
        for h in handles: h.remove()
        return value,outputs
    first,first_layers=forward(model,graph); repeats=[]
    for reconstructed in (False,False,True,True):
        other=graph.clone() if reconstructed else graph
        value,state=forward(model,other)
        earliest=next((name for name in first_layers if any(not np.array_equal(a,b) for a,b in zip(first_layers[name],state[name]))),None)
        repeats.append(dict(reconstructed=reconstructed,comparison=compare(first,value),first_divergent_module=earliest))
    # Isolate first-layer aggregation: identical upstream float32 inputs.
    layer=model.emb_net.layers[0]
    with torch.inference_mode():
        x=model.emb_net.act_fn(model.emb_net.v_lin0(graph.x)); w=model.emb_net.act_fn(model.emb_net.e_lin0(graph.edge_attr))
        messages=torch.sigmoid(w)*layer.v_lin2(x)[graph.edge_index[1]]
        a=array(layer.agg_fn(messages,graph.edge_index[0])); b=array(layer.agg_fn(messages,graph.edge_index[0]))
    cpu=copy.deepcopy(model).cpu(); cg=graph.cpu()
    c,_=forward(cpu,cg); d,_=forward(cpu,cg)
    return dict(cuda_repeats=repeats,isolated_first_layer_global_mean_pool=compare(a,b),cpu_repeat=compare(c,d))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--strict',action='store_true'); args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    if args.strict: strict_cuda()
    result=dict(settings=settings(),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
    try:
        torch.set_num_threads(2); faco.set_faco_cpp_threads(2)
        ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        model=Net(feats=2,edge_feats=6,logit_net=True).cuda().eval(); model.load_state_dict(ckpt['model_state_dict'],strict=True)
        coords=np.load(ROOT/'artifacts/r1/datasets/tsp100_seed20261031_v1.npy')[0]
        a,aa=capture(coords,model,out,'hold_a'); b,bb=capture(coords,model,out,'hold_b')
        result['refresh_comparisons']=[dict(iteration=x['iteration'],metadata_equal={k:x[k]==y[k] for k in ('model_eval','state_dict_sha256','coordinate_sha256')},
            arrays={key:compare(s[key],t[key]) for key in s}) for x,y,s,t in zip(a,b,aa,bb)]
        result['frozen_graph']=forward_probes(model,aa[0])
        result['status']='completed'
    except Exception: result['status']='failed'; result['error']=traceback.format_exc()
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result,indent=2),flush=True)
    if result['status']=='failed': raise SystemExit(1)

if __name__=='__main__': main()
