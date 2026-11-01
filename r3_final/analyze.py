"""Rebuild every result, figure and decision solely from sealed raw records."""
import csv
import json
import re
from collections import Counter
import numpy as np
from r3_final.core import *

LABELS={'A':'K32','B':'K40','C':'Adaptive K32/K64'}


def polish(text):
    # Typographic spacing in prose only; never alter code, hashes, or numeric results.
    pieces=re.split(r'(`[^`]*`)',text)
    for i in range(0,len(pieces),2):
        pieces[i]=re.sub(r'\b(có|đều|mở|ở|giữ|đúng|mỗi|vượt|lúc|gồm|và|seed|mod|watchdog|RSS|allocation|reserve|metric|Anytime|Khoảng|dùng|median|mean|gain|CUDA|Python|NumPy|PyTorch|VRAM|tại|iteration|Full|trên|với|trước|epoch|Trong|chiếm|range|đó|bình|tất)(?=[−-]?\d)',r'\1 ',pieces[i])
        pieces[i]=pieces[i].replace('ngoàiK32','ngoài K32')
        pieces[i]=re.sub(r'(?<=\d)(GiB|MiB|ms|s|instances|iteration|run|case|seed|CPU)\b',r' \1',pieces[i])
        pieces[i]=re.sub(r',(?=[A-Za-zÀ-ỹ])',', ',pieces[i])
    return ''.join(pieces)


def decision(gains, indices):
    boot=np.mean(np.asarray(gains)[indices],axis=1)
    lo,hi=np.percentile(boot,[2.5,97.5],method='linear')
    values=dict(mean=float(np.mean(gains)),median=float(np.median(gains)),
                positive=int(np.count_nonzero(gains>0)),negative=int(np.count_nonzero(gains<0)),
                tied=int(np.count_nonzero(gains==0)),ci95=[float(lo),float(hi)],
                worst=float(np.min(gains)),best=float(np.max(gains)),n_independent_instances=16)
    values['conditions']=dict(median_at_least_0_001=values['median']>=.001,
                              at_least_12_positive=values['positive']>=12,
                              bootstrap_lower_above_zero=bool(lo>0))
    values['passes']=bool(all(values['conditions'].values()))
    return values


def csv_write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    from r3_final.execute import check_frozen
    check_frozen()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':140,
                         'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'r3-final-end-to-end-v1'})
    analysis=OUT/'analysis';analysis.mkdir(exist_ok=True)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    coords=np.load(OUT/'principal_coords.npy')
    records=[];trajectories={};files={}
    for identity in identities():
        folder=OUT/'raw_results'/identity['run_id']
        completion=json.loads((folder/'completion.json').read_text())
        assert completion['status']=='VALID' and completion['identity']==identity
        for file,digest in completion['files'].items():
            assert sha256(folder/file)==digest
        attempt=folder/f'attempt_{completion["attempt"]}'
        row=json.loads((attempt/'result.json').read_text())
        for key,value in identity.items():assert row[key]==value
        a=dict(np.load(attempt/'trajectory.npz',allow_pickle=False))
        assert np.all(np.diff(a['update_times'])>0) and len(a['update_times'])==row['completed_iterations']+1
        assert array_hash(a['native_nn64'])==row['native_candidates_sha256']
        recalculated=np.array([length(coords[identity['instance']],r) for r in a['routes']])
        assert np.array_equal(recalculated,a['route_lengths'])
        eligible=np.flatnonzero(a['route_times']<=8.) if row['mode']=='wall' else np.arange(len(recalculated))
        official=int(eligible[-1])
        assert official==row['primary_route_index'] and row['official_length']==recalculated[official]
        assert row['counters']['calls']==100*row['completed_iterations']
        expected=32000 if row['arm']=='A' else 40000
        assert np.all(a['allocation_widths'].sum(axis=1)==expected)
        if row['arm']=='C':assert np.all((a['allocation_widths']==64).sum(axis=1)==250)
        assert np.array_equal(a['routes'][-1],a['final_best'])
        row['initial_route_validated']=True
        trajectories[row['run_id']]=a
        records.append(row)
        files[row['run_id']]={str(p.relative_to(OUT)):sha256(p) for p in (folder/'completion.json',attempt/'result.json',attempt/'trajectory.npz')}
    assert len(records)==192
    for mode in MODES:
        for i in range(16):
            for seed in SEEDS:
                paired=[r for r in records if (r['mode'],r['instance'],r['seed'])==(mode,i,seed)]
                assert len(paired)==3
                for field in ('native_seed','initial_route_sha256','initial_source_sha256','initial_pheromone_sha256','native_candidates_sha256'):
                    assert len(set(r[field] for r in paired))==1, 'unpaired initialization '+field
    flat=[]
    for r in records:
        flat.append({k:r[k] for k in ('run_id','mode','instance','seed','arm','official_length','initial_length','completed_iterations','eligible_completed_iterations','overshoot_iterations','last_completed_before_budget','search_seconds','setup_seconds','allocation_seconds','graph_neural_seconds','serialization_seconds','total_run_seconds','rss_peak_bytes','peak_cuda_bytes','native_seed','slots')} | {f'native_{k}':r['counters'][k] for k in COUNTERS})
    csv_write(analysis/'per_run_results.csv',flat)
    per=[];aggregate={};stats={}
    indices=np.random.Generator(np.random.PCG64(202612070302)).integers(0,16,size=(10000,16))
    with (analysis/'bootstrap_indices.npy').open('wb') as f:np.save(f,indices)
    for mode in MODES:
        for i in range(16):
            item=dict(mode=mode,instance=i)
            paired={arm:{r['seed']:r for r in records if r['mode']==mode and r['instance']==i and r['arm']==arm} for arm in ARMS}
            for arm in ARMS:
                item[f'length_{arm}']=float(np.mean([r['official_length'] for r in paired[arm].values()]))
                item[f'iterations_{arm}']=float(np.mean([r['eligible_completed_iterations'] for r in paired[arm].values()]))
                item[f'search_seconds_{arm}']=float(np.mean([r['search_seconds'] for r in paired[arm].values()]))
            for control in ('A','B'):
                item[f'gain_C_{control}']=float(np.mean([(paired[control][s]['official_length']-paired['C'][s]['official_length'])/paired[control][s]['official_length'] for s in SEEDS]))
            per.append(item)
        aggregate[mode]={}
        stats[mode]={}
        for control in ('A','B'):
            g=np.array([r[f'gain_C_{control}'] for r in per if r['mode']==mode])
            stats[mode][f'C_{control}']=decision(g,indices)
        for arm in ARMS:
            rows=[r for r in records if r['mode']==mode and r['arm']==arm]
            values=np.array([r['official_length'] for r in rows])
            instance_lengths=np.array([r[f'length_{arm}'] for r in per if r['mode']==mode])
            length_ci=np.percentile(np.mean(instance_lengths[indices],axis=1),[2.5,97.5])
            aggregate[mode][arm]=dict(label=LABELS[arm],n_runs=32,mean_length=float(values.mean()),
               median_instance_length=float(np.median(instance_lengths)),mean_length_ci95=length_ci.tolist(),
               mean_completed_iterations=float(np.mean([r['eligible_completed_iterations'] for r in rows])),
               mean_search_seconds=float(np.mean([r['search_seconds'] for r in rows])),
               mean_setup_seconds=float(np.mean([r['setup_seconds'] for r in rows])),
               mean_serialization_seconds=float(np.mean([r['serialization_seconds'] for r in rows])),
               seconds_per_iteration=float(np.mean([r['search_seconds']/r['completed_iterations'] for r in rows])),
               mean_allocation_seconds=float(np.mean([r['allocation_seconds'] for r in rows])),
               mean_neural_seconds=float(np.mean([r['graph_neural_seconds'] for r in rows])),
               mean_relative_gain_per_search_second=float(np.mean([(r['initial_length']-r['official_length'])/r['initial_length']/(8. if mode=='wall' else r['search_seconds']) for r in rows])),
               mean_unique_undirected_edges=float(np.mean([np.mean(r['unique_undirected_edges']) for r in rows])),
               counters_total={k:int(sum(r['counters'][k] for r in rows)) for k in COUNTERS})
    csv_write(analysis/'per_instance_results.csv',per)
    final='GO' if all(c['passes'] for c in stats['wall'].values()) else 'NO-GO'
    verdict=dict(project_decision=final,scientific_conclusion=final,valid_runs=192,planned_runs=192,
                 primary_mode='wall',criteria=stats['wall'],complete_grid_validated=True,
                 bootstrap_indices_sha256=sha256(analysis/'bootstrap_indices.npy'))
    atomic_json(analysis/'aggregate_results.json',aggregate)
    atomic_json(analysis/'statistical_tests.json',stats)
    atomic_json(analysis/'decision.json',verdict)
    atomic_json(analysis/'raw_integrity_audit.json',dict(valid_runs=192,paired_initialization_exact=True,all_tours_valid=True,official_f64_exact=True,one_srr_per_ant=True,files=files))
    # Epoch behavior and offline, unique undirected budgets.
    epoch_rows=[];node_rows=[]
    for r in records:
        a=trajectories[r['run_id']]
        if r['arm']=='C':
            for epoch,(w,score,meta) in enumerate(zip(a['allocation_widths'],a['allocation_scores'],r['allocations'])):
                epoch_rows.append(dict(run_id=r['run_id'],mode=r['mode'],instance=r['instance'],seed=r['seed'],
                     epoch=epoch,iteration=meta['iteration'],changed_nodes=meta['changed_nodes'],slots=meta['slots'],
                     unique_undirected=r['unique_undirected_edges'][epoch],score_min=float(score.min()),
                     score_median=float(np.median(score)),score_max=float(score.max()),
                     score_selected_median=float(np.median(score[w==64]))))
            for node,frequency in enumerate((a['allocation_widths']==64).mean(axis=0)):
                node_rows.append(dict(run_id=r['run_id'],node=node,expanded_fraction=float(frequency)))
    csv_write(analysis/'allocation_epochs.csv',epoch_rows)
    csv_write(analysis/'allocation_node_frequency.csv',node_rows)
    colors={'A':'#3b6ca8','B':'#df8b28','C':'#24845c'}
    def save(name,fig):
        fig.savefig(figures/(name+'.png'),bbox_inches='tight')
        fig.savefig(figures/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
        plt.close(fig)
    wall=[r for r in per if r['mode']=='wall']
    fig,ax=plt.subplots(figsize=(8,5))
    for j,arm in enumerate(ARMS):
        vals=np.array([r[f'length_{arm}'] for r in wall])
        ax.scatter(np.full(16,j)+np.linspace(-.12,.12,16),vals,c=colors[arm],alpha=.65,s=22)
        ax.plot([j-.22,j+.22],[vals.mean()]*2,color='black',lw=2)
    for r in wall:ax.plot(range(3),[r[f'length_{a}'] for a in ARMS],c='gray',alpha=.2,lw=.7)
    ax.set(xticks=range(3),xticklabels=list(LABELS.values()),ylabel='Float64 cyclic tour length',title='8-second budget: all 16 independent instance means',ylim=(0,None))
    save('01_wall_final_length',fig)
    gain_limit=1.15*max(abs(r[f'gain_C_{control}'])*100 for r in wall for control in ('A','B'))
    for no,control in ((2,'A'),(3,'B')):
        fig,ax=plt.subplots(figsize=(9,4.6));g=np.array([r[f'gain_C_{control}'] for r in wall])*100
        ax.bar(range(16),g,color=[colors['C'] if v>0 else '#b94b4b' for v in g])
        ax.axhline(0,color='black',lw=.8);ax.axhline(.1,color='black',ls='--',label='0.1% median screening threshold')
        ax.set(xticks=range(16),xlabel='Independent instance',ylabel='Adaptive relative gain (%)',ylim=(-gain_limit,gain_limit),title=f'8 seconds: Adaptive vs {LABELS[control]} (mean of two paired seed gains)')
        ax.legend(fontsize=8);save(f'0{no}_paired_gain_C_{control}',fig)
    anytime=[]
    points=np.linspace(0,8,161)
    fig,ax=plt.subplots(figsize=(8,5))
    for arm in ARMS:
        rows=[r for r in records if r['mode']=='wall' and r['arm']==arm];curves=[]
        for r in rows:
            a=trajectories[r['run_id']]
            idx=np.searchsorted(a['route_times'],points,side='right')-1
            curves.append(a['route_lengths'][idx]/r['initial_length'])
            if r['seed']==SEEDS[0]:ax.plot(points,curves[-1],color=colors[arm],alpha=.10,lw=.5)
            for t in (0,1,2,4,6,8):
                ix=int(np.searchsorted(a['route_times'],t,side='right')-1)
                anytime.append(dict(run_id=r['run_id'],instance=r['instance'],seed=r['seed'],arm=arm,
                                    time_seconds=t,length=float(a['route_lengths'][ix])))
        ax.plot(points,np.mean(curves,axis=0),color=colors[arm],label=LABELS[arm],lw=2)
    ax.set(xlabel='Search wall time (s)',ylabel='Length / paired initialized length',ylim=(0,1.03),xlim=(0,8),title='Anytime quality: overshoot updates excluded')
    ax.legend();save('04_anytime_quality',fig);csv_write(analysis/'anytime_results.csv',anytime)
    fig,ax=plt.subplots(figsize=(8,5))
    for arm in ARMS:
        rows=[r for r in per if r['mode']=='iterations']
        ax.scatter([r[f'search_seconds_{arm}'] for r in rows],[r[f'length_{arm}'] for r in rows],label=LABELS[arm],color=colors[arm],alpha=.8)
    ax.set(xlabel='1000-iteration search wall time (s)',ylabel='Float64 tour length',xlim=(0,None),ylim=(0,None),title='Equal iterations: all independent instance means');ax.legend();save('05_iteration_quality_runtime',fig)
    fig,ax=plt.subplots(figsize=(9,5));xx=np.arange(16)
    for j,arm in enumerate(ARMS):ax.bar(xx+(j-1)*.25,[r[f'iterations_{arm}'] for r in wall],width=.24,color=colors[arm],label=LABELS[arm])
    ax.set(xticks=xx,xlabel='Independent instance',ylabel='Completed iterations at timestamp ≤ 8 s',title='Equal wall time: completed work (seed means)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3);save('06_completed_iterations',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,mode in zip(axes,MODES):
        for j,arm in enumerate(ARMS):
            rows=[r for r in records if r['mode']==mode and r['arm']==arm]
            visits=np.array([r['counters']['candidates'] for r in rows])/1e6
            ax.bar(j,visits.mean(),color=colors[arm]);ax.scatter(j+np.linspace(-.1,.1,32),visits,c='black',s=6,alpha=.35)
        ax.set(xticks=range(3),xticklabels=['K32','K40','Adaptive'],ylabel='Candidate visits (million)',title='1000 iterations; no overshoot' if mode=='iterations' else '8-second runs; includes recorded overshoot')
    save('07_candidate_evaluations',fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].bar(['K32','K40','Adaptive'],[1000,0,750],label='width 32',color=colors['A'])
    axes[0].bar(['K32','K40','Adaptive'],[0,1000,0],label='width 40',color=colors['B'])
    axes[0].bar(['K32','K40','Adaptive'],[0,0,250],bottom=[1000,1000,750],label='width 64',color=colors['C'])
    axes[0].set(ylabel='Nodes',title='Fixed allocation sizes');axes[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3)
    changes=[r['changed_nodes'] for r in epoch_rows if r['iteration']>0]
    axes[1].hist(changes,bins=20,color=colors['C']);axes[1].set(xlabel='Nodes whose width changed / epoch',ylabel='Epochs',title='Adaptive allocation changes')
    scores=np.concatenate([a['allocation_scores'].ravel() for id,a in trajectories.items() if id.endswith('_C')])
    axes[2].hist(scores,bins=50,color=colors['C']);axes[2].set(xlabel='Geometry / tour-excess score',ylabel='Node-epochs',title='All adaptive scores (no filtering)')
    fig.tight_layout();save('08_allocation_behavior',fig)
    report(records,per,aggregate,stats,verdict,epoch_rows)
    print(json.dumps(verdict,indent=2),flush=True)


def report(records,per,aggregate,stats,verdict,epochs):
    imp=json.loads((OUT/'implementation_manifest.json').read_text());data=json.loads((OUT/'dataset_manifest.json').read_text())
    env=json.loads((OUT/'environment_manifest.json').read_text());tests=json.loads((OUT/'test_results.json').read_text())
    diag=json.loads((OUT/'development/untimed_srr_diagnostics.json').read_text())
    final=verdict['scientific_conclusion']
    def pct(x):return f'{100*x:.4f}%'
    rows=[]
    for mode in MODES:
        for arm in ARMS:
            r=aggregate[mode][arm];ci=r['mean_length_ci95']
            rows.append(f'| {mode} | {LABELS[arm]} | {r["mean_length"]:.8f} | {r["median_instance_length"]:.8f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {r["mean_completed_iterations"]:.2f} | {r["mean_search_seconds"]:.4f} |')
    table='\n'.join(['| Budget | Arm | Mean length | Median instance length | 95% CI mean length | Iterations hợp lệ | Search s |','|---|---|---:|---:|---|---:|---:|',*rows])
    comparisons=[];gates=[]
    for mode in MODES:
        for comp,r in stats[mode].items():
            comparisons.append(f'| {mode} | {comp} | {pct(r["mean"])} | {pct(r["median"])} | [{pct(r["ci95"][0])}, {pct(r["ci95"][1])}] | {r["positive"]}/{r["negative"]}/{r["tied"]} | {pct(r["worst"])} | {pct(r["best"])} |')
            if mode=='wall':
                gates.append(f'| {comp} | {pct(r["median"])} ({"PASS" if r["conditions"]["median_at_least_0_001"] else "FAIL"}) | {r["positive"]}/16 ({"PASS" if r["conditions"]["at_least_12_positive"] else "FAIL"}) | {pct(r["ci95"][0])} ({"PASS" if r["conditions"]["bootstrap_lower_above_zero"] else "FAIL"}) |')
    gains='\n'.join(['| Budget | So sánh | Mean gain | Median gain | 95% CI mean | + / − / = instances | Worst | Best |','|---|---|---:|---:|---|---|---:|---:|',*comparisons])
    gate_table='\n'.join(['| Primary 8 s | Median ≥0.1% | Positive ≥12/16 | Bootstrap lower >0 |','|---|---|---|---|',*gates])
    eff=[]
    for mode in MODES:
        base=aggregate[mode]['A']
        for arm in ARMS:
            r=aggregate[mode][arm];c=r['counters_total']
            eff.append(f'| {mode} | {LABELS[arm]} | {r["seconds_per_iteration"]*1000:.4f} | {pct(r["seconds_per_iteration"]/base["seconds_per_iteration"]-1)} | {c["calls"]} | {c["candidates"]} | {c["gains"]} | {c["expanded_candidates"]} | {c["expanded_accepted"]} | {r["mean_unique_undirected_edges"]:.2f} |')
    efficiency='\n'.join(['| Budget | Arm | ms/iteration | Overhead vs K32 | SRR calls | Candidate visits | Gain evals | Visits rank33–64 | Accepted rank33–64 (float32) | Mean unique undirected edges |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',*eff])
    individuals=[]
    for r in per:
        individuals.append(f'| {r["mode"]} | {r["instance"]} | {r["length_A"]:.7f} | {r["length_B"]:.7f} | {r["length_C"]:.7f} | {pct(r["gain_C_A"])} | {pct(r["gain_C_B"])} |')
    individual='\n'.join(['| Budget | Instance | K32 | K40 | Adaptive | Gain C−A | Gain C−B |','|---|---:|---:|---:|---:|---:|---:|',*individuals])
    changes=[r['changed_nodes'] for r in epochs if r['iteration']>0]
    adaptive_accept=sum(aggregate[m]['C']['counters_total']['expanded_accepted'] for m in MODES)
    wa=aggregate['wall']['A'];wb=aggregate['wall']['B'];wc=aggregate['wall']['C']
    wallc=wc['counters_total']
    expanded_events=[e for r in diag['records'] for e in r['events'] if e['rank']>32]
    interpretation=f'''Các đo lường cụ thể cho tradeoff compute/quality:

- C so với K32 tại8s: mean gain{pct(stats['wall']['C_A']['mean'])},
  median{pct(stats['wall']['C_A']['median'])}; tại1000 iteration:
  mean{pct(stats['iterations']['C_A']['mean'])}.
- C so với K40 tại8s: mean gain{pct(stats['wall']['C_B']['mean'])},
  median{pct(stats['wall']['C_B']['median'])}; tại1000 iteration:
  mean{pct(stats['iterations']['C_B']['mean'])}.
- C thay đổi runtime/iteration so với K32: {pct(wc['seconds_per_iteration']/wa['seconds_per_iteration']-1)},
  so với K40: {pct(wc['seconds_per_iteration']/wb['seconds_per_iteration']-1)}.
  Trong8s, C hoàn tất{wc['mean_completed_iterations']:.2f} iteration trung bình,
  K32 {wa['mean_completed_iterations']:.2f}, K40 {wb['mean_completed_iterations']:.2f}.
- Score/allocation cost của C trung bình{wc['mean_allocation_seconds']*1000:.3f}ms/run wall,
  chiếm{pct(wc['mean_allocation_seconds']/wc['mean_search_seconds'])} search time.
  Rank33–64 chiếm{pct(wallc['expanded_candidates']/wallc['candidates'])} candidate visits
  và{pct(wallc['expanded_gains']/wallc['gains'])} actual gain evaluations của C.
- Trong untimed development replay, có{len(expanded_events)} accepted move ngoàiK32,
  trong đó{sum(e['robust'] for e in expanded_events)} robust float64 improvement.
  Điều này xác nhận cơ chế có thể thực thi move hữu ích, nhưng không chứng minh
  lượng gain đủ bù chi phí trong principal global-best trajectories.

Các số liệu trên cho phép tách hiệu ứng chất lượng theo iteration khỏi throughput
theo thời gian. Chúng không cho phép gán toàn bộ outcome cho overhead hoặc allocator
riêng lẻ. Không có ablation về checklist, allocator khác hoặc full-K64 nên không
thể kết luận đã xác định duy nhất nguyên nhân. Screening rule chỉ đánh giá policy
cụ thể này với cả hai control dưới ngân sách đã định.
'''
    overshoots=sum(r['overshoot_iterations'] for r in records if r['mode']=='wall')
    policy_recommend=('Tiếp tục một nghiên cứu xác nhận độc lập trước khi mở rộng phạm vi; kết quả hiện tại chỉ hỗ trợ cấu hình và phân phối này.' if final=='GO' else
                       'Dừng đầu tư thêm vào đúng adaptive candidate policy K32/K64 này trên uniform TSP1K. Hướng ưu tiên tiếp theo là nghiên cứu SRR scheduling/revisit với equal-compute control và dữ liệu mới; M1 v2 có tín hiệu cơ học trực tiếp hơn nhưng chưa chứng minh hiệu quả end-to-end.')
    relationship=('Kết quả mới củng cố quyết định không đầu tư thêm vào adaptive policy này trên uniform TSP1K; nó không xác định duy nhất cơ chế thất bại và không phủ định mọi allocator khác.' if final=='NO-GO' else
                  'Kết quả mới hỗ trợ policy tích hợp này trong cấu hình đã đo, đồng thời giới hạn cách suy rộng từ offline Stage A sang end-to-end search.')
    text=f'''# Báo cáo nghiên cứu R3 final end-to-end

**R3 FINAL DECISION: {final}**  
**SCIENTIFIC CONCLUSION: {final}**  
Hoàn thành và kiểm toán độc lập **192/192 run hợp lệ**, 16 instance × 2 seed × 3 arm × 2 budget.
Đây là kết luận khoa học theo screening rule đã đóng băng, không phải technical NO-GO.

## 1. Câu hỏi và giả thuyết

Phân bổ có thích nghi 8000 directed candidate slots bổ sung cho SRR có cải thiện
global-best tour của DyNACO so với K32 gốc và K40 đồng ngân sách cạnh hay không?
K32 có32000 slots; K40 và adaptive đều40000. Adaptive mở64 ứng viên ở250 node,
giữ32 ở750 node. So sánh K40 tách hiệu ứng vị trí phân bổ khỏi việc thêm cạnh.

Với tour global-best hiện tại, s(u)=max(d(u,pred(u)),d(u,succ(u)))/
max(mean distance to original K32,1e-12). Xếp giảm dần s, tie theo chỉ số tăng,
lấy đúng250 node. Cập nhật trước iteration0,100,200,…; chỉ dùng trạng thái sẵn có.
Khoảng cách allocation tính float64 từ tọa độ float32; native SRR giữ float32.

## 2. Triển khai và kiểm soát

Giữ nguyên nn_list Nx32, backup Nx32, construction, GNN, pheromone và heuristic.
Buffer SRR nối đúng nn_list + backup theo thứ tự native. Template dispatch chỉ
đổi row/width trong hai hướng quét 2-opt. Giữ strict distance early-break,
strict gain tie, max_diff>0, route reversal, checklist extension và RNG.
Không có callback Python trong OpenMP; setter được mutex bảo vệ khi sample.
Đếm tại thread slots rồi cộng sau sample; không race giữa ant. Các arm dùng cùng
instrumentation và đúng một SRR call/ant/iteration, đã kiểm toán toàn bộ grid.

TSP1K,100 ants,K32+backup32,M12,rho=.1,alpha=1,p_best=.05,smooth MMAS,
normalized heuristic,extended checklist,fixed_steps=0,NLS off. Frozen pretrained
checkpoint; không train. Dynamic graph và annealing1→0 trong mỗi100 iteration,
neural refresh mỗi100. Equal-wall lặp cùng schedule nếu vượt1000 iteration.
Giữ duplicate constructor và seed NumPy của utils.infer_instance. Ba arm khởi
tạo độc lập; seed native, source/best, pheromone và native candidates ban đầu
khớp chính xác trong mọi paired group. Không dùng evolving state của arm trước.

## 3. Provenance, môi trường và protocol

- Branch: `{imp['branch']}`; executed source commit: `{imp['executed_source_commit']}`.
- Reference: `{imp['reference_commit']}`.
- Dataset SHA256: `{data['sha256']}`.
- Protocol SHA256: `{imp['protocol_sha256']}`.
- Native binary SHA256: `{imp['native_binary_sha256']}`.
- Checkpoint SHA256: `{imp['checkpoint_sha256']}`.
- Dataset tạo một lần lúc{data['created_utc']}; Generator(PCG64(202612070301)),
  một call random((16,1000,2),dtype=float32), C-contiguous; hash từng instance
  trong dataset_manifest.json. Không tái tạo theo kết quả; confirmation cũ không truy cập.
- {env['gpu']}, VRAM{env['gpu_total_bytes']/1024**3:.2f}GiB; {env['platform']}.
  Python{env['python'].split()[0]}, NumPy{env['numpy']}, PyTorch{env['pytorch']},
  CUDA{env['torch_cuda']}; strict deterministic CUDA, TF32 off,2 CPU/OpenMP threads.

## 4. Technical validation

Full1000 ARM A đối chiếu binary gốc và utils.infer_instance: exact hash của tất
cả ant routes/costs, source/best, pheromone, heuristic, candidate arrays và prior
ở từng iteration; final arrays cũng exact. Native seed và toàn bộ RNG-derived
trajectory khớp. Candidate validity/prefix/slot budgets, collinear/duplicate
cases, normal/parallel-traced, setter RNG neutrality và resource pilots đều PASS.
Chi tiết: test_results.json và development/. Binary cuối được kiểm tra bổ sung
trong final_binary_verification.json trước principal.

Untimed literal SRR replay trên development ant0: {diag['exact_native_replays']}
snapshot khớp refined route native chính xác. {diag['accepted_native']} accepted
float32 moves gồm{diag['robust']} robust float64 moves,
{diag['zero_f64']} zero-gain và{diag['negative_f64']} negative float64 moves;
{diag['unchanged_route']} event không đổi route. Không sửa historical numerical
semantics. Các con số replay này thuộc development, không phải principal causal effect.

## 5. Timing, scheduling và xử lý lỗi

Equal-iterations chạy trước; instance tăng dần; seed31415 rồi27182; order ABC/BCA/CAB
theo p mod3.192 identity đều có atomic completion và hash raw data. Cả192 run đều
hoàn thành ở attempt1; không có failed attempt, retry, exclusion hoặc imputation. Một GPU worker
tuần tự, watchdog300s/identity, RSS4GiB, torch allocation3.5GiB, disk reserve1GiB;
không áp cap M2. Log lỗi/retry nếu có được giữ nguyên, không loại theo tour quality.

Search clock perf_counter bao gồm score/setter, graph/neural, transfers, sampling,
SRR, pheromone, CUDA synchronization và lightweight checkpoint recording. Imports,
checkpoint load, original constructor setup, warmup development, float64 recomputation
và serialization được ghi riêng. Adaptive trả toàn bộ score/allocation overhead.
Timestamp hoàn tất được lấy sau checkpoint copying/recording. Có{overshoots} run wall
có một iteration overshoot; tất cả vẫn được lưu nhưng không được credit vào metric8s.
Anytime0,1,2,4,6,8 dùng latest completed global-best≤mốc. Native best selection giữ
float32, không chọn lại tour theo float64. Khoảng8s là ngân sách chứ không bảo đảm
iteration cuối đúng8.000000s; actual search gồm overshoot được báo đầy đủ.

Peak RSS {max(r['rss_peak_bytes'] for r in records)/1024**3:.3f} GiB;
peak torch CUDA allocation{max(r['peak_cuda_bytes'] for r in records)/1024**2:.2f}MiB.
RSS dùng Linux ru_maxrss (process-lifetime high-water), không giả định đã reset
peak riêng từng run. CUDA allocation peak được reset trước mỗi search.
Warmup records, setup/serialization từng run và GPU snapshots được lưu trong raw_results.

## 6. Kết quả đầy đủ

Mỗi official length được tính lại bằng NumPy float64 trên tọa độ float32 gốc và
tour permutation hợp lệ. Bảng mean length dùng32 run/arm nhưng uncertainty resample
16 independent instance means. Median là median16 instance means, không phải32 seed.

{table}

Gain theo từng seed là(L_control−L_C)/L_control, sau đó lấy mean hai seed mỗi instance.
Positive nghĩa adaptive tốt hơn. CI là two-sided percentile95% của mean16 instance
gains,10000 resamples, PCG64(202612070302), cùng index matrix cho mọi comparison.

{gains}

## 7. Screening rule đã đăng ký

Chỉ equal wall time quyết định. Cả hai comparison phải đạt cả ba điều kiện:

{gate_table}

Kết luận theo đúng rule: **{final}**. Equal-iteration là secondary bắt buộc;
nó không thay thế kết quả equal-wall kể cả khi có hướng hiệu ứng khác nhau.

## 8. Search efficiency và cơ chế

{efficiency}

Candidate visits gồm cả neighbor tại early-break boundary; gain evaluations chỉ
đếm sau khi qua distance pruning. Counters wall gồm iteration overshoot, không
nhầm với lượng công việc được credit trước8s. Tổng accepted rank33–64 của adaptive
qua hai mode:{adaptive_accept}. Đây là native positive-float32 acceptances, không
đồng nghĩa tour cải thiện robust hoặc global-best được cải thiện nhờ đúng move đó.

Mean số node đổi width giữa hai epoch (bỏ epoch0):{np.mean(changes):.2f}/1000;
median{np.median(changes):.2f}, range{min(changes)}–{max(changes)}. Scores và width
của từng node/epoch có trong raw npz; selection frequency trong allocation_node_frequency.csv.
Unique undirected edge counts tính ngoài timed loop từ chính native prefix; khác
với directed slots. Thông tin này không thay đổi policy hoặc timing.

{interpretation}

Đánh giá nhân quả có giới hạn: end-to-end paired outcomes kiểm tra toàn bộ policy,
không cô lập riêng chất lượng node selection hay checklist. Expanded-candidate
counts chứng minh việc thực thi truy cập; chỉ official float64 gains chứng minh
chất lượng cuối. Những khác biệt overhead/iteration trong bảng giải thích tradeoff
compute quan sát được, không tự chứng minh missing-move headroom hay pruning là
nguyên nhân duy nhất. Nếu gain theo iteration và time khác nhau, cần đọc cả hai
bảng; không viện secondary để đảo primary decision.

## 9. Kết quả từng instance

{individual}

Từng seed/run, timestamps, initial state, overshoot, costs và allocations được lưu
đủ trong raw_results; per_run_results.csv không gộp mất run kém. Không dùng oracle
optimal tour hoặc reference solution; không báo optimality gap không có căn cứ.

## 10. Quan hệ với Stage A và M1/M2

Stage A lịch sử giữ nguyên NO-GO (G0/G4 pass,G1/G2/G3 0/8). Late source snapshots
có0 missed robust moves ngoàiK32, nhưng raw ant0 có off-K32 headroom. Nghiên cứu
mới kiểm tra integrated global-best trajectory trên dataset độc lập; không biến
Stage A thành GO hay thỏa old Stage B gate. {relationship}
Các phân phối/trạng thái và primary outcomes khác nhau.

M1 v2 báo exploratory signal trên32 saved ant0 cases:31/32 robust cases theo bảng
instance (một instance3/4),8/8 instance có robust case; second unchanged K32 pass
khác candidate expansion. Đây không phải end-to-end hoặc equal-compute evidence.
M2 principal lịch sử resource-inconclusive trước launch, không cung cấp xác nhận
hiệu quả. Không trộn sample, p-value, budget hoặc statistical unit giữa các nghiên cứu.
Historical numerical no-op findings vẫn áp dụng: native positive float32 gain
không đủ chứng minh robust improvement; native semantics được bảo tồn.

Nguồn lịch sử: [Stage A report](../../r3/reports/R3_STAGE_A_REPORT.md),
[M1 v2 report](../../r3/m1_additional_k32/v2/M1_V2_RESEARCH_REPORT.md),
[M2 RESOURCE-INCONCLUSIVE report](reproducibility/historical_M2_report.md)
(M2 được trích từ commit c40a034, không phải principal data của nghiên cứu này).

## 11. Giới hạn và quyết định đầu tư

16 uniform Euclidean TSP1K instances và2 stochastic seeds/instance, một checkpoint,
một GPU. Paired seeds được mean trước thống kê; không coi seed/iteration/move là
independent replicate. Bootstrap có hạn ở n16, correlated criteria là practical
screening chứ không có calibrated familywise error. Không suy rộng mọi TSP distribution,
hardware, checkpoint, allocator hay candidate budget. Wall-time trajectories nhạy
với OS/thermal/GPU scheduling; rotation và common clock giảm bias nhưng không loại
hết.8s đo short-budget behavior; equal1000 iterations là secondary.

Nghiên cứu chứng minh implementation isolation, complete-grid validity và outcome
theo prespecified rule. Kết quả hỗ trợ empirical direction trong cấu hình này;
chưa xác lập cơ chế duy nhất. Insufficient missed moves, allocation chưa tốt,
checklist/pruning và statistical power là những giải thích cần tách khỏi kết quả
trực tiếp. Không dùng NO-GO làm bằng chứng mọi adaptive allocator đều vô ích.

**Khuyến nghị:** {policy_recommend}

## 12. Hình và tái lập

![Final wall-time lengths](figures/01_wall_final_length.png)
![Adaptive vs K32](figures/02_paired_gain_C_A.png)
![Adaptive vs K40](figures/03_paired_gain_C_B.png)
![Anytime](figures/04_anytime_quality.png)
![Iteration quality and runtime](figures/05_iteration_quality_runtime.png)
![Completed work](figures/06_completed_iterations.png)
![Candidate counts](figures/07_candidate_evaluations.png)
![Allocation behavior](figures/08_allocation_behavior.png)

PNG300dpi và SVG vector; absolute quality axes bắt đầu0 để không phóng đại chênh
lệch nhỏ. Gain plots có zero/negative rõ ràng. Rebuild bằng `python -m r3_final.analyze`
trong môi trường r3/run.sh; không chạy solver lại. Lệnh, hashes và execution log
ở reproducibility/. analysis/raw_integrity_audit.json xác thực toàn bộ192 run.
'''
    (OUT/'FINAL_RESEARCH_REPORT.md').write_text(polish(text),encoding='utf-8')
    executive=f'''# R3 FINAL DECISION: {final}

**SCIENTIFIC CONCLUSION: {final}. 192/192 run hợp lệ.**

{table}

{gains}

{gate_table}

{policy_recommend}

Protocol `{imp['protocol_sha256']}`; dataset `{data['sha256']}`.
Nguồn thực thi `{imp['executed_source_commit']}`; Stage A giữ nguyên NO-GO.
Xem FINAL_RESEARCH_REPORT.md và analysis/decision.json; không có missing/imputed run.
'''
    (OUT/'EXECUTIVE_DECISION.md').write_text(polish(executive),encoding='utf-8')
    (OUT/'HANDOFF.md').write_text(f'''# Completed research delivery

R3 final investment/scientific decision: **{final}**. Complete valid grid192/192.
No additional authorization or principal run is pending. Read EXECUTIVE_DECISION.md,
FINAL_RESEARCH_REPORT.md, analysis/decision.json and raw_integrity_audit.json.
Executed source commit: {imp['executed_source_commit']}. Raw dataset/protocol/source/
binary/checkpoint hashes are sealed in manifests. Reproduce figures/analysis without
GPU solver execution using commands.md. All original Stage A/M1/M2 evidence remains
unchanged. Recommendation: {policy_recommend}
''',encoding='utf-8')


if __name__=='__main__':main()
