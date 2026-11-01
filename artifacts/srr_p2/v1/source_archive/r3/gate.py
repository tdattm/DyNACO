"""Frozen instance-level G0-G4, with explicit undefined ratios."""
import statistics

def ratio(a,b):return a/b if b>0 else None

def evaluate(rows,technical_valid,complete,resource_valid,expected_count=8):
    per=[]
    for row in rows:
        u=ratio(row['geometry_gain'],row['uniform_gain']);r=ratio(row['geometry_gain'],row['random_gain'])
        per.append(dict(**row,ratio_vs_uniform=u,ratio_vs_random=r,G1=row['missed_count']>0,
            G2=row['verified_best_relative_gain']>=1e-4,
            G3=u is not None and r is not None and u>=1.1 and r>=1.1))
    medians={name:(statistics.median([p[name] for p in per]) if len(per)==expected_count and all(p[name] is not None for p in per) else None) for name in ('ratio_vs_uniform','ratio_vs_random')}
    conditions=dict(G0=bool(technical_valid and complete),G1=sum(p['G1'] for p in per)>=6,
        G2=sum(p['G2'] for p in per)>=6,G3=sum(p['G3'] for p in per)>=6 and all(v is not None and v>=1.1 for v in medians.values()),
        G4=bool(resource_valid and complete))
    if not resource_valid:decision='RESOURCE-INCONCLUSIVE'
    elif not technical_valid or not complete or len(rows)!=expected_count:decision='INCONCLUSIVE'
    else:decision='GO' if all(conditions.values()) else 'NO-GO'
    return dict(decision=decision,conditions=conditions,per_instance=per,median_ratios=medians,
                passing_instance_counts={g:sum(p[g] for p in per) for g in ('G1','G2','G3')},
                stage_b_authorized=decision=='GO')
