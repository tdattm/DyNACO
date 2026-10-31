"""Explicit paired summaries and right-continuous anytime measurements."""
import statistics


def stats(values):
    values = list(values)
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def cost_at(run, seconds):
    best = run['initial_cost']
    for row in run['iterations']:
        if row['elapsed_seconds'] > seconds:
            break
        best = row['best_so_far']
    return best


def paired(runs, metric='final_cost'):
    blocks = {}
    for r in runs:
        blocks.setdefault((r['n'], r['instance_id'], r['seed']), {})[r['method']] = r
    result = []
    for (n, instance, seed), methods in sorted(blocks.items()):
        if 'event_stagnation' not in methods:
            continue
        event = methods['event_stagnation']
        for name, control in methods.items():
            if name == 'event_stagnation':
                continue
            result.append(dict(n=n, instance_id=instance, seed=seed, control=name,
                               event_value=event[metric], fixed_value=control[metric],
                               difference=event[metric]-control[metric],
                               cost_percent=100*(event['final_cost']/control['final_cost']-1),
                               gnn_call_difference=event['gnn_calls']-control['gnn_calls'],
                               wall_difference=event['wall_seconds']-control['wall_seconds']))
    return result
