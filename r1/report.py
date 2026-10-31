"""Generate summaries and figures exclusively from persisted successful runs."""
import csv
import json
from pathlib import Path

import numpy as np
from r1.metrics import stats, paired, cost_at


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(directory):
    out = Path(directory)
    raw_rows = list(csv.DictReader((out/'runs.csv').open()))
    failures = [r for r in raw_rows if r['status'] != 'ok']
    runs = [json.loads((out/'runs'/f'{r["run_id"]}.json').read_text()) for r in raw_rows if r['status']=='ok']
    sizes = sorted({r['n'] for r in runs})
    methods = sorted({r['method'] for r in runs})
    summary = []
    for n in sizes:
        for method in methods:
            selected = [r for r in runs if r['n']==n and r['method']==method]
            row = dict(n=n, method=method, count=len(selected))
            for key in ('final_cost', 'wall_seconds', 'gnn_calls', 'gnn_seconds', 'graph_seconds', 'transfer_seconds', 'decision_seconds', 'peak_gpu_bytes'):
                row.update({f'{key}_{name}': value for name, value in stats(r[key] for r in selected).items() if name!='count'})
            summary.append(row)
    csv_write(out/'summary.csv', summary)
    differences = paired(runs)
    csv_write(out/'paired_differences.csv', differences)
    pair_summary = []
    for n in sizes:
        for method in methods:
            if method == 'event_stagnation':
                continue
            rows = [d for d in differences if d['n']==n and d['control']==method]
            pair_summary.append(dict(n=n, control=method, **stats(r['cost_percent'] for r in rows),
                                     mean_wall_difference=stats(r['wall_difference'] for r in rows)['mean']))
    csv_write(out/'paired_summary.csv', pair_summary)
    # Predefined budgets are included only within the common observed horizon.
    wall = []
    for n in sizes:
        selected = [r for r in runs if r['n']==n]
        horizon = min(r['wall_seconds'] for r in selected)
        for budget in (.1, .25, .5, 1., 2., 5.):
            if budget <= horizon:
                for r in selected:
                    wall.append(dict(n=n, run_id=r['run_id'], method=r['method'], instance_id=r['instance_id'], seed=r['seed'], budget_seconds=budget, best_cost=cost_at(r, budget)))
    csv_write(out/'wall_budget_results.csv', wall)
    call_matches = []
    for event in [r for r in runs if r['method']=='event_stagnation']:
        controls = [r for r in runs if r['method']!='event_stagnation' and (r['n'],r['instance_id'],r['seed'])==(event['n'],event['instance_id'],event['seed'])]
        if not controls:
            continue
        nearest = min(controls, key=lambda r: abs(r['gnn_calls']-event['gnn_calls']))
        relative = abs(nearest['gnn_calls']-event['gnn_calls']) / max(1, event['gnn_calls'])
        call_matches.append(dict(n=event['n'], instance_id=event['instance_id'], seed=event['seed'], control=nearest['method'], event_calls=event['gnn_calls'], fixed_calls=nearest['gnn_calls'], relative_call_difference=relative, comparable_within_10pct=relative<=.1, event_minus_fixed_cost=event['final_cost']-nearest['final_cost']))
    csv_write(out/'gnn_budget_matches.csv', call_matches)
    result = dict(schema_version=1, successful_runs=len(runs), failed_runs=len(failures), failures=failures,
                  summary=summary, paired_cost_percent=pair_summary, gnn_budget_matches=call_matches,
                  interpretation='Exploratory descriptive evidence; four instance-seed pairs at each size represent only two independent instances. No significance test or optimum/gap claim.',
                  wall_budget_protocol='Predefined budgets .1,.25,.5,1,2,5 seconds; only budgets <= minimum observed run wall time at a size. Right-continuous incumbent, initialized from native initial tour at time zero.')
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if not runs:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = out/'figures'
    figures.mkdir(exist_ok=False)
    colors = {method: f'C{i}' for i, method in enumerate(methods)}

    def panels():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for ax, n in zip(axes.flat, sizes):
            ax.set_title(f'TSP-{n} (2 instances × 2 solver seeds)')
            ax.set_ylabel('Euclidean tour length')
            ax.grid(alpha=.2)
        return fig, list(axes.flat)

    for axis_type in ('iterations', 'wall_time'):
        fig, axes = panels()
        for ax, n in zip(axes, sizes):
            all_n = [r for r in runs if r['n']==n]
            horizon = min(r['wall_seconds'] for r in all_n)
            for method in methods:
                selected = [r for r in all_n if r['method']==method]
                if not selected:
                    continue
                if axis_type=='iterations':
                    x = np.arange(1, len(selected[0]['iterations'])+1)
                    y = np.mean([[v['best_so_far'] for v in r['iterations']] for r in selected], axis=0)
                else:
                    x = np.linspace(0, horizon, 201)
                    y = np.mean([[cost_at(r, t) for t in x] for r in selected], axis=0)
                ax.plot(x, y, label=method, color=colors[method], linewidth=1.4)
            ax.set_xlabel('Completed ACO iterations' if axis_type=='iterations' else 'Synchronized search wall time (s)')
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figures/f'anytime_{axis_type}.png', dpi=160); plt.close(fig)
    for key, label in [('gnn_calls', 'Neural forward calls'), ('wall_seconds', 'Search wall time (s)')]:
        fig, axes = panels()
        for ax, n in zip(axes, sizes):
            for method in methods:
                selected = [r for r in runs if r['n']==n and r['method']==method]
                if not selected:
                    continue
                ax.scatter([r[key] for r in selected], [r['final_cost'] for r in selected], color=colors[method], alpha=.55, s=24)
                ax.scatter(np.mean([r[key] for r in selected]), np.mean([r['final_cost'] for r in selected]), color=colors[method], marker='X', s=80, label=method)
            ax.set_xlabel(label)
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figures/f'quality_vs_{key}.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    representative = [r for r in runs if r['n']==max(sizes) and r['seed']==1234 and r['instance_id'].endswith('-000')]
    reason_colors = dict(initial='black', fixed_interval='C0', stagnation='C1', max_interval='C2')
    for i, r in enumerate(sorted(representative, key=lambda r:r['method'])):
        for reason, color in reason_colors.items():
            events = [e['iteration'] for e in r['refresh_events'] if e['reason']==reason]
            ax.scatter(events, [i]*len(events), marker='|', s=180, color=color, label=reason if i==0 else None)
    ax.set_yticks(range(len(representative)), [r['method'] for r in sorted(representative, key=lambda r:r['method'])])
    ax.set_xlabel('Refresh before absolute ACO iteration'); ax.set_title(f'TSP-{max(sizes)}, instance 000, solver seed 1234')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figures/'refresh_timeline.png', dpi=160); plt.close(fig)
    fig, axes = panels()
    for ax, n in zip(axes, sizes):
        for m, method in enumerate(methods):
            selected = sorted([r for r in runs if r['n']==n and r['method']==method], key=lambda r:(r['instance_id'],r['seed']))
            ax.scatter([m + .05*(j-1.5) for j in range(len(selected))], [r['final_cost'] for r in selected], color=colors[method], s=30)
        ax.set_xticks(range(len(methods)), methods, rotation=25, ha='right', fontsize=8)
    fig.tight_layout(); fig.savefig(figures/'instance_seed_variability.png', dpi=160); plt.close(fig)
    print(f'Report generated: {out}, {len(runs)} successful, {len(failures)} failed', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('directory')
    generate(p.parse_args().directory)
