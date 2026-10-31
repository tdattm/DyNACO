from r1.metrics import paired, cost_at, stats


def test_wall_budget_is_right_continuous_and_respects_initial_state():
    r = dict(initial_cost=10., iterations=[dict(elapsed_seconds=.2, best_so_far=9.), dict(elapsed_seconds=.5, best_so_far=8.)])
    assert cost_at(r, .1)==10. and cost_at(r, .2)==9. and cost_at(r, .49)==9. and cost_at(r, .5)==8.


def test_pairs_use_same_instance_and_seed():
    base = dict(n=100, instance_id='a', seed=1, gnn_calls=10, wall_seconds=2.)
    rows = [dict(base, method='fixed_s100', final_cost=10.), dict(base, method='event_stagnation', final_cost=9.),
            dict(base, instance_id='unpaired', method='fixed_s100', final_cost=100.)]
    differences = paired(rows)
    assert len(differences)==1 and differences[0]['difference']==-1.
    assert abs(differences[0]['cost_percent']+10.)<1e-10
    assert stats([1,2,3])['median']==2
