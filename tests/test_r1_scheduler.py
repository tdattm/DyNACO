from dataclasses import replace
import pytest
from r1.scheduler import GuidanceScheduler, SchedulerConfig


def simulate(config, costs=None):
    s = GuidanceScheduler(config)
    events = []
    for i in range(config.total_iterations):
        cost = 10. if costs is None else costs(i)
        reason = s.decide(i, cost)
        if reason:
            events.append((i, reason))
        s.observe(i + 1, cost)
    assert s.decide(config.total_iterations, 10.) is None
    return s, events


def test_fixed_boundaries_partial_tail_and_count():
    s, events = simulate(SchedulerConfig(total_iterations=107, fixed_s=25))
    assert events == [(0, 'initial'), (25, 'fixed_interval'), (50, 'fixed_interval'), (75, 'fixed_interval'), (100, 'fixed_interval')]
    assert s.completed_iterations == 107 and s.refresh_count == 5


def test_no_final_refresh():
    s, events = simulate(SchedulerConfig(total_iterations=100, fixed_s=25))
    assert [i for i, _ in events] == [0, 25, 50, 75]
    assert s.refresh_count == 4


def test_initial_refresh_disabled():
    _, events = simulate(SchedulerConfig(total_iterations=51, fixed_s=25, initial_refresh=False))
    assert events == [(25, 'fixed_interval'), (50, 'fixed_interval')]


def test_stagnation_and_minimum_interval():
    s, events = simulate(SchedulerConfig(mode='event_stagnation', total_iterations=101))
    assert events == [(0, 'initial'), (50, 'stagnation'), (75, 'stagnation'), (100, 'stagnation')]
    assert all(b[0]-a[0] >= 25 for a, b in zip(events, events[1:]))
    assert s.last_improvement == 0


def test_maximum_safeguard_with_continued_improvements():
    _, events = simulate(SchedulerConfig(mode='event_stagnation', total_iterations=451), lambda i: 10. - i * .001)
    assert events == [(0, 'initial'), (200, 'max_interval'), (400, 'max_interval')]


def test_stagnation_reset_by_global_improvement():
    s = GuidanceScheduler(SchedulerConfig(mode='event_stagnation'))
    assert s.decide(0, 10.) == 'initial'
    s.observe(49, 9.)
    assert s.decide(50, 9.5) is None  # A worsening iteration/source cost cannot replace global best.
    assert s.global_best == 9. and s.last_improvement == 49
    assert s.decide(98, 9.) is None
    assert s.decide(99, 9.) == 'stagnation'


def test_tolerance_and_maximum_priority():
    s = GuidanceScheduler(SchedulerConfig(mode='event_stagnation', max_interval=50))
    s.decide(0, 10.)
    s.observe(40, 10. - 5e-7)
    assert s.last_improvement == 0
    assert s.decide(50, 10. - 5e-7) == 'max_interval'


@pytest.mark.parametrize('kwargs', [dict(min_interval=201), dict(fixed_s=0), dict(total_iterations=0),
                                    dict(mode='bad'), dict(mode='event_stagnation', annealing_mode='original_fixed'),
                                    dict(improvement_tolerance=-1)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SchedulerConfig(**kwargs)
