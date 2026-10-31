"""Cheap deterministic scheduling from completed iterations and global incumbent."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SchedulerConfig:
    mode: str = 'fixed_s'
    total_iterations: int = 1000
    fixed_s: int = 100
    min_interval: int = 25
    max_interval: int = 200
    stagnation_threshold: int = 50
    initial_refresh: bool = True
    annealing_mode: str = 'no_anneal'
    improvement_tolerance: float = 1e-6

    def __post_init__(self):
        if self.mode not in ('fixed_s', 'event_stagnation'):
            raise ValueError('Unknown scheduling mode')
        if min(self.total_iterations, self.fixed_s, self.min_interval, self.max_interval, self.stagnation_threshold) <= 0:
            raise ValueError('Budgets and intervals must be positive')
        if self.min_interval > self.max_interval:
            raise ValueError('min_interval exceeds max_interval')
        if not math.isfinite(self.improvement_tolerance) or self.improvement_tolerance < 0:
            raise ValueError('Invalid improvement tolerance')
        if self.annealing_mode not in ('no_anneal', 'original_fixed'):
            raise ValueError('Unknown annealing mode')
        if self.mode == 'event_stagnation' and self.annealing_mode != 'no_anneal':
            raise ValueError('Event policy requires no_anneal to avoid annealing confound')


class GuidanceScheduler:
    """decide(j, best) precedes iteration j; observe(j+1, best) follows its update.

    Stagnation starts at completed iteration 0 and resets only when the global
    incumbent beats the last significant incumbent by > absolute tolerance.
    Refreshing guidance does not reset stagnation. Maximum interval takes priority.
    """
    def __init__(self, config=SchedulerConfig()):
        self.config = config
        self.last_refresh = 0
        self.last_improvement = 0
        self.global_best = math.inf
        self.significant_best = math.inf
        self.completed_iterations = 0
        self.refresh_count = 0

    def observe(self, completed_iterations, global_best):
        if not math.isfinite(global_best):
            raise ValueError('Global incumbent must be finite')
        if completed_iterations < self.completed_iterations:
            raise ValueError('Iterations must be monotone')
        if completed_iterations > self.config.total_iterations:
            raise ValueError('Iteration budget exceeded')
        self.completed_iterations = completed_iterations
        self.global_best = min(self.global_best, float(global_best))
        if self.global_best < self.significant_best - self.config.improvement_tolerance:
            self.significant_best = self.global_best
            self.last_improvement = completed_iterations

    def decide(self, iteration, global_best):
        if iteration < 0:
            raise ValueError('Negative iteration')
        if iteration >= self.config.total_iterations:
            return None
        self.observe(iteration, global_best)
        c = self.config
        since_refresh = iteration - self.last_refresh
        reason = None
        if iteration == 0 and c.initial_refresh:
            reason = 'initial'
        elif c.mode == 'fixed_s':
            if iteration > 0 and iteration % c.fixed_s == 0:
                reason = 'fixed_interval'
        elif since_refresh >= c.max_interval:
            reason = 'max_interval'
        elif since_refresh >= c.min_interval and iteration - self.last_improvement >= c.stagnation_threshold:
            reason = 'stagnation'
        if reason is not None:
            self.last_refresh = iteration
            self.refresh_count += 1
        return reason
