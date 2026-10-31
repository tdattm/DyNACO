"""Single intervention after deterministic replay, without native snapshot restore."""
from dataclasses import asdict
import hashlib
import time
import numpy as np
from r1.engine import RunConfig, run
from r1.scheduler import SchedulerConfig

def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()

class ProbeSchedule:
    def __init__(self,t,refresh,total):
        if t<=0 or t%100==0 or total<=t: raise ValueError('Probe must lie inside a macro-step')
        self.t=t; self.refresh=refresh
        self.config=SchedulerConfig(total_iterations=total)
        self.last_improvement=0
    def decide(self,i,best):
        if i<self.t and i%100==0: return 'initial' if i==0 else 'fixed_prefix'
        if i==self.t and self.refresh: return 'probe_refresh'
        return None
    def observe(self,i,best): pass

def branch(coords,model,seed,t,refresh,horizon=100,capture_priors=False):
    state={}; last=0; significant=float('inf'); solution_hashes=[]; guidance_hashes=[]; final_solver=None
    prior_pair={}
    def sampled(i,costs,tours,prior):
        solution_hashes.append(digest(costs)+':'+digest(tours))
        guidance_hashes.append(digest(prior))
        if capture_priors and i==t:
            prior_pair['new']=np.ascontiguousarray(prior).copy()
    def observe(i,solver,prior,elapsed):
        nonlocal last, significant, final_solver
        final_solver=solver
        if solver.best_cost<significant-1e-6:
            significant=solver.best_cost; last=i
        if i==t:
            if capture_priors:
                prior_pair['old']=np.ascontiguousarray(prior.detach().cpu().numpy()).copy()
            state.update(iteration=i,incumbent=solver.best_cost,source_cost=solver.source_cost,
                stagnation_iterations=i-last,prior_age_iterations=i%100,
                pheromone_sha256=digest(solver.pheromone_sparse_np),
                best_tour_sha256=digest(solver.best_route),source_tour_sha256=digest(solver.source_route),
                candidate_sha256=digest(solver.nn_list),backup_sha256=digest(solver.backup_list),
                heuristic_sha256=digest(solver.heuristic_sparse_np),cached_prior_sha256=digest(prior.cpu().numpy()),
                tau_min=solver.tau_min,tau_max=solver.tau_max,boundary_elapsed_seconds=elapsed)
    result=run(coords,model,RunConfig(total_iterations=t+horizon,seed=seed),
               scheduler=ProbeSchedule(t,refresh,t+horizon),observer=observe,sample_observer=sampled)
    result['iteration_solution_hashes']=solution_hashes
    result['native_guidance_hashes']=guidance_hashes
    result['final_search_state_hashes']={name:digest(getattr(final_solver,name)) for name in ('pheromone_sparse_np','best_route','source_route','nn_list','backup_list')}
    result['branch_state']=state
    result['suffix_wall_seconds']=result['wall_seconds']-state['boundary_elapsed_seconds']
    if capture_priors:
        result['intervention_priors']=prior_pair
    return result

def assert_prefix(a,b,t):
    assert a['config']==b['config']
    assert a['native_guidance_hashes'][:t]==b['native_guidance_hashes'][:t], 'Native input prior mismatch'
    assert a['iteration_solution_hashes'][:t]==b['iteration_solution_hashes'][:t], 'All-ant solution prefix mismatch'
    assert a['trace'][:t]==b['trace'][:t], 'Prefix sampled-cost mismatch'
    assert [r['best_so_far'] for r in a['iterations'][:t]]==[r['best_so_far'] for r in b['iterations'][:t]]
    for key,value in a['branch_state'].items():
        if key!='boundary_elapsed_seconds': assert value==b['branch_state'][key], key
    assert a['instance_seed']==b['instance_seed']

def observations(refresh,hold,t):
    assert_prefix(refresh,hold,t)
    assert refresh['gnn_calls']==hold['gnn_calls']+1
    assert [e['iteration'] for e in refresh['refresh_events']]==[e['iteration'] for e in hold['refresh_events']]+[t]
    output=[]
    for horizon in (25,50,100):
        a=refresh['iterations'][t+horizon-1]['best_so_far']; b=hold['iterations'][t+horizon-1]['best_so_far']
        incumbent=hold['branch_state']['incumbent']
        output.append(dict(probe_iteration=t,horizon_iterations=horizon,
            prior_age_iterations=t%100,phase=t//300,stagnation_iterations=hold['branch_state']['stagnation_iterations'],
            shared_incumbent=incumbent,refresh_cost=a,hold_cost=b,
            refresh_improvement_raw=incumbent-a,hold_improvement_raw=incumbent-b,
            refresh_benefit_raw=b-a,refresh_benefit_percent=100*(b-a)/b,
            additional_gnn_calls=1,marginal_gnn_seconds=refresh['refresh_events'][-1]['gnn_seconds'],
            suffix_wall_difference_seconds=(refresh['iterations'][t+horizon-1]['elapsed_seconds']-refresh['branch_state']['boundary_elapsed_seconds'])-(hold['iterations'][t+horizon-1]['elapsed_seconds']-hold['branch_state']['boundary_elapsed_seconds']),
            total_replay_wall_difference_seconds=refresh['wall_seconds']-hold['wall_seconds']))
    return output
