"""Resource pilot for the final scanner build, before principal data exists."""
import resource
import time
import numpy as np
from r3.common import ROOT,save_json,strict_cuda
from r3.state_capture import solver
from r3.diagnostics import diagnose

def main():
    strict_cuda();records=[]
    for n in (100,200,500,1000):
        with np.load(ROOT/f'artifacts/r3/baseline/verification_v1/n{n}_smoke_final.npz') as state:
            x=state['coordinates'];tour=state['source_route']
        owner=solver(x,n_ants=1,k=64);nn=owner.nn_list.copy()
        start=time.perf_counter();diagnostic=diagnose(x,tour,nn,20261102+n)
        record=dict(n=n,wall_seconds=time.perf_counter()-start,scanner_seconds=diagnostic['scan_wall_seconds'],
            enumeration_seconds=diagnostic['enumeration_and_bounded_sample_seconds'],
            classification_seconds=diagnostic['robust_move_classification_seconds'],
            gain_verification_seconds=diagnostic['gain_verification_seconds'],
            rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        records.append(record);print(record,flush=True)
    # 192 scans (128 frozen source/best + 64 raw), pessimistic 2x pilot scan cost,
    # plus 16 full trajectories at 2x measured pilot baseline time.
    import json
    baseline=json.loads((ROOT/'artifacts/r3/baseline/verification_v1/full1000_verification.json').read_text())
    estimate=192*records[-1]['wall_seconds']*2+16*baseline['timing']['search_wall_seconds']*2
    if estimate>4320:raise TimeoutError('pilot exceeds frozen estimate cap')
    save_json(ROOT/'artifacts/r3/pilots/scanner_final_build_v1.json',dict(records=records,
        estimated_principal_seconds=estimate,estimate_method='2x 192 complete snapshot diagnoses +2x16 complete trajectories',
        resource_precheck='PASS',principal_data_inspected=False))
if __name__=='__main__':main()
