import numpy as np
from r1.replay_pilot import prior_change_metrics

def test_prior_change_units_and_row_offset():
    a=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    unchanged=prior_change_metrics(a,a.copy())
    assert all(v==0 for v in unchanged.values())
    shifted=prior_change_metrics(a,a+2)
    assert shifted['prior_change_mean_absolute']==2 and shifted['prior_change_rms']==2
    assert shifted['prior_change_centered_row_rms']==0
    assert np.isclose(shifted['prior_change_relative_l2'],4/np.linalg.norm(a.astype(np.float64)))
