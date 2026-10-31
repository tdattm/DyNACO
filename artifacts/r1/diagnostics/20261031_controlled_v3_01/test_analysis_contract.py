"""CPU-only checks: held-out labels must not enter fitting/scaling."""
import copy
import unittest
import numpy as np
from analyze_v3 import design,fit_predict

class AnalysisContract(unittest.TestCase):
    def test_prediction_does_not_use_held_out_targets(self):
        train=[]
        for phase in range(3):
            for age in (25,75):
                for duration in (0,10,100):
                    value=.03*(age==75)-.02*phase+.01*np.log1p(duration)
                    train.append(dict(prior_age_iterations=age,phase=phase,stagnation_iterations=duration,refresh_benefit_raw=value))
        test=[dict(prior_age_iterations=75,phase=2,stagnation_iterations=33,refresh_benefit_raw=-999)]
        a,info=fit_predict(train,test,3)
        altered=copy.deepcopy(test); altered[0]['refresh_benefit_raw']=999999
        b,_=fit_predict(train,altered,3)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(info['rank'],5)
        self.assertAlmostEqual(a[0],.03-.04+.01*np.log1p(33),places=12)

    def test_nested_designs_keep_targets_out(self):
        rows=[dict(prior_age_iterations=25,phase=0,stagnation_iterations=0,refresh_benefit_raw=1e8),dict(prior_age_iterations=75,phase=2,stagnation_iterations=100,refresh_benefit_raw=-1e8)]
        for model,width in ((1,2),(2,4),(3,5)):
            self.assertEqual(design(rows,model).shape,(2,width))
        np.testing.assert_array_equal(design(rows,1),design(rows,3)[:,:2])

if __name__=='__main__': unittest.main()
