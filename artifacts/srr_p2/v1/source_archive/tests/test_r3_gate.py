from r3.gate import evaluate,ratio

def rows(geometry=1.2):
    return [dict(instance_id=i,geometry_gain=geometry,uniform_gain=1.,random_gain=1.,
                 missed_count=1,verified_best_relative_gain=.0002) for i in range(8)]
def test_all_gates_required():
    assert evaluate(rows(),True,True,True)['decision']=='GO'
    assert evaluate(rows(1.09),True,True,True)['decision']=='NO-GO'
    assert evaluate(rows(),False,True,True)['decision']=='INCONCLUSIVE'
    assert evaluate(rows(),True,False,True)['decision']=='INCONCLUSIVE'
    assert evaluate(rows(),True,True,False)['decision']=='RESOURCE-INCONCLUSIVE'
def test_zero_denominator_is_undefined_and_conservative():
    r=rows();r[0]['uniform_gain']=0
    assert ratio(0,0) is None and ratio(1,0) is None
    assert evaluate(r,True,True,True)['decision']=='NO-GO'
def test_six_instances_and_median_both_required():
    r=rows()
    for item in r[:3]:item['geometry_gain']=1.
    assert evaluate(r,True,True,True)['passing_instance_counts']['G3']==5
    assert evaluate(r,True,True,True)['decision']=='NO-GO'
