"""Strict complete JSON-representation comparison; no approximate or dropped fields."""
import copy
import json
import unittest
from r3.common import ROOT

def canonical(value):
    return json.dumps(value,allow_nan=False,sort_keys=True,separators=(',',':'))

def assert_equal(native,saved):
    if canonical(native)!=canonical(saved):
        raise AssertionError('Complete JSON-compatible event structure differs')

class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((ROOT/'artifacts/r3/m0_srr_replay/i0_s31415_t499_events.json').read_text())
    def changed(self,mutation):
        other=copy.deepcopy(self.base);mutation(other)
        with self.assertRaises(AssertionError):assert_equal(other,self.base)
    def test_tuple_list(self):assert_equal({'a':[(1,2)]},{'a':[[1,2]]})
    def test_route(self):self.changed(lambda r:r['route'].__setitem__(0,-1))
    def test_checklist(self):self.changed(lambda r:r['checklist'].reverse())
    def test_gain(self):self.changed(lambda r:r.__setitem__('total_change_f32',r['total_change_f32']+2**-20))
    def test_event_delete(self):self.changed(lambda r:r['events'].pop())
    def test_event_order(self):self.changed(lambda r:r['events'].reverse())
    def test_rank(self):
        def mutate(r):
            e=next(e for e in r['events'] if e['evaluations']);e['evaluations'][0]['rank']+=1
        self.changed(mutate)
    def test_nonfinite(self):
        for v in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError):assert_equal({'v':v},{'v':v})
    def test_complete_record(self):
        native=copy.deepcopy(self.base)
        for e in native['events']:
            if 'route_updates' in e:e['route_updates']=[tuple(u) for u in e['route_updates']]
        assert_equal(native,self.base)
    def test_flags_and_fields(self):
        self.changed(lambda r:r['events'][0].__setitem__('new_field',1))
        self.changed(lambda r:r['events'][0].__setitem__('node',True))

def test_comparator():
    import io
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ComparatorTests))
    return dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),log=stream.getvalue(),native_invocations=0)
