"""The wheel and CaModel import this same launch planner."""
import sys
from copy import deepcopy
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'integrations/cannbench/ci/swiglu_performance'))
from geometry import DTYPES, all_specs, layout, validate


def band(end,route,tile):
    return dict(through=end,route=route,tile=tile,unroll=2,
                block_cap=5,tiles_per_block=2,unroll_waves=3)


class Policy(unittest.TestCase):
    def setUp(self):
        self.policy={d:[band(8,'slab',32),band(63,'rows',16),band(None,'flat',128)] for d in DTYPES}

    def test_all_dispatched_specializations_are_compiled(self):
        self.policy['float32'][1]['width_step_bytes']=32
        specs=set(all_specs(self.policy))
        for dtype in DTYPES:
            for half in (1,2,7,8,9,48,63,64,129,12345):
                for rows in (1,8,81,528):
                    for cores in (1,3,72):
                        _,_,_,route,const,blocks=layout((rows,half*2),1,dtype,self.policy,cores)
                        self.assertIn((dtype,route,const),specs)
                        self.assertTrue(1<=blocks<=cores)
                        if route!='_swiglu_pair_groups':
                            self.assertLessEqual(blocks,5)
        self.assertEqual(layout((528,100),1,'float32',self.policy,72)[-2:],((16,56,2),5))
        self.assertEqual(layout((81,100),1,'float32',self.policy,72)[-2:],((16,56,1),3))
        self.assertEqual(layout((0,100),1,'float16',self.policy,4)[-1],0)

    def test_rejects_unbounded_or_case_specific_policy(self):
        for index,changes in [(0,{'case':12}),(0,{'through':100}),(1,{'through':5}),
                              (-1,{'through':1000}),(1,{'block_cap':0}),(0,{'tile':True})]:
            policy=deepcopy(self.policy)
            policy['float16'][index].update(changes)
            with self.assertRaises(ValueError):
                validate(policy)
