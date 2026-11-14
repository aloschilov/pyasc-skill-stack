"""The wheel and CaModel import this same launch planner."""
import io
import sys
from copy import deepcopy
from pathlib import Path
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'integrations/cannbench/ci/swiglu_performance'))
from geometry import DTYPES, all_specs, full_row_probes, layout, validate
from generate import compiler_overflow, pre_submission_overflow


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

    def test_three_stages_require_an_exact_fp32_row_tile(self):
        row_band = self.policy['float32'][1]
        row_band.update(through=64, tile=2, unroll=3, unroll_waves=8, block_cap=None, tiles_per_block=1)
        shapes = full_row_probes(self.policy)
        self.assertEqual(shapes,[(50,126),(50,128),(51,128)])
        self.assertEqual([layout(s,1,'float32',self.policy,3)[-2][-1] for s in shapes],[1,3,1])
        compiled = set(all_specs(self.policy))
        for rows, half, expected in [(48,64,3),(50,64,3),(49,64,1),(48,63,1),(46,64,1)]:
            _,_,_,route,const,blocks = layout((rows,2*half),1,'float32',self.policy,3)
            self.assertEqual((const[-1],blocks),(expected,3))
            self.assertIn(('float32',route,const),compiled)
        for dtype,index in [('float16',1),('bfloat16',1),('float32',0),('float32',2)]:
            invalid = deepcopy(self.policy)
            invalid[dtype][index]['unroll'] = 3
            with self.assertRaises(ValueError):
                validate(invalid)

    def test_rejects_unbounded_or_case_specific_policy(self):
        for index,changes in [(0,{'case':12}),(0,{'through':100}),(1,{'through':5}),
                              (-1,{'through':1000}),(1,{'block_cap':0}),(0,{'tile':True})]:
            policy=deepcopy(self.policy)
            policy['float16'][index].update(changes)
            with self.assertRaises(ValueError):
                validate(policy)

    def test_policy_repair_requires_failed_camodel_and_skipped_upload(self):
        job = {'name':'swiglu-performance / evaluate', 'id':15, 'conclusion':'failure',
               'steps':[{'name':'Run numerical CaModel controls and lower the same bytes to AscendC',
                         'conclusion':'failure'},
                        {'name':'Submit once and collect service timings','conclusion':'skipped'}]}
        self.assertEqual(pre_submission_overflow([job]),job['name'])
        for steps in (job['steps'][:1], [job['steps'][0],dict(job['steps'][1],conclusion='failure')],
                      [dict(job['steps'][0],conclusion='success'),job['steps'][1]]):
            with self.assertRaises(ValueError):
                pre_submission_overflow([dict(job,steps=steps)])

    def test_ub_diagnostic_must_come_from_the_evaluation(self):
        message = 'UB overflow: 253952 bytes are available, 344064 bytes are used.'
        def logs(evaluation):
            data = io.BytesIO()
            with zipfile.ZipFile(data,'w') as archive:
                archive.writestr('3_swiglu-performance _ evaluate.txt',evaluation)
                archive.writestr('1_pr-gate.txt',message)
            return data.getvalue()
        self.assertEqual(compiler_overflow(logs('RuntimeError: '+message),'swiglu-performance / evaluate'),message)
        with self.assertRaises(ValueError):
            compiler_overflow(logs('Numerical mismatch'),'swiglu-performance / evaluate')
