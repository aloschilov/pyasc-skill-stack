"""Pairing and qualification failures; no Docker, compiler, model or NPU calls."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('comparison',ROOT/'scripts/simulator_comparison.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def fixtures(self):
        intent={'warmup':1,'samples':3,'input_hashes':{'candidate.py':'1'*64,'reference.asc':'2'*64}}
        reports={}
        for n,ticks in [('reference',30),('candidate',20),('fusion',10)]:
            def sample(i,t):return {'index':i,'tick_before':123,'tick_after':123+t,'ticks':t,'launches':1,'numerical_checks':3}
            reports[n]={'status':'passed','variant':n,'task':comparison.TASK,'metric':comparison.METRIC,
                        'input_hashes':intent['input_hashes'],'runtime_hashes':{'model.so':'3'*64,'@loaded_runtime_wrapper':'5'*64},
                        'binary_sha256':'4'*64,'commands':[{'kind':'compile','argv':['compiler']}],
                        'warmup':[sample(0,9999)],'samples':[sample(i,ticks+i) for i in range(3)],
                        'control':{'status':'rejected','launch_attempts':1,'launches':0},
                        'requested_options':{'vf_fusion':n=='fusion','opt_level':3},
                        'effective_options':{'vf_fusion':n=='fusion','opt_level':3,'vector_length':256}}
        return copy.deepcopy(intent),copy.deepcopy(reports)

    def test_warmup_excluded_and_ratios_use_matching_sample_indices(self):
        i,r=self.fixtures();result=comparison.summarize(i,r)
        self.assertEqual(result['median_ticks'],{'reference':31,'candidate':21,'fusion':11})
        self.assertEqual(result['reference_over_candidate'],31/21)
        self.assertEqual(result['paired_sample_ratios']['candidate_over_fusion'],[20/10,21/11,22/12])
        self.assertNotIn('speedup',result)

    def test_failed_blocked_static_only_and_missing_variant_never_pass(self):
        for status in ('failed','blocked','compiled','not-run'):
            i,r=self.fixtures();r['reference']['status']=status
            with self.subTest(status=status),self.assertRaises(ValueError):comparison.summarize(i,r)
        i,r=self.fixtures();del r['fusion']
        with self.assertRaisesRegex(ValueError,'Missing'):comparison.summarize(i,r)

    def test_cannot_pair_different_task_input_or_runtime(self):
        for field,value in [('task',dict(comparison.TASK,blocks=1)),('input_hashes',{'candidate.py':'5'*64}),
                            ('runtime_hashes',{'model.so':'6'*64}),('metric','wall_time'),('variant','reference')]:
            i,r=self.fixtures();r['candidate'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):comparison.summarize(i,r)

    def test_every_sample_requires_launch_numerics_and_valid_tick_delta(self):
        changes=[('launches',0),('launches',2),('numerical_checks',0),('index',9),('ticks',0),
                 ('ticks',-1),('ticks',float('nan')),('ticks',float('inf')),('ticks',True),('tick_after',0),('tick_before',-1)]
        for field,value in changes:
            i,r=self.fixtures();r['candidate']['samples'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):comparison.summarize(i,r)
        i,r=self.fixtures();r['candidate']['samples'].pop()
        with self.assertRaisesRegex(ValueError,'Missing measurements'):comparison.summarize(i,r)

    def test_suppression_must_reach_launch_and_reject_numerics(self):
        for control in ({'status':'skipped'},{'status':'rejected','launch_attempts':0,'launches':0},
                        {'status':'passed','launch_attempts':1,'launches':0},
                        {'status':'rejected','launch_attempts':1,'launches':1}):
            i,r=self.fixtures();r['fusion']['control']=control
            with self.assertRaisesRegex(ValueError,'suppression'):comparison.summarize(i,r)

    def test_fusion_is_the_only_requested_and_effective_option_change(self):
        for key in ('requested_options','effective_options'):
            for change in ({'opt_level':2},{'vf_fusion':False},{'new_option':True}):
                i,r=self.fixtures();r['fusion'][key].update(change)
                with self.subTest(key=key,change=change),self.assertRaisesRegex(ValueError,'Fusion'):comparison.summarize(i,r)

    def test_missing_hashes_or_commands_rejected(self):
        for key,value in [('runtime_hashes',{}),('runtime_hashes',{'model.so':'3'*64}),
                          ('binary_sha256','unknown'),('binary_sha256',None),('commands',[])]:
            i,r=self.fixtures();r['reference'][key]=value
            with self.assertRaisesRegex(ValueError,'evidence'):comparison.summarize(i,r)

    def test_reference_extraction_rejects_unpinned_source(self):
        with self.assertRaisesRegex(ValueError,'reference bytes changed'):
            comparison.extract_reference('int32_t main() {}')


if __name__=='__main__':unittest.main()
