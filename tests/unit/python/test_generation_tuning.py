"""Synthetic measurement and native-role receipts; no model or device calls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/tools'))
import generation_tuning as tuning
import generation_review as review
from generation_executor import Blocked
from generation_plan import canonical, digest
import test_generation_review as review_fixtures


class TunerHandoffs(unittest.TestCase):
    def setUp(self):
        self.fixture=review_fixtures.ReviewReceipts();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root/'tuning';self.incumbent=self.fixture.run
        old=review.read(self.incumbent/'state.json')['cycles'][-1]
        self.receipt={'schema_version':1,'kind':'caller-device-measurement','backend':'NPU',
            'candidate_sha256':old['candidate_sha256'],'task_sha256':old['binding']['task_sha256'],
            'pyasc_revision':self.fixture.plan['common']['runtime']['source_revision'],
            'hardware':'offline-device-fixture','method':'offline-median-10-repeats',
            'measured_at':'2026-01-01T00:00:00+00:00','source_url':'https://example.invalid/immutable-test-evidence',
            'source_sha256':'a'*64,'cases':[{'id':f'seed-{s}','elapsed_us':10.0,'samples':10,
                'numerically_correct':True,'device_kernels':['offline_kernel']} for s in (0,1,7)]}
        self.path=self.fixture.root/'device.json';review.save(self.path,self.receipt)
        self.plan=tuning.prepare(self.root,self.incumbent,self.path)

    def propose(self):
        with patch.dict(os.environ,OPENCODE_API_KEY='offline',OPENCODE_BASE_URL='http://127.0.0.1'), \
             patch.object(review,'command',side_effect=self.fixture.command), \
             patch.object(review,'run_session',side_effect=self.fixture.client), \
             patch.object(review,'qualify',side_effect=self.fixture.verifier):
            return tuning.propose(self.root,self.plan['plan_sha256'],self.fixture.root/'shared-state')

    def measured_proposal(self, seconds=9.0):
        state=self.propose();self.assertEqual(state['status'],'awaiting-measurement')
        after=deepcopy(self.receipt);after.update(candidate_sha256=state['proposed_candidate_sha256'],source_sha256='b'*64)
        for row in after['cases']:row['elapsed_us']=seconds
        path=self.fixture.root/'proposed-device.json';review.save(path,after)
        return path

    def test_plan_retains_verified_incumbent_and_binds_one_proposal(self):
        self.assertEqual(self.plan['tuning']['binding']['candidate_sha256'],self.receipt['candidate_sha256'])
        self.assertEqual(self.plan['tuning']['budget']['proposals'],1)
        self.assertEqual(tuning.verify(self.root)[0],self.plan)
        self.assertFalse(any(p.name=='private' for p in (self.root/'incumbent').rglob('*')))
        with self.assertRaisesRegex(ValueError,'overwrite'):tuning.prepare(self.root,self.incumbent,self.path)
        with self.assertRaisesRegex(ValueError,'approval'):tuning.propose(self.root,'wrong',self.fixture.root/'state')

    def test_tuner_gets_immutable_handoff_then_fresh_review_and_qualification(self):
        result=self.propose();self.assertEqual(result['status'],'awaiting-measurement')
        self.assertEqual(result['selected_candidate_sha256'],self.receipt['candidate_sha256'])
        self.assertNotEqual(result['proposed_candidate_sha256'],self.receipt['candidate_sha256'])
        row=review.read(self.root/'proposal/state.json')['cycles'][0]
        self.assertEqual(row['writer']['role'],'tuner');self.assertNotEqual(row['writer']['session_id'],row['reviewer']['session_id'])
        self.assertIn('tuning-context.json',row['writer']['read_inputs'])
        context=review.read(self.root/'proposal/cycle-1/writer/context/inputs/tuning-context.json')
        self.assertEqual(context,{'measurement':self.plan['tuning']['measurement'],'budget':self.plan['tuning']['budget']})
        # One auxiliary request, sequential skill/input reads, two writes and a
        # final completion must fit without assuming parallel tool calls.
        minimum_requests=1+len(self.plan['common']['skills'])+1+len(tuning.handoff_inputs(self.plan['tuning']))+2+1
        self.assertLessEqual(minimum_requests,self.plan['spec']['writer']['rounds_per_attempt'])
        self.assertTrue((self.root/'proposal/cycle-1/verification/qualification.json').exists())
        with self.assertRaisesRegex(ValueError,'Never replay'):self.propose()

    def test_only_sufficient_comparable_measurement_promotes_once(self):
        path=self.measured_proposal();result=tuning.select(self.root,path)
        self.assertEqual(result['status'],'promoted');self.assertAlmostEqual(result['observed_speedup'],10/9)
        self.assertFalse(result['hardware_invoked']);self.assertFalse(result['general_performance_claim'])
        self.assertIn('caller-supplied',result['measurement_trust'])
        self.assertEqual(tuning.report(self.root)['selected_candidate_sha256'],result['selected_candidate_sha256'])
        with self.assertRaisesRegex(ValueError,'No unmeasured'):tuning.select(self.root,path)

    def test_retained_promotion_cannot_change_ratio_or_selected_bytes(self):
        path=self.measured_proposal();tuning.select(self.root,path)
        value=review.read(self.root/'decision.json');value['observed_speedup']=99
        review.save(self.root/'decision.json',value);review.seal(self.root)
        with self.assertRaisesRegex(ValueError,'ratio changed'):tuning.report(self.root)

    def test_missing_or_slower_measurement_retains_last_correct_incumbent(self):
        path=self.measured_proposal(seconds=11.0);result=tuning.select(self.root,path)
        self.assertEqual(result['status'],'kept-incumbent')
        self.assertEqual(result['selected_candidate_sha256'],self.receipt['candidate_sha256'])
        self.assertLess(result['observed_speedup'],1)

    def test_absent_measurement_finishes_without_promotion_or_retry(self):
        self.propose();result=tuning.select(self.root)
        self.assertEqual(result['status'],'kept-incumbent');self.assertIsNone(result['observed_speedup'])
        self.assertEqual(result['selected_candidate_sha256'],self.receipt['candidate_sha256'])

    def test_equal_candidate_cannot_reach_review_or_measurement(self):
        original=self.fixture.client
        def unchanged(*args):
            result=original(*args)
            if review.read(args[0]/'isolation.json')['role']=='tuner':
                (args[2]/'candidate.py').write_text(self.plan['tuning']['candidate'])
            return result
        self.fixture.client=unchanged
        result=self.propose();self.assertEqual(result['status'],'kept-incumbent')
        self.assertEqual(result['selected_candidate_sha256'],self.receipt['candidate_sha256'])
        self.assertFalse((self.root/'proposal/cycle-1/reviewer').exists())

    def test_different_hardware_cannot_promote(self):
        path=self.measured_proposal();value=review.read(path);value['hardware']='different-device';review.save(path,value)
        result=tuning.select(self.root,path);self.assertEqual(result['status'],'kept-incumbent')
        self.assertIsNone(result['observed_speedup'])

    def test_measurement_cannot_come_from_model_handoff_tree(self):
        path=self.measured_proposal();inside=self.root/'model-receipt.json';inside.write_bytes(path.read_bytes());review.seal(self.root)
        result=tuning.select(self.root,inside);self.assertEqual(result['status'],'kept-incumbent')
        self.assertIsNone(result['observed_speedup'])

    def test_invalid_scope_or_numerics_cannot_promote(self):
        path=self.measured_proposal();after=review.read(path);after['cases'][0]['numerically_correct']=False
        after['unexpected_sensitive_field']='must-not-be-published';review.save(path,after)
        result=tuning.select(self.root,path);self.assertEqual(result['status'],'kept-incumbent')
        self.assertFalse((self.root/'proposed-measurement.json').exists())
        self.assertNotIn('must-not-be-published',json.dumps(result))

    def test_receipt_rejects_simulator_partial_duplicate_nonfinite_and_drift(self):
        handoff=self.plan['tuning'];contract=review.task_contract(ROOT,self.plan['common']['cells'][0])
        changes=[lambda v:v.update(backend='Model'),lambda v:v.update(candidate_sha256='c'*64),
            lambda v:v.update(pyasc_revision='d'*40),lambda v:v['cases'].pop(),
            lambda v:v['cases'][1].update(id=v['cases'][0]['id']),
            lambda v:v['cases'][0].update(elapsed_us=float('nan')),lambda v:v['cases'][0].update(elapsed_us=0),
            lambda v:v['cases'][0].update(numerically_correct=False),lambda v:v['cases'][0].update(device_kernels=[])]
        for change in changes:
            value=deepcopy(self.receipt);change(value)
            with self.subTest(value=value),self.assertRaises(ValueError):
                tuning.measurement(value,handoff['binding'],self.plan['common']['runtime'],contract)

    def test_interrupted_proposal_is_not_replayed_and_does_not_replace_incumbent(self):
        with patch.object(tuning,'run_review',side_effect=Blocked('Unknown delivery')):
            with self.assertRaises(Blocked):tuning.propose(self.root,self.plan['plan_sha256'],self.fixture.root/'state')
        state=review.read(self.root/'state.json');self.assertEqual(state['status'],'blocked')
        self.assertEqual(state['selected_candidate_sha256'],self.receipt['candidate_sha256'])
        with self.assertRaisesRegex(ValueError,'Never replay'):self.propose()


if __name__=='__main__':unittest.main()
