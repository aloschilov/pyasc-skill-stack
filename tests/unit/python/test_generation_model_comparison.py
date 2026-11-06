"""Offline multi-model experiments; synthetic receipts are never qualification."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_generation_pairs as fixtures
from test_generation_pairs import ROOT, pairs, ci, receipts, signed_plan
from generation_plan import canonical, digest
sys.path.insert(0, str(ROOT/'scripts'))
import generation_operations as ops

NOW = datetime(2026, 11, 6, 10, tzinfo=timezone.utc)


class ModelDesign(unittest.TestCase):
    def setUp(self):
        self.plan = signed_plan('models')

    def test_same_tasks_limits_context_axis_and_counterbalanced_order(self):
        p = self.plan
        self.assertEqual(p['maximum_work'], {'model_attempts':24, 'output_tokens':4608000, 'execution_seconds':28800})
        self.assertEqual(len({t['id'] for t in p['trials']}), 24)
        for cell in p['common']['cells']:
            orders = [[(t['model_arm'], t['protocol']) for t in p['trials']
                       if t['cell'] == cell['id'] and t['repetition'] == rep] for rep in (1, 2)]
            self.assertEqual(orders[0], list(reversed(orders[1])))
            self.assertEqual(set(orders[0]), {(m, c) for m in ('glm52','glm53') for c in ('P3','P6')})
        for trial in p['trials']:
            value = pairs.trial_plan(p, trial)
            self.assertEqual(value['model'], p['models'][trial['model_arm']])
            self.assertEqual(value['runtime'], p['common']['runtime'])
            self.assertEqual(value['limits'], p['common']['limits'])
            self.assertEqual(value['cells'], [c for c in p['common']['cells'] if c['id'] == trial['cell']])
            self.assertEqual(value['skills'], p['common']['skills'] if trial['protocol']=='P6' else [])
            self.assertEqual(value['source_sha256'], p['common']['source_sha256'])
        self.assertFalse(p['schedule_enabled']); self.assertFalse(p['hardware_invoked'])

    def test_duplicate_models_unknown_arms_budgets_and_changed_settings_rejected(self):
        original = pairs.resource
        mutations = [lambda s: s['models'][1].update(id='glm52'),
            lambda s: s['models'][1].update(profile='cloud-glm-5.2-gateway',response_model='glm-5.2'),
            lambda s: s['models'][1].update(response_model='glm-5.2'),
            lambda s: s['models'][0].update(profile='cloud-glm-5.3-low'),
            lambda s: s.update(max_output_tokens=4607999), lambda s: s.update(max_execution_seconds=28799),
            lambda s: s.update(output_tokens_per_response=True), lambda s: s.update(protocols=['P3','P4']),
            lambda s: s.update(repetitions=3)]
        for mutate in mutations:
            def changed(root, name):
                raw = original(root, name)
                if name == pairs.MODELS_SPEC:
                    value = json.loads(raw); mutate(value); return canonical(value)
                return raw
            with self.subTest(mutation=mutations.index(mutate)), patch.object(pairs, 'resource', side_effect=changed):
                with self.assertRaises(ValueError): pairs.build_pairs(experiment='models')
        with self.assertRaisesRegex(ValueError, 'Unknown model-comparison trial'):
            pairs.trial_plan(self.plan, dict(self.plan['trials'][0], model_arm='absent'))

    def test_current_input_validation_cannot_switch_model_lineage_or_lane(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'plan.json'; receipts.write(path, self.plan)
            self.assertEqual(canonical(pairs.validate_pairs(path, 'models')), canonical(self.plan))
            with self.assertRaisesRegex(ValueError, 'current inputs'): pairs.validate_pairs(path)
            altered = deepcopy(self.plan); altered['models']['glm52']['expected_response']='glm-5.3'
            altered.pop('plan_sha256'); altered['plan_sha256']=digest(canonical(altered)); receipts.write(path, altered)
            with self.assertRaisesRegex(ValueError, 'current inputs'): pairs.validate_pairs(path, 'models')

    def test_report_keeps_model_denominators_missing_arms_and_all_repetitions(self):
        observations = {t['id']: {'status':'valid','qualified':t['model_arm']=='glm53' and t['protocol']=='P6',
            'outcome':'qualified' if t['model_arm']=='glm53' and t['protocol']=='P6' else 'candidate-failed',
            'attempts':1,'seconds':2,'output_tokens':3,'input_tokens':4,'cost':None} for t in self.plan['trials']}
        report = pairs.summarize(self.plan, observations)
        self.assertEqual(set(report['comparisons']), {'glm52:P6-P3','glm53:P6-P3','P3:glm53-glm52','P6:glm53-glm52'})
        self.assertEqual(report['comparisons']['glm53:P6-P3']['qualification_delta'], 1)
        self.assertEqual(report['comparisons']['glm52:P6-P3']['qualification_delta'], 0)
        for item in report['comparisons'].values(): self.assertEqual(item['denominator'], 6)
        trial = next(t for t in self.plan['trials'] if t['model_arm']=='glm52' and t['protocol']=='P3')
        observations[trial['id']]={'status':'blocked','outcome':'unresolved-infrastructure'}
        report=pairs.summarize(self.plan, observations)
        self.assertEqual(report['status'],'incomplete')
        self.assertIsNone(report['comparisons']['glm52:P6-P3']['qualification_delta'])
        self.assertEqual(report['models']['glm52']['protocols']['P3']['coverage_gaps'],1)
        self.assertEqual(report['models']['glm52']['protocols']['P3']['candidate_failed'],5)


class ModelEvidence(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.PairedReceipts();self.fixture.experiment='models'
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.plan=self.fixture.plan

    def test_full_receipts_bind_model_and_wire_settings_per_arm(self):
        report=pairs.report_pairs(self.root)
        self.assertEqual((report['status'], report['trial_denominator']),('complete',24))
        for trial in self.plan['trials']:
            value=report['observations'][trial['id']]
            self.assertEqual(value['model']['observed'],self.plan['models'][trial['model_arm']]['expected_response'])
            self.assertEqual(value['request_settings']['max_tokens'],24000)
            self.assertIsNone(value['model']['configuration']['provider_revision'])
            self.assertEqual(len(value['candidate_sha256']),1)

    def test_rehashed_envelope_cannot_change_an_arm(self):
        altered=deepcopy(self.plan);altered['models']['glm52']['requested']='gateway/substitute'
        altered.pop('plan_sha256');altered['plan_sha256']=digest(canonical(altered));receipts.write(self.root/'plan.json',altered)
        state=pairs.read(self.root/'state.json');state['plan_sha256']=altered['plan_sha256'];receipts.write(self.root/'state.json',state)
        pairs.seal(self.root)
        with self.assertRaisesRegex(ValueError,'retained inputs'): pairs.report_pairs(self.root)

    def test_known_wire_difference_prevents_comparability(self):
        trial=self.plan['trials'][-1];directory=self.root/trial['id'];path=directory/trial['cell']/'attempt-1/usage.json'
        usage=pairs.read(path)
        from generation_provider import request_settings
        usage['requests'][0]['request_settings']=request_settings({'max_tokens':24000,'temperature':0.3,
            'reasoning_effort':'high','thinking':{'type':'enabled'}})
        receipts.write(path,usage);state=pairs.read(directory/'state.json');state['cells'][0]['attempts'][0]['usage']=usage
        receipts.write(directory/'state.json',state);receipts.seal(directory);pairs.seal(self.root)
        report=pairs.report_pairs(self.root)
        self.assertEqual(report['status'],'incomplete')
        self.assertIn('Wire settings differ',report['observations'][trial['id']]['reason'])

    def test_equal_missing_reasoning_settings_do_not_prove_requested_options(self):
        from generation_models import inspect_comparison_settings
        from generation_provider import request_settings
        model=self.plan['models']['glm52']
        with self.assertRaisesRegex(ValueError,'not serialized'):
            inspect_comparison_settings(model,request_settings({'max_tokens':24000}))
        with self.assertRaisesRegex(ValueError,'not serialized'):
            inspect_comparison_settings(model,request_settings({'max_tokens':24000,'reasoning_effort':'low','thinking':{'type':'enabled'}}))


class ModelController(unittest.TestCase):
    def test_isolated_disabled_policy_and_workflows(self):
        import yaml
        config=ops.policy(NOW,paired='models');self.assertFalse(config['enabled'])
        self.assertNotEqual(ops.policy_path(True),ops.policy_path('models'))
        workflow=yaml.safe_load((ROOT/ops.MODELS_WORKFLOW).read_text())
        self.assertEqual(set(workflow[True]),{'pull_request','workflow_dispatch'})
        self.assertEqual(workflow['jobs']['paired']['with']['experiment'],'models')
        self.assertIn('plan --models',str(workflow['jobs']['prepare']))
        with tempfile.TemporaryDirectory() as d, patch.object(ops,'api',side_effect=AssertionError('Disabled policy must not query')):
            report=ops.monitor(Path(d),config,NOW,paired='models')
            self.assertEqual((report['status'],report['experiment']),('disabled','models'))

    def test_caller_and_plan_cannot_cross_model_and_context_lanes(self):
        plan=signed_plan('models')
        env={'GENERATION_EXPERIMENT':'models','GENERATION_ACTION':'execute','APPROVE_PLAN':plan['plan_sha256'],
             'GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_REPOSITORY':ops.REPO,'GITHUB_REF':'refs/heads/codex/model-coverage',
             'GITHUB_WORKFLOW_REF':ops.REPO+'/'+ops.MODELS_WORKFLOW+'@refs/heads/codex/model-coverage'}
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,env,clear=True), \
                patch.object(ci.subprocess,'check_output',return_value=plan['checkout_revision']):
            root=Path(d)
            self.assertEqual(ci.prepare(root/'right'),0)
            with patch.dict(os.environ,GITHUB_WORKFLOW_REF=env['GITHUB_WORKFLOW_REF'].replace('generation-models','generation-paired')):
                with self.assertRaisesRegex(ValueError,'explicit controller'):ci.prepare(root/'wrong')
            with patch.dict(os.environ,APPROVE_PLAN=signed_plan()['plan_sha256']):
                with self.assertRaisesRegex(ValueError,'digest mismatch'):ci.prepare(root/'context-plan')

    def test_model_cadence_does_not_accept_context_run_history(self):
        config=dict(ops.policy(NOW,paired='models'),enabled=True,enabled_since='2026-11-01T00:00:00Z',alert_issue=123)
        run={'id':99,'event':'schedule','path':ops.MODELS_WORKFLOW,'head_branch':'main','created_at':NOW.isoformat()}
        with patch.dict(os.environ,GITHUB_RUN_ID='99',GITHUB_RUN_ATTEMPT='1'), \
                patch.object(ops,'api',return_value={'total_count':1,'workflow_runs':[run]}) as api:
            self.assertTrue(ops.paired_due(config,NOW,paired='models')[0])
            self.assertIn('generation-models.yml/runs?',api.call_args.args[0])
            run['path']=ops.PAIRED_WORKFLOW
            with self.assertRaisesRegex(ValueError,'Unexpected paired cadence'):ops.paired_due(config,NOW,paired='models')


if __name__=='__main__':unittest.main()
