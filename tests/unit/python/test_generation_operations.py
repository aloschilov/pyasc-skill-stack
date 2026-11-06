"""Controller/monitor contracts with synthetic receipts; no model or network."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
import generation_operations as ops
import test_generation_compare as receipts
from generation_compare import Bundle
from generation_plan import canonical, digest

NOW = receipts.NOW


class GenerationOperations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy_file = self.root/'policy.json'
        self.config = {'schema_version': 1, 'enabled': True, 'enabled_since': '2026-11-03T00:00:00Z',
            'baseline_ref': 'release/reviewed', 'baseline_sha256': 'a'*64,
            'max_age_hours': 36, 'max_run_hours': 2, 'alert_issue': 123}
        self.write_policy()
        p = patch.object(ops, 'POLICY', self.policy_file); p.start(); self.addCleanup(p.stop)
        self.env = {'GITHUB_REPOSITORY': ops.REPO, 'GITHUB_EVENT_NAME': 'workflow_dispatch',
            'GITHUB_REF': 'refs/heads/main', 'GITHUB_WORKFLOW_REF': ops.REPO+'/'+ops.WORKFLOW+'@refs/heads/main',
            'GITHUB_SHA': 'synthetic-offline-fixture'}
        self.fixture = receipts.Comparisons(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        for path in (self.fixture.base, self.fixture.new):
            self.fixture.edit('state.json', lambda s: s.update(checkout_revision='synthetic-offline-fixture'), root=path)
        self.baseline = Bundle(self.fixture.base)
        self.evidence = self.root/'evidence'; self.evidence.mkdir()
        shutil.copytree(self.fixture.new, self.evidence/'run')
        self.run = {'id': 99, 'run_attempt': 1, 'event': 'schedule', 'path': ops.WORKFLOW,
            'head_branch': 'main', 'head_sha': 'synthetic-offline-fixture', 'created_at': '2026-11-06T03:59:00Z',
            'status': 'completed', 'conclusion': 'success', 'html_url': 'https://github.com/'+ops.REPO+'/actions/runs/99'}
        self.origin()

    def write_policy(self):
        self.policy_file.write_text(json.dumps(self.config))

    def origin(self):
        run = self.run
        receipts.write(self.evidence/'ci.json', {'action': 'execute',
            'baseline_ref': self.config['baseline_ref'], 'baseline_sha256': self.config['baseline_sha256'],
            'context': {'GITHUB_RUN_ID': str(run['id']), 'GITHUB_RUN_ATTEMPT': str(run['run_attempt']),
                'GITHUB_REPOSITORY': ops.REPO, 'GITHUB_EVENT_NAME': run['event'],
                'GITHUB_WORKFLOW_REF': ops.REPO+'/'+run['path']+'@refs/heads/'+run['head_branch'],
                'GITHUB_SHA': run['head_sha']}})
        receipts.write(self.evidence/'result.json', {'status': 'passed', 'model_invoked': True,
            'hardware_invoked': False, 'compared': True})

    def monitor(self, runs=None, now=NOW, count=None, inspect_run=None, download_error=False):
        runs = [self.run] if runs is None else runs
        history = {'total_count': len(runs) if count is None else count, 'workflow_runs': runs}
        def download(root, config, run=None):
            if run is not None and download_error: raise OSError('missing archive')
            return self.evidence, 'b'*64
        with patch.object(ops, 'download', side_effect=download), patch.object(ops, 'baseline_at', return_value=self.baseline), \
             patch.object(ops, 'api', return_value=self.run if inspect_run else history):
            return ops.monitor(self.root, self.config, now, inspect_run)

    def test_policy_requires_activation_time_baseline_checksum_and_alert_destination(self):
        self.assertTrue(ops.policy(NOW)['enabled'])
        for field, value in [('enabled', 'true'), ('enabled_since', None), ('enabled_since', '2027-01-01T00:00:00Z'),
                             ('baseline_ref', 'latest'), ('baseline_sha256', 'bad'), ('alert_issue', None), ('max_age_hours', True)]:
            with self.subTest(field=field):
                original = self.config[field]; self.config[field] = value; self.write_policy()
                with self.assertRaises((ValueError, TypeError)): ops.policy(NOW)
                self.config[field] = original

    def test_disabled_monitor_is_get_free_and_cannot_claim_fresh_coverage(self):
        self.config.update(enabled=False, enabled_since=None, alert_issue=None); self.write_policy()
        with patch.object(ops, 'api', side_effect=AssertionError('No network')), patch.object(ops, 'download', side_effect=AssertionError('No fetch')):
            result = ops.monitor(self.root, self.config, NOW)
        self.assertEqual(result['status'], 'disabled'); self.assertIsNone(result['last_qualified'])

    def test_plan_approval_pr_rejection_and_scheduled_opt_in(self):
        value = ops.build_plan(); value['checkout_revision'] = 'synthetic-offline-fixture'; approval = digest(canonical(value))
        cases = [('workflow_dispatch', 'plan', '', False, 'plan'),
                 ('workflow_dispatch', 'execute', approval, False, 'execute'),
                 ('pull_request', 'plan', '', False, 'plan'),
                 ('schedule', 'plan', '', False, 'none'), ('schedule', 'plan', '', True, 'execute')]
        for i, (event, mode, approved, enabled, expected) in enumerate(cases):
            dest = self.root/str(i); dest.mkdir(); self.config['enabled'] = enabled
            with patch.dict(os.environ, dict(self.env, GITHUB_EVENT_NAME=event)), \
                 patch.object(ops.subprocess, 'check_output', return_value='synthetic-offline-fixture\n'):
                result = ops.plan(dest, self.config, mode, approved)
            self.assertEqual(result['action'], expected); self.assertFalse(result['model_invoked'])
        for event, mode, approved in [('workflow_dispatch', 'execute', ''), ('pull_request', 'execute', approval)]:
            dest = self.root/(event+'-reject'); dest.mkdir()
            with patch.dict(os.environ, dict(self.env, GITHUB_EVENT_NAME=event)), \
                 patch.object(ops.subprocess, 'check_output', return_value='synthetic-offline-fixture\n'):
                with self.assertRaisesRegex(ValueError, 'planning-only|reviewed plan'): ops.plan(dest, self.config, mode, approved)
        with patch.dict(os.environ, dict(self.env, GITHUB_EVENT_NAME='schedule', GITHUB_REF='refs/heads/other')):
            with self.assertRaisesRegex(ValueError, 'main-only'): ops.plan(self.root, self.config, 'plan', '')

    def test_health_requires_numerical_evidence_not_green_workflow_status(self):
        healthy = self.monitor()
        self.assertEqual(healthy['status'], 'healthy'); self.assertEqual(healthy['last_qualified']['id'], 99)
        missing = self.monitor(download_error=True)
        self.assertEqual(missing['status'], 'alert'); self.assertIn('unverifiable-run', missing['alerts'])
        with (self.evidence/'run/add-f32-tail/attempt-1/candidate.py').open('a') as out: out.write('changed')
        corrupt = self.monitor()
        self.assertEqual(corrupt['status'], 'alert'); self.assertIsNone(corrupt['last_qualified'])

    def test_wrong_origin_and_baseline_never_establish_coverage(self):
        for field, value in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_EVENT_NAME', 'workflow_dispatch'),
                             ('GITHUB_SHA', 'different'), ('GITHUB_WORKFLOW_REF', ops.REPO+'/.github/workflows/ci.yml@refs/heads/main')]:
            self.origin(); ci = ops.read(self.evidence/'ci.json'); ci['context'][field] = value
            receipts.write(self.evidence/'ci.json', ci)
            self.assertIn('unverifiable-run', self.monitor()['alerts'])
        self.origin(); ci = ops.read(self.evidence/'ci.json'); ci['baseline_sha256'] = 'c'*64
        receipts.write(self.evidence/'ci.json', ci)
        self.assertIn('unverifiable-run', self.monitor()['alerts'])

    def test_missing_failed_stalled_and_stale_runs_remain_alerts(self):
        self.assertIn('stale-coverage', self.monitor([])['alerts'])
        bad = dict(self.run, id=100, conclusion='failure', created_at='2026-11-06T05:00:00Z')
        mixed = self.monitor([bad, self.run])
        self.assertIn('run-failed', mixed['alerts']); self.assertIsNotNone(mixed['last_qualified'])
        stalled = dict(self.run, status='queued', conclusion=None)
        self.assertIn('run-stalled', self.monitor([stalled])['alerts'])
        stale = self.monitor(now=NOW+timedelta(hours=37))
        self.assertIn('stale-coverage', stale['alerts']); self.assertEqual(stale['status'], 'alert')
        self.config['enabled_since'] = (NOW-timedelta(hours=1)).isoformat()
        self.assertEqual(self.monitor([])['status'], 'pending')
        self.assertIn('history-truncated', self.monitor([], count=101)['alerts'])

    def test_manual_inspection_cannot_qualify_or_notify_a_schedule(self):
        self.run.update(event='workflow_dispatch', path='.github/workflows/ci.yml'); self.origin()
        report = self.monitor(inspect_run='99/1')
        self.assertEqual(report['status'], 'verified-manual'); self.assertEqual(report['scope'], 'manual-inspection')
        with self.assertRaisesRegex(ValueError, 'Manual inspection'): ops.notify(report, self.config, NOW)
        self.run['path'] = ops.WORKFLOW; self.origin()
        self.assertEqual(self.monitor(inspect_run='99/1')['status'], 'verified-manual')

    def test_notifications_are_marked_idempotent_updates_and_confirm_delivery(self):
        report = self.monitor([])
        issue = {'body': ops.MARKER+'\nEmpty alert channel', 'state': 'closed', 'html_url': 'https://github.com/'+ops.REPO+'/issues/123'}
        writes = []
        def api(endpoint, payload=None):
            self.assertTrue(endpoint.endswith('/issues/123'))
            if payload is not None: writes.append(payload); issue.update(payload)
            return deepcopy(issue)
        with patch.object(ops, 'api', side_effect=api):
            self.assertEqual(ops.notify(report, self.config, NOW)['status'], 'confirmed')
            report['checked_at'] = (NOW+timedelta(minutes=1)).isoformat()
            self.assertEqual(ops.notify(report, self.config, NOW+timedelta(minutes=1))['status'], 'unchanged')
            healthy = self.monitor()
            self.assertEqual(ops.notify(healthy, self.config, NOW)['status'], 'confirmed')
        self.assertEqual(len(writes), 2); self.assertEqual(issue['state'], 'closed')
        with patch.object(ops, 'api', return_value=dict(issue, body='unrelated issue')):
            with self.assertRaisesRegex(ValueError, 'unmarked'): ops.notify(report, self.config, NOW+timedelta(minutes=1))
        with self.assertRaisesRegex(ValueError, 'too old'): ops.notify(report, self.config, NOW+timedelta(minutes=30))

    def test_unknown_patch_is_reconciled_once_without_replay(self):
        report = self.monitor([])
        for applied in (False, True):
            issue = {'body': ops.MARKER, 'state': 'closed', 'html_url': 'url'}; writes = []
            def api(endpoint, payload=None):
                if payload is not None:
                    writes.append(payload)
                    if applied: issue.update(payload)
                    raise OSError('unknown delivery')
                return deepcopy(issue)
            with patch.object(ops, 'api', side_effect=api):
                if applied: self.assertEqual(ops.notify(report, self.config, NOW)['status'], 'confirmed')
                else:
                    with self.assertRaisesRegex(ValueError, 'unconfirmed'): ops.notify(report, self.config, NOW)
            self.assertEqual(len(writes), 1)

    def test_workflows_are_disabled_by_default_and_pr_planning_has_no_secret(self):
        import yaml
        nightly = yaml.safe_load((ROOT/'.github/workflows/generation-nightly.yml').read_text())
        health = yaml.safe_load((ROOT/'.github/workflows/generation-health.yml').read_text())
        self.assertEqual(set(nightly[True]), {'pull_request', 'workflow_dispatch'})
        self.assertEqual(set(health[True]), {'workflow_dispatch'})
        self.assertFalse(json.loads((ROOT/'tests/generation/nightly.json').read_text())['enabled'])
        self.assertNotIn('secrets', nightly['jobs']['prepare'])
        self.assertNotIn('OPENCODE_API_KEY', str(nightly['jobs']['prepare']))
        self.assertIn("github.event_name != 'pull_request'", nightly['jobs']['qualify']['if'])
        self.assertEqual(health['jobs']['observe']['runs-on'], 'ubuntu-latest')
        self.assertFalse(health[True]['workflow_dispatch']['inputs']['notify']['default'])
        delivery = next(s for s in health['jobs']['observe']['steps'] if s.get('name') == 'Update configured alert issue')
        self.assertIn("github.event_name == 'schedule' || inputs.notify", delivery['if'])
        self.assertIn("inputs.inspect_run == ''", delivery['if'])
        self.assertNotIn('OPENCODE', str(health)); self.assertNotIn('BENCHSITE', str(health))


class PairedObservation(unittest.TestCase):
    def setUp(self):
        import test_generation_pairs as paired_receipts
        from generation_pairs import seal
        self.fixture = paired_receipts.PairedReceipts(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.evidence = self.fixture.root.parent/'evidence'; self.evidence.mkdir()
        shutil.copytree(self.fixture.root, self.evidence/'run')
        self.config = ops.policy(NOW)
        self.run = {'id': 123, 'run_attempt': 1, 'event': 'workflow_dispatch',
            'path': '.github/workflows/generation-paired.yml', 'head_branch': 'codex/paired',
            'head_sha': self.fixture.plan['checkout_revision'], 'created_at': '2026-11-06T03:00:00Z',
            'status': 'completed', 'conclusion': 'success', 'html_url': 'https://github.com/'+ops.REPO+'/actions/runs/123'}
        state = ops.read(self.evidence/'run/state.json')
        state.update(started_at='2026-11-06T04:00:00Z', finished_at='2026-11-06T04:10:00Z')
        receipts.write(self.evidence/'run/state.json', state); seal(self.evidence/'run')
        run = self.run; plan = self.fixture.plan
        receipts.write(self.evidence/'ci.json', {'experiment': 'paired', 'action': 'execute',
            'approve_plan': plan['plan_sha256'], 'context': {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1',
            'GITHUB_REPOSITORY': ops.REPO, 'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_SHA': run['head_sha'],
            'GITHUB_WORKFLOW_REF': ops.REPO+'/'+run['path']+'@refs/heads/'+run['head_branch']}})
        receipts.write(self.evidence/'result.json', {'status': 'complete', 'model_invoked': True,
            'hardware_invoked': False, 'plan_sha256': plan['plan_sha256'], 'trial_denominator': 24,
            'cells': {t['id']: 'qualified' for t in plan['trials']}})

    def monitor(self, now=NOW):
        def download(root, config, run=None):
            self.assertIsNotNone(run, 'Paired inspection must not fetch a single-run baseline')
            return self.evidence, 'a'*64
        with patch.object(ops, 'api', return_value=self.run) as api, patch.object(ops, 'download', side_effect=download):
            report = ops.monitor(self.evidence.parent, self.config, now, '123/1', paired=True)
        api.assert_called_once_with('repos/'+ops.REPO+'/actions/runs/123/attempts/1')
        return report

    def test_paired_inspection_verifies_all_pairs_without_claiming_scheduled_coverage(self):
        report = self.monitor()
        self.assertEqual(report['status'], 'verified-manual')
        self.assertEqual(report['scope'], 'manual-inspection')
        summary = report['last_qualified']['paired_summary']
        self.assertEqual(summary['trial_denominator'], 24)
        self.assertEqual(summary['comparisons']['P6-P3']['valid_pairs'], 6)
        self.assertFalse(summary['general_skills_benefit_claim'])
        self.assertIn('stale-coverage', self.monitor(NOW+timedelta(hours=37))['alerts'])
        with patch.object(ops, 'api', side_effect=AssertionError('No network')):
            with self.assertRaisesRegex(ValueError, 'explicit RUN_ID'):
                ops.monitor(self.evidence.parent, self.config, NOW, paired=True)
        with self.assertRaisesRegex(ValueError, 'not configured'):
            ops.notify(report, self.config, NOW)

    def test_green_paired_run_needs_matching_origin_complete_cells_and_intact_evidence(self):
        for filename, field, value in [('ci.json', 'approve_plan', 'bad'), ('result.json', 'cells', {}),
                                       ('result.json', 'trial_denominator', 23), ('result.json', 'model_invoked', False)]:
            path = self.evidence/filename; original = path.read_text(); item = json.loads(original); item[field] = value
            receipts.write(path, item)
            self.assertIn('unverifiable-run', self.monitor()['alerts'])
            path.write_text(original)
        self.run['head_sha'] = 'other'
        self.assertIn('unverifiable-run', self.monitor()['alerts'])
        self.run['head_sha'] = self.fixture.plan['checkout_revision']
        t = self.fixture.plan['trials'][0]
        (self.evidence/'run'/t['id']/t['cell']/'attempt-1/candidate.py').write_text('corrupted')
        report = self.monitor()
        self.assertIn('unverifiable-run', report['alerts']); self.assertIsNone(report['last_qualified'])


if __name__ == '__main__': unittest.main()
