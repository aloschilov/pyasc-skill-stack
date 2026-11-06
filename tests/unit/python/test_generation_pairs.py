"""Synthetic paired receipts and offline protocol tests, never qualification evidence."""
from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import generation_pairs as pairs
import generation_protocols as protocols
import generation_executor as executor
import generation_pairs_ci as ci
from generation_adapters import task_contract, write_probes
from generation_executor import Blocked, configure
from generation_plan import canonical, digest
import test_generation_compare as receipts


def signed_plan():
    p = pairs.build_pairs(); p['checkout_revision'] = 'synthetic-paired-fixture'
    return dict(p, plan_sha256=digest(canonical(p)))


class PairedDesign(unittest.TestCase):
    def setUp(self):
        self.plan = signed_plan()

    def test_plan_is_predeclared_bounded_and_counterbalances_each_task(self):
        self.assertEqual(self.plan, signed_plan())
        self.assertEqual(len(self.plan['trials']), 24)
        self.assertEqual(self.plan['maximum_work'], {'model_attempts': 24, 'output_tokens': 1572864, 'execution_seconds': 14400})
        self.assertIsNone(self.plan['generation_seed'])
        for cell in self.plan['common']['cells']:
            orders = [[t['protocol'] for t in self.plan['trials'] if t['cell'] == cell['id'] and t['repetition'] == rep] for rep in (1, 2)]
            self.assertEqual(orders[0], list(reversed(orders[1])))
            self.assertEqual(set(orders[0]), set(protocols.PROTOCOLS))

    def test_only_the_declared_intervention_changes_between_legs(self):
        plans = [pairs.trial_plan(self.plan, t) for t in self.plan['trials'][:4]]
        for p in plans:
            q = deepcopy(p)
            for key in ('skills', 'intervention', 'plan_sha256'): q.pop(key)
            reference = deepcopy(plans[0])
            for key in ('skills', 'intervention', 'plan_sha256'): reference.pop(key)
            self.assertEqual(q, reference)
            context = protocols.context_files(p); pid = p['intervention']['id']
            self.assertEqual(any(n.startswith('skills/') for n in context), pid == 'P6')
            self.assertEqual(protocols.BASELINE in context, pid == 'P4')
            self.assertFalse(any('target/' in n or 'fixtures/' in n for n in context))

    def test_read_permissions_match_opencode_worktree_relative_requests(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d)/'cell/attempt/project';context = Path(d)/'context'
            for t in self.plan['trials'][:4]:
                plan = pairs.trial_plan(self.plan, t)
                config = configure(pairs.read(ROOT/'integrations/opencode/profiles/cloud-glm-5.3.json'), plan, context, project)
                rules = config['permission']['read']
                for name in protocols.context_files(plan):
                    relative = os.path.relpath(context/name, project)
                    self.assertEqual(rules.get(relative, rules['*']), 'deny' if name.startswith('integrations/') else 'allow')
                self.assertEqual(rules.get('../../../../source/oracle.py', rules['*']), 'deny')

    def test_control_generation_does_not_require_unselected_skill_files(self):
        # Exercise generate itself with a fake client, not merely context_files.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);context = root/'context';attempt = root/'attempt';attempt.mkdir()
            t = next(t for t in self.plan['trials'] if t['protocol'] == 'P2')
            plan = pairs.trial_plan(self.plan, t)
            for name in protocols.context_files(plan):
                dest = context/name;dest.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/name, dest)
            usage = {'usage_complete': True, 'requests': [{'usage': {'observed_model': plan['model']['expected_response']},
                'audit': {'prompt_present': True, 'tool_schema_sha256': 'a'*64, 'visible_skills': []}}]}
            def client(command, directory, name, *args, **kwargs):
                if name == 'generation':
                    (attempt/'generation.stdout').write_text('{}\n')
                    (attempt/'project/candidate.py').write_text('offline fake candidate')
                return {'exit_code': 0}
            with patch.object(executor, 'BudgetGate') as factory, patch.object(executor, 'command', side_effect=client), \
                 patch.object(executor, 'budget_server', return_value=nullcontext('http://127.0.0.1/v1')), \
                 patch.object(executor, 'native_trace', return_value={}), patch.object(protocols, 'audit_trace', return_value={}), \
                 patch.dict(os.environ, OPENCODE_API_KEY='offline', OPENCODE_BASE_URL='http://127.0.0.1'):
                factory.return_value.error = None;factory.return_value.token = 'offline';factory.return_value.snapshot.return_value = usage
                executor.generate(attempt, plan, task_contract(ROOT, plan['cells'][0]), context)
            self.assertTrue((attempt/'candidate.py').is_file())
            self.assertFalse((context/'skills').exists())

    def test_missing_cost_and_invalid_control_cannot_create_a_skills_gain(self):
        values = {t['id']: {'status': 'valid', 'qualified': t['protocol'] == 'P6', 'attempts': 1,
            'seconds': 5, 'output_tokens': 10, 'input_tokens': 20, 'cost': None} for t in self.plan['trials']}
        r = pairs.summarize(self.plan, values)
        self.assertEqual(r['comparisons']['P6-P3']['qualification_delta'], 1)
        self.assertIsNone(r['comparisons']['P6-P3']['mean_deltas']['cost'])
        self.assertFalse(r['general_skills_benefit_claim'])
        control = next(t for t in self.plan['trials'] if t['protocol'] == 'P3')
        values[control['id']] = {'status': 'invalid-instrumentation'}
        r = pairs.summarize(self.plan, values); c = r['comparisons']['P6-P3']
        self.assertEqual((r['status'], c['denominator'], c['valid_pairs']), ('incomplete', 6, 5))
        self.assertIsNone(c['qualification_delta']); self.assertIsNone(c['uncertainty']['interval'])
        values.pop(control['id'])
        with self.assertRaisesRegex(ValueError, 'Missing/extra'): pairs.summarize(self.plan, values)

    def test_negative_effect_and_small_sample_uncertainty_remain_visible(self):
        values = {t['id']: {'status': 'valid', 'qualified': t['protocol'] != 'P6', 'attempts': 1,
            'seconds': 5, 'output_tokens': 10, 'input_tokens': 20, 'cost': None} for t in self.plan['trials']}
        c = pairs.summarize(self.plan, values)['comparisons']['P6-P3']
        self.assertEqual(c['qualification_delta'], -1)
        self.assertGreater(c['uncertainty']['interval'][1], 0)
        self.assertEqual(len(c['paired_rows']), 6)

    def test_provider_audit_checks_actual_context_and_schema_not_file_names(self):
        with tempfile.TemporaryDirectory() as d:
            context = Path(d); p = context/protocols.BASELINE;p.parent.mkdir(parents=True);p.write_text('First\nSecond\n')
            (context/'skills/demo').mkdir(parents=True);(context/'skills/demo/SKILL.md').write_text('---\nname: demo\n---\nPinned skill body')
            body = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'task "with quotes"\nline'}]}],
                'tools': [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}]}
            a = protocols.request_audit(body, 'task "with quotes"\nline', 'P2', ['demo'], context)
            self.assertTrue(a['prompt_present']);self.assertFalse(a['visible_skills'])
            body['messages'].append({'role': 'tool', 'content': '1: First\n2: Second'})
            self.assertTrue(protocols.request_audit(body, 'task', 'P4', ['demo'], context)['baseline_visible'])
            body['messages'].append({'role': 'tool', 'content': '<skill_content name="demo">Pinned skill body</skill_content>'})
            with self.assertRaisesRegex(ValueError, 'Unselected skill'): protocols.request_audit(body, 'task', 'P3', ['demo'], context)
            b = protocols.request_audit(body, 'task', 'P6', ['demo'], context)
            self.assertEqual(b['visible_skills'], ['demo']);self.assertEqual(b['tool_schema_sha256'], a['tool_schema_sha256'])

    def test_unknown_trial_stops_the_whole_experiment_and_retains_shared_latch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(pairs, 'execute', return_value={'status': 'blocked', 'observed_output_tokens': 7}) as call, \
                 patch.object(pairs, 'report_pairs', return_value={'status': 'incomplete'}):
                pairs.run_pairs(self.plan, root/'run', root/'state')
                self.assertEqual(call.call_count, 1)
                self.assertIsNotNone(call.call_args.kwargs['_lease'])
            state = pairs.read(root/'run/state.json')
            self.assertEqual(state['status'], 'blocked')
            self.assertEqual([t['status'] for t in state['trials']], ['blocked']+['not-run']*23)
            self.assertTrue((root/'state/active.json').exists())
            with self.assertRaisesRegex(Blocked, 'Unresolved'):
                pairs.run_pairs(self.plan, root/'other', root/'state')

    def test_invalid_instrumentation_stops_before_spending_remaining_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(pairs, 'execute', return_value={'status': 'passed', 'observed_output_tokens': 7}) as call, \
                 patch.object(pairs, 'inspect_trial', side_effect=ValueError('Missing provider audit')), \
                 patch.object(pairs, 'report_pairs', return_value={'status': 'incomplete'}):
                pairs.run_pairs(self.plan, root/'run', root/'state')
                self.assertEqual(call.call_count, 1)
            state = pairs.read(root/'run/state.json')
            self.assertEqual(state['trials'][0]['status'], 'invalid-instrumentation')
            self.assertTrue(all(t['status'] == 'not-run' for t in state['trials'][1:]))
            self.assertTrue((root/'state/active.json').exists())


class PairedReceipts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'experiment';self.root.mkdir()
        self.plan = signed_plan();receipts.write(self.root/'plan.json', self.plan)
        self.fixture = receipts.Comparisons();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        for t in self.plan['trials']: self.make_trial(t)
        receipts.write(self.root/'state.json', {'status': 'completed', 'plan_sha256': self.plan['plan_sha256'],
            'trials': [dict(t, status='passed') for t in self.plan['trials']]})
        pairs.seal(self.root)

    def make_trial(self, t):
        root = self.root/t['id'];shutil.copytree(self.fixture.base, root)
        plan = pairs.trial_plan(self.plan, t);receipts.write(root/'plan.json', plan)
        for name in plan['source_sha256']:
            target = root/'source'/name;target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/name, target)
        for name in protocols.context_files(plan):
            target = root/'context'/name;target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/name, target)
        state = pairs.read(root/'state.json');state.update(checkout_revision=self.plan['checkout_revision'])
        state['cells'] = [c for c in state['cells'] if c['id'] == t['cell']]
        p = root/t['cell']/'attempt-1'
        contract = task_contract(ROOT, plan['cells'][0]);write_probes(p, contract)
        q = pairs.read(p/'qualification.json')
        q['input_sha256'].update({n: digest((p/n).read_bytes()) for n in ('adapter.json', 'probe.py', 'compile_probe.py')})
        q['phases']['jit']['report']['launcher_calls'] = len(contract['seeds'])
        q['phases']['execution']['report']['launcher_calls'] = len(contract['seeds'])
        q['phases']['control']['report']['launcher_calls'] = len(contract['seeds'])
        q['phases']['execution']['report']['numerical_checks'] = [q['phases']['execution']['report']['numerical_checks'][0]]*(len(contract['seeds'])*len(contract['arguments']))
        receipts.write(p/'qualification.json', q)
        old = Path('/recorded/paired')/t['id'];project = old/t['cell']/'attempt-1/project';context = old/'context'
        config = configure(json.loads((ROOT/f'integrations/opencode/profiles/{plan["model"]["profile"]}.json').read_text()), plan, context, project)
        config['provider']['gateway']['options'] = {'baseURL': 'http://127.0.0.1:1234/v1', 'apiKey': '{env:PYASC_GATE_TOKEN}'}
        receipts.write(p/'model-config.json', config)
        receipts.write(p/'isolation.json', {'protocol': plan['intervention'], 'context_sha256': protocols.context_files(plan),
            'project': str(project), 'context': str(context), 'private': str(old/'private'),
            'fresh_project': True, 'fresh_private': True, 'permission': config['permission']})
        events = []
        def event(tool, data, meta, output=''):
            events.append({'sessionID': 'session-'+t['id'], 'part': {'tool': tool, 'state': {
                'status': 'completed', 'input': data, 'metadata': meta, 'output': output}}})
        for skill in plan['skills']:
            event('skill', {'name': skill}, {'dir': str(context/'skills'/skill), 'truncated': False},
                  protocols.skill_body((ROOT/'skills'/skill/'SKILL.md').read_text()))
        if t['protocol'] == 'P4':
            event('read', {'filePath': str(context/protocols.BASELINE)}, {'loaded': [], 'truncated': False,
                'display': {'lineStart': 1, 'text': (ROOT/protocols.BASELINE).read_text().rstrip('\n')}})
        event('write', {'filePath': str(project/'candidate.py'), 'content': 'offline double'}, {})
        raw = '\n'.join(json.dumps(e) for e in events)
        self.fixture.command(p, 'generation', raw)
        text = protocols.prompt(task_contract(ROOT, plan['cells'][0]), t['protocol'], plan['skills'], context)
        (p/'prompt.txt').write_text(text)
        attribution = protocols.audit_trace(raw, project, context, plan, source_context=root/'context')
        attribution.update(requested_model=plan['model']['requested'], observed_model=plan['model']['expected_response'],
            tool_schema_sha256='e'*64, prompt_sha256=digest(text.encode()))
        receipts.write(p/'attribution.json', attribution)
        usage = {'usage_complete': True, 'error': None, 'observed_output_tokens': 10, 'provider_cost': None, 'requests': [
            {'status': 'complete', 'usage': {'observed_model': plan['model']['expected_response'], 'prompt_tokens': 20,
                'completion_tokens': 10, 'total_tokens': 30}, 'audit': {'protocol': t['protocol'], 'prompt_present': True,
                'tool_schema_sha256': 'e'*64, 'visible_skills': plan['skills'], 'baseline_visible': t['protocol'] == 'P4'}}]}
        receipts.write(p/'usage.json', usage);state['cells'][0]['attempts'][0].update(usage=usage, duration_seconds=5)
        receipts.write(root/'state.json', state);receipts.seal(root)

    def edit_trial(self, t, name, modify):
        p = self.root/t['id']/t['cell']/'attempt-1'/name
        value = pairs.read(p);modify(value);receipts.write(p, value);receipts.seal(self.root/t['id']);pairs.seal(self.root)

    def test_complete_pair_evidence_proves_fixed_denominator_not_benefit(self):
        r = pairs.report_pairs(self.root)
        self.assertEqual((r['status'], r['trial_denominator']), ('complete', 24))
        self.assertEqual(r['comparisons']['P6-P3']['qualification_delta'], 0)
        self.assertIsNone(r['comparisons']['P6-P3']['mean_deltas']['cost'])

    def test_status_and_nonfinite_metrics_are_not_valid_pair_observations(self):
        t = self.plan['trials'][0]
        s = pairs.read(self.root/'state.json');s['trials'][0]['status'] = 'failed'
        receipts.write(self.root/'state.json', s);pairs.seal(self.root)
        self.assertEqual(pairs.report_pairs(self.root)['observations'][t['id']]['status'], 'invalid-instrumentation')
        s['trials'][0]['status'] = 'passed';receipts.write(self.root/'state.json', s)
        trial = self.root/t['id'];state = pairs.read(trial/'state.json')
        state['cells'][0]['attempts'][0]['duration_seconds'] = -1
        receipts.write(trial/'state.json', state);receipts.seal(trial);pairs.seal(self.root)
        self.assertEqual(pairs.report_pairs(self.root)['observations'][t['id']]['status'], 'invalid-instrumentation')

    def test_model_or_protocol_mismatch_does_not_become_failed_control(self):
        t = next(t for t in self.plan['trials'] if t['protocol'] == 'P3')
        self.edit_trial(t, 'model-config.json', lambda v: v.update(small_model='different/model'))
        r = pairs.report_pairs(self.root)
        self.assertEqual(r['observations'][t['id']]['status'], 'invalid-instrumentation')
        self.assertIsNone(r['comparisons']['P6-P3']['qualification_delta'])

    def test_unexpected_global_instruction_load_and_unindexed_records_fail_closed(self):
        t = next(t for t in self.plan['trials'] if t['protocol'] == 'P4')
        p = self.root/t['id']/t['cell']/'attempt-1'
        events = [json.loads(s) for s in (p/'generation.stdout').read_text().splitlines()]
        events[0]['part']['state']['metadata']['loaded'] = ['/foreign/AGENTS.md']
        self.fixture.command(p, 'generation', '\n'.join(json.dumps(e) for e in events))
        receipts.seal(self.root/t['id']);pairs.seal(self.root)
        self.assertEqual(pairs.report_pairs(self.root)['observations'][t['id']]['status'], 'invalid-instrumentation')
        index = pairs.read(self.root/'files.json');index.pop('state.json');receipts.write(self.root/'files.json', index)
        with self.assertRaisesRegex(ValueError, 'not indexed'): pairs.report_pairs(self.root)

    def test_changed_tool_schema_invalidates_the_pair(self):
        t = next(t for t in self.plan['trials'] if t['protocol'] == 'P3')
        self.edit_trial(t, 'usage.json', lambda u: u['requests'][0]['audit'].update(tool_schema_sha256='f'*64))
        # Keep the state receipt in sync, to test the cross-leg invariant itself.
        root = self.root/t['id'];s = pairs.read(root/'state.json')
        s['cells'][0]['attempts'][0]['usage'] = pairs.read(root/t['cell']/'attempt-1/usage.json')
        receipts.write(root/'state.json', s);receipts.seal(root);pairs.seal(self.root)
        self.assertEqual(pairs.report_pairs(self.root)['status'], 'incomplete')

    def test_changed_client_version_invalidates_the_pair(self):
        t = self.plan['trials'][1];root = self.root/t['id']
        self.fixture.command(root, 'opencode-version', 'different-client\n');receipts.seal(root);pairs.seal(self.root)
        r = pairs.report_pairs(self.root)
        self.assertEqual(r['observations'][t['id']]['status'], 'invalid-instrumentation')
        self.assertIn('OpenCode versions differ', r['observations'][t['id']]['reason'])

    def test_reused_session_is_not_an_independent_pair(self):
        first, t = self.plan['trials'][:2]
        p = self.root/t['id']/t['cell']/'attempt-1'
        events = [json.loads(s) for s in (p/'generation.stdout').read_text().splitlines()]
        for event in events: event['sessionID'] = 'session-'+first['id']
        self.fixture.command(p, 'generation', '\n'.join(json.dumps(e) for e in events))
        self.edit_trial(t, 'attribution.json', lambda a: a.update(session_id='session-'+first['id']))
        r = pairs.report_pairs(self.root)
        self.assertEqual(r['observations'][t['id']]['status'], 'invalid-instrumentation')
        self.assertIn('Reused', r['observations'][t['id']]['reason'])

    def test_real_static_rejection_is_a_valid_negative_outcome(self):
        t = next(t for t in self.plan['trials'] if t['protocol'] == 'P6')
        root = self.root/t['id'];p = root/t['cell']/'attempt-1'
        source = 'import torch\n'  # The real static guard rejects this host fallback.
        for name in ('candidate.py', 'project/candidate.py'): (p/name).write_text(source)
        for name in ('qualification.json', 'qualification.command.json', 'qualification.stdout', 'qualification.stderr', 'container-cleanup.json'):
            (p/name).unlink()
        s = pairs.read(root/'state.json');s['status'] = s['cells'][0]['status'] = 'failed'
        s['cells'][0]['attempts'][0].update(status='failed', error='Static rejection', failure_stage='qualifying', candidate_sha256=digest(source.encode()))
        receipts.write(root/'state.json', s);receipts.seal(root)
        s = pairs.read(self.root/'state.json');next(i for i in s['trials'] if i['id'] == t['id'])['status'] = 'failed'
        receipts.write(self.root/'state.json', s);pairs.seal(self.root)
        r = pairs.report_pairs(self.root)
        self.assertEqual(r['status'], 'complete');self.assertEqual(r['observations'][t['id']]['outcome'], 'candidate-failed')
        self.assertEqual(r['comparisons']['P6-P3']['qualification_delta'], -1/6)

    def test_changed_numerical_oracle_is_invalid_even_with_self_consistent_hashes(self):
        t = self.plan['trials'][0];root = self.root/t['id'];p = root/t['cell']/'attempt-1'
        adapter = pairs.read(p/'adapter.json');adapter['rtol'] = 1000;receipts.write(p/'adapter.json', adapter)
        q = pairs.read(p/'qualification.json');q['input_sha256']['adapter.json'] = digest((p/'adapter.json').read_bytes());receipts.write(p/'qualification.json', q)
        receipts.seal(root);pairs.seal(self.root)
        r = pairs.report_pairs(self.root)
        self.assertEqual(r['observations'][t['id']]['status'], 'invalid-instrumentation')
        self.assertIn('oracle', r['observations'][t['id']]['reason'])

    def test_blocked_experiment_cannot_report_a_complete_effect(self):
        s = pairs.read(self.root/'state.json');s['status'] = 'blocked';receipts.write(self.root/'state.json', s);pairs.seal(self.root)
        r = pairs.report_pairs(self.root)
        self.assertEqual(r['status'], 'incomplete');self.assertIsNone(r['comparisons']['P6-P3']['qualification_delta'])


class PairedCI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'ci'
        self.env = {'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_REPOSITORY': 'aloschilov/pyasc-skill-stack',
            'GITHUB_REF': 'refs/heads/codex/paired', 'GITHUB_SHA': 'synthetic-revision',
            'GITHUB_WORKFLOW_REF': 'aloschilov/pyasc-skill-stack/.github/workflows/generation-paired.yml@refs/heads/codex/paired',
            'GENERATION_STATE_DIR': str(Path(self.tmp.name)/'state')}

    def test_workflow_is_plan_only_on_pr_and_has_no_cron(self):
        import yaml
        workflow = yaml.safe_load((ROOT/'.github/workflows/generation-paired.yml').read_text())
        self.assertEqual(set(workflow[True]), {'pull_request', 'workflow_dispatch'})
        self.assertEqual(workflow[True]['workflow_dispatch']['inputs']['mode']['default'], 'plan')
        job = workflow['jobs']['paired']
        self.assertEqual(job['uses'], './.github/workflows/generation-simulator.yml')
        self.assertIn("github.event_name == 'workflow_dispatch'", job['with']['action'])
        self.assertIn("|| 'plan'", job['with']['action'])

    def test_plan_and_archive_are_offline_and_private_bytes_are_excluded(self):
        with patch.dict(os.environ, self.env, clear=True), patch.object(ci.subprocess, 'check_output', return_value='synthetic-revision\n'), \
             patch.object(ci, 'run_pairs', side_effect=AssertionError('Planning cannot execute')):
            self.assertEqual(ci.prepare(self.root), 0)
            self.assertEqual(pairs.read(self.root/'result.json')['status'], 'not-run')
            secret = self.root/'run/private/token';secret.parent.mkdir(parents=True);secret.write_text('offline-secret')
            self.assertEqual(ci.collect(self.root), 0)
        from generation_ci import unpack
        archive = self.root/'public/generation.tar.gz';dest = Path(self.tmp.name)/'unpacked'
        unpack(archive, dest, digest(archive.read_bytes()))
        self.assertFalse((dest/'run/private/token').exists())
        self.assertEqual(pairs.read(dest/'plan/plan.json')['kind'], 'paired-generation-plan')

    def test_execution_requires_exact_reviewed_plan_and_manual_caller(self):
        p = pairs.build_pairs();p['checkout_revision'] = 'synthetic-revision'
        self.env.update(GENERATION_ACTION='execute', APPROVE_PLAN=digest(canonical(p)))
        with patch.dict(os.environ, self.env, clear=True), patch.object(ci.subprocess, 'check_output', return_value='synthetic-revision\n'):
            self.assertEqual(ci.prepare(self.root), 0)
        for key, value, message in [('GITHUB_EVENT_NAME', 'schedule', 'not activated'),
                                    ('GITHUB_EVENT_NAME', 'pull_request', 'manual controller'),
                                    ('GITHUB_WORKFLOW_REF', 'wrong', 'manual controller'),
                                    ('APPROVE_PLAN', '0'*64, 'digest mismatch')]:
            with self.subTest(key=key, value=value):
                env = dict(self.env, **{key: value})
                with patch.dict(os.environ, env, clear=True), patch.object(ci.subprocess, 'check_output', return_value='synthetic-revision\n'), \
                     self.assertRaisesRegex(ValueError, message):
                    ci.prepare(Path(self.tmp.name)/(key+'-'+value.replace('/', '_')))


if __name__ == '__main__': unittest.main()
