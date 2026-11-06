"""Offline handoff transition tests; doubles do not claim model qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import generation_review as review
import generation_review_evidence as receipts
from generation_adapters import write_probes
from generation_executor import Blocked
from generation_plan import canonical, digest
from generation_protocols import provider_prompt, skill_body


class Handoffs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.plan = review.build_review()
        self.plan['checkout_revision'] = 'offline-fixture'
        self.plan['plan_sha256'] = digest(canonical(self.plan))
        self.calls = []; self.verifications = []
        self.reject_first = False; self.unchanged = False; self.reuse = False; self.mutate = False

    def finding(self, disposition='open'):
        return {'id': 'R1', 'severity': 'blocking', 'location': 'candidate.py:1',
            'problem': 'Incorrect tail', 'repair': 'Fix tail bounds', 'disposition': disposition}

    def verdict(self, bound, reject=False, repaired=False):
        return {'schema_version': 1, 'binding': bound, 'verdict': 'reject' if reject else 'accept',
            'findings': [self.finding('fixed' if repaired else 'open')] if reject or repaired else [],
            'scope': 'Static review only; simulator verifier still required.'}

    def fake_session(self, attempt, plan, role, source, contract, inputs):
        attempt.mkdir(parents=True); self.calls.append((role, deepcopy(inputs)))
        number = len([r for r, _ in self.calls if r == 'writer'])
        review.save(attempt/'usage.json', {'usage_complete': True, 'observed_output_tokens': 15})
        if role == 'writer':
            (attempt/'candidate.py').write_text('candidate-v'+str(1 if self.unchanged else number))
            (attempt/'design.md').write_text('design-v'+str(number))
        else:
            review.save(attempt/'review.json', self.verdict(json.loads(inputs['binding.json']),
                reject=self.reject_first and number == 1, repaired=self.reject_first and number > 1))
            if self.mutate: (attempt.parent/'writer/candidate.py').write_text('changed-after-review')
        return {'session_id': 'same' if self.reuse else 'session-'+str(len(self.calls)),
            'observed_model': plan['common']['model']['expected_response'], 'role': role}

    def fake_qualify(self, path, plan, contract, source):
        self.verifications.append(path)
        review.save(path/'qualification.json', {'status': 'passed'})

    def run_double(self, session=None, qualify=None, validated=True):
        with patch.dict(os.environ, OPENCODE_API_KEY='offline', OPENCODE_BASE_URL='http://127.0.0.1'), \
             patch.object(review.subprocess, 'check_output', return_value='offline-fixture\n'), \
             patch.object(review, 'command', return_value={'exit_code': 0}), \
             patch.object(receipts, 'report_review', return_value={'validated_acceptance': validated}), \
             patch.object(review, 'session', side_effect=session or self.fake_session), \
             patch.object(review, 'qualify', side_effect=qualify or self.fake_qualify):
            return review.run_review(self.plan, self.root/'run', self.root/'state')

    def test_plan_is_offline_and_binds_roles_limits_and_source(self):
        self.assertEqual(self.plan['maximum_work'], {'model_sessions': 4, 'output_tokens': 772608, 'execution_seconds': 3000})
        self.assertIn('skills/pyasc-code-review/SKILL.md', self.plan['common']['source_sha256'])
        path = self.root/'plan.json'; review.save(path, self.plan)
        with patch.object(review.subprocess, 'Popen', side_effect=AssertionError('Offline planning')):
            self.assertEqual(review.validate_plan(path), self.plan)
        changed = deepcopy(self.plan); changed.pop('plan_sha256'); changed['spec']['max_repairs'] = 100
        changed['plan_sha256'] = digest(canonical(changed)); review.save(path, changed)
        with self.assertRaisesRegex(ValueError, 'inputs changed'): review.validate_plan(path)

    def test_acceptance_needs_fresh_review_and_verification_of_same_hash(self):
        result = self.run_double()
        self.assertEqual(result['status'], 'accepted'); self.assertEqual(len(self.verifications), 1)
        row = result['cycles'][0]
        self.assertNotEqual(row['writer']['session_id'], row['reviewer']['session_id'])
        self.assertEqual(row['binding'], row['review']['binding'])
        self.assertEqual(result['accepted_candidate_sha256'], digest((self.verifications[0]/'candidate.py').read_bytes()))
        self.assertFalse(result['independent_model']); self.assertTrue(result['usage_complete'])
        self.assertEqual(result['reserved_output_tokens'], 386304)
        self.assertFalse((self.root/'state/active.json').exists())
        self.assertTrue((self.root/'run/files.json').exists())

    def test_rejection_repairs_with_lineage_then_new_review_and_verification(self):
        self.reject_first = True; result = self.run_double()
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual([r for r, _ in self.calls], ['writer', 'reviewer', 'writer', 'reviewer'])
        first, last = result['cycles']
        self.assertEqual(first['status'], 'rejected')
        self.assertEqual(last['parent_candidate_sha256'], first['candidate_sha256'])
        self.assertNotEqual(last['candidate_sha256'], first['candidate_sha256'])
        self.assertEqual(last['review']['findings'][0]['disposition'], 'fixed')
        self.assertIn('prior-review.json', self.calls[-1][1])
        self.assertEqual(len(self.verifications), 1)

    def test_rejection_without_repair_budget_never_verifies_or_accepts(self):
        self.reject_first = True; self.plan['spec']['max_repairs'] = 0
        result = self.run_double()
        self.assertEqual(result['status'], 'rejected'); self.assertIsNone(result['accepted_candidate_sha256'])
        self.assertFalse(self.verifications)

    def test_unchanged_repair_stops_before_reviewer_or_verifier(self):
        self.reject_first = self.unchanged = True; result = self.run_double()
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('Unchanged repair', result['cycles'][-1]['error'])
        self.assertEqual(len(self.calls), 3); self.assertFalse(self.verifications)

    def test_verifier_failure_requires_repair_and_review_again(self):
        def verify(path, plan, contract, source):
            self.verifications.append(path)
            if len(self.verifications) == 1:
                review.save(path/'qualification.json', {'status': 'failed', 'phases': {'jit': {'exit_code': 1, 'report': {'error': 'bad rank'}}}})
                raise ValueError('Candidate did not qualify')
        result = self.run_double(qualify=verify)
        self.assertEqual(result['status'], 'accepted'); self.assertEqual(len(self.calls), 4)
        feedback = json.loads(self.calls[2][1]['feedback.json'])
        self.assertEqual(feedback['verification']['phases']['jit']['error'], 'bad rank')

    def test_unknown_or_interrupted_role_keeps_latch_and_cannot_restart(self):
        for failure in (Blocked('Unknown delivery'), KeyboardInterrupt('Interrupted')):
            with self.subTest(failure=type(failure).__name__):
                if (self.root/'run').exists():
                    # Separate independent fixture, never a runtime recovery step.
                    self.root = self.root/'another'; self.root.mkdir()
                result = self.run_double(session=lambda *a: (_ for _ in ()).throw(failure))
                self.assertEqual(result['status'], 'blocked'); self.assertTrue((self.root/'state/active.json').exists())
                with self.assertRaisesRegex(ValueError, 'Never replay'): self.run_double()

    def test_known_budget_rejection_stops_and_preserves_usage(self):
        def exhausted(path, *args):
            path.mkdir(parents=True); review.save(path/'usage.json', {'usage_complete': True, 'observed_output_tokens': 8192})
            raise review.RoleRejected('Role budget exhausted')
        result = self.run_double(session=exhausted)
        self.assertEqual(result['status'], 'rejected'); self.assertEqual(result['observed_output_tokens'], 8192)
        self.assertFalse((self.root/'state/active.json').exists()); self.assertFalse(self.verifications)

    def test_failed_evidence_recheck_clears_tentative_acceptance_and_keeps_latch(self):
        result=self.run_double(validated=False)
        self.assertEqual(result['status'],'blocked');self.assertIsNone(result['accepted_candidate_sha256'])
        self.assertTrue((self.root/'state/active.json').exists())

    def test_mutation_or_reused_session_blocks_acceptance(self):
        self.mutate = True; result = self.run_double()
        self.assertEqual(result['status'], 'blocked'); self.assertFalse(self.verifications)
        self.root = self.root/'next'; self.root.mkdir(); self.mutate = False; self.reuse = True
        result = self.run_double(); self.assertEqual(result['status'], 'blocked')
        self.assertIn('not fresh', result['error'])

    def test_review_schema_rejects_stale_verdict_unresolved_blockers_and_missing_dispositions(self):
        bound = {'candidate_sha256': 'a'*64, 'design_sha256': 'b'*64, 'task_sha256': 'c'*64}
        valid = self.verdict(bound); self.assertEqual(review.validate_review(valid, bound), valid)
        bad = self.verdict(bound, reject=True); bad['verdict'] = 'accept'
        with self.assertRaisesRegex(ValueError, 'unresolved'): review.validate_review(bad, bound)
        with self.assertRaisesRegex(ValueError, 'unbound'): review.validate_review(valid, dict(bound, candidate_sha256='d'*64))
        prior = self.verdict(bound, reject=True)
        with self.assertRaisesRegex(ValueError, 'dispositions'): review.validate_review(valid, bound, prior)
        repaired = self.verdict(bound, repaired=True); review.validate_review(repaired, bound, prior)
        with self.assertRaisesRegex(ValueError, 'predecessor'): review.validate_review(repaired, bound)
        downgraded=self.verdict(bound,reject=True);downgraded['verdict']='accept';downgraded['findings'][0]['severity']='advisory'
        with self.assertRaisesRegex(ValueError,'severity downgrade'):review.validate_review(downgraded,bound,prior)

    def test_native_role_context_is_readonly_complete_and_attributed(self):
        contract = review.task_contract(ROOT, self.plan['common']['cells'][0])
        candidate, design = b'candidate fixture', b'design fixture'
        bound = review.binding(candidate, design, contract)
        inputs = {'candidate.py': candidate, 'design.md': design, 'binding.json': canonical(bound)}
        def fake_run(attempt, plan, project, config, message, observe):
            context = attempt/'context'; private = attempt/'private'; private.mkdir()
            self.assertEqual(config['permission']['glob'], 'deny')
            self.assertEqual(config['permission']['grep'], 'deny')
            self.assertNotIn('candidate.py', config['permission']['edit'])
            self.assertEqual(config['permission']['edit']['review.json'], 'allow')
            parts = []; content = []
            for skill in plan['skills']:
                directory = context/'skills'/skill; body = skill_body((directory/'SKILL.md').read_text()); content.append(body)
                parts.append({'tool': 'skill', 'state': {'status': 'completed', 'input': {'name': skill},
                    'metadata': {'dir': str(directory)}, 'output': body}})
            for p in (context/'inputs').iterdir():
                body = p.read_text().rstrip('\n'); content.append('\n'.join(f'{i}: {line}' for i, line in enumerate(body.splitlines(), 1)))
                parts.append({'tool': 'read', 'state': {'status': 'completed', 'input': {'filePath': str(p)},
                    'metadata': {'display': {'lineStart': 1, 'text': body}}, 'output': body}})
            if getattr(self, 'bad_access', False):
                parts.append({'tool': 'read', 'state': {'status': 'completed', 'input': {'filePath': '/not-allowed'}, 'output': 'unplanned'}})
            body = {'messages': [{'role': 'user', 'content': provider_prompt(message)}, {'role': 'tool', 'content': '\n'.join(content)}]}
            audit = observe(1, body, canonical(body))
            usage = {'requests': [{'audit': audit}], 'usage_complete': True, 'error': None, 'observed_output_tokens': 15}
            (attempt/'generation.stdout').write_text('\n'.join(json.dumps({'sessionID': 'review-session', 'part': p}) for p in parts))
            review.save(project/'review.json', self.verdict(bound))
            return {'exit_code': 0}, usage
        with patch.object(review, 'run_session', side_effect=fake_run):
            record = review.session(self.root/'native', self.plan, 'reviewer', ROOT, contract, inputs)
            self.assertEqual(record['read_inputs'], sorted(['task.json', *inputs]))
            self.assertEqual(record['loaded_skills'], sorted(review.REVIEW_SKILLS))
            self.assertEqual(record['observed_model'], self.plan['common']['model']['expected_response'])
            self.bad_access = True
            with self.assertRaisesRegex(Blocked, 'Unexpected role file access'):
                review.session(self.root/'bad-native', self.plan, 'reviewer', ROOT, contract, inputs)


class ReviewReceipts(unittest.TestCase):
    """Synthetic full receipts exercise the real auditor; never a live claim."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.run = self.root/'run'; self.phase = 0
        self.plan = review.build_review(); self.plan['checkout_revision'] = 'offline-fixture'
        self.plan['plan_sha256'] = digest(canonical(self.plan))
        with patch.dict(os.environ, OPENCODE_API_KEY='offline', OPENCODE_BASE_URL='http://127.0.0.1'), \
             patch.object(review.subprocess, 'check_output', return_value='offline-fixture\n'), \
             patch.object(review, 'command', side_effect=self.command), \
             patch.object(review, 'run_session', side_effect=self.client), \
             patch.object(review, 'qualify', side_effect=self.verifier):
            result = review.run_review(self.plan, self.run, self.root/'state')
        self.assertEqual(result['status'], 'accepted', result.get('error'))

    def command(self, argv, path, name, *args):
        out = '1.19.0\n' if name == 'opencode-version' else json.dumps([{'RepoDigests': [self.plan['common']['runtime']['image']]}])
        return self.receipt(path, name, argv, out)

    def receipt(self, path, name, argv, out):
        (path/(name+'.stdout')).write_text(out); (path/(name+'.stderr')).write_text('')
        record = {'argv': argv, 'status': 'completed', 'exit_code': 0, 'duration_seconds': 1.0,
            'stdout_sha256': digest(out.encode()), 'stderr_sha256': digest(b'')}
        review.save(path/(name+'.command.json'), record); return record

    def client(self, attempt, plan, project, config, message, observe):
        self.phase += 1; context = attempt/'context'; private = attempt/'private'; private.mkdir()
        role = review.read(attempt/'isolation.json')['role']
        provider = plan['model']['requested'].split('/')[0]
        config['provider'][provider]['options'] = {'baseURL': 'http://127.0.0.1:1234/v1', 'apiKey': '{env:PYASC_GATE_TOKEN}'}
        review.save(attempt/'model-config.json', config); (attempt/'prompt.txt').write_text(message)
        parts = []; content = []
        for skill in plan['skills']:
            directory = context/'skills'/skill; body = skill_body((directory/'SKILL.md').read_text()); content.append(body)
            parts.append({'tool': 'skill', 'state': {'status': 'completed', 'input': {'name': skill},
                'metadata': {'dir': str(directory)}, 'output': body}})
        for p in (context/'inputs').iterdir():
            body = p.read_text().rstrip('\n'); content.append('\n'.join(f'{i}: {line}' for i, line in enumerate(body.splitlines(), 1)))
            parts.append({'tool': 'read', 'state': {'status': 'completed', 'input': {'filePath': str(p)},
                'metadata': {'display': {'lineStart': 1, 'text': body}}, 'output': body}})
        body = {'messages': [{'role': 'user', 'content': provider_prompt(message)}, {'role': 'tool', 'content': '\n'.join(content)}]}
        audit = observe(1, body, canonical(body))
        usage = {'requests': [{'audit': audit, 'status': 'complete', 'usage': {'observed_model': plan['model']['expected_response'],
            'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}],
            'usage_complete': True, 'error': None, 'observed_output_tokens': 20}
        review.save(attempt/'usage.json', usage)
        out = '\n'.join(json.dumps({'sessionID': 'session-'+str(self.phase), 'part': p}) for p in parts)
        cmd = self.receipt(attempt, 'generation', ['opencode', 'run', '--pure', '--format', 'json', '--dir', str(project), '-m', plan['model']['requested'], message], out)
        if role != 'reviewer':
            candidate=(ROOT/'tests/generation/fixtures/add.py').read_bytes()
            if role=='tuner': candidate+=b'\n# Offline tuning proposal fixture, not a measured optimization.\n'
            (project/'candidate.py').write_bytes(candidate)
            (project/'design.md').write_text('Synthetic design fixture; not a model result.')
        else:
            review.save(project/'review.json', {'schema_version': 1, 'binding': review.read(context/'inputs/binding.json'),
                'verdict': 'accept', 'findings': [], 'scope': 'Synthetic static-review fixture; no live reviewer.'})
        return cmd, usage

    def verifier(self, path, plan, contract, source):
        write_probes(path, contract)
        check = {'passed': True, 'after_launch': True, 'api': 'torch.allclose', 'line': 10}
        real = {'launcher_calls': len(contract['seeds']), 'mock_launch': False,
            'numerical_checks': [check]*(len(contract['seeds'])*len(contract['arguments']))}
        control = {'launcher_calls': len(contract['seeds']), 'mock_launch': True, 'status': 'fail', 'exit_code': 1,
            'numerical_failure': dict(check, passed=False)}
        q = {'status': 'passed', 'backend': 'Model', 'inputs_unchanged': True, 'verification_level': 'numerically_verified', 'static': {'status': 'passed'},
            'input_sha256': {n: digest((path/n).read_bytes()) for n in ('candidate.py', 'adapter.json', 'probe.py', 'compile_probe.py')},
            'phases': {'jit': {'exit_code': 0, 'report': {'launcher_calls': len(contract['seeds'])}},
                'execution': {'exit_code': 0, 'report': real}, 'control': {'exit_code': 1, 'report': control}}}
        review.save(path/'qualification.json', q); review.save(path/'container-cleanup.json', {'confirmed': True})
        self.receipt(path, 'qualification', ['offline-verifier-double'], 'Synthetic verification receipt, never hardware evidence.')

    def edit(self, name, change):
        path = self.run/name; value = review.read(path); change(value); review.save(path, value); review.seal(self.run)

    def test_report_revalidates_handoff_and_numerical_proof_without_execution(self):
        with patch.object(review, 'run_session', side_effect=AssertionError('No model')), \
             patch.object(review, 'qualify', side_effect=AssertionError('No verifier')):
            report = receipts.report_review(self.run)
        self.assertTrue(report['validated_acceptance']); self.assertEqual(len(report['sessions']), 2)
        self.assertEqual(report['repair_attempts'], 0); self.assertEqual(report['output_tokens'], 40)
        self.assertFalse(report['performance_claim']); self.assertFalse(report['independent_model'])
        self.assertEqual(report['model_binding_schema_version'], 1)

    def reseal_plan(self, plan):
        plan.pop('plan_sha256')
        plan['plan_sha256'] = digest(canonical(plan))
        review.save(self.run/'plan.json', plan)
        state = review.read(self.run/'state.json'); state['plan_sha256'] = plan['plan_sha256']
        review.save(self.run/'state.json', state); review.seal(self.run)

    def test_legacy_receipt_is_readable_without_claiming_new_binding_or_allowing_execution(self):
        # Synthetic old closure/schema, never evidence of a real model run.
        plan = review.read(self.run/'plan.json')
        for field in ('profile_sha256', 'configuration'): plan['common']['model'].pop(field)
        helper = 'tests/tools/generation_models.py'
        plan['common']['source_sha256'].pop(helper); (self.run/'source'/helper).unlink()
        self.reseal_plan(plan)
        report = receipts.report_review(self.run)
        self.assertTrue(report['validated_acceptance']); self.assertIsNone(report['model_binding_schema_version'])
        with self.assertRaisesRegex(ValueError, 'inputs changed'): review.validate_plan(self.run/'plan.json')

    def test_rehashed_new_binding_cannot_be_removed_or_changed(self):
        original = review.read(self.run/'plan.json')
        for mutate in (lambda model: model.pop('configuration'),
                       lambda model: model['configuration'].update(provider_revision='invented'),
                       lambda model: model.update(profile_sha256='0'*64)):
            plan = deepcopy(original); mutate(plan['common']['model']); self.reseal_plan(plan)
            with self.assertRaisesRegex(ValueError, 'binding'): receipts.report_review(self.run)

    def test_rehashed_stale_review_cannot_accept_the_candidate(self):
        self.edit('cycle-1/reviewer/review.json', lambda v: v['binding'].update(candidate_sha256='0'*64))
        with self.assertRaises(ValueError): receipts.report_review(self.run)

    def test_rehashed_missing_negative_control_cannot_accept(self):
        self.edit('cycle-1/verification/qualification.json', lambda v: v['phases']['control']['report'].update(mock_launch=False))
        with self.assertRaisesRegex(ValueError, 'negative control'): receipts.report_review(self.run)

    def test_rehashed_role_model_drift_cannot_accept(self):
        self.edit('cycle-1/reviewer/usage.json', lambda v: v['requests'][0]['usage'].update(observed_model='another-model'))
        with self.assertRaisesRegex(ValueError, 'response model'): receipts.report_review(self.run)

    def test_rehashed_underreservation_cannot_accept(self):
        self.edit('state.json', lambda v: v.update(reserved_output_tokens=40))
        with self.assertRaisesRegex(ValueError, 'Reservations'): receipts.report_review(self.run)

    def test_changed_evaluator_cannot_accept_even_after_rehash(self):
        self.edit('cycle-1/verification/adapter.json', lambda v: v.update(rtol=1e6))
        self.edit('cycle-1/verification/qualification.json', lambda v: v['input_sha256'].update(
            {'adapter.json': digest((self.run/'cycle-1/verification/adapter.json').read_bytes())}))
        with self.assertRaisesRegex(ValueError, 'oracle'): receipts.report_review(self.run)

    def test_unindexed_file_and_false_accepted_state_fail_closed(self):
        (self.run/'extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'index'): receipts.report_review(self.run)
        (self.run/'extra.json').unlink()
        self.edit('state.json', lambda v: v.update(accepted_candidate_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'candidate changed'): receipts.report_review(self.run)


class ReviewCI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = {'GITHUB_RUN_ID': '99', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_REPOSITORY': 'aloschilov/pyasc-skill-stack',
            'GITHUB_SHA': 'offline-revision', 'GITHUB_REF': 'refs/heads/codex/review', 'GITHUB_EVENT_NAME': 'workflow_dispatch',
            'GITHUB_WORKFLOW_REF': 'aloschilov/pyasc-skill-stack/.github/workflows/generation-review.yml@refs/heads/codex/review',
            'GENERATION_STATE_DIR': str(self.root/'shared-state')}

    def test_plan_archive_without_provider_and_no_private_data(self):
        from generation_ci import collect, unpack
        with patch.dict(os.environ, self.env, clear=True), \
             patch.object(review.subprocess, 'check_output', return_value='offline-revision\n'), \
             patch.object(review, 'run_review', side_effect=AssertionError('No execution')):
            self.assertEqual(review.prepare_ci(self.root/'ci'), 0)
            result = review.read(self.root/'ci/result.json')
            self.assertFalse(result['model_invoked']); self.assertEqual(result['status'], 'not-run')
            private = self.root/'ci/run/private'; private.mkdir(parents=True); (private/'credential').write_text('offline-sensitive-fixture')
            collect(self.root/'ci')
            sha = (self.root/'ci/public/generation.tar.gz.sha256').read_text().split()[0]
            unpack(self.root/'ci/public/generation.tar.gz', self.root/'unpacked', sha)
            self.assertFalse(any(p.name == 'private' for p in (self.root/'unpacked').rglob('*')))

    def test_execution_rejects_wrong_digest_trigger_caller_and_revision(self):
        plan = review.build_review(); plan['checkout_revision'] = 'offline-revision'; approval = digest(canonical(plan))
        cases = [({'GENERATION_ACTION': 'execute', 'APPROVE_PLAN': approval}, True),
            ({'GENERATION_ACTION': 'execute'}, False),
            ({'GENERATION_ACTION': 'execute', 'APPROVE_PLAN': approval, 'GITHUB_EVENT_NAME': 'pull_request'}, False),
            ({'GENERATION_ACTION': 'execute', 'APPROVE_PLAN': approval, 'GITHUB_EVENT_NAME': 'schedule'}, False),
            ({'GENERATION_ACTION': 'execute', 'APPROVE_PLAN': approval, 'GITHUB_WORKFLOW_REF': 'unexpected'}, False),
            ({'GITHUB_SHA': 'different'}, False)]
        for i, (changes, expected) in enumerate(cases):
            with patch.dict(os.environ, dict(self.env, **changes), clear=True), \
                 patch.object(review.subprocess, 'check_output', return_value='offline-revision\n'), \
                 patch.object(review, 'run_review', side_effect=AssertionError('No execution during prepare')):
                if expected: self.assertEqual(review.prepare_ci(self.root/str(i)), 0)
                else:
                    with self.assertRaises(ValueError): review.prepare_ci(self.root/str(i))

    def test_workflow_reuses_worker_and_only_plans_on_pr(self):
        import yaml
        workflow = yaml.safe_load((ROOT/'.github/workflows/generation-review.yml').read_text())
        self.assertEqual(set(workflow[True]), {'pull_request', 'workflow_dispatch'})
        self.assertEqual(workflow[True]['workflow_dispatch']['inputs']['mode']['default'], 'plan')
        job = workflow['jobs']['review']
        self.assertEqual(job['uses'], './.github/workflows/generation-simulator.yml')
        self.assertEqual(job['with']['experiment'], 'review')
        self.assertIn("github.event_name == 'workflow_dispatch'", job['if'])
        self.assertEqual(job['with']['action'],'execute')
        prepare=workflow['jobs']['prepare']
        self.assertEqual(prepare['runs-on'],'ubuntu-latest')
        self.assertNotIn('secrets',prepare)
        self.assertNotIn('OPENCODE_API_KEY',json.dumps(prepare))
        worker = yaml.safe_load((ROOT/'.github/workflows/generation-simulator.yml').read_text())
        self.assertEqual(worker['concurrency']['group'], 'generation-simulator')
        self.assertIn("inputs.experiment != 'review'", worker['jobs']['qualify']['if'])
        self.assertIn('generation_review.py', worker['jobs']['qualify']['env']['GENERATION_HELPER'])


if __name__ == '__main__': unittest.main()
