"""Offline policy tests using synthetic receipts, never generation/qualification evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
from generation_compare import Bundle, compare
from generation_plan import build_plan, canonical, digest

NOW = datetime(2026, 11, 6, 6, tzinfo=timezone.utc)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(root):
    plan = json.loads((root/'plan.json').read_text()); plan.pop('plan_sha256', None)
    plan['plan_sha256'] = digest(canonical(plan)); write(root/'plan.json', plan)
    state = json.loads((root/'state.json').read_text()); state['plan_sha256'] = plan['plan_sha256']
    write(root/'state.json', state)
    files = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in root.rglob('*')
             if p.is_file() and p.name != 'files.json'}
    write(root/'files.json', files)
    return digest((root/'files.json').read_bytes())


class Comparisons(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.base = self.root/'baseline'; self.base.mkdir()
        plan = build_plan(); plan['checkout_revision'] = 'synthetic-offline-fixture'
        write(self.base/'plan.json', plan)
        for name in plan['source_sha256']:
            dest = self.base/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
        self.command(self.base, 'opencode-version', '1.19.0\n')
        state = {'status': 'passed', 'inputs_unchanged': True, 'cells': [],
                 'started_at': '2026-11-06T04:00:00+00:00', 'finished_at': '2026-11-06T04:05:00+00:00'}
        for cell in plan['cells']:
            name = cell['id']; p = self.base/name/'attempt-1'; (p/'project').mkdir(parents=True)
            # Deliberately not executable kernel code: these are verifier unit doubles.
            for file in ('candidate.py', 'project/candidate.py'): (p/file).write_text('offline double')
            sha = digest((p/'candidate.py').read_bytes())
            write(p/'adapter.json', {'seeds': [0], 'arguments': ['x', 'out']})
            check = {'passed': True, 'after_launch': True, 'api': 'torch.allclose', 'line': 10}
            real = {'launcher_calls': 1, 'mock_launch': False, 'numerical_checks': [check, check]}
            control = {'launcher_calls': 1, 'mock_launch': True, 'status': 'fail', 'exit_code': 1,
                       'numerical_failure': dict(check, passed=False)}
            q = {'status': 'passed', 'backend': 'Model', 'inputs_unchanged': True,
                 'verification_level': 'numerically_verified', 'static': {'status': 'passed'},
                 'input_sha256': {'candidate.py': sha, 'adapter.json': digest((p/'adapter.json').read_bytes())},
                 'phases': {'jit': {'exit_code': 0, 'report': {'launcher_calls': 1}},
                            'execution': {'exit_code': 0, 'report': real}, 'control': {'exit_code': 1, 'report': control}}}
            write(p/'qualification.json', q)
            write(p/'container-cleanup.json', {'confirmed': True})
            for cmd in ('generation', 'qualification'): self.command(p, cmd, 'synthetic receipt')
            write(p/'attribution.json', {'session_id': 'baseline-'+name, 'loaded_skills': plan['skills'],
                'requested_model': plan['model']['requested'], 'observed_model': plan['model']['expected_response']})
            usage = {'usage_complete': True, 'error': None, 'requests': [
                {'status': 'complete', 'usage': {'observed_model': plan['model']['expected_response']}}]}
            write(p/'usage.json', usage)
            state['cells'].append({'id': name, 'status': 'passed', 'attempts': [
                {'number': 1, 'status': 'passed', 'candidate_sha256': sha, 'usage': usage}]})
        write(self.base/'state.json', state); self.pin = seal(self.base)
        self.new = self.root/'candidate'; shutil.copytree(self.base, self.new)
        for p in self.new.glob('*/attempt-1/attribution.json'):
            value = json.loads(p.read_text()); value['session_id'] = value['session_id'].replace('baseline-', 'candidate-'); write(p, value)
        seal(self.new)

    def command(self, path, name, out):
        (path/(name+'.stdout')).write_text(out); (path/(name+'.stderr')).write_text('')
        write(path/(name+'.command.json'), {'status': 'completed', 'exit_code': 0,
            'stdout_sha256': digest(out.encode()), 'stderr_sha256': digest(b'')})

    def edit(self, file, change, root=None):
        root = root or self.new; p = root/file
        obj = json.loads(p.read_text()); change(obj); write(p, obj); seal(root)

    def compare(self):
        return compare(Bundle(self.base, self.pin), Bundle(self.new), NOW)

    def test_fresh_complete_run_passes_without_model_runtime_or_commands(self):
        result = self.compare()
        self.assertEqual((result['status'], result['passed'], result['denominator']), ('passed', 3, 3))
        self.assertEqual(result['alerts'], [])

    def test_new_binding_cannot_be_removed_but_legacy_receipts_remain_readable(self):
        plan = json.loads((self.base/'plan.json').read_text())
        for field in ('profile_sha256', 'configuration'): plan['model'].pop(field)
        write(self.base/'plan.json', plan); seal(self.base)
        with self.assertRaisesRegex(ValueError, 'Missing model profile binding'): Bundle(self.base)
        helper = 'tests/tools/generation_models.py'
        plan['source_sha256'].pop(helper); (self.base/'source'/helper).unlink()
        write(self.base/'plan.json', plan); seal(self.base)
        with self.assertRaisesRegex(ValueError, 'source closure'): Bundle(self.base)
        planner = 'tests/tools/generation_plan.py'
        source = b'# Synthetic pre-binding planner source, not live evidence.\n'
        (self.base/'source'/planner).write_bytes(source); plan['source_sha256'][planner] = digest(source)
        write(self.base/'plan.json', plan); seal(self.base)
        self.assertIsNone(Bundle(self.base).model_binding_schema_version)

    def test_one_lost_pass_is_regression_not_an_arbitrary_aggregate_threshold(self):
        def fail(s):
            s['status'] = 'failed'; s['cells'][0]['status'] = 'failed'
            s['cells'][0]['attempts'][0].update(status='failed', error='Candidate did not qualify')
        self.edit('state.json', fail)
        result = self.compare()
        self.assertEqual((result['status'], result['passed'], result['denominator']), ('regression', 2, 3))
        self.assertEqual(result['regressions'], ['add-f32-tail'])

    def test_partial_and_stale_results_do_not_shrink_the_denominator(self):
        def block(s):
            s['status'] = 'blocked'; s['cells'][1]['status'] = 'blocked'; s['cells'][1]['attempts'][0]['status'] = 'blocked'
            s['cells'][2].update(status='not-run', attempts=[])
        self.edit('state.json', block)
        result = self.compare()
        self.assertEqual((result['status'], result['passed'], result['denominator']), ('incomplete', 1, 3))
        self.assertEqual(result['coverage_gaps'], ['reduce-sum', 'broadcast-f32'])
        self.edit('state.json', lambda s: s.update(started_at='2026-11-01T00:00:00Z', finished_at='2026-11-01T00:05:00Z'))
        self.assertIn('stale-coverage', self.compare()['alerts'])

    def test_missing_future_or_naive_completion_time_is_not_fresh(self):
        for value in (None, '2026-11-07T00:00:00Z', '2026-11-06T05:00:00'):
            self.edit('state.json', lambda s: s.update(finished_at=value))
            self.assertIn('unknown-coverage-time', self.compare()['alerts'])

    def test_complete_stale_run_fails_and_freshness_boundary_is_explicit(self):
        self.edit('state.json', lambda s: s.update(started_at='2026-11-04T17:00:00Z', finished_at='2026-11-04T18:00:00Z'))
        self.assertEqual(self.compare()['status'], 'passed')  # Exactly 36 hours.
        self.edit('state.json', lambda s: s.update(finished_at='2026-11-04T17:59:59Z'))
        result = self.compare()
        self.assertEqual((result['status'], result['passed']), ('incomplete', 3))
        self.assertEqual(result['alerts'], ['stale-coverage'])

    def test_regression_and_gaps_are_both_visible(self):
        def partial(s):
            s['status'] = 'blocked'
            s['cells'][0]['status'] = 'failed'; s['cells'][0]['attempts'][0].update(status='failed', error='Candidate did not qualify')
            s['cells'][1]['status'] = 'blocked'; s['cells'][1]['attempts'][0]['status'] = 'blocked'
            s['cells'][2].update(status='not-run', attempts=[])
        self.edit('state.json', partial)
        result = self.compare()
        self.assertEqual(result['status'], 'regression')
        self.assertEqual(result['alerts'], ['lost-pass', 'incomplete-run'])
        self.assertEqual((result['passed'], result['denominator']), (0, 3))

    def test_coherently_reduced_inventory_still_cannot_improve_the_denominator(self):
        self.edit('plan.json', lambda p: p['cells'].pop())
        self.edit('state.json', lambda s: s['cells'].pop())
        result = self.compare()
        self.assertEqual(result['status'], 'incompatible')
        self.assertIn('cells', result['incompatible_fields']); self.assertEqual(result['denominator'], 3)

    def test_skills_and_stack_revision_may_change_but_are_reported(self):
        name = 'skills/pyasc-api-patterns/SKILL.md'
        p = self.new/'source'/name; p.write_text(p.read_text()+'\nNew guidance.\n')
        self.edit('plan.json', lambda p: p['source_sha256'].update({name: digest((self.new/'source'/name).read_bytes())}))
        self.edit('state.json', lambda s: s.update(checkout_revision='another-commit'))
        result = self.compare()
        self.assertEqual(result['status'], 'passed'); self.assertEqual(result['changed_skill_files'], [name])

    def test_changed_budget_runtime_model_client_or_verifier_is_incompatible(self):
        for key, change in [
            ('limits', lambda p: p['limits'].update(rounds_per_attempt=9)),
            ('runtime', lambda p: p['runtime'].update(image='different@sha256:'+'f'*64)),
        ]:
            original = (self.new/'plan.json').read_text()
            self.edit('plan.json', change)
            self.assertIn(key, self.compare()['incompatible_fields'])
            (self.new/'plan.json').write_text(original); seal(self.new)
        plan = json.loads((self.new/'plan.json').read_text())
        old = 'integrations/opencode/profiles/'+plan['model']['profile']+'.json'
        new = 'integrations/opencode/profiles/different-config.json'
        shutil.copyfile(self.new/'source'/old, self.new/'source'/new)
        plan['model']['profile'] = 'different-config'; plan['source_sha256'][new] = plan['source_sha256'][old]
        write(self.new/'plan.json', plan); seal(self.new)
        self.assertIn('model', self.compare()['incompatible_fields'])
        (self.new/'plan.json').write_text(original); (self.new/'source'/new).unlink(); seal(self.new)
        self.command(self.new, 'opencode-version', '1.20.0\n'); seal(self.new)
        self.assertIn('opencode_version', self.compare()['incompatible_fields'])
        self.command(self.new, 'opencode-version', '1.19.0\n')
        name = 'tests/tools/generation_adapters.py'
        (self.new/'source'/name).write_text('changed oracle')
        self.edit('plan.json', lambda p: p['source_sha256'].update({name: digest(b'changed oracle')}))
        self.assertIn('source_sha256', self.compare()['incompatible_fields'])

    def test_dropped_cell_same_session_and_unpinned_baseline_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, 'Baseline.*digest'): Bundle(self.base, '0'*64)
        with self.assertRaisesRegex(ValueError, 'Same run/session'): compare(Bundle(self.base), Bundle(self.base), NOW)
        self.edit('add-f32-tail/attempt-1/attribution.json', lambda a: a.update(session_id='baseline-add-f32-tail'))
        with self.assertRaisesRegex(ValueError, 'Same run/session'): self.compare()
        self.edit('state.json', lambda s: s['cells'].pop())
        with self.assertRaisesRegex(ValueError, 'cell results'): self.compare()

    def test_hash_mismatch_missing_index_symlink_and_escaping_path_fail_closed(self):
        p = self.new/'add-f32-tail/attempt-1/candidate.py'; p.write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'): Bundle(self.new)
        seal(self.new)
        with self.assertRaisesRegex(ValueError, 'Candidate changed'): Bundle(self.new)
        p.unlink(); p.symlink_to(self.base/'add-f32-tail/attempt-1/candidate.py')
        seal(self.new)
        with self.assertRaisesRegex(ValueError, 'Symlink'): Bundle(self.new)
        index = self.new/'files.json'; write(index, {'../outside': '0'*64})
        with self.assertRaisesRegex(ValueError, 'Unsafe'): Bundle(self.new)

    def test_missing_required_index_entry_and_nonterminal_baseline_are_rejected(self):
        index = self.new/'files.json'; value = json.loads(index.read_text())
        value.pop('add-f32-tail/attempt-1/container-cleanup.json'); write(index, value)
        with self.assertRaisesRegex(ValueError, 'not indexed'): Bundle(self.new)
        seal(self.new)
        self.edit('state.json', lambda s: s.update(status='running'))
        with self.assertRaisesRegex(ValueError, 'not terminal'): Bundle(self.new)

    def test_partial_baseline_and_empty_failure_are_not_accepted(self):
        self.edit('state.json', lambda s: s.update(status='blocked'), root=self.base)
        self.pin = digest((self.base/'files.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Baseline must be'): self.compare()
        def empty_failure(s):
            s['status'] = 'failed'; s['cells'][0].update(status='failed', attempts=[])
        self.edit('state.json', empty_failure)
        with self.assertRaisesRegex(ValueError, 'failure evidence'): Bundle(self.new)

    def test_rehashed_missing_negative_control_and_wrong_model_still_fail(self):
        p = 'add-f32-tail/attempt-1/qualification.json'
        original = (self.new/p).read_text()
        self.edit(p, lambda q: q['phases']['control'].update(exit_code=0))
        with self.assertRaisesRegex(ValueError, 'negative-control'): Bundle(self.new)
        (self.new/p).write_text(original); seal(self.new)
        self.edit('add-f32-tail/attempt-1/attribution.json', lambda a: a.update(observed_model='unexpected'))
        with self.assertRaisesRegex(ValueError, 'attribution'): Bundle(self.new)

    def test_cli_writes_external_report_and_never_overwrites_it(self):
        report = self.root/'comparison.json'
        argv = [sys.executable, str(ROOT/'tests/tools/generation_compare.py'), '--baseline', str(self.base),
                '--baseline-sha256', self.pin, '--candidate', str(self.new), '--report', str(report)]
        # Synthetic wall-clock receipts are not current: test stable invalid-evidence exit instead.
        argv[argv.index(self.pin)] = '0'*64
        p = subprocess.run(argv, capture_output=True)
        self.assertEqual(p.returncode, 2); self.assertEqual(json.loads(report.read_text())['status'], 'invalid-evidence')
        before = report.read_bytes(); p = subprocess.run(argv, capture_output=True)
        self.assertNotEqual(p.returncode, 0); self.assertEqual(report.read_bytes(), before)

    def test_cli_exit_codes_distinguish_success_regression_and_incomplete(self):
        stamp = datetime.now(timezone.utc).isoformat()
        self.edit('state.json', lambda s: s.update(started_at=stamp, finished_at=stamp))
        for expected_code, status in [(0, 'passed'), (1, 'regression'), (2, 'incomplete')]:
            if expected_code == 1:
                def fail(s):
                    s['status'] = 'failed'; s['cells'][0]['status'] = 'failed'
                    s['cells'][0]['attempts'][0].update(status='failed', error='wrong result')
                self.edit('state.json', fail)
            if expected_code == 2:
                def block(s):
                    s['status'] = 'blocked'; s['cells'][0]['status'] = 'blocked'
                    s['cells'][0]['attempts'][0]['status'] = 'blocked'
                self.edit('state.json', block)
            report = self.root/f'report-{expected_code}.json'
            p = subprocess.run([sys.executable, str(ROOT/'tests/tools/generation_compare.py'),
                '--baseline', str(self.base), '--baseline-sha256', self.pin,
                '--candidate', str(self.new), '--report', str(report)], capture_output=True)
            self.assertEqual(p.returncode, expected_code, p.stderr)
            self.assertEqual(json.loads(report.read_text())['status'], status)


if __name__ == '__main__': unittest.main()
