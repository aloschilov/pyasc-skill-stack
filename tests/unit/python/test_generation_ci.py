"""Offline CI routing, immutable artifacts and pre-dispatch rejection contracts."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import generation_ci as ci
from generation_plan import canonical, digest
import test_generation_compare as receipts


class GenerationCI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'ci'
        self.env = {'GENERATION_ACTION': 'plan', 'APPROVE_PLAN': '', 'BASELINE_REF': '', 'BASELINE_SHA256': '',
            'GENERATION_STATE_DIR': str(self.root.parent/'shared-state'),
            'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_REPOSITORY': 'owner/repo',
            'GITHUB_SHA': 'synthetic-offline-fixture',
            'GITHUB_EVENT_NAME': 'workflow_dispatch'}

    def prepare(self, execute=False):
        if execute:
            plan = ci.build_plan(); plan['checkout_revision'] = 'synthetic-offline-fixture'
            self.env.update(GENERATION_ACTION='execute', APPROVE_PLAN=digest(canonical(plan)))
        with patch.dict(os.environ, self.env), patch.object(ci.subprocess, 'check_output', return_value='synthetic-offline-fixture\n'):
            ci.prepare(self.root)

    def fixture(self):
        fixture = receipts.Comparisons(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        return fixture.base

    def test_plan_only_never_calls_executor_and_collects_an_explicit_not_run(self):
        with patch.object(ci, 'execute', side_effect=AssertionError('No model calls')):
            self.prepare(); ci.collect(self.root)
        self.assertEqual(ci.read(self.root/'result.json')['status'], 'not-run')
        with tarfile.open(self.root/'public/generation.tar.gz') as t:
            self.assertIn('plan/plan.json', t.getnames()); self.assertNotIn('run/state.json', t.getnames())
        self.assertIn('not a generation result', (self.root/'public/SUMMARY.md').read_text())

    def test_wrong_approval_and_incomplete_baseline_reference_fail_before_execution(self):
        self.env.update(GENERATION_ACTION='execute', APPROVE_PLAN='0'*64)
        with patch.dict(os.environ, self.env), patch.object(ci.subprocess, 'check_output', return_value='synthetic-offline-fixture'), \
             patch.object(ci, 'execute', side_effect=AssertionError('No model calls')):
            with self.assertRaisesRegex(ValueError, 'digest'): ci.prepare(self.root)
        other = self.root.parent/'other'; self.env.update(BASELINE_REF='123/1')
        with patch.dict(os.environ, self.env):
            with self.assertRaisesRegex(ValueError, 'are a pair'): ci.prepare(other)

    def test_approved_execution_qualifies_but_never_promotes_a_baseline(self):
        self.prepare(execute=True); source = self.fixture()
        def execute(plan, run, state_dir):
            self.assertEqual(state_dir, (self.root.parent/'shared-state').resolve())
            shutil.copytree(source, run)
            return ci.read(run/'state.json')
        with patch.object(ci, 'execute', side_effect=execute) as invoked, \
             patch.object(ci.subprocess, 'check_output', return_value='1.19.0\n'), patch.object(ci.subprocess, 'run'):
            self.assertEqual(ci.run(self.root), 0); self.assertEqual(invoked.call_count, 1)
        result = ci.read(self.root/'result.json')
        self.assertEqual(result['status'], 'qualified'); self.assertFalse(result['compared'])
        self.assertTrue(result['model_invoked']); self.assertFalse(result['hardware_invoked'])

    def test_incompatible_baseline_is_rejected_before_a_provider_call(self):
        self.prepare(execute=True); source = self.fixture()
        shutil.copytree(source, self.root/'baseline/run')
        request = ci.read(self.root/'ci.json'); request.update(baseline_ref='123/1', baseline_index_sha256=digest((source/'files.json').read_bytes()))
        ci.save(self.root/'ci.json', request)
        with patch.object(ci.subprocess, 'check_output', return_value='different-client'), patch.object(ci.subprocess, 'run'), \
             patch.object(ci, 'execute', side_effect=AssertionError('No model calls')):
            with self.assertRaisesRegex(ValueError, 'incompatible'): ci.run(self.root)
        self.assertFalse((self.root/'run').exists())

    def test_public_archive_round_trip_excludes_private_bytes(self):
        self.prepare(); shutil.copytree(self.fixture(), self.root/'run')
        private = self.root/'run/add-f32-tail/attempt-1/private'; private.mkdir()
        (private/'raw-response.bin').write_text('private-wire-secret-sentinel')
        ci.collect(self.root); archive = self.root/'public/generation.tar.gz'
        with tarfile.open(archive) as t:
            self.assertTrue(all('private' not in Path(name).parts for name in t.getnames()))
            self.assertTrue(all(b'private-wire-secret-sentinel' not in t.extractfile(m).read() for m in t))
        ci.unpack(archive, self.root/'restored', digest(archive.read_bytes()))
        restored = ci.Bundle(self.root/'restored/run')
        self.assertEqual(restored.state['status'], 'passed')
        with self.assertRaisesRegex(ValueError, 'Never replace'): ci.unpack(archive, self.root/'restored', digest(archive.read_bytes()))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'): ci.unpack(archive, self.root/'bad', '0'*64)

    def test_tar_traversal_symlink_duplicate_and_corrupt_members_are_rejected(self):
        for i, (name, symlink, duplicate) in enumerate([('../escape', False, False), ('run/link', True, False),
                                                       ('run/file', False, True), ('run/file', False, False)]):
            archive = self.root.parent/f'bad-{i}.tar.gz'
            with tarfile.open(archive, 'w:gz') as t:
                m = tarfile.TarInfo(name); data = b'x'; m.size = len(data)
                if symlink: m.type = tarfile.SYMTYPE; m.linkname = '/secret'
                t.addfile(m, io.BytesIO(data))
                if duplicate: t.addfile(m, io.BytesIO(data))
                index = json.dumps({name: 'wrong-hash'}).encode(); n = tarfile.TarInfo('archive-index.json'); n.size = len(index)
                t.addfile(n, io.BytesIO(index))
            with self.assertRaises(ValueError): ci.unpack(archive, self.root.parent/f'out-{i}', digest(archive.read_bytes()))
        self.assertFalse((self.root.parent/'escape').exists())

    def test_failed_preflight_collection_is_incomplete_not_green_qualification(self):
        self.prepare(execute=True); ci.collect(self.root)
        result = ci.read(self.root/'result.json')
        self.assertEqual(result['status'], 'incomplete'); self.assertIsNone(result['model_invoked'])
        self.assertEqual(result['alerts'], ['missing-terminal-result'])

    def test_cli_replay_preserves_original_result_and_never_dispatches(self):
        self.prepare(); before = (self.root/'result.json').read_bytes()
        for action in ('prepare', 'execute'):
            (self.root/'run').mkdir(exist_ok=True)
            with patch.object(sys, 'argv', ['generation_ci.py', action, '--root', str(self.root)]), \
                 patch.object(ci, 'execute', side_effect=AssertionError('No model calls')):
                self.assertEqual(ci.main(), 2)
            self.assertEqual((self.root/'result.json').read_bytes(), before)

    def test_selected_run_and_release_use_exact_downloads_and_validate_origins(self):
        self.prepare(execute=True); shutil.copytree(self.fixture(), self.root/'run')
        state = ci.read(self.root/'run/state.json'); state['checkout_revision'] = 'synthetic-offline-fixture'
        ci.save(self.root/'run/state.json', state); receipts.seal(self.root/'run')
        ci.save(self.root/'result.json', {'status': 'qualified', 'hardware_invoked': False})
        ci.collect(self.root)
        archive = self.root/'public/generation.tar.gz'; sha = digest(archive.read_bytes())
        for ref in ('123/1', 'release/reviewed-baseline'):
            consumer = self.root.parent/ref.replace('/', '-')
            consumer.mkdir(); request = ci.read(self.root/'ci.json')
            request.update(action='execute', baseline_ref=ref, baseline_sha256=sha); ci.save(consumer/'ci.json', request)
            def download(argv, **kwargs):
                shutil.copyfile(archive, Path(argv[argv.index('--dir')+1])/'generation.tar.gz')
            with patch.object(ci.subprocess, 'run', side_effect=download) as called:
                self.assertEqual(ci.fetch(consumer), 0)
                argv = called.call_args.args[0]
                self.assertEqual(argv[:2], ['gh', 'release' if ref.startswith('release/') else 'run'])
                self.assertNotIn('latest', argv)
            self.assertIn('baseline_index_sha256', ci.read(consumer/'ci.json'))
        consumer = self.root.parent/'wrong-attempt'; consumer.mkdir()
        request.update(baseline_ref='123/2'); ci.save(consumer/'ci.json', request)
        with patch.object(ci.subprocess, 'run', side_effect=download):
            with self.assertRaisesRegex(ValueError, 'run/attempt mismatch'): ci.fetch(consumer)

        wrong = self.root.parent/'wrong-revision'
        shutil.copytree(self.root, wrong, ignore=shutil.ignore_patterns('public'))
        origin = ci.read(wrong/'ci.json'); origin['context']['GITHUB_SHA'] = 'different-revision'; ci.save(wrong/'ci.json', origin)
        ci.collect(wrong); archive = wrong/'public/generation.tar.gz'
        consumer = self.root.parent/'wrong-revision-consumer'; consumer.mkdir()
        request.update(baseline_ref='123/1', baseline_sha256=digest(archive.read_bytes())); ci.save(consumer/'ci.json', request)
        with patch.object(ci.subprocess, 'run', side_effect=download):
            with self.assertRaisesRegex(ValueError, 'revision mismatch'): ci.fetch(consumer)

    def test_workflow_has_no_schedule_or_implicit_model_hardware_execution(self):
        import yaml
        workflow = yaml.safe_load((ROOT/'.github/workflows/generation-simulator.yml').read_text())
        self.assertEqual(set(workflow[True]), {'workflow_call'})
        self.assertFalse(workflow['concurrency']['cancel-in-progress'])
        job = workflow['jobs']['qualify']; self.assertIn('workflow_dispatch', job['if'])
        steps = job['steps']; execute = next(s for s in steps if 'generation_ci.py execute' in s.get('run', ''))
        self.assertEqual(execute['if'], "inputs.action == 'execute'")
        self.assertEqual(set(execute['env']), {'OPENCODE_BASE_URL', 'OPENCODE_API_KEY'})
        for s in steps:
            if s is not execute: self.assertNotIn('OPENCODE_API_KEY', s.get('env', {}))
        self.assertIn('always()', next(s for s in steps if 'generation_ci.py collect' in s.get('run', ''))['if'])
        ci_workflow = yaml.safe_load((ROOT/'.github/workflows/ci.yml').read_text())
        self.assertNotIn('schedule', ci_workflow[True])
        self.assertEqual(ci_workflow[True]['workflow_dispatch']['inputs']['generation_action']['default'], 'none')
        self.assertIn('workflow_dispatch', ci_workflow['jobs']['generation-simulator']['if'])
        self.assertIn('generation_action', ci_workflow['jobs']['merge-gate']['if'])
        storage = next(s for s in steps if s.get('name') == 'Select persistent external evidence directory')['run']
        self.assertIn('RUNNER_WORKSPACE', storage)
        self.assertIn('GENERATION_STATE_DIR=$base/state', storage)
        self.assertNotIn('GENERATION_ROOT=$HOME', storage)
        dispatch = next(s for s in ci_workflow['jobs']['pr-gate']['steps'] if s.get('name') == 'Validate manual dispatch selection')
        import subprocess
        for hardware, generation, success in [('none', 'execute', True), ('generate', 'none', True), ('generate', 'execute', False)]:
            result = subprocess.run(['bash', '-e', '-c', dispatch['run']], env=dict(os.environ,
                HARDWARE_ACTION=hardware, GENERATION_ACTION=generation), capture_output=True)
            self.assertEqual(result.returncode == 0, success)


if __name__ == '__main__': unittest.main()
