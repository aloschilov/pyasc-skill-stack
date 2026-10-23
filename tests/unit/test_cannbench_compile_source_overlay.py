import importlib
import hashlib
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integrations/cannbench/workers'))
gate = importlib.import_module('local_compile_gate')
driver = importlib.import_module('driver')
prompts = importlib.import_module('prompts')


class CompileOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'cases.yaml'
        self.cases = [{'case_id': 1, 'attrs': {'epsilon': '1e-5'}}, {'case_id': 2}]
        self.path.write_text(json.dumps({'cases': self.cases}))

    def test_restoration_preserves_all_cases_and_evidence(self):
        restored = {'case_id': 1, 'attrs': {'epsilon': 1e-5}}
        with patch('reconcile_case_source.load_verified_overlay', side_effect=[(restored, {'source_sha256': 'source'}), None]) as load:
            cases, evidence = gate.load_compile_cases(self.path, '/overlay', 'hash')
        self.assertEqual(cases, [restored, self.cases[1]])
        self.assertEqual(evidence, {'1': {'source_sha256': 'source'}})
        self.assertEqual(load.call_count, 2)
        self.assertEqual(json.loads(self.path.read_text())['cases'], self.cases)

    def test_corrupted_overlay_fails_without_raw_fallback(self):
        with patch('reconcile_case_source.load_verified_overlay', side_effect=ValueError('hash mismatch')):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                gate.load_compile_cases(self.path, '/overlay', 'hash')

    def test_overlay_requires_hash_and_cannot_change_case_identity(self):
        with self.assertRaisesRegex(ValueError, 'paired'):
            gate.load_compile_cases(self.path, '/overlay')
        with patch('reconcile_case_source.load_verified_overlay', return_value=({'case_id': 99}, {})):
            with self.assertRaisesRegex(ValueError, 'identity'):
                gate.load_compile_cases(self.path, '/overlay', 'hash')

    def test_historical_raw_mode_remains_explicit(self):
        self.assertEqual(gate.load_compile_cases(self.path), (self.cases, {}))

    def test_driver_binds_overlay_to_container_relative_path(self):
        root = Path(self.tmp.name).resolve()
        overlay = root / 'overlay'
        overlay.mkdir()
        (overlay / 'MANIFEST.json').write_text('{"records": []}')
        digest = hashlib.sha256((overlay / 'MANIFEST.json').read_bytes()).hexdigest()
        with patch.object(driver, 'REPO_ROOT', root), patch.dict(os.environ, {
                'CANNBENCH_SOURCE_OVERLAY': str(overlay),
                'CANNBENCH_SOURCE_OVERLAY_SHA256': digest}):
            self.assertEqual(driver.compile_overlay_arguments(),
                             ['--source-overlay', 'overlay', '--source-overlay-sha256', digest])
            (overlay / 'MANIFEST.json').write_text('changed')
            with patch.object(driver.subprocess, 'Popen') as launch:
                result = driver.local_evaluate('exp', root/'candidate.py', root, root)
            self.assertTrue(result['hard_failure'])
            self.assertIn('hash mismatch', result['error'])
            launch.assert_not_called()

    def test_driver_rejects_overlay_outside_mounted_repository(self):
        with patch.object(driver, 'REPO_ROOT', Path(self.tmp.name)), patch.dict(os.environ, {
                'CANNBENCH_SOURCE_OVERLAY': '/outside/overlay',
                'CANNBENCH_SOURCE_OVERLAY_SHA256': 'hash'}):
            with self.assertRaises(ValueError):
                driver.compile_overlay_arguments()

    def test_generation_context_uses_same_restored_cases(self):
        for name in ('desc.md', 'proto.yaml', 'golden.py'):
            (self.path.parent / name).write_text('unchanged source')
        restored = [{'case_id': 1, 'attrs': {'epsilon': 1e-5}}, self.cases[1]]
        with patch.object(gate, 'load_compile_cases', return_value=(restored, {'1': {'source_sha256': 'source'}})) as load:
            message = prompts.build_generation_prompt('exp', 'exp', 'exp', '-', self.path.parent,
                source_overlay='/overlay', source_overlay_sha256='hash')
        load.assert_called_once_with(self.path, '/overlay', 'hash')
        import yaml
        rendered = message.split('## cases.yaml\n```yaml\n', 1)[1].split('\n```', 1)[0]
        self.assertEqual(yaml.safe_load(rendered)['cases'], restored)
        self.assertIn('Overlay manifest SHA256: hash', message)
        self.assertEqual(json.loads(self.path.read_text())['cases'], self.cases)

    def test_generation_context_rejects_invalid_restoration(self):
        with patch.object(gate, 'load_compile_cases', side_effect=ValueError('source drift')):
            with self.assertRaisesRegex(ValueError, 'source drift'):
                prompts.build_generation_prompt('exp', 'exp', 'exp', '-', self.path.parent,
                    source_overlay='/overlay', source_overlay_sha256='hash')


if __name__ == '__main__':
    unittest.main()
