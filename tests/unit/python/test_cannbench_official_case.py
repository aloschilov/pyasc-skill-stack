"""The local case check can fail before Docker and never needs service credentials."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SMOKE = Path(__file__).resolve().parents[3] / 'integrations/cannbench/smoke.py'


def load_checker():
    spec = importlib.util.spec_from_file_location('official_case', SMOKE.parent / 'ci/official_case.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


class OfficialCaseTests(unittest.TestCase):
    def command(self, base, key):
        return subprocess.run([
            sys.executable, str(SMOKE), 'verify-case', '--case', key,
            '--run-dir', str(base / 'result'), '--candidate', str(base / 'missing.py'),
            '--benchmark', str(base / 'unused'),
        ], capture_output=True, text=True)

    def test_missing_source_retained_and_replay_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = self.command(base, 'exp/1')
            self.assertEqual(first.returncode, 1)
            evidence = base / 'result/result.json'
            raw = evidence.read_bytes()
            result = json.loads(raw)
            self.assertEqual((result['stage'], result['status']), ('inputs', 'failed'))
            self.assertEqual(result['benchmark_revision'], '597a915ec49735ee7d53057554bde426ef8c1d96')
            self.assertEqual(result['phase_timeout_seconds'], 300)
            self.assertIn('missing.py', result['error'])
            self.assertFalse(result['phases'])
            self.assertNotIn('candidate_sha256', result)
            rerun = self.command(base, 'exp/1')
            self.assertEqual(rerun.returncode, 1)
            self.assertIn('new results directory', rerun.stderr)
            self.assertEqual(evidence.read_bytes(), raw)

    def test_new_cases_select_the_same_official_revision(self):
        for key in ('maximum/1', 'strided_slice/1', 'gather/1'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                self.assertEqual(self.command(base, key).returncode, 1)
                report = json.loads((base / 'result/result.json').read_text())
                self.assertEqual(report['case'], key)
                self.assertEqual(report['benchmark_revision'], '08d519c503843bce5fd4672ffa2259abeb22fb00')
                self.assertEqual(report['phase_timeout_seconds'], 600)
                self.assertIn('missing.py', report['error'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires the simulator image’s CPU torch')
    def test_suppression_supports_integer_scratch_and_poisons_float_output(self):
        import torch

        candidate_torch = load_checker().PoisonAllocations(torch)
        index = candidate_torch.empty((2, 3), dtype=torch.int32)
        self.assertEqual(index.dtype, torch.int32)
        self.assertEqual(index.count_nonzero().item(), 0)
        source = torch.ones((2, 3), dtype=torch.float16)
        self.assertTrue(candidate_torch.empty_like(source).isnan().all().item())
        self.assertTrue(source.eq(1).all().item())

    def test_simulator_timeout_retains_completed_seed(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            candidate = base / 'candidate.py'
            candidate.write_text('# test input')
            run = base / 'run'

            def simulate(argv, **kwargs):
                if argv[0] == 'git':
                    Path(argv[argv.index('--output') + 1]).write_bytes(b'fixture')
                    return subprocess.CompletedProcess(argv, 0)
                self.assertEqual(argv[0], 'docker')
                self.assertEqual(argv[argv.index('timeout') + 3], '42')
                checker.write(run / 'real/probe.json', {
                    'stage': 'compile-and-launch', 'status': 'running', 'active_seed': 23,
                    'checks': [{'seed': 19, 'numerical_pass': True}],
                })
                return subprocess.CompletedProcess(argv, 124)

            with patch.object(checker.subprocess, 'run', side_effect=simulate):
                self.assertEqual(checker.verify('gather/1', candidate, base, run, phase_seconds=42), 1)
            report = json.loads((run / 'result.json').read_text())
            self.assertEqual(report['phase_timeout_seconds'], 42)
            real = report['phases']['real']
            self.assertEqual((real['status'], real['active_seed'], real['exit_code']), ('timed_out', 23, 124))
            self.assertEqual(real['checks'], [{'seed': 19, 'numerical_pass': True}])
            self.assertNotIn('suppressed', report['phases'])

    def test_invalid_phase_timeout_creates_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cmd = subprocess.run([sys.executable, str(SMOKE), 'verify-case', '--case', 'gather/1',
                                  '--run-dir', str(base/'run'), '--candidate', str(base/'candidate.py'),
                                  '--benchmark', str(base), '--phase-timeout-seconds', '0'],
                                 capture_output=True, text=True)
            self.assertNotEqual(cmd.returncode, 0)
            self.assertIn('positive', cmd.stderr)
            self.assertFalse((base/'run').exists())

    def test_unknown_case_does_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            command = self.command(base, 'transpose/1')
            self.assertEqual(command.returncode, 1)
            self.assertIn('Supported cases', command.stderr)
            self.assertFalse((base / 'result').exists())
