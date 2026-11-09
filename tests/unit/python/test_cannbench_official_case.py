"""The local case check can fail before Docker and never needs service credentials."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SMOKE = Path(__file__).resolve().parents[3] / 'integrations/cannbench/smoke.py'


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
            self.assertIn('missing.py', result['error'])
            self.assertFalse(result['phases'])
            self.assertNotIn('candidate_sha256', result)
            rerun = self.command(base, 'exp/1')
            self.assertEqual(rerun.returncode, 1)
            self.assertIn('new results directory', rerun.stderr)
            self.assertEqual(evidence.read_bytes(), raw)

    def test_unknown_case_does_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            command = self.command(base, 'transpose/1')
            self.assertEqual(command.returncode, 1)
            self.assertIn('Supported cases', command.stderr)
            self.assertFalse((base / 'result').exists())
