"""Permanent historical fixtures preserve scoring behavior across implementation changes."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
from score_kernel import score


class HistoricalScoring(unittest.TestCase):
    def test_frozen_evidence_is_byte_identical(self):
        hashes = json.loads((ROOT/'tests/fixtures/frozen-evidence-sha256.json').read_text())
        for name, digest in hashes.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)

    def test_score_outcomes_preserve_historical_schema_and_behavior(self):
        for path in sorted((ROOT/'docs/evidence/generation-gate-20261028').glob('run-*/score.json')):
            with self.subTest(path=str(path)):
                frozen = json.loads(path.read_text())
                current = score(str(path.parent/'kernel.py'))
                self.assertEqual(sorted(c.name for c in current.checks), sorted(frozen['checks']))
                self.assertEqual({c.name: c.passed for c in current.checks},
                                 {key: value['passed'] for key, value in frozen['checks'].items()})
                self.assertEqual(current.failure_category, frozen['failure_category'])
                self.assertEqual(current.gating_detail, frozen['gating_detail'])


if __name__=='__main__':unittest.main()
