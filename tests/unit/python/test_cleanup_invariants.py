"""The prose rename must not rewrite historical evidence or legacy detectors."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
from score_kernel import score


class CleanupInvariants(unittest.TestCase):
    def test_frozen_evidence_and_detectors_are_byte_identical(self):
        baseline = json.loads((ROOT/'docs/audits/cleanup-invariants-20261030.json').read_text())
        for name, digest in baseline['sha256'].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)

    def test_score_check_names_preserve_historical_schema(self):
        baseline = json.loads((ROOT/'docs/audits/cleanup-invariants-20261030.json').read_text())
        for name, serialized in baseline['check_names_json'].items():
            with self.subTest(path=name):
                frozen = json.loads((ROOT/name).read_text())
                self.assertEqual(json.dumps(sorted(frozen['checks']),separators=(',',':')), serialized)
                current = score(str(ROOT/Path(name).parent/'kernel.py'))
                self.assertEqual(json.dumps(sorted(c.name for c in current.checks),separators=(',',':')), serialized)


if __name__=='__main__':unittest.main()
