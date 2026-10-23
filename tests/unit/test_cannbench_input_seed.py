"""Canonical official seed binding without executing torch/CANN."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'integrations/cannbench/workers'))
from run_camodel_cases import official_input_seed

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'integrations/cannbench/catalogs/official-tasks-1.1.2-20261001'


class OfficialSeedTests(unittest.TestCase):
    def test_all_1060_canonical_ids_and_offsets(self):
        manifest = json.loads((CATALOG / 'manifest.json').read_text())
        seen = set()
        for op in manifest['operators']:
            for case_id in op['case_ids']:
                num = int(case_id.rsplit('_', 1)[1])
                result = official_input_seed(CATALOG / 'tasks' / op['function_name'], num, 0)
                self.assertEqual(result['case_id'], case_id)
                self.assertTrue(0 <= result['case_seed'] < 2**31)
                plus = official_input_seed(CATALOG / 'tasks' / op['function_name'], num, 2**31 + 7)
                self.assertEqual(plus['case_seed'], (result['case_seed'] + 7) % (2**31))
                seen.add(result['case_id'])
        self.assertEqual(len(seen), 1060)
        self.assertEqual(official_input_seed(CATALOG / 'tasks/exp', 1, 0)['case_seed'], 895033750)

    def test_invalid_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            official_input_seed(CATALOG / 'tasks/exp', 999, 0)

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_bytes((CATALOG / 'manifest.json').read_bytes())
            task = root / 'tasks/exp'
            task.mkdir(parents=True)
            (task / 'cases.yaml').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                official_input_seed(task, 1, 0)

    def test_boolean_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer'):
            official_input_seed(CATALOG / 'tasks/exp', 1, True)
