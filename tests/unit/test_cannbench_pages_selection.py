import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('select_pages_evidence', ROOT / 'integrations/cannbench/ci/select_pages_evidence.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class PagesSelectionTests(unittest.TestCase):
    def test_exact_revision_takes_precedence_over_a_newer_run(self):
        runs = [{'id': 3, 'head_sha': 'newer'}, {'id': 2, 'head_sha': 'current'}, {'id': 1, 'head_sha': 'older'}]
        self.assertEqual(selector.select(runs, 'current')['id'], 2)

    def test_dashboard_change_retains_latest_completed_measurements(self):
        runs = [{'id': 3, 'head_sha': 'measured'}, {'id': 2, 'head_sha': 'older'}]
        self.assertEqual(selector.select(runs, 'dashboard-only')['id'], 3)

    def test_first_deployment_has_no_measurement(self):
        self.assertIsNone(selector.select([], 'current'))


if __name__ == '__main__':
    unittest.main()
