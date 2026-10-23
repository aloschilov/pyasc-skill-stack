import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cannbench_dashboard', ROOT / 'tests/tools/generate_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class DashboardEvidenceTests(unittest.TestCase):
    def test_missing_report_replaces_stale_output_with_every_official_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / 'site'; public = output / 'cannbench'; public.mkdir(parents=True)
            (public / 'hardware-summary.json').write_text(json.dumps({'gate_passed': True, 'correct_cases': 1060}))
            with patch.object(dashboard, 'CANNBENCH_SUMMARY_FILE', root / 'missing/hardware-summary.json'), patch.object(dashboard.sys, 'argv', ['generate_dashboard.py', '--output-dir', str(output)]):
                dashboard.main()
            report = json.loads((public / 'hardware-summary.json').read_text())
            self.assertFalse(report['gate_passed'])
            self.assertFalse(report['skill_stack_gate_passed'])
            self.assertEqual((report['required_operators'], report['required_cases'], report['correct_cases']), (53, 1060, 0))
            self.assertEqual(sum(len(op['cases']) for op in report['operators']), 1060)

    def test_mismatched_report_scope_cannot_display_passed_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'hardware-summary.json'
            source.write_text(json.dumps({'benchmark_slug': 'official-tasks', 'benchmark_version': 'old', 'required_operators': 9, 'required_cases': 180, 'gate_passed': True, 'correct_cases': 180}))
            with patch.object(dashboard, 'CANNBENCH_SUMMARY_FILE', source):
                catalog, report = dashboard.cannbench_documents()
            self.assertEqual(catalog['required_cases'], 1060)
            self.assertFalse(report['gate_passed'])
            self.assertEqual(report['correct_cases'], 0)
            self.assertIn('disagrees with catalog', report['errors'][0])


if __name__ == '__main__':
    unittest.main()
