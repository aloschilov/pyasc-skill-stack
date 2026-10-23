"""Exercise the real CLI rejection and workflow argument transport without VM work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


class NumericalInputTests(unittest.TestCase):
    def test_cli_rejects_missing_reports_before_creating_campaign(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output = Path(tmp) / 'campaign'
            for extra in ([], ['--numerical-reports', str(Path(tmp) / 'absent.json')]):
                result = subprocess.run(
                    [sys.executable, str(ROOT / 'integrations/cannbench/ci/run.py'),
                     '--output', str(output), '--generate', '--submit', *extra],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--numerical-reports', result.stderr)
                self.assertFalse(output.exists())

    def test_workflow_transports_literal_report_path_and_rejects_missing_file(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/cannbench.yml').read_text())
        step = next(s for s in workflow['jobs']['cannbench-gate']['steps']
                    if s.get('name') == 'Capture full catalog, generate and validate exact hardware results')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / 'argv.json'
            stub = root / 'python3'
            stub.write_text('#!' + sys.executable + '\n'
                            'import json,os,sys\n'
                            'if sys.argv[1] == "-":\n'
                            ' sys.stdin.read(); print("{}")\n'
                            'else:\n'
                            ' open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            reports = root / 'reports $(touch SHOULD_NOT_EXIST) `literal`.json'
            reports.write_text('[]')
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       CAPTURE=str(capture), GENERATE='true', SUBMIT='true',
                       CONTINUE_SUBMISSIONS='false', REUSE_RUN_ID='',
                       NUMERICAL_REPORTS=str(reports), CANNBENCH_OUTPUT=str(root / 'output'))
            result = subprocess.run(['bash', '-c', step['run']], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(capture.read_text())
            self.assertEqual(args[args.index('--numerical-reports') + 1], str(reports))
            self.assertFalse((root / 'SHOULD_NOT_EXIST').exists())
            capture.unlink()
            reports.unlink()
            result = subprocess.run(['bash', '-c', step['run']], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())


if __name__ == '__main__':
    unittest.main()
