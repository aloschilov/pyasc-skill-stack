"""Run the documented import probes against fake packages, without PyASC/NPU."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT/'skills/asctile-env-check'
SCRIPT = (SKILL/'scripts/check_env.sh').read_text()
PROBES = {
    'quick command': re.search(r'python3 -c "\n(.*?)\n"', (SKILL/'SKILL.md').read_text(), re.S)[1],
    'shell helper': re.search(r"ASCTILE_PROBE='\n(.*?)\n'", SCRIPT, re.S)[1],
}


class ImportDiagnostics(unittest.TestCase):
    def fixture(self, root, files):
        for name, source in {'asc/__init__.py': '', **files}.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def run_probe(self, root, source):
        # Isolated mode excludes any user-site runtime; the fixture is first.
        command = 'import sys; sys.path.insert(0, ' + repr(str(root)) + ')\n' + source
        return subprocess.run([sys.executable, '-I', '-c', command], capture_output=True, text=True, timeout=10)

    def test_modern_is_preferred_and_true_legacy_layout_still_works(self):
        modern = {'asc/experimental/__init__.py': '', 'asc/experimental/asctile.py': "LAYOUT = 'modern'\n"}
        legacy = {'asctile.py': "LAYOUT = 'legacy'\n"}
        for files, selected in [(modern, 'modern'), (legacy, 'legacy'), ({**modern, **legacy}, 'modern'),
                                ({'asc/experimental/__init__.py': '', **legacy}, 'legacy')]:
            for name, source in PROBES.items():
                with self.subTest(probe=name, selected=selected, files=list(files)), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp); self.fixture(root, files)
                    result = self.run_probe(root, source + "\nprint('selected:', asctile.LAYOUT)")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('selected: ' + selected, result.stdout)

    def test_internal_dependency_and_native_errors_never_select_legacy(self):
        missing = 'raise ModuleNotFoundError("No module named \'pybind11\'", name="pybind11")\n'
        native = 'raise ImportError("native extension: undefined symbol fixture_abi")\n'
        for failing_file, error in [('asc/__init__.py', missing), ('asc/experimental/__init__.py', missing),
                                    ('asc/experimental/asctile.py', missing), ('asc/experimental/asctile.py', native)]:
            for name, source in PROBES.items():
                with self.subTest(probe=name, failure=failing_file, error=error), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.fixture(root, {'asc/experimental/__init__.py': '', 'asc/experimental/asctile.py': '',
                                        'asctile.py': "print('LEGACY_LOADED')\n", failing_file: error})
                    result = self.run_probe(root, source)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('LEGACY_LOADED', result.stdout)
                    self.assertNotIn('During handling of the above exception', result.stderr)
                    self.assertTrue(result.stderr.rstrip().endswith(
                        "ModuleNotFoundError: No module named 'pybind11'" if error == missing
                        else 'ImportError: native extension: undefined symbol fixture_abi'), result.stderr)

    def test_no_runtime_or_asctile_is_reported_without_fallback_masking(self):
        for name, source in PROBES.items():
            with self.subTest(probe=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.fixture(root, {})
                result = self.run_probe(root, source)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr.rstrip().endswith("ModuleNotFoundError: No module named 'asc.experimental'"))

    def test_shell_emits_metadata_before_preserved_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, {
                'asc/__init__.py': 'raise ModuleNotFoundError("No module named \'pybind11\'", name="pybind11")\n',
                'pyasc_dev-0.0.dist-info/METADATA':
                    'Metadata-Version: 2.1\nName: pyasc-dev\nVersion: 0.0\nRequires-Dist: pybind11\n',
            })
            wrapper = root/'python-fixture'
            wrapper.write_text(f'#!{sys.executable} -I\nimport sys\nsys.path.insert(0, {str(root)!r})\n'
                               "if sys.argv[1] == '--version':\n    print('Python 3.12.0')\n"
                               "else:\n    exec(sys.argv[2])\n")
            wrapper.chmod(0o700)
            # Only metadata and runtime checks; never run framework or device probes.
            prefix = SCRIPT.split('# numpy\n', 1)[0]
            result = subprocess.run(['bash', '-c', prefix], env={**os.environ, 'PYTHON': str(wrapper)},
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10)
            output = result.stdout
            self.assertIn('Python SOABI:', output)
            self.assertIn('Requires-Dist:', output)
            self.assertLess(output.index('Requires-Dist:'), output.index('Traceback'))
            self.assertIn("ModuleNotFoundError: No module named 'pybind11'", output)
            self.assertIn('[FAIL] pyasc/asctile import failed', output)
            self.assertNotIn('pip install', output)


if __name__ == '__main__':
    unittest.main()
