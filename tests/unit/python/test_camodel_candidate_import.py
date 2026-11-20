"""Smoke source attribution, using ordinary Python packages and no tensor runtime."""
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


WORKERS = Path(__file__).resolve().parents[3] / 'integrations/cannbench/workers'
spec = importlib.util.spec_from_file_location('candidate_import', WORKERS / 'candidate_import.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CandidateImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.saved_modules = dict(sys.modules)
        self.saved_path = list(sys.path)
        self.addCleanup(self.restore_imports)

    def restore_imports(self):
        sys.path[:] = self.saved_path
        for name in list(sys.modules):
            if name == 'cann_bench' or name.startswith(('cann_bench.', '_camodel_candidate_')) or name == 'foreign':
                if name in self.saved_modules:
                    sys.modules[name] = self.saved_modules[name]
                else:
                    del sys.modules[name]

    def package(self, parent, marker):
        root = self.base / parent / 'cann_bench'
        root.mkdir(parents=True)
        (root / '__init__.py').write_text('from .exp import exp\n')
        (root / '_pyasc_runtime.py').write_text('def ensure_npu_platform(): pass\n')
        (root / 'helper.py').write_text(f'value = {marker!r}\n')
        (root / 'exp.py').write_text(
            'from ._pyasc_runtime import ensure_npu_platform\n'
            'def exp():\n'
            '    ensure_npu_platform()\n'
            '    from .helper import value\n'
            '    return value\n')
        return root

    def test_installed_and_cached_competitors_do_not_replace_requested_package(self):
        requested = self.package('requested', 'requested')
        installed = self.package('installed', 'installed')
        sys.path.insert(0, str(installed.parent))
        for cached in (False, True):
            with self.subTest(cached=cached):
                if cached:
                    other = importlib.import_module('cann_bench')
                    self.assertEqual(other.exp(), 'installed')
                with loader.load_candidate_package(requested, ['exp']) as (package, binding):
                    self.assertEqual(package.exp(), 'requested')
                    self.assertEqual(binding['resolved_root'], str(requested.resolve()))
                    for key, source in [('package', requested / '__init__.py'),
                                        ('exp', requested / 'exp.py')]:
                        record = binding['package'] if key == 'package' else binding['operators'][key]
                        self.assertEqual(record['requested_path'], str(source.resolve()))
                        self.assertEqual(record['imported_path'], str(source.resolve()))
                        self.assertEqual(record['requested_sha256'], digest(source))
                        self.assertEqual(record['imported_sha256'], digest(source))
                    name = package.__name__
                    self.assertEqual(binding['operators']['exp']['function_code_path'],
                                     str((requested / 'exp.py').resolve()))
                    self.assertIn(name + '.helper', sys.modules)  # Late relative import works.
                    if cached:
                        self.assertIs(sys.modules['cann_bench'], other)
                self.assertFalse(any(n == name or n.startswith(name + '.') for n in sys.modules))

    def test_missing_file_and_namespace_roots_rejected_despite_installed_package(self):
        installed = self.package('installed', 'installed')
        sys.path.insert(0, str(installed.parent))
        other = importlib.import_module('cann_bench')
        namespace = self.base / 'namespace' / 'cann_bench'
        namespace.mkdir(parents=True)
        (namespace / 'exp.py').write_text('def exp(): return "requested"\n')
        for root in (self.base / 'missing' / 'cann_bench', namespace, namespace / 'exp.py'):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, 'containing __init__.py'):
                with loader.load_candidate_package(root, ['exp']):
                    self.fail('Invalid requested root was accepted')
        self.assertIs(sys.modules['cann_bench'], other)

    def test_successive_same_named_packages_keep_relative_helpers_separate(self):
        first = self.package('first', 'first')
        second = self.package('second', 'second')
        with loader.load_candidate_package(first, ['exp']) as (left, _):
            with loader.load_candidate_package(second, ['exp']) as (right, _):
                self.assertNotEqual(left.__name__, right.__name__)
                self.assertEqual(left.exp(), 'first')
                self.assertEqual(right.exp(), 'second')
            self.assertEqual(left.exp(), 'first')

    def test_foreign_export_is_rejected_and_private_modules_are_removed(self):
        root = self.package('requested', 'requested')
        (self.base / 'foreign.py').write_text('def exp(): return "foreign"\n')
        sys.path.insert(0, str(self.base))
        (root / '__init__.py').write_text('from foreign import exp\n')
        before = set(sys.modules)
        with self.assertRaisesRegex(ValueError, 'wrapper from the requested package'):
            with loader.load_candidate_package(root, ['exp']):
                self.fail('External export was accepted')
        self.assertFalse(any(n.startswith('_camodel_candidate_') for n in set(sys.modules) - before))

    def test_function_code_location_must_match_even_if_module_attribute_is_changed(self):
        root = self.package('requested', 'requested')
        (self.base / 'foreign.py').write_text('def exp(): return "foreign"\n')
        sys.path.insert(0, str(self.base))
        (root / '__init__.py').write_text('from foreign import exp\nexp.__module__ = __name__\n')
        with self.assertRaisesRegex(ValueError, 'Imported source differs'):
            with loader.load_candidate_package(root, ['exp']):
                self.fail('Module metadata substituted for actual function code location')

    def test_timestamp_valid_stale_bytecode_is_ignored_without_touching_source_caches(self):
        root = self.package('requested', 'requested')
        source = root / 'exp.py'
        source.write_text('def exp():\n    from .helper import value\n    return "old" + value\n')
        helper = root / 'helper.py'
        helper.write_text('value = "old"\n')
        old_caches = {}
        for path in (source, helper):
            os.utime(path, (1700000000, 1700000000))
            cache = Path(py_compile.compile(str(path), doraise=True,
                                            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP))
            old_caches[cache] = cache.read_bytes()
            # Both entrypoint and late helper keep the old cache's size/mtime.
            path.write_text(path.read_text().replace('"old"', '"new"'))
            os.utime(path, (1700000000, 1700000000))
        original_files = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        saved_prefix = sys.pycache_prefix
        with loader.load_candidate_package(root, ['exp']) as (package, binding):
            self.assertNotEqual(sys.pycache_prefix, saved_prefix)
            self.assertEqual(package.exp(), 'newnew')
            self.assertEqual(binding['operators']['exp']['imported_sha256'], digest(source))
            loader.verify_sources_unchanged(binding)
            self.assertTrue(binding['sources_unchanged'])
        self.assertEqual(sys.pycache_prefix, saved_prefix)
        self.assertEqual({p: p.read_bytes() for p in old_caches}, old_caches)
        self.assertEqual({p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}, original_files)

    def test_source_mutated_during_import_is_rejected_and_cache_prefix_restored(self):
        root = self.package('requested', 'requested')
        (root / '__init__.py').write_text(
            'from pathlib import Path\n'
            'source = Path(__file__).with_name("exp.py")\n'
            'source.write_text(source.read_text() + "# changed during import\\n")\n'
            'from .exp import exp\n')
        saved_prefix = sys.pycache_prefix
        with self.assertRaisesRegex(ValueError, 'changed during import'):
            with loader.load_candidate_package(root, ['exp']):
                self.fail('Modified source was attributed to its original bytes')
        self.assertEqual(sys.pycache_prefix, saved_prefix)

    def test_symlinked_operator_outside_root_is_rejected(self):
        root = self.package('requested', 'requested')
        outside = self.base / 'outside.py'
        outside.write_text('def exp(): return "outside"\n')
        (root / 'exp.py').unlink()
        (root / 'exp.py').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside the requested root'):
            with loader.load_candidate_package(root, ['exp']):
                self.fail('Escaping source was accepted')

    def worker(self):
        """Import the real CLI with inert runtime modules; no numerical route runs."""
        modules = {name: types.ModuleType(name) for name in (
            'torch', 'torch.nn', 'torch.nn.functional', 'asc', 'asc.runtime',
            'asc.runtime.config', 'asc.lib', 'asc.lib.runtime')}
        for name, module in modules.items():
            parent, _, attr = name.rpartition('.')
            if parent:
                setattr(modules[parent], attr, module)
        config = modules['asc.runtime.config']
        config.Backend = types.SimpleNamespace(Model='Model')
        config.Platform = types.SimpleNamespace(Ascend950PR_9599='Ascend950PR_9599')
        config.set_platform = Mock()
        modules['asc.lib.runtime'].current_tick = lambda: 0
        with patch.dict(sys.modules, {**modules, 'candidate_import': loader}):
            spec = importlib.util.spec_from_file_location('smoke_under_test', WORKERS / 'run_camodel_smoke.py')
            worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(worker)
        return worker

    def test_cli_report_contains_actual_source_binding(self):
        root = self.package('requested', 'requested')
        worker = self.worker()
        output = self.base / 'result.json'
        # Replace only numerical work: the real CLI, package imports, invocation
        # and report-writing execute. This test makes no numerical PASS claim.
        def run_case(op, package):
            self.assertEqual(op, 'exp')
            self.assertEqual(package.exp(), 'requested')
        with patch.object(worker, 'run_case', side_effect=run_case), patch.object(
            sys, 'argv', ['smoke', '--candidate-root', str(root), '--ops', 'exp', '--output', str(output)]
        ), patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(worker.main(), 0)
        report = json.loads(output.read_text())
        self.assertEqual(report['source_binding']['operators']['exp']['imported_sha256'], digest(root / 'exp.py'))
        self.assertEqual(report['source_binding']['package']['requested_path'], str((root / '__init__.py').resolve()))
        self.assertIn('not the complete official task case set', report['limitations'])
        self.assertTrue(report['source_binding']['sources_unchanged'])
        worker.config.set_platform.assert_called_once()

    def test_cli_source_drift_does_not_write_a_passed_qualification(self):
        root = self.package('requested', 'requested')
        worker = self.worker()
        output = self.base / 'drift.json'

        def run_case(op, package):
            self.assertEqual(package.exp(), 'requested')
            (root / 'helper.py').write_text('value = "changed after import"\n')

        with patch.object(worker, 'run_case', side_effect=run_case), patch.object(
            sys, 'argv', ['smoke', '--candidate-root', str(root), '--ops', 'exp', '--output', str(output)]
        ), patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(worker.main(), 1)
        report = json.loads(output.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['source_binding']['sources_unchanged'])
        self.assertIn('sources changed', report['error'])

    def test_cli_missing_root_fails_before_platform_setup(self):
        worker = self.worker()
        with patch.object(sys, 'argv', ['smoke', '--candidate-root', str(self.base / 'missing'), '--ops', 'exp']), patch(
            'sys.stderr', new_callable=io.StringIO
        ), self.assertRaises(SystemExit) as caught:
            worker.main()
        self.assertEqual(caught.exception.code, 2)
        worker.config.set_platform.assert_not_called()


if __name__ == '__main__':
    unittest.main()
