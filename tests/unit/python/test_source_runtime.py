"""Runtime build handoff invariants; no compiler, Docker, model or NPU calls."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'integrations/cannbench'))
import runtime_bundle as bundle
import smoke
spec = importlib.util.spec_from_file_location('source_runtime', ROOT/'scripts/source_runtime.py')
source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)


class SourceRuntime(unittest.TestCase):
    def fixtures(self, root):
        payloads = {
            'pyasc_dev': {'asc/__init__.py': b'', 'asc/experimental/asctile/__init__.py': b'',
                          bundle.NATIVE: b'fixture-native'},
            'numpy': {'numpy/__init__.py': b'', 'numpy.libs/libfixture.so': b'numpy-native'},
            'pybind11': {'pybind11/__init__.py': b'', 'pybind11/include/fixture.h': b'header'},
            'typing_extensions': {'typing_extensions.py': b'fixture = True\n'},
        }
        paths = []
        for name, files in payloads.items():
            path = root/(name+'-0.0.0-cp312-cp312-linux_x86_64.whl')
            files[name+'-0.0.0.dist-info/licenses/LICENSE'] = b'license-'+name.encode()
            bundle.write_wheel(path, files, name)
            paths.append(path)
        passport = {'wheel': paths[0].name, 'wheel_sha256': source.sha(paths[0]),
                    'runtime_dependency_wheels': {p.name: source.sha(p) for p in paths[1:]},
                    'source_sha': source.SOURCE, 'recipe_commit': source.RECIPE,
                    'target': 'cp312-x86_64', 'llvm': {'zstd_soname': None}}
        return paths, passport

    def test_bundle_and_submission_preserve_exact_runtime_bytes_and_licenses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths, passport = self.fixtures(root)
            runtime = root/'pyasc_runtime-0.0.0-cp312-cp312-linux_x86_64.whl'
            manifest = bundle.assemble(paths[0], paths[1:], passport, runtime)
            files = bundle.read_wheel(runtime, source.sha(runtime))
            self.assertEqual(len(manifest['payload_sha256']), 8)
            for package in ('pyasc_dev', 'numpy', 'pybind11', 'typing_extensions'):
                self.assertTrue(any('/licenses/'+package+'/' in n for n in files))
            (root/'runtime_probe.py').write_text('def runtime_probe(): pass\n')
            (root/'_pyasc_runtime.py').write_text('# simulator-only\n')
            smoke.make_bundle(runtime, source.sha(runtime), root, '2.0.0', root/'submission.zip', 'runtime_probe')
            with zipfile.ZipFile(root/'submission.zip') as package:
                wheel = package.read('vendor/cann_bench-2.0.0-cp312-cp312-linux_x86_64.whl')
            with zipfile.ZipFile(io.BytesIO(wheel)) as assembled:
                for name, checksum in manifest['payload_sha256'].items():
                    self.assertEqual(hashlib.sha256(assembled.read(name)).hexdigest(), checksum)
                self.assertNotIn('asctile/__init__.py', assembled.namelist())

    def test_corruption_and_missing_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); paths, passport = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, 'exact PyASC'):
                bundle.assemble(paths[0], paths[1:-1], passport, root/'unused.whl')
            changed = copy.deepcopy(passport); changed['target'] = 'cp311-aarch64'
            with self.assertRaisesRegex(ValueError, 'cp312'):
                bundle.assemble(paths[0], paths[1:], changed, root/'unused.whl')
            with zipfile.ZipFile(paths[0], 'a') as z:
                z.writestr('asc/new.py', b'injected')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                bundle.read_wheel(paths[0], passport['wheel_sha256'])
            with self.assertRaisesRegex(ValueError, 'RECORD inventory'):
                bundle.read_wheel(paths[0], source.sha(paths[0]))

    def test_unsafe_and_duplicate_wheel_members_rejected(self):
        for names in (['../escape'], ['/escape'], ['back\\slash'], ['dup', 'dup']):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as tmp:
                wheel=Path(tmp)/'bad.whl'
                with zipfile.ZipFile(wheel, 'w') as z:
                    for name in names:
                        z.writestr(name, b'test')
                with self.assertRaises(ValueError):
                    bundle.read_wheel(wheel, source.sha(wheel))

    def test_record_hash_and_undeclared_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); paths, passport = self.fixtures(root)
            files = bundle.read_wheel(paths[0], source.sha(paths[0]))
            files['asc/__init__.py'] = b'changed'
            with zipfile.ZipFile(paths[0], 'w') as z:
                for name, data in files.items(): z.writestr(name, data)
            with self.assertRaisesRegex(ValueError, 'RECORD mismatch'):
                bundle.read_wheel(paths[0], source.sha(paths[0]))

    def test_in_checkout_output_and_unsupported_host_fail_before_build(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            source.external(ROOT/'result')
        with patch.object(source.sys, 'platform', 'darwin'), patch.object(source, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'native Linux'):
                source.build(Path('/never-created'), source.SOURCE)
            run.assert_not_called()
        with patch.object(source, 'native') as native:
            with self.assertRaisesRegex(ValueError, 'full lowercase'):
                source.build(Path('/never-created'), 'main')
            native.assert_not_called()

    def test_docker_failure_cleans_only_own_exact_container(self):
        with patch.object(source, 'run', side_effect=TimeoutError), patch.object(source.subprocess, 'run') as cleanup:
            with self.assertRaises(TimeoutError):
                source.container(['docker', 'run'], 'pyasc-source-owned', Path('/unused'), 1)
            self.assertEqual(cleanup.call_args.args[0], ['docker', 'rm', '-f', 'pyasc-source-owned'])

    def test_qualification_refuses_changed_artifact_code_or_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'output'; output.mkdir()
            filename='scripts/source_runtime.py'
            intent={'stack_files': {filename:source.sha(ROOT/filename)},
                    'source_sha':source.SOURCE, 'recipe_commit':source.RECIPE}
            source.save(root/'intent.json', intent)
            source.save(output/'bundle-manifest.json', {'source_sha':source.SOURCE, 'recipe_commit':source.RECIPE})
            source.save(output/'files.json', {'bundle-manifest.json':source.sha(output/'bundle-manifest.json')})
            self.assertEqual(source.check_build(root), output)
            with patch.object(source, 'sha', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'code differs'): source.check_build(root)
            (output/'bundle-manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'): source.check_build(root)
            source.save(output/'bundle-manifest.json', {'source_sha':'other', 'recipe_commit':source.RECIPE})
            source.save(output/'files.json', {'bundle-manifest.json':source.sha(output/'bundle-manifest.json')})
            with self.assertRaisesRegex(ValueError, 'source/recipe'): source.check_build(root)


if __name__ == '__main__':
    unittest.main()
