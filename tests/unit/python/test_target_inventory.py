"""Inventory contracts are independent of model calls and runtime execution."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import target_inventory as inventory


class Inventory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'checkout'; self.root.mkdir()
        for name in ('test_add.py', 'nested/test_reduce.py', 'conftest.py'):
            path = self.root/inventory.PREFIX/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# pinned test fixture\n')
        upstream = {n: inventory.digest((self.root/inventory.PREFIX/n).read_bytes())
                    for n in ('test_add.py', 'nested/test_reduce.py')}
        self.write('reference/asctile/PIN.json', {'commit': 'a'*40, 'files': upstream})
        (self.root/'reference/__init__.py').write_text('')
        self.write('reference/images/ENV.json', {'source_sha': 'a'*40,
            'dev_image': {'ref': 'ghcr.io/example/simulator@sha256:'+'b'*64},
            'image': {'simulator': 'Ascend950PR_9599', 'python_bin': 'python3.11'}})
        self.ids = ['test_add.py::test_add[1-float32]', 'test_add.py::test_add[199-float32]',
                    'nested/test_reduce.py::test_reduce[2-257-float16]']
        self.contract = inventory.derive(self.ids, self.root)
        self.write(inventory.CONTRACT, self.contract)

    def write(self, name, value):
        p = self.root/name; p.parent.mkdir(parents=True, exist_ok=True); inventory.save(p, value)

    def test_exact_ids_are_order_independent_and_normalize_only_target_prefix(self):
        other = [inventory.PREFIX+i for i in reversed(self.ids)]
        self.assertEqual(inventory.validate(other, self.root), self.contract)
        self.assertEqual(self.contract['case_count'], 3)
        self.assertEqual(len(self.contract['modules']), 2)
        self.assertEqual(self.contract['case_ids_sha256'], inventory.digest(inventory.canonical(sorted(self.ids))))

    def test_empty_duplicate_missing_unexpected_and_changed_cases_are_rejected(self):
        for ids in ([], self.ids+[self.ids[0]], self.ids[:-1], self.ids[1:],
                    self.ids+['test_other.py::test_new'],
                    self.ids+['../test_other.py::test_new'], self.ids+['/test_other.py::test_new']):
            with self.subTest(ids=ids), self.assertRaises(ValueError): inventory.validate(ids, self.root)
        changed = list(self.ids); changed[1] = changed[1].replace('199', '200')
        with self.assertRaisesRegex(ValueError, 'drift'): inventory.validate(changed, self.root)

    def test_upstream_and_local_fixture_drift_are_distinct(self):
        p = self.root/inventory.PREFIX/'test_add.py'; original = p.read_bytes(); p.write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'Upstream source drift'): inventory.validate(self.ids, self.root)
        p.write_bytes(original)
        (self.root/inventory.PREFIX/'conftest.py').write_text('# locally changed\n')
        with self.assertRaisesRegex(ValueError, 'source drift'): inventory.validate(self.ids, self.root)

    def test_runtime_revision_image_and_module_drift_cannot_reuse_inventory(self):
        path = self.root/'reference/images/ENV.json'; original = inventory.read(path)
        for mutate in (lambda e: e.update(source_sha='c'*40),
                       lambda e: e['dev_image'].update(ref='ghcr.io/example/simulator@sha256:'+'d'*64),
                       lambda e: e['image'].update(simulator='Ascend910B1')):
            changed = deepcopy(original); mutate(changed); inventory.save(path, changed)
            with self.assertRaises(ValueError): inventory.validate(self.ids, self.root)
        inventory.save(path, original)
        (self.root/inventory.PREFIX/'test_new.py').write_text('# unpinned\n')
        with self.assertRaisesRegex(ValueError, 'module inventory'): inventory.validate(self.ids, self.root)

    def test_bounded_collection_preserves_ids_and_never_claims_numerics(self):
        def run(argv, **kwargs):
            if argv[1] == 'run':
                kwargs['stdout'].write(('\n'.join(inventory.PREFIX+n for n in self.ids)+'\n').encode())
                self.assertIn('--collect-only', argv); self.assertIn('none', argv)
                self.assertNotIn('NPU', argv); self.assertNotIn('--compile-only', argv)
                self.assertEqual(kwargs['timeout'], 180)
            return subprocess.CompletedProcess(argv, 0, b'', b'')
        destination = Path(self.tmp.name)/'collection'
        with patch.object(inventory.subprocess, 'run', side_effect=run): result = inventory.collect(destination, self.root)
        self.assertEqual(result['status'], 'collected'); self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(result['numerically_qualified_cases'], 0)
        self.assertEqual(inventory.read(destination/'nodeids.json'), sorted(self.ids))
        self.assertEqual(result['artifact_sha256']['nodeids.json'], inventory.digest((destination/'nodeids.json').read_bytes()))
        with self.assertRaises(FileExistsError): inventory.collect(destination, self.root)

    def test_empty_or_timed_out_collection_stays_blocked(self):
        for timeout in (False, True):
            def run(argv, **kwargs):
                if argv[1] == 'run' and timeout: raise subprocess.TimeoutExpired(argv, 180)
                return subprocess.CompletedProcess(argv, 0, b'', b'')
            destination = Path(self.tmp.name)/str(timeout)
            with patch.object(inventory.subprocess, 'run', side_effect=run): result = inventory.collect(destination, self.root)
            self.assertEqual(result['status'], 'blocked'); self.assertTrue(result['cleanup_confirmed'])
            self.assertEqual(result['numerically_qualified_cases'], 0)
            self.assertFalse((destination/'inventory.json').exists())


if __name__ == '__main__': unittest.main()
