"""Reference selection regressions; no torch/CANN import on the Mac."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'integrations/cannbench/workers'))
from run_camodel_cases import reference_functions


class ReferenceHookTests(unittest.TestCase):
    def test_legacy_loader_uses_proto_name_and_distinct_references(self):
        golden = lambda x: x + 100
        oracle = lambda x: x * 2
        bench = lambda x: x * 3
        module = SimpleNamespace(proto_name_oracle=oracle, proto_name_bench=bench)
        loader = SimpleNamespace(_load_module=lambda path: module,
                                 _get_function_name=lambda path: 'proto_name')
        actual = reference_functions(loader, 'different_directory', golden)
        self.assertEqual([f(4) for f in actual], [8, 12])

    def test_absent_hooks_fall_back_to_golden(self):
        golden = lambda x: x
        loader = SimpleNamespace(_load_module=lambda path: SimpleNamespace(),
                                 _get_function_name=lambda path: 'op')
        self.assertEqual(reference_functions(loader, 'op', golden), (golden, golden))

    def test_modern_accessors_are_authoritative(self):
        golden = lambda x: x
        oracle = lambda x: x + 1
        loader = SimpleNamespace(get_oracle_function=lambda path: oracle,
                                 get_bench_function=lambda path: None)
        self.assertEqual(reference_functions(loader, 'op', golden), (oracle, golden))

    def test_invalid_declared_hook_is_not_silently_ignored(self):
        loader = SimpleNamespace(get_oracle_function=lambda path: 0)
        with self.assertRaisesRegex(ValueError, 'Non-callable oracle'):
            reference_functions(loader, 'op', lambda x: x)

if __name__ == '__main__':
    unittest.main()
