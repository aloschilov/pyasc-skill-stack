"""Legacy gate regressions, migrated from the former unregistered tools test.

The asc2 spellings here are intentional legacy detector inputs, not current
kernel authoring examples. These sources are scored statically, never executed.
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from score_kernel import score


CLEAN_KERNEL = """\
import asc
import asc.runtime.config as config
import asc2

@asc2.jit(always_compile=True)
def clean_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                 size: int, tile_size: asc.ConstExpr[int],
                 tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2):
        off = base + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[off])
        out = asc2.exp(x)
        asc2.store(out, out_gm, offsets=[off])
"""


class ScoreKernelBans(unittest.TestCase):
    def score_source(self, source, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'kernel.py'
            path.write_text(source)
            return score(str(path), **kwargs)

    def assert_legacy_gate(self, source, finding):
        result = self.score_source(source)
        self.assertEqual(result.failure_category, 'F_legacy_api_import')
        self.assertIn(finding, result.gating_detail['legacy_api_findings'])

    def test_clean_kernel_no_gate(self):
        result = self.score_source(CLEAN_KERNEL)
        self.assertIsNone(result.failure_category)
        self.assertGreaterEqual(sum(c.passed for c in result.checks), 12)
        self.assertEqual(result.gating_detail['legacy_api_findings'], [])
        self.assertIs(result.gating_detail['missing_asc2_import'], False)
        checks = {c.name: c.passed for c in result.checks}
        for name in ('asc2_tensor', 'asc2_load_store', 'jit_decorator', 'loop_range'):
            self.assertTrue(checks[name], name)

    def test_legacy_tik_gates(self):
        self.assert_legacy_gate('import tik\n' + CLEAN_KERNEL, 'import tik')

    def test_legacy_ascendcl_gates(self):
        self.assert_legacy_gate('from ascendcl import acl_init\n' + CLEAN_KERNEL, 'import ascendcl')

    def test_legacy_tbe_gates(self):
        self.assert_legacy_gate('from tbe.dsl import build\n' + CLEAN_KERNEL, 'import tbe')

    def test_tposition_gates(self):
        self.assert_legacy_gate(CLEAN_KERNEL + '\nTPOSITION_UB = TPosition.UB\n', 'TPosition')

    def test_npu_dispatch_gates(self):
        self.assert_legacy_gate(CLEAN_KERNEL + '\nimport torch\nx = torch.zeros(8).npu()\n', '.npu(')

    def test_missing_asc2_import_gates(self):
        result = self.score_source(CLEAN_KERNEL.replace('import asc2\n', ''))
        self.assertEqual(result.failure_category, 'F_missing_asc2_import')
        self.assertIs(result.gating_detail['missing_asc2_import'], True)

    def test_allow_legacy_apis_opt_out(self):
        result = self.score_source('import tik\n' + CLEAN_KERNEL, allow_legacy_apis=True)
        self.assertIsNone(result.failure_category)
        self.assertIs(result.gating_detail['allow_legacy_apis'], True)


if __name__ == '__main__':
    unittest.main()
