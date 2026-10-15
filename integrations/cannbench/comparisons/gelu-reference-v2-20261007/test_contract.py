"""Host-only checks: these do not establish compiled-kernel accuracy."""
import ast
import math
from pathlib import Path
import struct
import unittest


def float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


class ContractTests(unittest.TestCase):
    def test_constants_match_cpp_float_initializers(self):
        self.assertEqual(float32(1 / .044715), 22.363859176635742)
        self.assertEqual(float32(-1.595769121 * .044715), -0.07135481387376785)
        self.assertEqual(float32(.707106781), .7071067690849304)

    def test_balanced_tiles_have_exact_nonoverlapping_coverage(self):
        for size in (1, 63, 2047, 2048, 2049, 8193, 512*2049, 255*8193):
            for tile in (1024, 2048, 4096, 8192, 15872):
                total = math.ceil(size / tile)
                for blocks in (1, 8, 32, 64, 72):
                    indices = []
                    for block in range(blocks):
                        quotient, remainder = divmod(total, blocks)
                        first = block*quotient + min(block, remainder)
                        count = quotient + min(max(remainder-block, 0), 1)
                        indices.extend(range(first, first+count))
                    self.assertEqual(indices, list(range(total)))
                    self.assertEqual(sum(min(tile, size-i*tile) for i in indices), size)

    def test_authored_inline_is_only_erf_selection(self):
        tree = ast.parse(Path(__file__).with_name('kernel.py').read_text())
        inlines = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr.startswith('inline')]
        self.assertEqual(len(inlines), 1)
        self.assertEqual(inlines[0].func.attr, 'inline_vf')
        body = ast.literal_eval(inlines[0].args[0])
        self.assertIn('AscendC::Erf<float, false, config>', body)
        self.assertNotIn('RegTensor', body)
        self.assertNotIn('MicroAPI', body)


if __name__ == '__main__':
    unittest.main()
