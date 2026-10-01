import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integrations/cannbench/workers'))
gate = importlib.import_module('local_compile_gate')


class ShapeMetadataTests(unittest.TestCase):
    def test_foreach_norm_scalar_output_squeeze_preserves_metadata(self):
        output = gate.FakeTensor((1,), gate.DTYPES['float32'], 'npu:2').squeeze()
        self.assertEqual(output.shape, ())
        self.assertEqual(output.numel(), 1)
        self.assertEqual(output.dtype, gate.DTYPES['float32'])
        self.assertEqual(output.device, 'npu:2')
        self.assertEqual(output.squeeze(-1).shape, ())
        self.assertEqual(output.unsqueeze(-1).shape, (1,))

    def test_dimension_specific_views_preserve_element_count(self):
        tensor = gate.FakeTensor((1, 3, 1, 4), gate.DTYPES['float16'])
        self.assertEqual(tensor.squeeze((0, -2)).shape, (3, 4))
        self.assertEqual(tensor.squeeze(1).shape, tensor.shape)
        self.assertEqual(tensor.unsqueeze(-2).shape, (1, 3, 1, 1, 4))
        self.assertEqual(tensor.flatten(1, -2).shape, (1, 3, 4))
        self.assertEqual(tensor.flatten().numel(), tensor.numel())
        with self.assertRaises(IndexError):
            tensor.squeeze(4)
        with self.assertRaises(RuntimeError):
            tensor.squeeze((0, -4))
        with self.assertRaises(RuntimeError):
            tensor.flatten(3, 1)


if __name__ == '__main__':
    unittest.main()
