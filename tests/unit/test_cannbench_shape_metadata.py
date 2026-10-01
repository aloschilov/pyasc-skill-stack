import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integrations/cannbench/workers'))
gate = importlib.import_module('local_compile_gate')
driver = importlib.import_module('driver')


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

    def test_rotary_broadcast_expansion_and_contiguous_copy_metadata(self):
        cosine = gate.FakeTensor((2, 1, 16, 32), gate.DTYPES['float16'], 'npu:2')
        expanded = cosine.expand(2, 8, 16, 32)
        self.assertEqual(expanded.shape, (2, 8, 16, 32))
        self.assertEqual(expanded.stride(), (512, 0, 32, 1))
        self.assertFalse(expanded.is_contiguous())
        packed = expanded.contiguous()
        self.assertTrue(packed.is_contiguous())
        self.assertEqual(packed.stride(), (4096, 512, 32, 1))
        self.assertEqual(packed.dtype, cosine.dtype)
        self.assertEqual(packed.device, cosine.device)
        self.assertEqual(expanded.reshape(-1).shape, (8192,))
        self.assertEqual(cosine.shape, (2, 1, 16, 32))

    def test_expand_preserves_existing_dimensions_and_rejects_invalid_shapes(self):
        tensor = gate.FakeTensor((1, 3), gate.DTYPES['float32'])
        self.assertEqual(tensor.expand(2, -1).shape, (2, 3))
        self.assertEqual(tensor.expand([4, 2, 3]).shape, (4, 2, 3))
        self.assertEqual(gate.FakeTensor((3,), tensor.dtype).expand(1, 3).stride(), (3, 1))
        self.assertEqual(tensor.expand(0, 3).numel(), 0)
        for sizes in [(3,), (2, 4), (-1, 2, 3), (2, -2)]:
            with self.assertRaises(RuntimeError):
                tensor.expand(*sizes)

    def test_host_output_data_writes_rejected_but_jit_writes_remain_allowed(self):
        import tempfile
        source = """import torch
import asctile
from ._pyasc_runtime import ensure_npu_platform
@asctile.jit
def kernel(x):
    x[0] = 1

def exp(x, base=-1.0, scale=1.0, shift=0.0):
    ensure_npu_platform()
    out = torch.empty_like(x)
    out.copy_(x)
    out[...] = x
    return out
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.py'
            path.write_text(source)
            problems = driver.static_check(path, 'exp')
            self.assertTrue(any('host Tensor.copy_' in p for p in problems))
            self.assertTrue(any('host Tensor indexed assignment' in p for p in problems))
            path.write_text(source.replace('    out.copy_(x)\n', '').replace('    out[...] = x\n', ''))
            self.assertEqual(driver.static_check(path, 'exp'), [])


if __name__ == '__main__':
    unittest.main()
