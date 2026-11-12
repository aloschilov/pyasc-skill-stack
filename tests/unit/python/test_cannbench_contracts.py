"""Retained task/static contracts without a campaign, tensor runtime or service."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'integrations/cannbench/workers'))
from contracts import case_arguments, static_check
from evalqueue import load_credentials
from profile_unary import check_exp_host_casts


class CandidateContracts(unittest.TestCase):
    def test_profile_excludes_exp_host_casts_but_accepts_jit_casts(self):
        kernel = "@asctile.jit(reuse_alloc=2)\ndef kernel(tile): return tile.to(asctile.float32)\n"
        check_exp_host_casts(kernel + "def exp(x): return x.new_empty(x.shape)\n")
        for call in ("x.to(torch.float32)", "x.float()", "tmp.type_as(x)"):
            with self.subTest(call=call), self.assertRaisesRegex(ValueError, "Exp host tensor conversion"):
                check_exp_host_casts(kernel + "def exp(x): return " + call)

    def test_reference_and_mutated_candidates(self):
        source = (ROOT / 'integrations/cannbench/smoke-kernel/swi_glu.py').read_text()
        task = ROOT / 'integrations/cannbench/tasks/swi_glu'
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / 'swi_glu.py'
            candidate.write_text(source)
            self.assertEqual(static_check(candidate, 'swi_glu', task), [])
            mutations = {
                'signature': source.replace('def swi_glu(', 'def other('),
                'jit': source.replace('@asctile.jit', '# @asctile.jit'),
                'math': source + '\ndef forbidden(x):\n    return torch.exp(x)\n',
                'pointer': source + '\ndef forbidden(x):\n    return x.data_ptr()\n',
                'host-copy': source + '\ndef forbidden(x, y):\n    y.copy_(x)\n',
                'host-store': source + '\ndef forbidden(x):\n    y = torch.empty_like(x)\n    y[:] = x\n',
            }
            for reason, changed in mutations.items():
                with self.subTest(reason=reason):
                    self.assertNotEqual(source, changed)
                    candidate.write_text(changed)
                    self.assertTrue(static_check(candidate, 'swi_glu', task))

    def test_defaults_come_from_selected_task_not_operator_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            (task / 'golden.py').write_text('def new_op(x, axis=0): pass\n')
            source = ('@asctile.jit\ndef kernel(): pass\n'
                      'def new_op(x, axis=1):\n    ensure_npu_platform()\n')
            candidate = task / 'candidate.py'; candidate.write_text(source)
            self.assertIn('Public callable defaults differ from official golden',
                          static_check(candidate, 'new_op', task))
            candidate.write_text(source.replace('axis=1', 'axis=0'))
            self.assertEqual(static_check(candidate, 'new_op', task), [])

    def test_tensor_lists_optional_slots_and_conflicting_attrs(self):
        proto = {'operator': {'inputs': [{'name': 'xs'}, {'name': 'bias', 'optional': True}]}}
        case = {'input_shape': [[[2, 3], None, [4]], None],
                'dtype': [['float32', 'float32', 'float16'], 'float32'], 'attrs': {'axis': -1}}
        values = case_arguments(proto, case, lambda shape, dtype: (shape, dtype))
        self.assertEqual(values, {'xs': [([2, 3], 'float32'), None, ([4], 'float16')],
                                  'bias': None, 'axis': -1})
        case['attrs']['xs'] = 1
        with self.assertRaisesRegex(ValueError, 'conflicts'): case_arguments(proto, case, lambda s, d: (s, d))

    def test_explicit_credentials_preserve_environment_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / 'test.env'
            path.write_text('# fixture only\nexport BENCHSITE_API_TOKEN="test-token"\nBENCHSITE_URL=fixture\n')
            os.environ['BENCHSITE_URL'] = 'existing'
            load_credentials(path)
            self.assertEqual(os.environ['BENCHSITE_API_TOKEN'], 'test-token')
            self.assertEqual(os.environ['BENCHSITE_URL'], 'existing')
            path.unlink()
            load_credentials(path)  # Existing token needs no file access.


if __name__ == '__main__': unittest.main()
