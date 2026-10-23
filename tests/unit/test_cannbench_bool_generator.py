"""Bool dispatch regression without CANN or torch imports on Mac."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'integrations/cannbench/workers'))
import run_camodel_cases as runner

class BoolGeneratorTests(unittest.TestCase):
    def test_bool_ranges_preserved_and_other_dtypes_unchanged(self):
        calls=[]
        class Tensor:
            def to(self,dtype):calls.append(('cast',dtype));return self
        fake=SimpleNamespace(bool='bool',int64='int64',
            full=lambda shape,value,dtype: calls.append(('full',shape,value,dtype)),
            randint=lambda lo,hi,shape,**kw: (calls.append(('randint',lo,hi,shape,kw)) or Tensor()))
        class Generator:
            def generate_input_tensor(self,shape,dtype,value_range=None):
                return ('original',shape,dtype,value_range)
            def _parse_range(self,value_range,dtype):return (0,100) if value_range is None else value_range
        g=Generator()
        with patch.dict(sys.modules,{'torch':fake}):
            provenance=runner.install_official_bool_generation(g,SimpleNamespace(str_to_torch_dtype=lambda x:x))
            self.assertEqual(provenance['source_sha256'],runner.BOOL_GENERATOR_SHA256)
            self.assertEqual(g.generate_input_tensor([3],'float32',[-1,1]),('original',[3],'float32',[-1,1]))
            self.assertIsNone(g.generate_input_tensor(None,'bool',[0,1]))
            g.generate_input_tensor([3],'bool',[1,1]);self.assertEqual(calls[-1],('full',[3],True,'bool'))
            g.generate_input_tensor([3],'bool',[0,0]);self.assertEqual(calls[-1],('full',[3],False,'bool'))
            g.generate_input_tensor([3],'bool',[0,1]);self.assertEqual(calls[-2][:4],('randint',0,2,[3]))
            g.generate_input_tensor([3],'bool',None);self.assertEqual(calls[-2][:4],('randint',0,2,[3]))
            g.generate_input_tensor([3],'bool',[1,0]);self.assertEqual(calls[-2][:4],('randint',0,2,[3]))

    def test_source_drift_rejected(self):
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}),patch.object(runner,'_sha256_file',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                runner.install_official_bool_generation(None,None)
