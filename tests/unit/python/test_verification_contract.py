import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import run_and_verify as rv
from numerical_checks import NumericalChecks, proves_numerics
from run_artifacts import RunArtifacts, ROOT


class Contract(unittest.TestCase):
    def test_control_failures_do_not_prove_numerics(self):
        real = (0, {'status': 'pass', 'exit_code': 0, 'launcher_calls': 1}, '')
        for control in [(1, {}, 'timeout'), (1, {}, 'no report'),
                        (1, {'error': 'unrelated exception'}, ''),
                        (2, {}, 'missing runtime'), (-11, {}, 'signal'), (0, {}, '')]:
            with self.subTest(control=control), patch.object(rv, '_run_execute_once', side_effect=[real, control]):
                self.assertEqual(rv._run_execute('unused', 'Model', 'platform')[2], 'executed')

    def test_exit_zero_requires_valid_report_and_launch(self):
        for payload in ({}, {'status':'pass','exit_code':0,'launcher_calls':0},
                        {'status':'fail','exit_code':1,'launcher_calls':1}):
            with patch.object(rv, '_run_execute_once', return_value=(0,payload,'')):
                self.assertEqual(rv._run_execute('unused','Model','p')[0],1)

    def test_real_failure_never_runs_control(self):
        with patch.object(rv, '_run_execute_once', return_value=(1,{'launcher_calls':1,'error':'boom'},'')) as call:
            result=rv._run_execute('unused','Model','p')
            self.assertEqual(result[0],1); self.assertIsNone(result[2]); self.assertEqual(result[3],1)
            call.assert_called_once()

    def test_same_check_site_required(self):
        event={'api':'torch.allclose','line':1,'after_launch':True,'passed':True}
        real={'numerical_checks':[event]}
        failure=dict(event, passed=False)
        control={'status':'fail','exit_code':1,'mock_launch':True,'launcher_calls':1,'numerical_failure':failure}
        self.assertTrue(proves_numerics(real,control,1))
        for key,value in [('launcher_calls',0),('mock_launch',False),('status','skip')]:
            self.assertFalse(proves_numerics(real,dict(control,**{key:value}),1))
        self.assertFalse(proves_numerics(real,dict(control,numerical_failure=dict(failure,line=9)),1))

    def observe(self, source):
        fake=types.ModuleType('torch'); fake.allclose=lambda a,b: a == b
        with tempfile.TemporaryDirectory() as tmp:
            kernel=Path(tmp)/'kernel.py'; kernel.write_text(source)
            checks=NumericalChecks(kernel, {'calls':1})
            def module(name):
                if name=='torch': return fake
                raise ImportError(name)
            with patch('numerical_checks.importlib.import_module',side_effect=module), patch.dict(sys.modules,{'torch':fake}):
                checks.install()
                try:
                    exec(compile(source,str(kernel),'exec'),{})
                except BaseException as exc:
                    return checks.failure(exc)
                finally:
                    checks.restore()
            return None

    def test_direct_checks_attributed(self):
        self.assertIsNotNone(self.observe('import torch\nassert torch.allclose(1,2)\n'))
        self.assertIsNotNone(self.observe('import torch\nif not torch.allclose(1,2):\n    raise ValueError("bad")\n'))

    def test_unrelated_exception_after_false_comparison_not_attributed(self):
        self.assertIsNone(self.observe('import torch\ntorch.allclose(1,2)\nraise AssertionError("unrelated")\n'))
        self.assertIsNone(self.observe('import torch\nassert torch.allclose(1,1)\nraise ValueError("unrelated")\n'))

    def test_strict_generated_qualification_rejects_skip_fail_and_weak_pass(self):
        import qualify_generated as q
        from unittest.mock import MagicMock
        for runtime_code, runtime_report, expected in [
            (2, {'status':'SKIP'}, 2), (1, {'status':'FAIL'}, 1),
            (0, {'status':'PASS','verification_level':'executed'}, 1),
            (0, {'status':'PASS','verification_level':'numerically_verified'}, 0)]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); kernel=root/'generated.py'; kernel.write_text('# immutable input')
                output=root/'result'; output.mkdir()
                run=MagicMock(); run.path=output; run.data={}
                run.command.side_effect=[(0,{}),(0,{'accepted':True}),
                    (0,{'status':'PASS','verification_level':'compiled'}),(runtime_code,runtime_report)]
                with patch.object(q,'RunArtifacts',return_value=run), \
                     patch.object(q,'pinned_sim_image',return_value='image@sha256:test'), \
                     patch.object(sys,'argv',['qualify',str(kernel),'--results-dir',str(output)]):
                    self.assertEqual(q.main(),expected)
                self.assertEqual(kernel.read_text(),'# immutable input')

    def test_tuple_shape_loop_scoring(self):
        import ast
        from score_kernel import _has_literal_shape_checks
        for sizes in ('(1024, 1025)', '[1024, 1025]'):
            source=f'for size in {sizes}:\n    out = run(size)\n    assert torch.allclose(out, expected(size))\n'
            self.assertTrue(_has_literal_shape_checks(ast.parse(source)))
        for source in (
            'for size in (1024,):\n    assert torch.allclose(run(size), expected(size))',
            'for size in (1024,1024):\n    assert torch.allclose(run(size), expected(size))',
            'for size in (1024,1025):\n    print(size)',
            'for size in (1024,1025):\n    assert torch.allclose(x,y)',
        ):
            self.assertFalse(_has_literal_shape_checks(ast.parse(source)))

    def test_artifacts_outside_checkout(self):
        with self.assertRaises(ValueError): RunArtifacts(ROOT/'should-not-exist','test')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'; run=RunArtifacts(path,'test'); run.finish('passed')
            self.assertTrue(json.loads((path/'manifest.json').read_text())['inputs_unchanged'])
            with self.assertRaises(FileExistsError): RunArtifacts(path,'test')


if __name__=='__main__': unittest.main()
