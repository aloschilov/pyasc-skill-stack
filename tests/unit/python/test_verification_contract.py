import importlib
import contextlib
import io
import json
from pathlib import Path
import sys
import shutil
import subprocess
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
        for name in ('allclose', 'close'):
            self.assertIsNotNone(self.observe(f'from torch import allclose as {name}\nassert {name}(1,2)\n'))
        self.assertIsNotNone(self.observe('import torch\nif not torch.allclose(1,2):\n    print("bad values")\n    raise ValueError("bad")\n'))

    def test_unrelated_exception_after_false_comparison_not_attributed(self):
        self.assertIsNone(self.observe('import torch\ntorch.allclose(1,2)\nraise AssertionError("unrelated")\n'))
        self.assertIsNone(self.observe('import torch\nassert torch.allclose(1,1)\nraise ValueError("unrelated")\n'))
        self.assertIsNone(self.observe('from torch import allclose\nallclose(1,2)\nraise AssertionError("unrelated")\n'))
        self.assertIsNone(self.observe('import torch\nif not torch.allclose(1,2):\n    raise ValueError("earlier unrelated error")\n    raise ValueError("numerical rejection")\n'))

    def invoke_cli(self, *args):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['verify', __file__, '--json', *args]), \
             contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            rv.main()
        return caught.exception.code, json.loads(output.getvalue())

    def test_docker_exit_contract_preserves_raw_error(self):
        for raw_code, expected in [(125,2),(126,2),(127,2),(137,1),(-9,1),(23,1)]:
            with self.subTest(raw_code=raw_code), \
                 patch.object(rv, '_docker_available', return_value=True), \
                 patch.object(rv, '_run_in_docker', return_value=(raw_code, 'partial output', 'raw failure')):
                code, report = self.invoke_cli('--runtime-backend', 'docker')
                self.assertEqual(code, expected)
                self.assertEqual(report['status'], 'SKIP' if expected == 2 else 'FAIL')
                self.assertIsNone(report['verification_level'])
                self.assertEqual(report['evidence']['docker'], {'exit_code':raw_code, 'log':'raw failure'})

    def test_auto_fallback_retains_both_reports(self):
        def execution(*args, evidence, **kwargs):
            evidence['execution'] = {'exit_code':2, 'log':'unavailable'}
            return 2, 'unavailable', None, 0
        def jit(kernel, use_json, platform, scalars, evidence):
            evidence['jit'] = {'exit_code':0, 'log':'compiled', 'argv':['jit',kernel]}
            return 0, 'compiled'
        with patch.object(rv, '_can_import', return_value=True), \
             patch.object(rv, '_run_execute', side_effect=execution), \
             patch.object(rv, '_run_jit', side_effect=jit):
            code, report = self.invoke_cli('--mode', 'auto')
        self.assertEqual(code, 0)
        self.assertEqual(report['verification_level'], 'compiled')
        self.assertEqual(report['mode'], 'auto (jit fallback)')
        self.assertEqual(report['evidence']['execution']['exit_code'], 2)
        self.assertEqual(report['evidence']['jit']['log'], 'compiled')
        self.assertTrue(report['evidence']['jit']['argv'])

    def test_docker_budget_and_offline_execution(self):
        for timeout in (10, 120, 300):
            with patch.object(rv, '_docker_available', return_value=True), \
                 patch.object(rv, '_run_in_docker', return_value=(2, '', 'unavailable')) as run:
                self.invoke_cli('--runtime-backend', 'docker', '--timeout', str(timeout))
            inner = run.call_args.kwargs['timeout']
            self.assertEqual(inner, rv.docker_run_timeout(timeout))
            self.assertGreater(rv.verification_command_timeout(timeout),
                               inner + rv.DOCKER_PROBE_TIMEOUT + rv.DOCKER_CLEANUP_TIMEOUT)
        with patch.object(rv.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as run:
            rv._run_in_docker(__file__, ['--mode','execute'], 'image@sha256:test', 360)
        command = run.call_args_list[0].args[0]
        self.assertEqual(command[command.index('--pull')+1], 'never')
        self.assertEqual(command[command.index('--network')+1], 'none')

    def test_skill_scalar_overrides_survive_insertion(self):
        import skill_jit_examples as skills
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT/'skills', root/'skills')
            with patch.object(skills, 'ROOT', root):
                skills.extract_examples(root/'before')
                earlier = root/'skills/aaa-new/SKILL.md'
                earlier.parent.mkdir()
                earlier.write_text('```python\nimport asc\n@asc.jit\ndef extra():\n    pass\n```\n')
                skills.extract_examples(root/'after')
            before = json.loads((root/'before/inventory.json').read_text())
            after = json.loads((root/'after/inventory.json').read_text())
            target = 'skills/pyasc-target-operator/SKILL.md'
            old = next(e for e in before if e['file'] == target)
            new = next(e for e in after if e['file'] == target)
            self.assertNotEqual(old['label'], new['label'])
            self.assertEqual(new['scalar_args'], {'size':1025, 'tile_size':1024})
            self.assertTrue(all(not e['scalar_args'] for e in after
                                if not e['excluded'] and e['file'] != target))

    def test_qualification_preserves_original_error_when_inputs_change(self):
        import qualify_generated as q
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); kernel = root/'kernel.py'; kernel.write_text('# input')
            output = root/'result'
            with patch('run_artifacts.input_hashes', side_effect=[{'input':'before'}, {'input':'after'}]), \
                 patch.object(RunArtifacts, 'command', side_effect=OSError('original command error')), \
                 patch.object(q, 'pinned_sim_image', return_value='image@sha256:test'), \
                 patch.object(sys, 'argv', ['qualify', str(kernel), '--results-dir', str(output)]):
                with self.assertRaisesRegex(OSError, 'original command error'):
                    q.main()
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed-inputs-changed')
            self.assertFalse(manifest['inputs_unchanged'])
            self.assertEqual(manifest['error']['message'], 'original command error')
            with patch('run_artifacts.input_hashes', side_effect=[{}, {'new':'input'}]):
                run = RunArtifacts(root/'changed', 'test')
                with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
                    run.finish('passed')

    def test_contract_preserves_interrupt_and_records_input_integrity(self):
        import verification_contract as contract
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/'result'
                error = KeyboardInterrupt('operator cancelled')
                before = {'input':'before'}
                after = {'input':'after'} if changed else before
                with patch('run_artifacts.input_hashes', side_effect=[before, after]), \
                     patch.object(RunArtifacts, 'command', side_effect=error), \
                     patch.object(contract, 'pinned_sim_image', return_value='image@sha256:test'), \
                     patch.object(sys, 'argv', ['contract', '--results-dir', str(output)]):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        contract.main()
                self.assertIs(caught.exception, error)
                manifest = json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'failed-inputs-changed' if changed else 'interrupted')
                self.assertEqual(manifest['inputs_unchanged'], not changed)
                self.assertEqual(manifest['error'], {'type':'KeyboardInterrupt', 'message':'operator cancelled'})

    def test_skill_scalar_overrides_reject_stale_sources(self):
        import skill_jit_examples as skills
        target = 'skills/pyasc-target-operator/SKILL.md'
        for change in ('bogus-key', 'removed', 'renamed', 'marked-fragment'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT/'skills', root/'skills')
                overrides = dict(skills.SCALAR_ARGS)
                stale = target
                if change == 'bogus-key':
                    stale = 'skills/missing/SKILL.md'
                    overrides[stale] = {'size':1}
                elif change == 'removed':
                    (root/target).unlink()
                elif change == 'renamed':
                    (root/target).rename(root/target.replace('SKILL.md', 'renamed.md'))
                else:
                    path = root/target
                    path.write_text(path.read_text().replace('```python', '<!-- skill-code: fragment -->\n```python'))
                with patch.object(skills, 'ROOT', root), patch.object(skills, 'SCALAR_ARGS', overrides):
                    with self.assertRaisesRegex(ValueError, 'Stale scalar argument overrides') as caught:
                        skills.extract_examples(root/'examples')
                self.assertIn(stale, str(caught.exception))

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
