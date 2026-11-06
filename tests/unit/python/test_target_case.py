from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
from target_case import NumericalAudit, proves_case
from target_inventory import PREFIX


class Audit(unittest.TestCase):
    def test_call_through_preserves_arguments_results_and_exceptions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); name = PREFIX+'test_add.py'; path = root/name
            namespace = {}
            exec(compile('def verify(check):\n    return check(2, 3, atol=0.01, rtol=0.02)\n', str(path), 'exec'), namespace)
            audit = NumericalAudit(root, [name], False)
            calls = []
            def launch(*args, **kwargs): calls.append((args, kwargs)); return 'launched'
            self.assertEqual(audit.launcher(launch)(7, x=9), 'launched')
            self.assertEqual(calls, [((7,), {'x':9})]); self.assertEqual(audit.completed_launches, 1)
            def check(a, b, **kwargs):
                self.assertEqual((a,b,kwargs),(2,3,{'atol':0.01,'rtol':0.02})); return 'checked'
            self.assertEqual(namespace['verify'](audit.assertion(check)), 'checked')
            self.assertTrue(audit.events[0]['passed']); self.assertTrue(audit.events[0]['after_completed_launch'])
            error = AssertionError('mismatch')
            def fail(*args, **kwargs): raise error
            with self.assertRaises(AssertionError): namespace['verify'](audit.assertion(fail))
            self.assertEqual(audit.failure(error)['source'], name)
            self.assertIsNone(audit.failure(AssertionError('mismatch')))
            # Calls outside the pinned source cannot earn numerical evidence.
            audit.assertion(check)(2,3,atol=0.01,rtol=0.02)
            self.assertEqual(len(audit.events), 2)

    def test_control_never_invokes_original_and_failed_launch_does_not_count_completed(self):
        def fail(): raise RuntimeError('launch failed')
        control = NumericalAudit(ROOT, [], True)
        self.assertIsNone(control.launcher(fail)())
        self.assertEqual((control.launches, control.completed_launches), (1,0))
        real = NumericalAudit(ROOT, [], False)
        with self.assertRaises(RuntimeError): real.launcher(fail)()
        self.assertEqual((real.launches, real.completed_launches), (1,0))


class Qualification(unittest.TestCase):
    def setUp(self):
        event = {'source':PREFIX+'test_add.py','line':10,'api':'torch.testing.assert_close',
            'tolerances':{'atol':1e-6,'rtol':1e-5},'passed':True,'after_launch':True,'after_completed_launch':True}
        self.real = {'schema_version':1,'kind':'target-case','status':'completed','mode':'numerical',
            'nodeid':'test_add.py::test_add[tail]', 'exit_code':0,'inputs_unchanged':True,'backend':'Model',
            'platform':'Ascend950PR_9599','runtime':{'launcher_sha256':'a'*64},'source_sha256':{'test_add.py':'b'*64},
            'collected':[PREFIX+'test_add.py::test_add[tail]'],'launcher_calls':1,'completed_launches':1,
            'phases':{p:{'outcome':'passed','xfail':None} for p in ('setup','call','teardown')},
            'numerical_checks':[event],'numerical_failure':None}
        self.control = deepcopy(self.real)
        self.control.update(mode='suppressed-launch',exit_code=1,completed_launches=0)
        self.control['phases']['call']['outcome']='failed'
        self.control['numerical_checks'][0].update(passed=False,after_completed_launch=False)
        self.control['numerical_failure']=self.control['numerical_checks'][0]

    def test_real_execution_requires_matching_rejecting_control(self):
        self.assertTrue(proves_case(self.real, self.control))
        for mutate in (lambda v:v.update(launcher_calls=0), lambda v:v.update(completed_launches=0),
                lambda v:v.update(numerical_checks=[]),lambda v:v.update(inputs_unchanged=False),
                lambda v:v['phases']['call'].update(xfail='unsupported'),lambda v:v['phases'].pop('teardown'),
                lambda v:v.update(backend='NPU'),lambda v:v.update(collected=[])):
            real=deepcopy(self.real);mutate(real)
            with self.assertRaises(ValueError): proves_case(real,self.control)

    def test_unrelated_failure_changed_tolerance_runtime_and_skips_are_not_controls(self):
        for mutate in (lambda v:v.update(numerical_failure=None),lambda v:v.update(exit_code=0),
                lambda v:v.update(completed_launches=1),lambda v:v['numerical_failure'].update(line=11),
                lambda v:v['numerical_failure']['tolerances'].update(atol=1),
                lambda v:v['runtime'].update(launcher_sha256='c'*64),
                lambda v:v['phases']['call'].update(outcome='skipped')):
            control=deepcopy(self.control);mutate(control)
            with self.assertRaises(ValueError): proves_case(self.real,control)


if __name__ == '__main__': unittest.main()
