from pathlib import Path
import sys
RECIPE = Path(__file__).resolve().parents[3]/"examples/tuning/swiglu"
sys.path.insert(0, str(RECIPE))
"""Offline only: no framework, Docker, provider or model invocation."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from parameter_guard import validate
from consumer_interface import commands, validate_bash, tool_config, request_audit, audit_events
from selection import select
from tool import host_projection, prepare_command, cpu_gate
ROOT=RECIPE
BASE=(ROOT/'baseline.py').read_bytes()

def altered(old,new):return BASE.replace(old.encode(),new.encode())

class ParameterTests(unittest.TestCase):
    def test_baseline(self): self.assertFalse(validate(BASE,BASE)['ast_changed'])
    def test_initial_height(self): self.assertTrue(validate(BASE,altered('T = 128','T = 16'))['ast_changed'])
    def test_scalar_cap(self): self.assertTrue(validate(BASE,altered('T = max(T, 1)','T = min(T, 4)\n    T = max(T, 1)'))['ast_changed'])
    def test_resource_expression(self):validate(BASE,altered('T = max(T, 1)','T = min(T, max(1, (R + aiv - 1) // aiv))\n    T = max(T, 1)'))
    def test_attack_mutations(self):
        changes=[('reuse_alloc=2','reuse_alloc=0'),('15000','30000'),('_AIV_FALLBACK = 72','_AIV_FALLBACK = 64'),('T = 128','T = 256'),('T = 128','T = True'),('W = 32','W = 64'),('T //= 2','T -= 1'),('tiles = (R + T - 1) // T','tiles = 1'),('return T, W, blocks','return T, W, 1'),('out_f = silu * value_f','out_f = silu + value_f'),('T = min(T, _pow2_ceil(R))','T = min(T, __import__("os").getpid())'),('T = min(T, _pow2_ceil(R))','T = min(T, [1, 2][R])'),('T = min(T, _pow2_ceil(R))','T = min(T, R ** H)'),('T = min(T, _pow2_ceil(R))','T = min(T, 0)'),('T = min(T, _pow2_ceil(R))','T = min(T, H if R == 5 else 1)')]
        for old,new in changes:
            with self.subTest(new=new),self.assertRaises(ValueError):validate(BASE,altered(old,new))
    def test_host_schema(self):
        p=host_projection(BASE)
        self.assertFalse(p['host_tensor_compute']);self.assertTrue(p['no_host_tensor_computation']);self.assertEqual(len(p['candidate_sha256']),64)

class SelectionTests(unittest.TestCase):
    def result(self,values):return {'status':'passed','analysis':[{'fixture':str(i),'order_A_over_B':{'ABBA':a,'BAAB':b}}for i,(a,b)in enumerate(values)]}
    def test_baseline_without_measurements(self):self.assertEqual(select({'status':'failed'},['0','1','2'])['selected'],'baseline')
    def test_tie(self):self.assertEqual(select(self.result([(1,1)]*3),['0','1','2'])['selected'],'baseline')
    def test_improvement(self):self.assertEqual(select(self.result([(1.02,1.03),(1,1),(1,1)]),['0','1','2'])['selected'],'variant')
    def test_shape_regression(self):self.assertEqual(select(self.result([(1.5,1.5),(.98,.98),(1,1)]),['0','1','2'])['selected'],'baseline')
    def test_order_disagreement(self):self.assertEqual(select(self.result([(1.1,1.1),(.999,1.001),(1,1)]),['0','1','2'])['selected'],'baseline')
    def test_invalid_ratio(self):
        with self.assertRaises(ValueError):select(self.result([(float('nan'),1)]*3),['0','1','2'])

class InterfaceTests(unittest.TestCase):
    def test_exact_command(self):self.assertEqual(validate_bash({'command':commands('/trusted/recipe')['compare'],'timeout':600000},'/trusted/recipe','/consumer'),'compare')
    def test_shell_and_timeout_rejected(self):
        good=commands('/trusted/recipe')['baseline']
        for command in (good+'; id',good+' && id',good+'\nid',good+' ',good.replace('baseline','baseline --retry'),'docker run image','python3 /tmp/tool.py baseline'):
            with self.subTest(command=command),self.assertRaises(ValueError):validate_bash({'command':command,'timeout':300000},'/trusted/recipe','/consumer')
        for timeout in (None,120000,299999,300001,600001,True):
            with self.assertRaises(ValueError):validate_bash({'command':good,'timeout':timeout},'/trusted/recipe','/consumer')
    def test_workdir(self):
        with self.assertRaises(ValueError):validate_bash({'command':commands('/trusted/recipe')['baseline'],'timeout':300000,'workdir':'/tmp'},'/trusted/recipe','/consumer')
    def test_permissions(self):
        c=tool_config({'permission':{'read':{'*':'deny'},'external_directory':{'*':'deny'}}},'/trusted/recipe','/consumer')
        self.assertEqual(c['permission']['bash']['*'],'deny');self.assertEqual(len(c['permission']['bash']),5)
        self.assertEqual(c['permission']['edit'].get('baseline.py'),'deny' if 'baseline.py' in c['permission']['edit'] else None)
    def test_provider_schema(self):
        body={'tools':[{'function':{'name':'bash','parameters':{'properties':{'command':{},'timeout':{}}}}}]}
        self.assertEqual(request_audit(body)['tool_names'],['bash'])
        body['tools'].append({'function':{'name':'network','parameters':{}}})
        with self.assertRaises(ValueError):request_audit(body)
    def test_fake_feedback_rejected(self):
        event={'part':{'tool':'bash','state':{'status':'completed','input':{'command':commands('/trusted/recipe')['baseline'],'timeout':300000},'output':'PASS'}}}
        with tempfile.TemporaryDirectory()as d,self.assertRaises(ValueError):audit_events([event],'/trusted/recipe','/consumer',d)

class PreparationTests(unittest.TestCase):
    def test_supervisor_template_no_execution(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'baseline';p.mkdir();(p/'candidate').mkdir();(p/'candidate/candidate.py').write_bytes(BASE)
            import tool
            fixture=Path(d)/'recipe';fixture.mkdir();(fixture/'engine/probe').mkdir(parents=True)
            (fixture/'engine/probe/qualification-plan.json').write_text(json.dumps({'runtime':json.loads((RECIPE/'engine-lock.json').read_text())['runtime']}))
            (fixture/'supervisor-template.py.txt').write_bytes((RECIPE/'supervisor-template.py.txt').read_bytes())
            with patch.object(tool,'ROOT',fixture):
                prepare_command(p,['/candidate:/candidate:ro'],['timeout','10','true'],240,6)
            code=(p/'owned-invocation.py').read_text();compile(code,'test','exec')
            self.assertNotIn('__WHOLE__',code);self.assertIn('deadline=start+240',code)
            command=json.loads((p/'command.json').read_text())['argv']
            self.assertIn('--network',command);self.assertNotIn('--entrypoint',command)
            self.assertFalse((p/'run-intent.json').exists())
            self.assertLess(code.index("(P/'out').mkdir()"),code.index('subprocess.Popen(argv'))
            self.assertEqual(command[command.index('--user')+1],str(__import__('os').getuid())+':'+str(__import__('os').getgid()))
            self.assertIn('HOME=/tmp',command);self.assertIn('PYTHONPYCACHEPREFIX=/tmp/pycache',command)
    def test_cpu_empty_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'receipt.json';p.write_text('{}')
            with self.assertRaises(AssertionError):cpu_gate(p,[{'id':'x','seeds':[19]}])


if __name__=='__main__':unittest.main()
