"""Offline execution of the native-log boundary; no PyASC, provider or service."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'integrations/cannbench/ci/swiglu_performance'))
import qualification_control as control
import evaluate
import qualify


def report():
    return {'status':'passed','checks':[{'positive':True,'suppressed_rejected':True,
        'real_launch_requests':1,'real_launches':1,'intercepted_launches':1,
        'suppressed_runtime_calls':0,'input_sha256_before':['abc'],'input_sha256_after':['abc']}]}


class NativeBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'generation'
        self.output = Path(self.temp.name)/'qualification'
        for name in ('project/candidate.py','project/policy.json',
                     'context/skills/asctile-api-patterns/templates/packed-bf16-swiglu.py'):
            p=self.root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('unchanged')

    def run_child(self, payload=None, stdout='', stderr='', mutate=False, exit_code=0):
        original = subprocess.Popen
        child = ('import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); '
                 'p.joinpath("camodel.json").write_text(sys.argv[2]); '
                 'print(sys.argv[3]); print(sys.argv[4],file=sys.stderr); '
                 + ('pathlib.Path(sys.argv[5]).write_text("mutated")' if mutate else 'pass')
                 + '; sys.exit('+str(exit_code)+')')
        def launch(command, **kw):
            self.assertIn('--worker',command)
            self.assertIn('--numerics-only',command)
            return original([sys.executable,'-c',child,str(self.output),json.dumps(payload or report()),
                             stdout,stderr,str(self.root/'project/candidate.py')],**kw)
        with patch.dict(os.environ,{},clear=True), patch.object(control.subprocess,'Popen',side_effect=launch):
            rc=control.supervise(self.root,self.output,True)
        return rc,json.loads((self.output/'camodel.json').read_text())

    def test_real_child_success_logs_and_separate_requests(self):
        rc,result=self.run_child(stdout='retained output',stderr='retained error stream')
        self.assertEqual(rc,0)
        self.assertEqual(result['child_exit_code'],0)
        self.assertEqual(result['source_sha256_before'],result['source_sha256_after'])
        self.assertIn('retained error stream',(self.output/'model.stderr.log').read_text())
        self.assertTrue((self.output/'child.json').is_file())

    def test_exit_zero_numerical_pass_with_native_dma_fails_both_streams(self):
        marker='[error] [core31.veccore1] [mte_gdma_illegal_burst_len_t0] MTE2 invalid'
        for stream in ('stdout','stderr'):
            with self.subTest(stream=stream):
                self.output=Path(self.temp.name)/stream
                rc,result=self.run_child(**{stream:marker})
                self.assertEqual((rc,result['status'],result['child_exit_code']),(1,'failed',0))
                self.assertEqual(result['diagnostics']['illegal_dma'],1)
                self.assertTrue((self.output/f'model.{stream}.log').exists())

    def test_intentional_arithmetic_is_counted_separately(self):
        rc,result=self.run_child(stderr='[error] [core0.veccore0] [vec_err_idata_inf_nan_t0] Inf')
        self.assertEqual(rc,0)
        self.assertEqual(result['diagnostics']['arithmetic_nonfinite'],1)
        self.assertEqual(result['diagnostics']['illegal_dma'],0)

    def test_dry_run_even_zero_refused_before_child(self):
        with patch.dict(os.environ,{'PYASC_DRY_RUN':'0'},clear=True), patch.object(control.subprocess,'Popen') as spawn:
            self.assertEqual(control.supervise(self.root,self.output,True),1)
            spawn.assert_not_called()
        self.assertIn('must be unset',json.loads((self.output/'camodel.json').read_text())['error'])

    def test_original_sources_mutated_fails(self):
        rc,result=self.run_child(mutate=True)
        self.assertEqual(rc,1)
        self.assertIn('sources changed',result['error'])

    def test_numerical_false_pass_without_runtime_launch_fails(self):
        p=report(); p['checks'][0]['real_launches']=0
        self.assertEqual(self.run_child(payload=p)[0],1)

    def test_input_mutation_and_nonrejecting_control_fail(self):
        for field,value in [('input_sha256_after',['different']),('suppressed_rejected',False),('suppressed_runtime_calls',1)]:
            self.output=Path(self.temp.name)/field
            p=report();p['checks'][0][field]=value
            self.assertEqual(self.run_child(payload=p)[0],1)

    def test_original_intent_prevents_reexecution(self):
        self.run_child()
        with patch.object(control.subprocess,'Popen') as spawn:
            with self.assertRaises(FileExistsError):
                control.supervise(self.root,self.output,True)
            spawn.assert_not_called()

    def test_nonzero_original_child_preserves_failure_logs(self):
        rc,result=self.run_child(stderr='original exception',exit_code=3)
        self.assertEqual((rc,result['status'],result['child_exit_code']),(1,'failed',3))
        self.assertIn('original exception',(self.output/'model.stderr.log').read_text())

    def test_deadline_stops_original_process_group_and_reaps(self):
        child=SimpleNamespace(pid=12345)
        from unittest.mock import Mock
        child.wait=Mock(side_effect=[subprocess.TimeoutExpired('qualify',1800),-9])
        with patch.dict(os.environ,{},clear=True), patch.object(control.subprocess,'Popen',return_value=child), \
             patch.object(control.os,'killpg') as kill:
            self.assertEqual(control.supervise(self.root,self.output,True),1)
        kill.assert_called_once_with(12345,control.signal.SIGKILL)
        self.assertEqual(child.wait.call_count,2)
        self.assertIn('deadline exceeded',json.loads((self.output/'camodel.json').read_text())['error'])

    def test_actual_dispatch_counts_and_restores(self):
        runtime=SimpleNamespace(launch_kernel=lambda:None)
        class Launcher:
            def run(self): runtime.launch_kernel()
        old=Launcher.run; native=runtime.launch_kernel
        def spy(asc,suppress):
            count={'calls':0}
            def run(self):
                count['calls']+=1
                if not suppress: return old(self)
            Launcher.run=run
            return Launcher,old,count
        self.assertEqual(control.dispatch_counted(lambda:Launcher().run(),None,runtime,spy,False),
                         {'requests':1,'runtime_calls':1})
        self.assertEqual(control.dispatch_counted(lambda:Launcher().run(),None,runtime,spy,True),
                         {'requests':1,'runtime_calls':0})
        self.assertIs(Launcher.run,old); self.assertIs(runtime.launch_kernel,native)
        with self.assertRaises(ValueError):
            control.dispatch_counted(lambda:None,None,runtime,spy,False)
        self.assertIs(Launcher.run,old); self.assertIs(runtime.launch_kernel,native)


class RestoreBinding(unittest.TestCase):
    def test_existing_actions_origin_is_bound_to_candidate_and_context(self):
        for mutation in (None,'project/candidate.py','context/skill.md'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                run={'run_attempt':1,'head_repository':{'full_name':'o/r'},'event':'workflow_dispatch',
                     'path':'.github/workflows/ci.yml','head_sha':'a'*40}
                jobs={'jobs':[{'name':'swiglu-performance / generate','conclusion':'success'}]}
                def download(command,check):
                    d=Path(command[-1]);(d/'project').mkdir();(d/'context').mkdir()
                    files=['project/candidate.py','project/policy.json','project/opencode.json',
                           'prompt.txt','policy-prompt.txt','context/skill.md']
                    for f in files:(d/f).write_text('original')
                    g={'run_id':'123','run_attempt':1,'revision':'a'*40,'model':evaluate.ci.MODEL,
                       'loaded_skills':list(evaluate.ci.SKILLS), 'context_hashes':evaluate.ci.hashes(d/'context')}
                    for key,path in [('candidate','project/candidate.py'),('policy','project/policy.json'),
                        ('profile','project/opencode.json'),('prompt','prompt.txt'),('policy_prompt','policy-prompt.txt')]:
                        g[key+'_sha256']=evaluate.sha(d/path)
                    (d/'generation.json').write_text(json.dumps(g))
                    if mutation:(d/mutation).write_text('changed')
                with patch.dict(os.environ,{'GITHUB_REPOSITORY':'o/r'}), \
                     patch.object(evaluate.subprocess,'check_output',side_effect=[json.dumps(run).encode(),json.dumps(jobs).encode()]), \
                     patch.object(evaluate.subprocess,'run',side_effect=download):
                    if mutation:
                        with self.assertRaises(ValueError):evaluate.restore(root,'123')
                    else:evaluate.restore(root,'123')


class PackageBoundary(unittest.TestCase):
    def test_package_refuses_retained_false_pass_and_numerics_only(self):
        for change in ({'status':'failed'}, {'diagnostics':{'illegal_dma':1}},
                       {'child_exit_code':1}, {'numerics_only':True}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'generation').mkdir();(root/'qualification').mkdir()
                (root/'generation/generation.json').write_text('{}')
                payload=dict(report(),child_exit_code=0,diagnostics={'illegal_dma':0},numerics_only=False)
                payload.update(change)
                (root/'qualification/camodel.json').write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError,'Missing clean full-bank CaModel qualification'):
                    evaluate.package(root)
                self.assertFalse((root/'prepared/submission.zip').exists())

    def test_package_rechecks_source_hashes_and_actual_launch_evidence(self):
        for change in ('source_before','source_after','runtime_calls'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);g=root/'generation';q=root/'qualification';q.mkdir()
                for f in ('project/candidate.py','project/policy.json',
                          'context/skills/asctile-api-patterns/templates/packed-bf16-swiglu.py'):
                    p=g/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('retained')
                (g/'generation.json').write_text('{}')
                payload=dict(report(),child_exit_code=0,diagnostics={'illegal_dma':0},numerics_only=False,
                             source_sha256_before=control.source_hashes(g),source_sha256_after=control.source_hashes(g))
                if change=='runtime_calls':payload['checks'][0]['real_launches']=0
                else:payload['source_sha256_'+('before' if change=='source_before' else 'after')]['candidate']='wrong'
                (q/'camodel.json').write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError,'Qualified generation sources changed|Incomplete numerical/launch/input evidence'):
                    evaluate.package(root)
                self.assertFalse((root/'prepared/submission.zip').exists())


class ProbeInventory(unittest.TestCase):
    def test_original_probes_preserved_and_new_boundary_capacity_explicit(self):
        def band(end,route,tile):
            return dict(through=end,route=route,tile=tile,unroll=2,
                        block_cap=5,tiles_per_block=2,unroll_waves=3)
        policy={d:[band(8,'slab',32),band(63,'rows',16),band(None,'flat',128)] for d in qualify.DTYPES}
        original=list(qualify.probe_cases(policy,False))
        numerical=list(qualify.probe_cases(policy,True))
        self.assertEqual(len(original),24)
        self.assertEqual(numerical[:24],original)
        self.assertEqual(numerical[24:],[('bfloat16',(67,34),1,19,64),('bfloat16',(67,34),1,23,64),
                                        ('bfloat16',(67,514),1,19,64),('bfloat16',(67,514),1,23,64)])
        policy['float32'][1].update(through=64,tile=2,unroll=3,unroll_waves=8,
                                    block_cap=None,tiles_per_block=1)
        self.assertEqual(len(list(qualify.probe_cases(policy,False))),30)
        self.assertEqual(len(list(qualify.probe_cases(policy,True))),34)


class WorkflowBoundary(unittest.TestCase):
    def test_direct_dispatch_has_only_origin_and_read_only_qualification_job(self):
        text=(ROOT/'.github/workflows/swiglu-performance.yml').read_text()
        dispatch=text.split('  workflow_dispatch:',1)[1].split('  workflow_call:',1)[0]
        self.assertEqual(re.findall(r'^      (\w+):$',dispatch,re.M),['origin'])
        blocks={m.group(1):m.group(2) for m in re.finditer(r'^  ([\w-]+):\n(.*?)(?=^  [\w-]+:\n|\Z)',text.split('jobs:\n',1)[1],re.M|re.S)}
        self.assertEqual(set(blocks),{'generate','evaluate','qualify-only'})
        self.assertIn("inputs.action != ''",blocks['generate'])
        self.assertIn("inputs.action != ''",blocks['evaluate'])
        def enabled(block, action, event='workflow_dispatch'):
            raw=block.split('    if: ',1)[1].split('    runs-on:',1)[0]
            raw=raw.split('    permissions:',1)[0]
            expression=' '.join(raw.removeprefix('>-').strip().splitlines())
            expression=expression.replace('always()','True').replace('!cancelled()','True')
            expression=expression.replace('needs.generate.result',repr('success'))
            expression=expression.replace('github.event_name',repr(event)).replace('inputs.action',repr(action))
            expression=expression.replace('&&',' and ').replace('||',' or ')
            return eval(expression, {'__builtins__':{}}, {})
        self.assertEqual({name:enabled(block,'') for name,block in blocks.items()},
                         {'generate':False,'evaluate':False,'qualify-only':True})
        for action in ('performance','performance-generate','performance-evaluate',
                       'performance-resume','performance-submit-prepared'):
            self.assertFalse(enabled(blocks['qualify-only'],action))
            self.assertEqual(enabled(blocks['generate'],action),action in ('performance','performance-generate'))
            self.assertEqual(enabled(blocks['evaluate'],action),action!='performance-generate')
        q=blocks['qualify-only']
        # Runner context is unavailable in job-level env; resolve its directory
        # in the running step and export it for subsequent steps/artifacts.
        self.assertNotIn('runner.', q.split('    steps:', 1)[0])
        self.assertIn('ROOT="$RUNNER_TEMP/swiglu-qualification-', q)
        self.assertIn('echo "ROOT=$ROOT" >> "$GITHUB_ENV"', q)
        self.assertIn("github.event_name == 'workflow_dispatch' && inputs.action == ''",q)
        self.assertIn('evaluate.py restore "$ROOT" --origin "$ORIGIN"',q)
        self.assertIn('qualify.py /generation /result --numerics-only',q)
        self.assertIn('if: always()',q)
        self.assertIn('contents: read',q)
        self.assertNotIn('contents: write',q)
        self.assertNotRegex('\n'.join(line for line in q.splitlines() if not line.lstrip().startswith('#')),r'secrets\.|setup-site-mcp|generate\.py|package\s|submit\s|reopen\s|resume\s|gh release')


if __name__=='__main__':unittest.main()
