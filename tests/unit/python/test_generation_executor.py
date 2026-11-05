"""Offline executor contracts. Provider/qualification doubles never claim real generation."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import generation_adapters as adapters
import generation_executor as executor
import generation_plan as planner
from generation_provider import BudgetError, BudgetGate, budget_server, response_usage


class ExecutorContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.plan = planner.build_plan()
        self.plan['checkout_revision'] = 'offline-fixture'
        self.plan['plan_sha256'] = planner.digest(planner.canonical(self.plan))

    def test_plan_validation_is_offline_and_rejects_rehashed_changes(self):
        path = self.path/'plan.json'; executor.save(path, self.plan)
        with patch.object(executor.subprocess, 'Popen', side_effect=AssertionError('Must remain offline')):
            self.assertEqual(executor.validate_plan(path), self.plan)
        changed = copy.deepcopy(self.plan); changed.pop('plan_sha256')
        changed['limits']['rounds_per_attempt'] = 100
        changed['plan_sha256'] = planner.digest(planner.canonical(changed)); executor.save(path, changed)
        with self.assertRaisesRegex(ValueError, 'current inputs'): executor.validate_plan(path)

    def test_adapter_contracts_and_static_host_fallback_rejection(self):
        for cell, name in zip(self.plan['cells'], ('add.py', 'reduce.py', 'broadcast.py')):
            contract = adapters.task_contract(ROOT, cell)
            source = (ROOT/'tests/generation/fixtures'/name).read_text()
            self.assertEqual(adapters.static_check(source, contract)['status'], 'passed')
            self.assertEqual(adapters.static_check(source.replace('@asctile.jit(always_compile=True)', '@asctile.jit'), contract)['status'], 'passed')
            compile(adapters.wrapper(contract), 'probe.py', 'exec')
            for addition in ('\nimport torch', '\nopen("secret")', '\nfrom asc.runtime import config'):
                with self.assertRaises(ValueError): adapters.static_check(source+addition, contract)
            with self.assertRaises(ValueError):
                adapters.static_check(source.replace('def launch(', 'def hidden('), contract)
            with self.assertRaises(ValueError):
                adapters.static_check(source.replace('    '+('add_kernel[1](x, y, out)' if name=='add.py' else
                    'sum_kernel[1](x, out)' if name=='reduce.py' else 'broadcast_kernel[4](x, out)'),
                    '    out.copy_(x)'), contract)
        bad = dict(self.plan['cells'][0], cases=['vadd_test_1'])
        with self.assertRaisesRegex(ValueError, 'Unsupported'): adapters.task_contract(ROOT, bad)

    def test_adapter_detects_upstream_contract_change_without_importing(self):
        task = self.path/'reference/asctile/target/test_vadd.py'; task.parent.mkdir(parents=True)
        task.write_text((ROOT/'reference/asctile/target/test_vadd.py').read_text().replace('[199]', '[198]'))
        with self.assertRaisesRegex(ValueError, 'differs from upstream'):
            adapters.task_contract(self.path, self.plan['cells'][0])

    def test_native_attribution_requires_each_selected_skill_and_unique_session(self):
        skills = {'first': self.path/'first'}
        event = {'sessionID':'ses_1', 'part': {'tool':'skill', 'state': {'status':'completed',
            'input':{'name':'first'}, 'metadata':{'dir':str(skills['first'])}}}}
        self.assertEqual(executor.native_trace(json.dumps(event), skills)['session_id'], 'ses_1')
        for events in ('{}', json.dumps(event)+'\n'+json.dumps(dict(event, sessionID='ses_2')),
                       json.dumps(event).replace(str(skills['first']), '/untracked/first')):
            with self.assertRaises(executor.Blocked): executor.native_trace(events, skills)

    def test_lock_and_unresolved_latch_prevent_duplicate_execution(self):
        state = self.path/'state'
        with executor.exclusive(state, self.path/'run'):
            with self.assertRaisesRegex(executor.Blocked, 'holds'):
                with executor.exclusive(state, self.path/'another'): pass
        with self.assertRaisesRegex(executor.Blocked, 'Unresolved'):
            with executor.exclusive(state, self.path/'another'): pass

    def test_permanent_provider_error_stops_client_without_waiting_for_timeout(self):
        with self.assertRaisesRegex(executor.Blocked, 'Provider gate'):
            executor.command([sys.executable, '-c', 'import time;time.sleep(30)'],
                             self.path, 'abort', 30, abort=lambda: True)
        record = json.loads((self.path/'abort.command.json').read_text())
        self.assertEqual(record['status'], 'blocked')
        self.assertLess(record['duration_seconds'], 3)
        self.assertNotEqual(record['exit_code'], 0)

    def test_timeout_kills_child_group_and_records_partial_command(self):
        child = self.path/'child-done'
        script = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'+repr(
            'import time,pathlib;time.sleep(0.7);pathlib.Path('+repr(str(child))+').touch()')+']);time.sleep(10)'
        with self.assertRaises(TimeoutError):
            executor.command([sys.executable, '-c', script], self.path, 'timeout', .2)
        time.sleep(.8)
        self.assertFalse(child.exists())
        record = json.loads((self.path/'timeout.command.json').read_text())
        self.assertEqual(record['status'], 'blocked'); self.assertLess(record['duration_seconds'], 3)
        self.assertIn('stdout_sha256', record)

    def run_double(self, generation=None, qualification=None):
        def fake_generate(attempt, plan, contract, context):
            (attempt/'candidate.py').write_text('fixed offline double\n')
            usage = {'observed_output_tokens': 10, 'usage_complete': True}
            executor.save(attempt/'usage.json', usage)
            return usage
        with patch.dict(os.environ, OPENCODE_API_KEY='local-test-only', OPENCODE_BASE_URL='http://127.0.0.1'), \
             patch.object(executor, 'command', return_value={'exit_code':0}), \
             patch.object(executor, 'generate', side_effect=generation or fake_generate), \
             patch.object(executor, 'qualify', side_effect=qualification or (lambda *args: {'status':'passed'})):
            return executor.execute(self.plan, self.path/'run', self.path/'state')

    def test_serial_execution_reserves_all_attempts_and_preserves_evidence(self):
        result = self.run_double()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['reserved_output_tokens'], 98304)
        self.assertEqual(result['observed_output_tokens'], 30)
        self.assertEqual(result['reserved_execution_seconds'], 1800)
        self.assertEqual([c['status'] for c in result['cells']], ['passed']*3)
        self.assertFalse((self.path/'state/active.json').exists())
        self.assertTrue((self.path/'run/files.json').is_file())
        with self.assertRaisesRegex(ValueError, 'never replayed'): self.run_double()

    def test_unknown_usage_stops_remaining_cells_and_latches_run(self):
        result = self.run_double(generation=lambda *args: (_ for _ in ()).throw(executor.Blocked('Unknown delivery')))
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual([c['status'] for c in result['cells']], ['blocked', 'not-run', 'not-run'])
        self.assertEqual(result['reserved_output_tokens'], 32768)
        self.assertTrue((self.path/'state/active.json').exists())

    def test_failed_numerics_never_silently_passes(self):
        result = self.run_double(qualification=lambda *args: (_ for _ in ()).throw(ValueError('wrong numerics')))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual([c['status'] for c in result['cells']], ['failed']*3)
        self.assertFalse((self.path/'state/active.json').exists())

    def test_aggregate_cap_is_enforced_before_dispatch(self):
        self.plan['limits']['max_output_tokens'] = 32768
        result = self.run_double()
        self.assertEqual([c['status'] for c in result['cells']], ['passed', 'not-run', 'not-run'])
        self.assertEqual(result['status'], 'blocked')


class ProviderContracts(unittest.TestCase):
    def setUp(self):
        self.requests, self.saves = [], []
        self.payload = {'model':'test-model', 'choices':[{'index':0, 'finish_reason':'stop'}], 'usage': {'prompt_tokens':3, 'completion_tokens':7,
            'total_tokens':10, 'completion_tokens_details':{'reasoning_tokens':4}}}
        test = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                test.requests.append(body)
                if body.get('stream'):
                    raw = ('data: '+json.dumps(test.payload)+'\n\ndata: [DONE]\n\n').encode()
                else: raw = json.dumps(test.payload).encode()
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
                self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)
        self.gate = BudgetGate('http://127.0.0.1:'+str(self.server.server_port), 'fixture-key', 'test-model',
            {'rounds_per_attempt':2, 'output_tokens_per_response':16}, lambda x:self.saves.append(copy.deepcopy(x)), time.monotonic()+10)

    def test_cap_applied_before_dispatch_and_round_three_never_reaches_provider(self):
        with budget_server(self.gate) as endpoint:
            for stream in (False, True):
                request = urllib.request.Request(endpoint+'/chat/completions',
                    data=json.dumps({'model':'test-model', 'max_tokens':100000, 'stream':stream}).encode(),
                    headers={'Authorization':'Bearer '+self.gate.token})
                with urllib.request.urlopen(request) as response: self.assertEqual(response.status, 200)
            with self.assertRaises(BudgetError): self.gate.dispatch({'model':'test-model'})
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(p['max_tokens']==16 for p in self.requests))
        self.assertEqual(self.requests[1]['stream_options'], {'include_usage':True})
        self.assertEqual(self.saves[0]['requests'][0]['status'], 'dispatched')
        self.assertEqual(self.gate.snapshot()['observed_output_tokens'], 14)  # reasoning not double-counted
        self.assertNotIn('fixture-key', json.dumps(self.saves))

    def test_missing_usage_or_excess_tokens_blocks_retry(self):
        for payload in ({'model':'test-model'}, {'model':'test-model','choices':[{'index':0,'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':17,'total_tokens':20}}):
            with self.subTest(payload=payload):
                self.payload = payload; self.gate.error = None; self.gate.records = []
                before = len(self.requests)
                with self.assertRaises(BudgetError): self.gate.dispatch({'model':'test-model'})
                with self.assertRaises(BudgetError): self.gate.dispatch({'model':'test-model'})
                self.assertEqual(len(self.requests), before+1)
                self.assertEqual(self.gate.snapshot()['usage_complete'], 'usage' in payload)
                self.assertEqual(self.gate.snapshot()['reserved_output_tokens'], 16)

    def test_rejected_response_retains_private_bytes_hash_and_safe_reason(self):
        self.payload['choices'][0]['finish_reason'] = 'length'
        captured = []
        self.gate.capture_response = lambda number, raw: captured.append((number, raw))
        with self.assertRaisesRegex(BudgetError, 'truncated'):
            self.gate.dispatch({'model':'test-model', 'stream':True})
        record = self.gate.records[0]
        self.assertEqual(record['status'], 'rejected')
        self.assertEqual(record['http_status'], 200)
        self.assertEqual(record['validation_error'], 'Model completion truncated at output-token cap')
        self.assertTrue(self.gate.snapshot()['usage_complete'])
        self.assertEqual(self.gate.snapshot()['observed_output_tokens'], 7)
        self.assertEqual(record['response_sha256'], planner.digest(captured[0][1]))
        self.assertEqual(captured[0][0], 1)
        self.assertNotIn('fixture-key', json.dumps(self.saves))
        with budget_server(self.gate) as endpoint:
            request = urllib.request.Request(endpoint+'/chat/completions',
                data=json.dumps({'model':'test-model'}).encode(),
                headers={'Authorization':'Bearer '+self.gate.token})
            with self.assertRaises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(request)
            self.assertEqual(exc.exception.code,400)
            exc.exception.close()
        self.assertEqual(len(self.requests),1)  # No retry reaches the provider.

    def test_opencode_adapter_uses_isolated_config_gate_and_native_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); attempt = root/'attempt'; attempt.mkdir()
            context = root/'context'; context.mkdir()
            plan = planner.build_plan()
            profile = 'integrations/opencode/profiles/'+plan['model']['profile']+'.json'
            target = context/profile; target.parent.mkdir(parents=True); target.write_bytes((ROOT/profile).read_bytes())
            for name in plan['source_sha256']:
                if name.startswith('skills/'):
                    dest = context/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((ROOT/name).read_bytes())
            self.payload['model'] = plan['model']['expected_response']
            real_command = executor.command
            def fake_client(argv, directory, name, seconds, env=None, abort=None, cwd=None):
                if name=='project-init': return real_command(argv, directory, name, seconds)
                self.assertEqual(cwd, attempt/'project')
                worktree = subprocess.check_output(['git', '-C', str(cwd), 'rev-parse', '--show-toplevel'], text=True).strip()
                self.assertEqual(Path(worktree).resolve(), cwd.resolve())
                self.assertEqual(argv[:4], ['opencode','run','--pure','--format'])
                self.assertNotIn('OPENCODE_API_KEY', env)
                config = json.loads(env['OPENCODE_CONFIG_CONTENT'])
                options = config['provider']['gateway']['options']
                self.assertEqual(options['apiKey'], '{env:PYASC_GATE_TOKEN}')
                request = urllib.request.Request(options['baseURL']+'/chat/completions',
                    data=json.dumps({'model':plan['model']['requested'].split('/')[1], 'stream':True}).encode(),
                    headers={'Authorization':'Bearer '+env['PYASC_GATE_TOKEN']})
                with urllib.request.urlopen(request) as response: response.read()
                events = [{'sessionID':'fresh-session', 'type':'tool_use', 'part':{'tool':'skill',
                    'state':{'status':'completed','input':{'name':s},'metadata':{'dir':str(context/'skills'/s)}}}}
                    for s in plan['skills']]
                (directory/'generation.stdout').write_text('\n'.join(json.dumps(e) for e in events))
                (directory/'project/candidate.py').write_bytes((ROOT/'tests/generation/fixtures/add.py').read_bytes())
                return {'exit_code':0}
            with patch.dict(os.environ, OPENCODE_API_KEY='never-pass-this-key-to-agent',
                    OPENCODE_BASE_URL='http://127.0.0.1:'+str(self.server.server_port)), \
                    patch.object(executor, 'command', side_effect=fake_client):
                usage = executor.generate(attempt, plan, adapters.task_contract(ROOT,plan['cells'][0]), context)
            self.assertTrue(usage['usage_complete'])
            self.assertEqual(len(self.requests),1)
            self.assertEqual(json.loads((attempt/'attribution.json').read_text())['session_id'],'fresh-session')
            self.assertNotIn('never-pass-this-key', (attempt/'model-config.json').read_text())
            self.assertEqual((attempt/'candidate.py').read_bytes(), (attempt/'project/candidate.py').read_bytes())

    def test_request_alias_requires_explicit_response_revision_binding(self):
        self.payload['model'] = 'test-model-revision-20261106'
        with self.assertRaisesRegex(BudgetError, 'different observed model'):
            self.gate.dispatch({'model':'test-model'})
        self.gate.records = []; self.gate.error = None
        self.gate.response_model = 'test-model-revision-20261106'
        self.gate.dispatch({'model':'test-model'})
        self.assertEqual(self.requests[-1]['model'],'test-model')
        self.assertEqual(self.gate.records[-1]['usage']['observed_model'],'test-model-revision-20261106')
        self.payload['model'] = 'unapproved-revision'
        with self.assertRaisesRegex(BudgetError,'different observed model'):
            self.gate.dispatch({'model':'test-model'})

    def test_usage_requires_model_completion_integer_counts_and_consistency(self):
        self.assertEqual(response_usage(json.dumps(self.payload).encode(),False,'test-model')['completion_tokens'],7)
        changes = [lambda p:p.update(model='fallback'), lambda p:p['usage'].update(completion_tokens=True),
                   lambda p:p['usage'].update(total_tokens=11), lambda p:p.pop('usage'),
                   lambda p:p['choices'][0].update(finish_reason='unexpected')]
        for change in changes:
            data=copy.deepcopy(self.payload); change(data)
            with self.assertRaises(BudgetError): response_usage(json.dumps(data).encode(),False,'test-model')
        with self.assertRaises(BudgetError): response_usage(('data: '+json.dumps(self.payload)).encode(),True,'test-model')
        with self.assertRaises(BudgetError): self.gate.dispatch({'model':'fallback'})
        self.assertFalse(self.requests)


if __name__ == '__main__': unittest.main()
