"""Offline executor contracts. Provider/qualification doubles never claim real generation."""
import copy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import time
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import generation_adapters as adapters
import generation_executor as executor
import generation_plan as planner
from generation_provider import BudgetError, BudgetGate, budget_server, response_usage, known_budget_exhaustion, request_settings


class ExecutorContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.plan = planner.build_plan()
        self.plan['checkout_revision'] = 'offline-fixture'
        self.plan['plan_sha256'] = planner.digest(planner.canonical(self.plan))

    def test_source_availability_keeps_partial_bytes_outside_the_candidate_contract(self):
        project=self.path/'project'
        project.mkdir()
        source=project/'candidate.py'
        def snapshot():return executor.retain_attempt_source(self.path, {'status':'failed'})
        self.assertEqual(snapshot(), {'status':'unavailable','reason':'no-file'})
        source.write_bytes(b'def kernel(\n')
        partial=snapshot()
        self.assertEqual(partial, {'status':'incomplete','reason':'generation-incomplete',
                                  'sha256':planner.digest(source.read_bytes())})
        self.assertEqual((self.path/'partial-candidate.py').read_bytes(), source.read_bytes())
        self.assertFalse((self.path/'candidate.py').exists())
        source.write_bytes(b'\xff'); self.assertEqual(snapshot()['reason'],'invalid-text')
        source.unlink();source.symlink_to(self.path/'partial-candidate.py')
        self.assertEqual(snapshot()['reason'],'invalid-file')

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
        cell = next(c for c in self.plan['cells'] if c['id']=='add-f32-tail')
        source = (ROOT/'tests/generation/fixtures/add.py').read_text().replace(
            'add_kernel[1](x, y, out)', 'add_kernel[1](x, y, out, len(x), 256)')
        with self.assertRaisesRegex(ValueError, r"Host launch argument 'len\(x\)' must be a literal"):
            adapters.static_check(source, adapters.task_contract(ROOT, cell))
        bad = dict(self.plan['cells'][0], cases=['vadd_test_1'])
        with self.assertRaisesRegex(ValueError, 'Unsupported'): adapters.task_contract(ROOT, bad)

    def test_unknown_public_api_inside_lowering_is_a_candidate_failure(self):
        from pytest_verify_kernel import _is_provisioning_error
        namespace = {}
        exec(compile("def resolve(module): return module.this_function_does_not_exist", "/fake/asc/codegen/visitor.py", "exec"), namespace)
        for module_name, provisioning in [('asc.experimental.asctile', False), ('asctile', False), ('asc.lib.native', True)]:
            try: namespace['resolve'](types.ModuleType(module_name))
            except AttributeError as exc:
                self.assertEqual(_is_provisioning_error(exc, '/fake/asc'), provisioning)

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
        self.assertEqual(result['reserved_output_tokens'], 196608)
        self.assertEqual(result['observed_output_tokens'], 30)
        self.assertEqual(result['reserved_execution_seconds'], 1800)
        self.assertEqual([c['status'] for c in result['cells']], ['passed']*3)
        self.assertLessEqual(datetime.fromisoformat(result['started_at']), datetime.fromisoformat(result['finished_at']))
        self.assertIsNotNone(datetime.fromisoformat(result['finished_at']).utcoffset())
        self.assertFalse((self.path/'state/active.json').exists())
        self.assertTrue((self.path/'run/files.json').is_file())
        with self.assertRaisesRegex(ValueError, 'never replayed'): self.run_double()

    def test_unknown_usage_stops_remaining_cells_and_latches_run(self):
        result = self.run_double(generation=lambda *args: (_ for _ in ()).throw(executor.Blocked('Unknown delivery')))
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual([c['status'] for c in result['cells']], ['blocked', 'not-run', 'not-run'])
        self.assertIn('finished_at', result)
        self.assertEqual(result['reserved_output_tokens'], 65536)
        self.assertTrue((self.path/'state/active.json').exists())

    def test_failed_numerics_never_silently_passes(self):
        result = self.run_double(qualification=lambda *args: (_ for _ in ()).throw(ValueError('wrong numerics')))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual([c['status'] for c in result['cells']], ['failed']*3)
        self.assertFalse((self.path/'state/active.json').exists())

    def test_aggregate_cap_is_enforced_before_dispatch(self):
        self.plan['limits']['max_output_tokens'] = 65536
        result = self.run_double()
        self.assertEqual([c['status'] for c in result['cells']], ['passed', 'not-run', 'not-run'])
        self.assertEqual(result['status'], 'blocked')


class ProviderContracts(unittest.TestCase):
    def test_public_wire_settings_distinguish_explicit_values_and_unknown_defaults(self):
        body = {'max_tokens': 8192, 'temperature': 0, 'seed': 0, 'reasoning_effort': 'high',
            'thinking': {'type': 'enabled'}, 'enable_thinking': False, 'messages': [{'content': 'private prompt'}],
            'metadata': {'api_key': 'private-secret'}, 'tools': [{'description': 'private tool content'}]}
        value = request_settings(body)
        self.assertEqual(value['sampling']['temperature'], 0); self.assertEqual(value['sampling']['seed'], 0)
        self.assertIsNone(value['sampling']['top_p']); self.assertNotIn('top_p', value['present_fields'])
        self.assertEqual(value['reasoning']['thinking'], {'type': 'enabled'})
        self.assertFalse(value['reasoning']['enable_thinking']); self.assertEqual(value['max_tokens'], 8192)
        self.assertNotIn('private', json.dumps(value))
        value['reasoning']['thinking']['type'] = 'disabled'
        self.assertEqual(body['thinking']['type'], 'enabled')

    def test_unsafe_or_nonfinite_wire_settings_are_rejected_without_echoing_values(self):
        for extra in ({'temperature': float('nan')}, {'seed': True}, {'seed': 1.5}, {'top_p': 'private-secret'},
                      {'reasoning_effort': 'private-secret'}, {'thinking': {'type': 'enabled', 'secret': 'private-secret'}},
                      {'thinking': {'type': 'enabled', 'budget_tokens': -1}}, {'enable_thinking': 'private-secret'}):
            with self.subTest(extra=extra), self.assertRaises(BudgetError) as error:
                request_settings(dict(max_tokens=10, **extra))
            self.assertNotIn('private-secret', str(error.exception))

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
                if getattr(test, 'truncate_stream', False): raw = raw.removesuffix(b'data: [DONE]\n\n')
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
                if getattr(test, 'missing_http_trailer', False) or getattr(test, 'truncated_http_chunk', False):
                    self.send_header('Transfer-Encoding', 'chunked'); self.end_headers()
                    truncated = getattr(test, 'truncated_http_chunk', False)
                    terminator=b'\r' if getattr(test, 'partial_chunk_terminator', False) else b'\r\n'
                    self.wfile.write(format(len(raw)+(8 if truncated else 0), 'x').encode()+b'\r\n'+raw+(b'' if truncated else terminator))
                    self.close_connection = True  # Deliberately omit the terminating HTTP chunk.
                    return
                self.send_header('Content-Length', str(len(raw)+(8 if getattr(test, 'short_content_length', False) else 0)))
                self.end_headers(); self.wfile.write(raw)
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
        self.assertEqual(known_budget_exhaustion(self.gate.snapshot(), self.gate.limits), 'request-limit')
        unknown = self.gate.snapshot();unknown['requests'][0] = dict(unknown['requests'][0], status='unknown')
        self.assertIsNone(known_budget_exhaustion(unknown, self.gate.limits))

    def test_context_audit_failure_never_reaches_provider_and_is_not_unknown_delivery(self):
        self.gate.observe_request = lambda *args: (_ for _ in ()).throw(ValueError('bad local context'))
        with self.assertRaises(BudgetError): self.gate.dispatch({'model': 'test-model'})
        self.assertEqual(self.requests, [])
        record = self.gate.snapshot()['requests'][0]
        self.assertEqual(record['status'], 'not-sent');self.assertFalse(record['dispatch_started'])
        self.assertIsNotNone(self.gate.error)

    def test_actual_gate_retains_capped_wire_settings_before_provider_dispatch(self):
        body = {'model': 'test-model', 'temperature': 0.2, 'reasoning_effort': 'high',
            'thinking': {'type': 'enabled', 'budget_tokens': 8}, 'max_tokens': 10000}
        self.gate.dispatch(body)
        value = self.gate.snapshot()['requests'][0]['request_settings']
        self.assertEqual(value, request_settings(self.requests[0])); self.assertEqual(value['max_tokens'], 16)
        before = [s['requests'][0] for s in self.saves if s['requests'] and s['requests'][0].get('dispatch_started')]
        self.assertEqual(before[0]['request_settings'], value); self.assertNotIn('usage', before[0])

    def test_invalid_wire_settings_stop_before_any_provider_request(self):
        with self.assertRaises(BudgetError): self.gate.dispatch({'model': 'test-model', 'temperature': 'private-secret'})
        record = self.gate.snapshot()['requests'][0]
        self.assertEqual(self.requests, []); self.assertEqual(record['status'], 'not-sent')
        self.assertFalse(record['dispatch_started']); self.assertNotIn('private-secret', json.dumps(self.saves))

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

    def test_complete_sse_before_missing_http_trailer_keeps_exact_usage_without_retry(self):
        self.missing_http_trailer = True
        captured = []; self.gate.capture_response = lambda number, raw: captured.append((number, raw))
        raw, _ = self.gate.dispatch({'model':'test-model','stream':True})
        record = self.gate.records[0]
        self.assertFalse(record['http_body_complete']); self.assertEqual(record['transport_error'],'IncompleteRead')
        self.assertEqual(record['status'],'complete'); self.assertEqual(record['usage']['observed_model'],'test-model')
        self.assertTrue(self.gate.snapshot()['usage_complete']); self.assertEqual(self.gate.snapshot()['observed_output_tokens'],7)
        self.assertEqual(captured,[(1,raw)]); self.assertEqual(record['response_sha256'],planner.digest(raw))
        self.assertEqual(len(self.requests),1)

    def test_incomplete_sse_is_captured_privately_and_never_replayed(self):
        self.missing_http_trailer = self.truncate_stream = True
        captured = []; self.gate.capture_response = lambda number, raw: captured.append((number,raw))
        with self.assertRaisesRegex(BudgetError,'Incomplete streaming response'):
            self.gate.dispatch({'model':'test-model','stream':True})
        record=self.gate.records[0]
        self.assertEqual(record['status'],'unknown'); self.assertNotIn('usage',record)
        self.assertFalse(self.gate.snapshot()['usage_complete']); self.assertEqual(record['response_bytes'],len(captured[0][1]))
        self.assertEqual(record['response_sha256'],planner.digest(captured[0][1]))
        with self.assertRaisesRegex(BudgetError,'Earlier request'): self.gate.dispatch({'model':'test-model','stream':True})
        self.assertEqual(len(self.requests),1); self.assertEqual(self.gate.snapshot()['reserved_output_tokens'],16)

    def test_incomplete_nonstreaming_http_body_cannot_be_accepted_as_sse(self):
        self.missing_http_trailer = True
        with self.assertRaisesRegex(BudgetError,'Incomplete non-streaming'):
            self.gate.dispatch({'model':'test-model','stream':False})
        self.assertEqual(self.gate.records[0]['status'],'unknown')
        self.assertIn('response_sha256',self.gate.records[0])

    def test_complete_sse_inside_truncated_http_chunk_keeps_all_received_bytes(self):
        self.truncated_http_chunk = True
        captured=[];self.gate.capture_response=lambda number,raw:captured.append(raw)
        raw,_=self.gate.dispatch({'model':'test-model','stream':True})
        self.assertTrue(raw.endswith(b'data: [DONE]\n\n'));self.assertEqual(captured,[raw])
        self.assertEqual(self.gate.records[0]['status'],'complete')
        self.assertEqual(self.gate.records[0]['transport_error'],'IncompleteRead')
        self.assertEqual(self.gate.snapshot()['observed_output_tokens'],7)
        self.assertEqual(len(self.requests),1)

    def test_partial_sse_inside_truncated_chunk_stays_unknown_without_replay(self):
        self.truncated_http_chunk=self.truncate_stream=True
        captured=[];self.gate.capture_response=lambda number,raw:captured.append(raw)
        with self.assertRaisesRegex(BudgetError,'Incomplete streaming response'):
            self.gate.dispatch({'model':'test-model','stream':True})
        self.assertTrue(captured[0].startswith(b'data: '));self.assertNotIn(b'[DONE]',captured[0])
        self.assertEqual(self.gate.records[0]['response_sha256'],planner.digest(captured[0]))
        self.assertEqual(self.gate.records[0]['status'],'unknown')
        with self.assertRaisesRegex(BudgetError,'Earlier request'):self.gate.dispatch({'model':'test-model','stream':True})
        self.assertEqual(len(self.requests),1)

    def test_short_content_length_nonstreaming_body_is_preserved_but_not_accepted(self):
        self.short_content_length=True
        captured=[];self.gate.capture_response=lambda number,raw:captured.append(raw)
        with self.assertRaisesRegex(BudgetError,'Incomplete non-streaming'):
            self.gate.dispatch({'model':'test-model'})
        self.assertEqual(json.loads(captured[0]),self.payload)
        self.assertEqual(self.gate.records[0]['status'],'unknown')

    def test_partial_http_chunk_terminator_does_not_enter_the_sse_body(self):
        self.missing_http_trailer=self.partial_chunk_terminator=True
        raw,_=self.gate.dispatch({'model':'test-model','stream':True})
        expected=('data: '+json.dumps(self.payload)+'\n\ndata: [DONE]\n\n').encode()
        self.assertEqual(raw,expected)
        self.assertEqual(self.gate.records[0]['response_sha256'],planner.digest(expected))

    def test_incremental_reader_stops_at_response_byte_cap_and_blocks_retry(self):
        captured=[];self.gate.capture_response=lambda number,raw:captured.append(raw)
        with patch('generation_provider.MAX_RESPONSE',64),self.assertRaisesRegex(BudgetError,'Provider response too large'):
            self.gate.dispatch({'model':'test-model','stream':True})
        self.assertEqual(len(captured[0]),65)
        self.assertEqual(self.gate.records[0]['status'],'unknown')
        with self.assertRaisesRegex(BudgetError,'Earlier request'):self.gate.dispatch({'model':'test-model','stream':True})
        self.assertEqual(len(self.requests),1)

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

    def test_captured_counter_contradiction_is_not_a_valid_budget_failure(self):
        # Redacted counters from cohort42167095606: keep strict accounting.
        self.payload['usage'] = {'prompt_tokens': 3721, 'completion_tokens': 8192, 'total_tokens': 11913,
            'completion_tokens_details': {'reasoning_tokens': 8195}}
        self.payload['choices'][0]['finish_reason'] = 'length'
        self.gate.limits['output_tokens_per_response'] = 8192
        captured = []; self.gate.capture_response = lambda number, raw: captured.append(raw)
        with self.assertRaisesRegex(BudgetError, 'Invalid reasoning usage'):
            self.gate.dispatch({'model': 'test-model', 'stream': True})
        self.assertEqual(len(self.requests), 1); self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].decode().strip().endswith('data: [DONE]'))
        self.assertFalse(self.gate.snapshot()['usage_complete'])
        self.assertIsNone(known_budget_exhaustion(self.gate.snapshot(), self.gate.limits))
        with self.assertRaises(BudgetError): self.gate.dispatch({'model': 'test-model'})
        self.assertEqual(len(self.requests), 1)

    def test_identical_final_usage_only_chunk_is_counted_once(self):
        # Minimal protocol fixture for the two final frames captured in
        # cohort42189411115, not a retained run payload or qualification.
        ending = dict(copy.deepcopy(self.payload), id='response-1')
        ending['choices'][0]['finish_reason'] = 'tool_calls'
        usage_only = dict(copy.deepcopy(ending), choices=[])
        def encode(parts):
            return ('\n\n'.join('data: '+json.dumps(p) for p in parts)+'\n\ndata: [DONE]\n').encode()
        parsed = response_usage(encode([ending, usage_only]), True, 'test-model')
        self.assertEqual((parsed['completion_tokens'], parsed['total_tokens'], parsed['finish_reason']), (7, 10, 'tool_calls'))
        with self.assertRaisesRegex(BudgetError, 'different observed model'):
            response_usage(encode([ending, usage_only]), True, 'another-model')
        mutations = [lambda p:p[-1]['usage'].update(completion_tokens=8,total_tokens=11),
            lambda p:p[-1].update(id='response-2'), lambda p:p[-1].pop('id'),
            lambda p:p[-1].update(choices=[{'index':0,'finish_reason':'tool_calls'}]),
            lambda p:p.insert(1,dict(id='response-1',model='test-model',choices=[])),
            lambda p:p.append(copy.deepcopy(p[-1])),
            lambda p:p.insert(0,dict(id='another-response',model='test-model',choices=[]))]
        for mutate in mutations:
            rows=copy.deepcopy([ending,usage_only]);mutate(rows)
            with self.subTest(mutation=mutations.index(mutate)), self.assertRaises(BudgetError):
                response_usage(encode(rows),True,'test-model')


if __name__ == '__main__': unittest.main()
