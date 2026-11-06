"""Profile/credential boundaries; loopback doubles never qualify a model."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/tools'))
import generation_models as models
import generation_executor as executor
import generation_review as review
from generation_plan import build_plan, canonical, digest
from generation_provider import request_settings


def profile(provider='selected', key='${SELECTED_API_KEY}', url='${SELECTED_BASE_URL}'):
    return {'model': provider+'/model-v1', 'provider': {provider: {
        'npm': '@ai-sdk/openai-compatible', 'options': {'baseURL': url, 'apiKey': key},
        'models': {'model-v1': {'limit': {'output': 8192}, 'options': {'temperature': 0.2}}}}}}


def planned(config):
    return {'requested': config['model'], 'configuration': models.configuration(config)}


class ProfileBindings(unittest.TestCase):
    def test_wire_settings_distinguish_omitted_and_explicit_null_without_guessing_defaults(self):
        for body in ({'max_tokens': 8192}, {'max_tokens': 8192, 'temperature': None},
                {'max_tokens': 8192, 'temperature': 0.2, 'seed': 123, 'reasoning_effort': 'low'}):
            value = request_settings(body)
            self.assertEqual(models.inspect_request_settings({'max_tokens': 8192, 'request_settings': value}, 8192, True), value)
            self.assertEqual('temperature' in value['present_fields'], 'temperature' in body)

    def test_wire_settings_reject_missing_changed_cap_and_inconsistent_metadata(self):
        original = {'max_tokens': 8192, 'request_settings': request_settings({'max_tokens': 8192, 'reasoning_effort': 'low'})}
        changes = [lambda r: r.pop('request_settings'), lambda r: r.update(max_tokens=4096),
            lambda r: r['request_settings'].update(max_tokens=4096),
            lambda r: r['request_settings'].update(schema_version=True),
            lambda r: r['request_settings']['present_fields'].append('reasoning_effort'),
            lambda r: r['request_settings']['sampling'].update(seed=7),
            lambda r: r['request_settings'].update(messages='private-value')]
        for mutate in changes:
            with self.subTest(mutation=changes.index(mutate)):
                row = deepcopy(original); mutate(row)
                with self.assertRaises(ValueError) as error: models.inspect_request_settings(row, 8192, True)
                self.assertNotIn('private-value', str(error.exception))

    def test_legacy_provider_source_never_acquires_wire_settings_proof(self):
        old = b'def dispatch(): pass\n'
        new = (ROOT/'tests/tools/generation_provider.py').read_bytes()
        self.assertFalse(models.records_wire_settings(old)); self.assertTrue(models.records_wire_settings(new))
        self.assertIsNone(models.inspect_request_settings({}, 8192, False))
        with self.assertRaisesRegex(ValueError, 'source implementation'):
            models.inspect_request_settings({'request_settings': request_settings({'max_tokens': 8192})}, 8192, False)

    def test_plan_binds_profile_settings_and_does_not_invent_provider_revision(self):
        plan = build_plan(); model = plan['model']
        raw = (ROOT/'integrations/opencode/profiles'/str(model['profile']+'.json')).read_bytes()
        self.assertEqual(model['profile_sha256'], digest(raw))
        self.assertEqual(model['configuration'], models.configuration(json.loads(raw)))
        self.assertIsNone(model['configuration']['provider_revision'])
        self.assertEqual(model['configuration']['credential'], {'environment': 'OPENCODE_API_KEY'})

    def test_both_environment_reference_formats_are_explicit_and_never_fall_back(self):
        for key in ('${SELECTED_API_KEY}', '{env:SELECTED_API_KEY}'):
            config = profile(key=key); plan = planned(config)
            env = {'SELECTED_BASE_URL': 'https://selected.invalid/v1', 'SELECTED_API_KEY': 'selected-secret',
                'OPENCODE_BASE_URL': 'https://wrong.invalid/v1', 'OPENCODE_API_KEY': 'wrong-secret'}
            self.assertEqual(models.resolve(config, plan, env), ('https://selected.invalid/v1', 'selected-secret'))
            del env['SELECTED_API_KEY']
            with self.assertRaisesRegex(ValueError, 'SELECTED_API_KEY') as error: models.resolve(config, plan, env)
            self.assertNotIn('wrong-secret', str(error.exception))
            self.assertNotIn('selected-secret', json.dumps(plan))

    def test_literal_endpoint_does_not_use_unrelated_endpoint_variable(self):
        config = profile(url='https://selected.invalid/v1/')
        self.assertEqual(models.resolve(config, planned(config), {'SELECTED_API_KEY': 'chosen',
            'OPENCODE_BASE_URL': 'https://wrong.invalid/v1'}), ('https://selected.invalid/v1', 'chosen'))

    def test_ollama_placeholder_cannot_send_an_ambient_cloud_credential(self):
        config = profile(provider='ollama', key='ollama', url='${OLLAMA_BASE_URL}')
        self.assertEqual(models.resolve(config, planned(config), {'OLLAMA_BASE_URL': 'http://localhost:11434/v1',
            'OPENCODE_API_KEY': 'cloud-secret'}), ('http://localhost:11434/v1', 'ollama'))

    def test_profile_model_settings_and_transport_drift_are_rejected(self):
        original = profile(); plan = planned(original)
        for mutate in (lambda c: c['provider']['selected']['models']['model-v1']['options'].update(temperature=1),
                       lambda c: c['provider']['selected']['options'].update(apiKey='${ANOTHER_API_KEY}'),
                       lambda c: c['provider']['selected'].update(npm='another-sdk'),
                       lambda c: c.update(model='selected/other-model'),
                       lambda c: c['provider'].update(other=deepcopy(c['provider']['selected']))):
            config = deepcopy(original); mutate(config)
            with self.assertRaises(ValueError): models.resolve(config, plan, {})

    def test_credentials_and_unsafe_endpoint_values_are_not_printed(self):
        for key in ('private-secret', '${KEY:-private-secret}'):
            with self.assertRaises(ValueError) as error: models.configuration(profile(key=key))
            self.assertNotIn('private-secret', str(error.exception))
        config = profile()
        for url in ('https://user:private-secret@host/v1', 'https://host/v1?key=private-secret',
                    'https://host/v1#private-secret', 'https://bad host/v1', 'https://[private-secret',
                    'https://host:private-secret/v1', 'https://host:65536/v1', 'https://host:0/v1',
                    'file:///private-secret'):
            with self.assertRaises(ValueError) as error:
                models.resolve(config, planned(config), {'SELECTED_BASE_URL': url, 'SELECTED_API_KEY': 'private-secret'})
            self.assertNotIn('private-secret', str(error.exception))
        with self.assertRaisesRegex(ValueError, 'credential') as error:
            models.resolve(config, planned(config), {'SELECTED_BASE_URL': 'https://host/v1',
                'SELECTED_API_KEY': 'private-secret\ninvalid'})
        self.assertNotIn('private-secret', str(error.exception))

    def local_plan(self, root, config):
        plan = build_plan(); name = 'integrations/opencode/profiles/offline-selected.json'
        path = root/name; path.parent.mkdir(parents=True); raw = canonical(config); path.write_bytes(raw)
        plan['source_sha256'][name] = digest(raw)
        plan['model'].update(requested=config['model'], profile='offline-selected', profile_sha256=digest(raw),
            expected_response='observed-model-v1', configuration=models.configuration(config))
        return plan

    def test_missing_selected_credential_blocks_both_entrypoints_before_commands(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); checkout = root/'checkout'; plan = self.local_plan(checkout, profile())
            with patch.object(executor, 'ROOT', checkout), patch.dict(os.environ, {
                    'SELECTED_BASE_URL': 'http://localhost:1234/v1', 'OPENCODE_API_KEY': 'wrong-secret'}, clear=True), \
                    patch.object(executor, 'command', side_effect=AssertionError('Must not start')):
                with self.assertRaisesRegex(executor.Blocked, 'SELECTED_API_KEY'):
                    executor.execute(plan, root/'single', root/'state')
                with self.assertRaisesRegex(executor.Blocked, 'SELECTED_API_KEY'):
                    review.run_review({'common': plan}, root/'review', root/'state')
            self.assertFalse((root/'single').exists()); self.assertFalse((root/'review').exists())
            self.assertFalse((root/'state/active.json').exists())

    def test_changed_profile_bytes_are_rejected_before_resolving_credentials(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = self.local_plan(root, profile())
            path = root/'integrations/opencode/profiles/offline-selected.json'; path.write_bytes(path.read_bytes()+b'\n')
            with patch.object(executor, 'ROOT', root), self.assertRaisesRegex(executor.Blocked, 'profile changed'):
                executor.provider_credentials(plan)

    def test_selected_provider_receives_only_its_credential_through_actual_gate(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.headers.get('Authorization'), body))
                data = canonical({'model': 'observed-model-v1', 'choices': [{'index': 0, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7}})
                self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); config = profile(); plan = self.local_plan(root, config)
            attempt = root/'attempt'; attempt.mkdir(); project = attempt/'project'; project.mkdir()
            original_command = executor.command
            def client(argv, directory, name, seconds, env=None, abort=None, cwd=None):
                if name == 'project-init': return original_command(argv, directory, name, seconds)
                self.assertNotIn('SELECTED_API_KEY', env); self.assertNotIn('OPENCODE_API_KEY', env)
                rendered = json.loads(env['OPENCODE_CONFIG_CONTENT']); route = rendered['provider']['selected']['options']
                self.assertEqual(route['apiKey'], '{env:PYASC_GATE_TOKEN}')
                request = urllib.request.Request(route['baseURL']+'/chat/completions', data=canonical({'model': 'model-v1'}),
                    headers={'Authorization': 'Bearer '+env['PYASC_GATE_TOKEN']})
                with urllib.request.urlopen(request) as response: response.read()
                return {'exit_code': 0}
            with patch.object(executor, 'ROOT', root), patch.dict(os.environ, {
                    'SELECTED_BASE_URL': 'http://127.0.0.1:'+str(server.server_port), 'SELECTED_API_KEY': 'chosen-secret',
                    'OPENCODE_BASE_URL': 'http://127.0.0.1:1', 'OPENCODE_API_KEY': 'wrong-secret'}), \
                    patch.object(executor, 'command', side_effect=client):
                result, usage = executor.run_session(attempt, plan, project, config, 'offline provider routing fixture')
            self.assertEqual(result['exit_code'], 0); self.assertTrue(usage['usage_complete'])
            self.assertEqual(len(received), 1); self.assertEqual(received[0][0], 'Bearer chosen-secret')
            self.assertEqual(received[0][1]['model'], 'model-v1')
            for path in (attempt/'usage.json', attempt/'model-config.json'):
                self.assertNotIn('chosen-secret', path.read_text()); self.assertNotIn('wrong-secret', path.read_text())


if __name__ == '__main__': unittest.main()
