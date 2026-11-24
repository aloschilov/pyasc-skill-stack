"""Offline HTTP evidence tests: mocked curl, no submission service or credentials."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('receipt_smoke', ROOT/'integrations/cannbench/smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
from evalqueue import EvalQueue, RemoteError, SubmissionRejected


class UploadReceipts(unittest.TestCase):
    def queue(self):
        queue = EvalQueue.__new__(EvalQueue)
        queue._client = Mock(token='private-bearer', base_url='https://example.invalid')
        queue._client.list_jobs.return_value = {'jobs': []}
        queue._client.get_credits.return_value = {'credits': {'unlimited': True}}
        queue.target_hardware = '950pr'
        queue.benchmark_slug = 'official-tasks'
        queue.upload_timeout = 30
        return queue

    def completed(self, body, status='201', rc=0, stderr=b''):
        return SimpleNamespace(returncode=rc, stdout=body + b'\n' + status.encode() + b' 12345 1.25', stderr=stderr)

    def prepared(self, root):
        run, ledger = root/'run', root/'ledger'
        run.mkdir(); ledger.mkdir()
        (run/'submission.zip').write_bytes(b'archive')
        manifest = {'files': {'submission.zip': smoke.sha(run/'submission.zip')},
                    'catalog': {'benchmark_slug': 'official-tasks', 'benchmark_version': '2.0.0',
                                'operators': [{'operator': 'SwiGlu', 'function_name': 'swi_glu'}]},
                    'target_hardware': '950pr', 'aggregation_token': 'private-aggregation'}
        smoke.save(run/'manifest.json', manifest)
        smoke.save(run/'task.json', {})
        return run, ledger, manifest

    def test_success_preserves_original_bytes_and_return_api(self):
        body = '{\r\n"job_id":"job_test","account":"владелец"\r\n}'.encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'response.json'
            queue = self.queue()
            with patch('evalqueue.subprocess.run', return_value=self.completed(body)) as post:
                returned = queue._submit_streaming('/unused', ['swi_glu'], 'tag', receipt_path=path)
            receipt = json.loads(path.read_text())
            self.assertEqual(returned, json.loads(body))
            self.assertEqual(json.loads(receipt['body_redacted']), json.loads(body))
            self.assertEqual(receipt['body_sha256'], hashlib.sha256(body).hexdigest())
            self.assertEqual(receipt['body_bytes'], len(body))
            self.assertEqual((receipt['http_status'], receipt['multipart_bytes'], receipt['duration_seconds']),
                             ('201', '12345', '1.25'))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn('text', post.call_args.kwargs)

    def test_receipt_precedes_json_and_id_validation_and_no_repeat(self):
        for body, error in [(b'not JSON\r\n\xff', RemoteError),
                            (b'{"job_id":"job_\xff"}', RemoteError),
                            (b'{"account_id":"owner"}', ValueError)]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                run, ledger, manifest = self.prepared(Path(tmp))
                queue = self.queue()
                with (patch.object(smoke, 'task_snapshot', return_value=(manifest['catalog'], {})),
                      patch('evalqueue.subprocess.run', return_value=self.completed(body)) as post):
                    with self.assertRaises(error):
                        smoke.submit_once(queue, run, ledger)
                    state = json.loads((run/'state.json').read_text())
                    path = Path(state['http_response_receipt'])
                    before = path.read_bytes()
                    receipt = json.loads(before)
                    self.assertEqual(receipt['body_sha256'], hashlib.sha256(body).hexdigest())
                    readable = body.decode('utf-8', errors='replace')
                    try:
                        decoded = json.loads(readable)
                    except ValueError:
                        self.assertEqual(receipt['body_redacted'], readable)
                    else:
                        self.assertEqual(json.loads(receipt['body_redacted']), decoded)
                    self.assertIn(state['phase'], ('ambiguous', 'upload-intent'))
                    self.assertEqual(smoke.submit_once(queue, run, ledger), state)
                    self.assertEqual(path.read_bytes(), before)
                    post.assert_called_once()

    def test_missing_or_malformed_trailer_preserves_stdout_as_unknown(self):
        for stdout in (b'{"job_id":"job_test"}', b'{"job_id":"job_test"}\n',
                       b'{"job_id":"job_test"}\n200 12345', b'{"job_id":"job_test"}\n200 bogus 1.25',
                       b'{"job_id":"job_test",\n"detail":"partial response"}'):
            with self.subTest(stdout=stdout), tempfile.TemporaryDirectory() as tmp:
                run, ledger, manifest = self.prepared(Path(tmp))
                queue = self.queue()
                completed = SimpleNamespace(returncode=0, stdout=stdout, stderr=b'')
                with (patch.object(smoke, 'task_snapshot', return_value=(manifest['catalog'], {})),
                      patch('evalqueue.subprocess.run', return_value=completed) as post):
                    with self.assertRaisesRegex(RemoteError, 'delivery is unknown'):
                        smoke.submit_once(queue, run, ledger)
                    state = json.loads((run/'state.json').read_text())
                    self.assertEqual(state['phase'], 'ambiguous')
                    self.assertIsNone(state['job_id'])
                    receipt = json.loads(Path(state['http_response_receipt']).read_text())
                    self.assertEqual(receipt['transfer_metadata'], 'missing-or-malformed')
                    self.assertEqual(receipt['body_source'], 'curl_stdout')
                    for field in ('http_status', 'multipart_bytes', 'duration_seconds'):
                        self.assertIsNone(receipt[field])
                    self.assertEqual(receipt['body_bytes'], len(stdout))
                    self.assertEqual(receipt['body_sha256'], hashlib.sha256(stdout).hexdigest())
                    try:
                        decoded = json.loads(stdout)
                    except ValueError:
                        self.assertEqual(receipt['body_redacted'], stdout.decode())
                    else:
                        self.assertEqual(json.loads(receipt['body_redacted']), decoded)
                    self.assertEqual(smoke.submit_once(queue, run, ledger), state)
                    post.assert_called_once()

    def test_credentials_redacted_from_success_error_and_stdout(self):
        secret = 'agg/with"quote'
        body = json.dumps({'job_id': 'job_test', 'echo': 'private-bearer ' + secret,
                           'aggregation_token': secret, 'access_token': 'unknown-response-secret',
                           'url': 'https://example.invalid/?signature=signed-secret'}).encode()
        for rc, status in [(0, '200'), (22, '403')]:
            with self.subTest(rc=rc), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'response.json'
                queue = self.queue()
                output = io.StringIO()
                with (patch('evalqueue.subprocess.run', return_value=self.completed(body, status, rc,
                            b'Authorization: Bearer private-bearer')),
                      contextlib.redirect_stdout(output), contextlib.redirect_stderr(output)):
                    if rc:
                        with self.assertRaises(SubmissionRejected) as caught:
                            queue._submit_streaming('/unused', [], 'tag', secret, receipt_path=path)
                        output.write(str(caught.exception))
                    else:
                        queue._submit_streaming('/unused', [], 'tag', secret, receipt_path=path)
                evidence = path.read_text() + output.getvalue()
                for value in ('private-bearer', 'agg/with', 'unknown-response-secret', 'signed-secret'):
                    self.assertNotIn(value, evidence)
                self.assertEqual(json.loads(path.read_text())['body_sha256'], hashlib.sha256(body).hexdigest())
                self.assertIn('job_test', evidence)

    def test_alternate_json_escapes_redacted_in_receipts_and_errors(self):
        secret = 'agg/with"quote'
        bodies = [
            rb'{"job_id":"job_test","nested":[{"private\u002dbearer":"invalid private\u002dbearer"}],"detail":"agg\/with\"quote"}',
            rb'{"detail":"invalid private\u002dbearer agg\/with\"quote',
        ]
        for body in bodies:
            for rc, status in [(0, '200'), (22, '403')]:
                with self.subTest(body=body, rc=rc), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/'response.json'
                    with patch('evalqueue.subprocess.run', return_value=self.completed(body, status, rc, body)):
                        error_text = ''
                        if rc or body == bodies[1]:
                            expected = SubmissionRejected if rc else RemoteError
                            with self.assertRaises(expected) as caught:
                                self.queue()._submit_streaming('/unused', [], 'tag', secret, receipt_path=path)
                            error_text = str(caught.exception)
                        else:
                            self.queue()._submit_streaming('/unused', [], 'tag', secret, receipt_path=path)
                    receipt = json.loads(path.read_text())
                    self.assertEqual(receipt['body_sha256'], hashlib.sha256(body).hexdigest())
                    self.assertEqual(receipt['body_bytes'], len(body))
                    evidence = receipt['body_redacted'] + receipt['stderr_redacted'] + error_text
                    for value in ('private-bearer', r'private\u002dbearer', 'agg/with', r'agg\/with'):
                        self.assertNotIn(value, evidence)
                    if body == bodies[0]:
                        decoded = json.loads(receipt['body_redacted'])
                        self.assertEqual(decoded['nested'], [{'[REDACTED]': 'invalid [REDACTED]'}])
                        self.assertEqual(decoded['detail'], '[REDACTED]')
                        if rc:
                            decoded_error = json.loads(error_text.split(': ', 1)[1])
                            self.assertEqual(decoded_error, decoded)

    def test_maintenance_retry_keeps_first_receipt_and_new_intent_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, ledger, manifest = self.prepared(Path(tmp))
            queue = self.queue()
            queue._client._get_json.return_value = {'maintenance': {'enabled': False}}
            responses = [self.completed(b'{"detail":"submission is paused during maintenance"}', '403', 22),
                         self.completed(b'{"job_id":"job_test","submission_id":"sub_test"}')]
            with (patch.object(smoke, 'task_snapshot', return_value=(manifest['catalog'], {})),
                  patch('evalqueue.subprocess.run', side_effect=responses) as post):
                with self.assertRaises(SubmissionRejected):
                    smoke.submit_once(queue, run, ledger)
                rejected = json.loads((run/'state.json').read_text())
                original_path = Path(rejected['http_response_receipt'])
                original = original_path.read_bytes()
                accepted = smoke.submit_once(queue, run, ledger, retry_maintenance=True)
                self.assertEqual(accepted['previous_intents'], [rejected])
                self.assertNotEqual(accepted['http_response_receipt'], str(original_path))
                self.assertEqual(original_path.read_bytes(), original)
                self.assertTrue(Path(accepted['http_response_receipt']).is_file())
                self.assertEqual(smoke.submit_once(queue, run, ledger), accepted)
                self.assertEqual(post.call_count, 2)

    def test_existing_receipt_or_repository_path_prevents_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'response.json'
            path.write_text('historical')
            with patch('evalqueue.subprocess.run') as post:
                with self.assertRaises(FileExistsError):
                    self.queue()._submit_streaming('/unused', [], 'tag', receipt_path=path)
                with self.assertRaises(ValueError):
                    self.queue()._submit_streaming('/unused', [], 'tag', receipt_path=ROOT/'receipt.json')
                post.assert_not_called()
            self.assertEqual(path.read_text(), 'historical')

    def test_receipt_write_failure_leaves_unknown_and_never_reposts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, ledger, manifest = self.prepared(Path(tmp))
            import evalqueue
            original_open, original_fsync = evalqueue._open_response_receipt, os.fsync
            receipt_fd = None
            def open_receipt(path):
                nonlocal receipt_fd
                file = original_open(path)
                receipt_fd = file.fileno()
                return file
            def sync(fd):
                nonlocal receipt_fd
                if fd == receipt_fd:
                    receipt_fd = None
                    raise OSError('disk failure')
                return original_fsync(fd)
            with (patch.object(smoke, 'task_snapshot', return_value=(manifest['catalog'], {})),
                  patch('evalqueue.subprocess.run', return_value=self.completed(b'{"job_id":"job_test"}')) as post,
                  patch('evalqueue._open_response_receipt', side_effect=open_receipt),
                  patch('evalqueue.os.fsync', side_effect=sync)):
                # Keep ledger writes working; only receipt fsync is made to fail.
                with self.assertRaises(OSError):
                    smoke.submit_once(self.queue(), run, ledger)
            state = json.loads((run/'state.json').read_text())
            self.assertEqual(state['phase'], 'ambiguous')
            with patch('evalqueue.subprocess.run') as next_post:
                self.assertEqual(smoke.submit_once(self.queue(), run, ledger), state)
                next_post.assert_not_called()
            post.assert_called_once()

    def test_completed_failures_keep_receipts_and_error_classification(self):
        for status, rc, error in [('400', 22, SubmissionRejected), ('500', 22, RemoteError),
                                  ('408', 22, RemoteError), ('000', 28, RemoteError)]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'response.json'
                with patch('evalqueue.subprocess.run', return_value=self.completed(b'', status, rc, b'Transfer failed')):
                    with self.assertRaises(error) as caught:
                        self.queue()._submit_streaming('/unused', [], 'tag', receipt_path=path)
                self.assertIs(type(caught.exception), error)
                self.assertIn('12345 multipart bytes in 1.25s', str(caught.exception))
                receipt = json.loads(path.read_text())
                self.assertEqual((receipt['http_status'], receipt['curl_returncode']), (status, rc))
                self.assertEqual(receipt['stderr_redacted'], 'Transfer failed')

    def test_process_timeout_does_not_expose_command_credentials(self):
        timeout = subprocess.TimeoutExpired(['curl', 'aggregation_token=private-aggregation'], 60)
        with patch('evalqueue.subprocess.run', side_effect=timeout):
            with self.assertRaisesRegex(RemoteError, 'delivery is unknown') as caught:
                self.queue()._submit_streaming('/unused', [], 'tag', 'private-aggregation')
        self.assertNotIn('private-aggregation', str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)


if __name__ == '__main__':
    unittest.main()
