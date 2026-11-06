"""Single-attempt OpenAI-compatible budget gate for OpenCode.

Reserve before dispatch, forbid retries after an uncertain response, and check
usage before allowing another request. OpenCode's agent step limit alone is not
such a gate (it can ask for a final summary). Raw response capture is delegated to a private evidence callback; public
records contain hashes and safe diagnostics, never credentials. This is a budget boundary, not an OS/network sandbox.
"""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import re
import threading
import time
import urllib.request
import urllib.error
from urllib.parse import urlsplit

MAX_BODY = 8 * 1024 * 1024
MAX_RESPONSE = 32 * 1024 * 1024


class BudgetError(ValueError):
    pass


def known_budget_exhaustion(usage, limits):
    """A measured cap is a failure outcome; uncertain delivery is never one."""
    rows = usage.get('requests', [])
    if not usage.get('usage_complete') or not rows: return None
    if any(r.get('status') not in ('complete', 'rejected') or 'usage' not in r for r in rows): return None
    if usage.get('error') == 'Request budget exhausted' and len(rows) == limits['rounds_per_attempt'] and all(r['status'] == 'complete' for r in rows):
        return 'request-limit'
    if (usage.get('error') == 'BudgetError: Model completion truncated at output-token cap'
            and all(r['status'] == 'complete' for r in rows[:-1]) and rows[-1]['status'] == 'rejected'
            and rows[-1]['usage'].get('finish_reason') == 'length'):
        return 'output-limit'
    return None


def response_usage(raw, streaming, model):
    if streaming:
        lines = [line[5:].strip() for line in raw.decode().splitlines() if line.startswith('data:')]
        if not lines or lines[-1] != '[DONE]': raise BudgetError('Incomplete streaming response')
        parts = [json.loads(s) for s in lines[:-1]]
    else:
        parts = [json.loads(raw)]
    models = {p.get('model') for p in parts if p.get('model')}
    # Provider aliases/revisions are not silently treated as the selected model.
    if models != {model}: raise BudgetError('Missing or different observed model')
    endings = [c for p in parts for c in p.get('choices', []) if c.get('finish_reason')]
    if (len(endings) != 1 or endings[0].get('index') != 0
            or endings[0]['finish_reason'] not in ('stop', 'tool_calls', 'length')):
        raise BudgetError('Incomplete/truncated model completion')
    usages = [p['usage'] for p in parts if p.get('usage')]
    if len(usages) != 1: raise BudgetError('Missing/ambiguous provider usage')
    usage = usages[0]
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        if type(usage.get(key)) is not int or usage[key] < 0: raise BudgetError('Invalid token usage')
    if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
        raise BudgetError('Inconsistent token usage')
    # Chat Completions completion_tokens includes reasoning; do not add it twice.
    reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
    if type(reasoning) is not int or not 0 <= reasoning <= usage['completion_tokens']:
        raise BudgetError('Invalid reasoning usage')
    return dict(usage, observed_model=model, finish_reason=endings[0]['finish_reason'])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BudgetError('Provider redirect rejected')


class BudgetGate:
    def __init__(self, base_url, api_key, model, limits, save, deadline, capture_response=None, response_model=None,
                 observe_request=None):
        url = urlsplit(base_url)
        if url.scheme not in ('https', 'http') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise BudgetError('Invalid provider endpoint')
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.key, self.model, self.limits, self.save, self.deadline = api_key, model, limits, save, deadline
        self.response_model = response_model or model
        self.capture_response = capture_response
        self.observe_request = observe_request
        self.records = []
        self.error = None
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)

    def snapshot(self):
        return {'requests': list(self.records), 'error': self.error,
                'reserved_output_tokens': len(self.records) * self.limits['output_tokens_per_response'],
                'observed_output_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in self.records),
                'provider_cost': None, 'usage_complete': bool(self.records) and all(r['status'] in ('complete', 'rejected') and 'usage' in r for r in self.records)}

    def dispatch(self, body):
        with self.lock:
            if self.error: raise BudgetError('Earlier request is unresolved; no retry')
            if len(self.records) >= self.limits['rounds_per_attempt']:
                self.error = 'Request budget exhausted'; self.save(self.snapshot())
                raise BudgetError(self.error)
            if time.monotonic() >= self.deadline: raise BudgetError('Generation deadline exhausted')
            if body.get('model') != self.model or body.get('n', 1) != 1:
                raise BudgetError('Unapproved model or response count')
            cap = self.limits['output_tokens_per_response']
            body.pop('max_completion_tokens', None)
            body['max_tokens'] = cap
            if body.get('stream'):
                body['stream_options'] = {'include_usage': True}
            payload = json.dumps(body).encode()
            record = {'status': 'dispatched', 'request_sha256': hashlib.sha256(payload).hexdigest(),
                      'model': self.model, 'max_tokens': cap, 'stream': bool(body.get('stream'))}
            self.records.append(record)
            self.save(self.snapshot())  # durable reservation BEFORE network input
            try:
                record['dispatch_started'] = False
                if self.observe_request:
                    record['audit'] = self.observe_request(len(self.records), body, payload)
                    self.save(self.snapshot())
                request = urllib.request.Request(self.url, data=payload,
                    headers={'Authorization': 'Bearer '+self.key, 'Content-Type': 'application/json'})
                record['dispatch_started'] = True
                self.save(self.snapshot())
                with urllib.request.build_opener(NoRedirect).open(request, timeout=max(.01, self.deadline-time.monotonic())) as response:
                    record['http_status'] = response.status
                    record['response_ids'] = {name: value for name in ('x-request-id', 'x-requestid', 'x-tt-logid')
                        if (value := response.headers.get(name)) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}', value)}
                    raw = response.read(MAX_RESPONSE + 1)
                    content_type = response.headers.get('Content-Type', 'application/json')
                record['response_sha256'] = hashlib.sha256(raw).hexdigest()
                record['response_bytes'] = len(raw)
                if self.capture_response:
                    self.capture_response(len(self.records), raw)
                if len(raw) > MAX_RESPONSE: raise BudgetError('Provider response too large')
                usage = response_usage(raw, bool(body.get('stream')), self.response_model)
                record['usage'] = usage
                if usage['completion_tokens'] > cap: raise BudgetError('Provider exceeded output-token cap')
                if usage['finish_reason'] == 'length': raise BudgetError('Model completion truncated at output-token cap')
                record['status'] = 'complete'
                self.save(self.snapshot())
                return raw, content_type
            except Exception as exc:
                record['status'] = 'rejected' if 'usage' in record else 'unknown' if record['dispatch_started'] else 'not-sent'
                # Only our fixed validation messages are safe to persist. External
                # exception strings can contain credentials, URLs or response text.
                if isinstance(exc, urllib.error.HTTPError):
                    record['http_status'] = exc.code
                record['validation_error'] = str(exc) if isinstance(exc, BudgetError) else type(exc).__name__
                self.error = type(exc).__name__ + ': ' + record['validation_error']
                self.save(self.snapshot())
                raise BudgetError(self.error) from None


@contextmanager
def budget_server(gate):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            try:
                if self.path != '/v1/chat/completions' or self.headers.get('Authorization') != 'Bearer '+gate.token:
                    raise BudgetError('Unexpected endpoint or client')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BODY: raise BudgetError('Invalid request size')
                raw, kind = gate.dispatch(json.loads(self.rfile.read(size)))
                self.send_response(200); self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
            except Exception:
                # A retryable 429 makes OpenCode sleep/retry after a permanent
                # rejection. The executor also watches gate.error and cancels it.
                raw = b'{"error":{"type":"budget_gate_rejected","message":"Generation budget gate rejected request; inspect usage.json"}}'
                self.send_response(400); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw))); self.end_headers()
                try: self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield 'http://127.0.0.1:' + str(server.server_port) + '/v1'
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
