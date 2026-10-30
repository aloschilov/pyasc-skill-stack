"""Conservative observation of supported host numerical checks, not stdout.

Only direct assert allclose(...), if not allclose(...): ...; raise, and native
assert_allclose/assert_close exceptions are attributable. Other check styles
remain executable but cannot earn the numerical level. This is a test harness,
not a sandbox against deliberately forged checks or reports.
"""
from __future__ import annotations

import ast
import functools
import importlib
import inspect
from pathlib import Path


class NumericalChecks:
    def __init__(self, kernel: Path, counter: dict):
        self.kernel = kernel.resolve()
        self.counter = counter
        self.events = []
        self.patches = []
        self.exceptions = []
        self.tree = ast.parse(kernel.read_text())

    def install(self):
        for module, name, assertion in (
            ('torch', 'allclose', False), ('numpy', 'allclose', False),
            ('numpy.testing', 'assert_allclose', True),
            ('torch.testing', 'assert_close', True),
        ):
            try:
                owner = importlib.import_module(module)
            except ImportError:
                continue
            original = getattr(owner, name)
            def wrap(original=original, api=module+'.'+name, assertion=assertion):
                @functools.wraps(original)
                def observed(*args, **kwargs):
                    frame = inspect.currentframe().f_back
                    local = Path(frame.f_code.co_filename).resolve() == self.kernel
                    event = {'api': api, 'line': frame.f_lineno,
                             'after_launch': self.counter['calls'] > 0}
                    try:
                        value = original(*args, **kwargs)
                    except AssertionError as exc:
                        if local and assertion:
                            event['passed'] = False
                            self.events.append(event)
                            self.exceptions.append((exc, event))
                        raise
                    if local:
                        event['passed'] = True if assertion else bool(value)
                        self.events.append(event)
                    return value
                return observed
            self.patches.append((owner, name, original))
            setattr(owner, name, wrap())

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)

    def failure(self, exc):
        for original, event in self.exceptions:
            if original is exc and event['after_launch']:
                return event
        if not isinstance(exc, (AssertionError, ValueError)) or not self.events:
            return None
        event = self.events[-1]
        if event['passed'] or not event['after_launch']:
            return None
        tb = exc.__traceback__
        while tb and tb.tb_next:
            tb = tb.tb_next
        if not tb or Path(tb.tb_frame.f_code.co_filename).resolve() != self.kernel:
            return None
        def direct_call(node):
            # The wrapper already established the API, including imported
            # aliases. Keep the AST link to the rejecting assertion/branch:
            # a false comparison followed by an unrelated raise is insufficient.
            return (isinstance(node, ast.Call)
                    and (isinstance(node.func, ast.Name)
                         or (isinstance(node.func, ast.Attribute) and node.func.attr == 'allclose'))
                    and node.lineno == event['line'])
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Assert) and direct_call(node.test)
                    and node.lineno <= tb.tb_lineno <= node.end_lineno):
                return event
            if (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                    and isinstance(node.test.op, ast.Not) and direct_call(node.test.operand)
                    and isinstance(node.body[-1], ast.Raise)
                    and node.body[-1].lineno <= tb.tb_lineno <= node.body[-1].end_lineno):
                return event
        return None


def proves_numerics(real: dict, control: dict, code: int) -> bool:
    failure = control.get('numerical_failure')
    checks = real.get('numerical_checks', [])
    return bool(
        code == 1 and control.get('status') == 'fail' and control.get('exit_code') == 1
        and control.get('mock_launch') is True and control.get('launcher_calls', 0) > 0
        and failure and failure.get('after_launch') and failure.get('passed') is False
        and checks and all(e.get('passed') for e in checks)
        and any(e.get('after_launch') and e.get('api') == failure.get('api')
                and e.get('line') == failure.get('line') for e in checks)
    )
