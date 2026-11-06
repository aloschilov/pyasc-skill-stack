"""Run one unchanged official pytest case with call-through numerical auditing.

The companion control suppresses device launches, not the official assertions.
This is evidence for trusted pinned fixtures, not a sandbox for hostile tests.
"""
import argparse
import functools
import importlib.metadata
import inspect
import os
from pathlib import Path
import sys
import time

from target_inventory import ROOT, PREFIX, normalize, read, require, save, sources, digest


def proves_case(real, control):
    """A pytest pass needs a real launch and a matching rejecting assertion."""
    for value, mode, exit_code, call_outcome in (
            (real, 'numerical', 0, 'passed'), (control, 'suppressed-launch', 1, 'failed')):
        require(value.get('kind') == 'target-case' and value.get('schema_version') == 1
            and value.get('status') == 'completed' and value.get('mode') == mode
            and value.get('exit_code') == exit_code and value.get('inputs_unchanged') is True,
            'Incomplete or unsuccessful case receipt')
        require(value.get('backend') == 'Model' and value.get('collected') == [PREFIX+value['nodeid']],
                'Backend or selected case mismatch')
        phases = value.get('phases', {})
        require(set(phases) == {'setup', 'call', 'teardown'}, 'Missing pytest phase')
        for phase, outcome in (('setup', 'passed'), ('call', call_outcome), ('teardown', 'passed')):
            require(phases[phase]['outcome'] == outcome and phases[phase].get('xfail') is None,
                    'Skipped, expected-failure or unsuccessful pytest phase')
        require(type(value.get('launcher_calls')) is int and value['launcher_calls'] > 0, 'No launch observed')
    for key in ('nodeid', 'backend', 'platform', 'runtime', 'source_sha256'):
        require(real[key] == control[key], 'Control provenance mismatch: '+key)
    require(real['completed_launches'] == real['launcher_calls'] and control['completed_launches'] == 0,
            'Missing real launch or unsuppressed control')
    checks = real.get('numerical_checks', [])
    require(checks and all(c.get('passed') is True and c.get('after_completed_launch') is True for c in checks),
            'Missing successful official assertions after execution')
    failure = control.get('numerical_failure')
    require(isinstance(failure, dict) and failure.get('passed') is False and failure.get('after_launch') is True
        and failure.get('after_completed_launch') is False and failure in control.get('numerical_checks', []),
        'Control did not fail at an attributable numerical assertion')
    require(any(all(failure[k] == c[k] for k in ('source', 'line', 'api', 'tolerances')) for c in checks),
            'Control assertion differs from real verification')
    return True


class NumericalAudit:
    def __init__(self, root, allowed, suppress):
        self.root, self.allowed, self.suppress = root.resolve(), set(allowed), suppress
        self.launches = self.completed_launches = 0
        self.events, self.failures = [], []

    def launcher(self, original):
        @functools.wraps(original)
        def observed(*args, **kwargs):
            self.launches += 1
            if self.suppress: return None
            value = original(*args, **kwargs)
            self.completed_launches += 1
            return value
        return observed

    def assertion(self, original):
        @functools.wraps(original)
        def observed(*args, **kwargs):
            caller = inspect.currentframe().f_back
            path = Path(caller.f_code.co_filename).resolve()
            try: relative = path.relative_to(self.root).as_posix()
            except ValueError: relative = None
            local = relative in self.allowed
            event = {'source': relative, 'line': caller.f_lineno, 'api': 'torch.testing.assert_close',
                'after_launch': self.launches > 0, 'after_completed_launch': self.completed_launches > 0,
                'tolerances': {k: kwargs.get(k) for k in ('atol', 'rtol', 'equal_nan', 'check_dtype', 'check_device')},
                'defaults': 'Omitted values use the recorded torch runtime defaults.'}
            try:
                value = original(*args, **kwargs)
            except AssertionError as exc:
                if local:
                    event['passed'] = False; self.events.append(event); self.failures.append((exc, event))
                raise
            if local:
                event['passed'] = True; self.events.append(event)
            return value
        return observed

    def failure(self, exc):
        return next((event for original, event in self.failures if original is exc and event['after_launch']), None)


def execute(nodeid, output, suppress=False, root=ROOT):
    import pytest
    import torch
    from asc.runtime.launcher import Launcher
    from asc.runtime import launcher
    nodeid = normalize([nodeid])[0]
    require(not os.environ.get('PYASC_DRY_RUN'), 'Dry-run environment cannot qualify numerics')
    require(os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1', 'Disable ambient pytest plugins')
    require(not Path(output).exists(), 'Do not overwrite a case receipt')
    inputs = sources(root)
    require(PREFIX+nodeid.split('::', 1)[0] in inputs, 'Unknown case module')
    audit = NumericalAudit(root, inputs, suppress)
    result = {'schema_version': 1, 'kind': 'target-case', 'nodeid': nodeid, 'backend': 'Model',
        'platform': read(root/'reference/images/ENV.json')['image']['simulator'],
        'mode': 'suppressed-launch' if suppress else 'numerical', 'status': 'running',
        'source_sha256': inputs, 'phases': {}, 'collected': [], 'numerical_failure': None,
        'runtime': {'pyasc': importlib.metadata.version('pyasc-dev'), 'torch': torch.__version__,
            'pytest': pytest.__version__, 'python': sys.version,
            'launcher_sha256': digest(Path(launcher.__file__).read_bytes()),
            'torch_assertion_sha256': digest(Path(inspect.getfile(torch.testing.assert_close)).read_bytes())}}
    save(output, result); started = time.monotonic()

    class Plugin:
        def pytest_collection_finish(self, session):
            result['collected'] = [item.nodeid for item in session.items]
            require(result['collected'] == [PREFIX+nodeid], 'Not exactly the requested official case')

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            outcome = yield
            report = outcome.get_result()
            result['phases'][report.when] = {'outcome': report.outcome, 'seconds': report.duration,
                'xfail': getattr(report, 'wasxfail', None),
                'reason': str(report.longrepr)[-2000:] if report.failed or report.skipped else None}
            if call.excinfo:
                result['numerical_failure'] = audit.failure(call.excinfo.value)

    original_launch, original_assertion = Launcher.run, torch.testing.assert_close
    try:
        Launcher.run = audit.launcher(original_launch)
        torch.testing.assert_close = audit.assertion(original_assertion)
        result['exit_code'] = int(pytest.main([PREFIX+nodeid, '-q', '-p', 'no:cacheprovider',
            '--backend', 'Model', '--platform', result['platform'], '--torch-seed', '0'], plugins=[Plugin()]))
        result['status'] = 'completed'
    finally:
        Launcher.run, torch.testing.assert_close = original_launch, original_assertion
        result.update(seconds=time.monotonic()-started, launcher_calls=audit.launches,
            completed_launches=audit.completed_launches, numerical_checks=audit.events,
            inputs_unchanged=sources(root) == inputs)
        save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodeid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suppress-launch', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    return execute(args.nodeid, args.output, args.suppress_launch)['exit_code']


if __name__ == '__main__': sys.exit(main())
