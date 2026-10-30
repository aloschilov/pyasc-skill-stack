"""Remote worker: verify payload, identify hardware, run the unchanged host harness."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from device_identity import probe


def hashes(root, names):
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def run_host(argv, timeout, cwd):
    # One process group includes the host harness and both execution children.
    # A timeout or worker interruption must not leave an NPU job behind.
    process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = process.communicate(timeout=timeout)
        return process.returncode, out, err
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def execute(root, request):
    evidence = {'payload_sha256': request['sha256'], 'remote_dir': str(root)}
    report = {'file': request['kernel'], 'status': 'FAIL', 'verification_level': None,
              'detail': '', 'evidence': {'ssh': evidence}}
    if hashes(root, request['sha256']) != request['sha256']:
        report['detail'] = 'SSH payload hash mismatch before execution'
        return 1, report
    identity = probe()
    report['device_identity'] = identity
    if identity['status'] != 'ready':
        report.update(status='SKIP', detail=identity['reason'])
        return 2, report
    soc = identity['soc_version']
    if request.get('platform') and request['platform'] != soc:
        report['detail'] = f"Requested platform {request['platform']} differs from detected {soc}"
        return 1, report
    # Cooperative serialization for this harness; no other checkout is removed
    # or overwritten, and concurrent hardware invocations fail closed.
    with open('/tmp/pyasc-hardware-device-0.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report.update(status='SKIP', detail='NPU lane is busy (device 0 lock)')
            return 2, report
        argv = [sys.executable, str(root/'tests/tools/run_and_verify.py'), str(root/request['kernel']),
                '--runtime-backend', 'host', '--backend', 'NPU', '--platform', soc,
                '--mode', request['mode'], '--timeout', str(request['timeout']), '--json',
                '--jit-scalar-args', request['scalar_args']]
        if request['no_prove_numerics']:
            argv.append('--no-prove-numerics')
        evidence['argv'] = argv
        try:
            code, out, err = run_host(argv, 2 * request['timeout'] + 120, root/'project')
            evidence.update(exit_code=code, stdout=out, stderr=err)
            inner = json.loads(out)
            if (code not in (0, 1, 2) or not isinstance(inner, dict)
                    or inner.get('status') != {0:'PASS', 1:'FAIL', 2:'SKIP'}[code]
                    or (code == 0 and inner.get('verification_level') not in
                        ('compiled', 'executed', 'numerically_verified'))):
                raise ValueError('Invalid host report/exit contract')
            report = inner
            report['device_identity'] = identity
            report.setdefault('evidence', {})['ssh'] = evidence
        except (ValueError, subprocess.TimeoutExpired) as exc:
            code = 1
            report['detail'] = f'Remote execution failed: {exc}'
        evidence['inputs_unchanged'] = hashes(root, request['sha256']) == request['sha256']
        if not evidence['inputs_unchanged']:
            code = 1
            report.update(status='FAIL', verification_level=None, detail='SSH payload changed during execution')
        return code, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'worker signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        code, report = execute(root, json.loads((root/'request.json').read_text()))
    except BaseException as exc:
        code = 1
        report = {'status':'FAIL', 'verification_level':None, 'detail':f'{type(exc).__name__}: {exc}'}
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
