"""Container-only unchanged-candidate adapter qualification, always Model backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from generation_adapters import static_check
from numerical_checks import proves_numerics


def qualify(directory, seconds):
    directory = Path(directory).resolve()
    source = directory/'candidate.py'
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [source, directory/'probe.py', directory/'compile_probe.py', directory/'adapter.json']}
    contract = json.loads((directory/'adapter.json').read_text())
    result = {'status': 'failed', 'backend': 'Model', 'phases': {}, 'input_sha256': originals}
    deadline = time.monotonic() + seconds
    try:
        result['static'] = static_check(source.read_text(), contract)
        for phase, filename, suppress in [('jit', 'compile_probe.py', True),
                                          ('execution', 'probe.py', False), ('control', 'probe.py', True)]:
            report = directory/(phase+'.json')
            command = [sys.executable, str(Path(__file__).with_name('execute_kernel.py')),
                       str(directory/filename), '--backend', 'Model', '--platform', 'Ascend950PR_9599',
                       '--json-out', str(report)]
            if suppress: command.append('--mock-launch')
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError('Qualification deadline exhausted')
            with (directory/(phase+'.log')).open('wb') as log:
                process = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                    timeout=remaining, env=dict(os.environ, PYTHONPATH=str(directory),
                        ASCEND_PLATFORM='Ascend950PR_9599', PYASC_BACKEND='Model'))
            payload = json.loads(report.read_text()) if report.exists() else {}
            result['phases'][phase] = {'exit_code': process.returncode, 'report': payload}
            if phase != 'control' and (process.returncode != 0 or payload.get('launcher_calls') != len(contract['seeds'])):
                if process.returncode == 2: result['status'] = 'blocked'
                raise ValueError(phase+' did not launch every probe')
        real = result['phases']['execution']['report']; control = result['phases']['control']
        expected_checks = len(contract['seeds']) * len(contract['arguments'])
        # One output check plus one check for every input tensor.
        if len(real.get('numerical_checks', [])) != expected_checks:
            raise ValueError('Missing adapter comparisons')
        if not proves_numerics(real, control['report'], control['exit_code']):
            raise ValueError('Suppressed-launch control did not reject output')
        result['status'] = 'passed'
        result['verification_level'] = 'numerically_verified'
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        result['inputs_unchanged'] = all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha
                                       for name, sha in originals.items() for p in [directory/name])
        if not result['inputs_unchanged']: result['status'] = 'failed'
        result['duration_seconds'] = seconds-(deadline-time.monotonic())
        (directory/'qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['status']=='passed' else 2 if result['status']=='blocked' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--seconds', type=int, required=True)
    args = parser.parse_args()
    sys.exit(qualify(args.directory, args.seconds))
