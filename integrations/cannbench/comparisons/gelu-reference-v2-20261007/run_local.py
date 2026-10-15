"""Run bounded sequential probes under the existing shared simulator locks."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CANN = Path('/usr/local/Ascend/cann-9.0.0')
LOCKS = Path('/home/aloschilov/workspace/pyasc-skill-stack/integrations/cannbench/comparisons/gelu-adaptive-20260910')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='qualification')
    parser.add_argument('--tile', type=int, default=15872)
    parser.add_argument('--cores', type=int, default=8)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--mode', choices=['none', 'tanh'])
    parser.add_argument('--workload-size', type=int, default=0)
    settings = parser.parse_args()
    handles = []
    for name in ('screen.lock', 'local.lock'):
        handle = (LOCKS/name).open('rb')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'native-installed'), PYTHONDONTWRITEBYTECODE='1',
               ASCEND_HOME_PATH=str(CANN), PYASC_COMPILER=str(CANN/'tools/bisheng_compiler/bin/bisheng'),
               LD_LIBRARY_PATH=f'{CANN}/tools/simulator/Ascend950PR_9599/lib:{CANN}/lib64')
    env['PATH'] = f'{CANN}/tools/bisheng_compiler/bin:' + env['PATH']
    summaries = []
    for dtype in ('float32', 'float16', 'bfloat16'):
        if settings.dtype and dtype != settings.dtype:
            continue
        for mode in ('none', 'tanh'):
            if settings.mode and mode != settings.mode:
                continue
            name = f'{settings.label}-{dtype}-{mode}'
            out = ROOT/'evidence'/name
            out.mkdir(exist_ok=False)
            command = [sys.executable, str(ROOT/'probe.py'), '--dtype', dtype, '--mode', mode,
                       '--tile', str(settings.tile), '--cores', str(settings.cores),
                       '--workload-size', str(settings.workload_size),
                       '--output', str(out/'result.json')]
            with (out/'stdout.log').open('w') as log:
                proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    code = proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    code = 124
            row = json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
            summaries.append(dict(route=name, exit_code=code, completed=row.get('completed'),
                                  accuracy_passed=row.get('accuracy_passed'), error=row.get('error')))
            (ROOT/f'{settings.label}-summary.json').write_text(json.dumps(summaries, indent=2))
            print(summaries[-1], flush=True)


if __name__ == '__main__':
    main()
