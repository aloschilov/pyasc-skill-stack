"""Native-log and immutable-input boundary for the existing SwiGLU qualifier."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

FATAL = re.compile(r'^\[error\]\s+\[[^\]\r\n]+\]\s+\[mte_gdma_illegal_burst_len_t0\]')
ARITHMETIC = re.compile(r'^\[error\]\s+\[[^\]\r\n]+\]\s+\[vec_err_idata_inf_nan_t0\]')


def require_no_dry_run(env):
    # Even the string "0" is not accepted: runtime versions differ in env parsing.
    if 'PYASC_DRY_RUN' in env:
        raise ValueError('PYASC_DRY_RUN must be unset')


def unchanged(before, after, label):
    if before != after:
        raise ValueError(label+' changed')


def source_hashes(root):
    return {name: sha256((root/path).read_bytes()).hexdigest() for name,path in {
        'candidate':'project/candidate.py', 'policy':'project/policy.json',
        'template':'context/skills/asctile-api-patterns/templates/packed-bf16-swiglu.py',
    }.items()}


def dispatch_counted(call, asc, runtime, install_spy, suppress):
    cls, original, count = install_spy(asc, suppress)
    launch = runtime.launch_kernel
    actual = 0
    def counted(*args, **kwargs):
        nonlocal actual
        value = launch(*args, **kwargs)
        actual += 1
        return value
    runtime.launch_kernel = counted
    try:
        call()
    finally:
        runtime.launch_kernel = launch
        cls.run = original
    if count['calls'] != 1 or actual != (0 if suppress else 1):
        raise ValueError(f'Launch evidence mismatch: requests={count["calls"]}, runtime calls={actual}, suppressed={suppress}')
    return {'requests':count['calls'], 'runtime_calls':actual}


def diagnostics(paths):
    counts = {'illegal_dma':0, 'arithmetic_nonfinite':0}
    examples = []
    for path in paths:
        with path.open(errors='replace') as stream:
            for line in stream:
                if FATAL.match(line):
                    counts['illegal_dma'] += 1
                    if len(examples) < 20:
                        examples.append(line.rstrip())
                if ARITHMETIC.match(line):
                    counts['arithmetic_nonfinite'] += 1
    return dict(counts, illegal_dma_examples=examples,
                arithmetic_note='Checks intentionally include NaN/Inf and overflowing intermediates; not a fatal DMA class.')


def accept(report, returncode, observed, before, after):
    unchanged(before, after, 'Generation sources')
    if returncode != 0 or report.get('status') != 'passed':
        raise ValueError('Qualification child did not pass')
    if observed['illegal_dma']:
        raise ValueError('Model reported illegal DMA burst length')
    checks = report.get('checks', [])
    if not checks or not all(c.get('positive') and c.get('suppressed_rejected')
            and c.get('real_launch_requests') == c.get('real_launches') == c.get('intercepted_launches') == 1
            and c.get('suppressed_runtime_calls') == 0
            and c.get('input_sha256_before') == c.get('input_sha256_after')
            and c.get('input_sha256_before') for c in checks):
        raise ValueError('Incomplete numerical/launch/input evidence')


def supervise(root, output, numerics_only):
    output.mkdir(parents=True, exist_ok=True)
    # No automatic replay or reuse of a previous execution directory.
    intent = output/'qualification-intent.json'
    with intent.open('x') as f:
        json.dump({'root':str(root),'numerics_only':numerics_only},f)
    paths = [output/'model.stdout.log', output/'model.stderr.log']
    report = {}; before = {}; after = {}; rc = None; error = None
    try:
        require_no_dry_run(os.environ)
        before = source_hashes(root)
        command = [sys.executable, str(Path(__file__).with_name('qualify.py')), str(root), str(output), '--worker']
        if numerics_only:
            command.append('--numerics-only')
        with paths[0].open('xb') as out, paths[1].open('xb') as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            (output/'child.json').write_text(json.dumps({'pid':child.pid,'command':command}))
            try:
                rc = child.wait(timeout=1800 if numerics_only else 7200)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise ValueError('Qualification deadline exceeded; process group stopped')
        if (output/'camodel.json').is_file():
            report = json.loads((output/'camodel.json').read_bytes())
        after = source_hashes(root)
        observed = diagnostics(paths)
        accept(report, rc, observed, before, after)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    observed = diagnostics([p for p in paths if p.exists()])
    result = dict(report, status='failed' if error else 'passed', child_exit_code=rc,
                  source_sha256_before=before, source_sha256_after=after,
                  diagnostics=observed, error=error)
    (output/'camodel.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'error':error,'diagnostics':observed}))
    return 1 if error else 0
