#!/usr/bin/env python3
"""Execute the simulator acceptance matrix; save all results outside checkout."""
import argparse
import json
import os
import shutil
from pathlib import Path
import sys
from run_artifacts import ROOT, RunArtifacts
from run_and_verify import pinned_sim_image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', required=True)
    args = ap.parse_args()
    run = RunArtifacts(args.results_dir, 'verification-contract')
    image = pinned_sim_image()
    if '@sha256:' not in image:
        raise ValueError('Acceptance requires a digest-pinned image')
    run.data['image'] = image
    fixtures = ROOT/'tests/fixtures/kernels'
    evidence = ROOT/'docs/evidence/generation-gate-20261028'
    template = ROOT/'skills/pyasc-codegen-workflow/templates/kernel-template.py'
    cases = [
        ('wrong-numerics', fixtures/'wrong_numerics.py', 'execute', 1, None, 1, 'verification failed'),
        ('raises-midrun', fixtures/'raises_midrun.py', 'execute', 1, None, 1, 'deliberate post-launch'),
        ('no-launch', fixtures/'no_launch.py', 'execute', 1, None, 0, 'no kernel launch'),
        ('legacy-prints-pass', fixtures/'prints_pass.py', 'simulator', 1, None, 0, 'no kernel launch'),
        ('prints-pass', fixtures/'prints_pass.py', 'execute', 1, None, 0, 'no kernel launch'),
        ('no-main', fixtures/'no_main.py', 'execute', 1, None, 0, 'no __main__'),
        ('unverified', fixtures/'unverified.py', 'execute', 0, 'executed', 1, None),
        ('template', template, 'execute', 0, 'numerically_verified', 3, None),
        ('run-1', evidence/'run-1/kernel.py', 'execute', 0, 'numerically_verified', 1, None),
        ('run-2', evidence/'run-2/kernel.py', 'execute', 1, None, 0, 'UnsupportedSyntaxError'),
        ('run-3', evidence/'run-3/kernel.py', 'execute', 0, 'numerically_verified', 1, None),
        ('auto-failure', fixtures/'raises_midrun.py', 'auto', 1, None, 1, 'deliberate post-launch'),
        ('template-jit', template, 'jit', 0, 'compiled', None, None),
    ]
    from skill_jit_examples import extract_examples
    for label, path in extract_examples(run.path/'skill-examples'):
        cases.append((label, path, 'jit', 0, 'compiled', None, None))
    failed = []
    try:
        for label, kernel, mode, expected_code, level, calls, reason in cases:
            # Compiler/simulator side effects belong in the external run area,
            # including for historical evidence and the shipped template.
            case_dir = run.path/'inputs'/label; case_dir.mkdir(parents=True)
            copied = case_dir/kernel.name; shutil.copy2(kernel, copied)
            if kernel.parent == fixtures:
                shutil.copy2(fixtures/'_base.py', case_dir/'_base.py')
            argv = [sys.executable, str(ROOT/'tests/tools/run_and_verify.py'), str(copied),
                    '--mode', mode, '--runtime-backend', 'docker', '--docker-image', image,
                    '--platform', 'Ascend950PR_9599', '--timeout', '120', '--json']
            if label == 'skill-jit-3':
                argv += ['--jit-scalar-args', json.dumps({'size':1025,'tile_size':1024})]
            code, report = run.command(label, argv, timeout=390)
            ok = (code == expected_code and report.get('status') == ('PASS' if code == 0 else 'FAIL')
                  and report.get('verification_level') == level
                  and (calls is None or report.get('launcher_calls') == calls)
                  and (reason is None or reason in report.get('detail', ''))
                  and (mode != 'auto' or report.get('mode') == 'auto (execute)'))
            if not ok: failed.append(label)
            print(f'{"PASS" if ok else "FAIL"}: {label} (exit={code}, level={report.get("verification_level")})', flush=True)
        run.data['failed_cases'] = failed
        run.finish('failed' if failed else 'passed')
    except BaseException:
        run.finish('interrupted'); raise
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
