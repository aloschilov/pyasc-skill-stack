"""Run the correctness contract on a real NPU; results stay outside checkout."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from device_identity import probe
from run_artifacts import ROOT, RunArtifacts
from ssh_runtime import command_timeout
from verification_contract import core_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--runtime-backend', choices=('ssh', 'host'), default='ssh')
    ap.add_argument('--timeout', type=int, default=120)
    args = ap.parse_args()
    run = RunArtifacts(args.results_dir, 'hardware-correctness')
    run.data.update(runtime_backend=args.runtime_backend, performance='not-run')
    failed = []
    try:
        identity = None
        if args.runtime_backend == 'host':
            identity = probe()
            run.data['device_identity'] = identity
            if identity['status'] != 'ready':
                run.finish('blocked')
                print('BLOCKED: '+identity['reason'])
                return 2
        for label, source, mode, expected, level, calls, reason in core_cases():
            if mode == 'jit': continue  # hardware claims require real launches
            directory = run.path/'inputs'/label
            directory.mkdir(parents=True)
            kernel = directory/source.name
            shutil.copyfile(source, kernel)
            if source.parent == ROOT/'tests/fixtures/kernels':
                shutil.copyfile(source.parent/'_base.py', directory/'_base.py')
            argv = [sys.executable, str(ROOT/'tests/tools/run_and_verify.py'), str(kernel),
                    '--runtime-backend', args.runtime_backend, '--mode', mode, '--backend','NPU',
                    '--timeout', str(args.timeout), '--json']
            if identity:
                argv += ['--platform', identity['soc_version']]
            code, report = run.command(label, argv, timeout=command_timeout(args.timeout))
            observed = report.get('device_identity') if args.runtime_backend == 'ssh' else identity
            if code == 2:
                run.data.update(blocked_case=label, detail=report.get('detail'), failed_cases=failed)
                run.finish('failed' if failed else 'blocked')
                print(f'BLOCKED: {label}: {report.get("detail")}', flush=True)
                return 1 if failed else 2
            ready = observed and observed.get('status') == 'ready'
            if ready and identity is None:
                identity = observed
                run.data['device_identity'] = identity
            same_device = ready and all(observed.get(k) == identity.get(k)
                for k in ('hostname','soc_version','cann','pyasc_version'))
            ok = (same_device and code == expected
                  and report.get('status') == ('PASS' if expected == 0 else 'FAIL')
                  and report.get('verification_level') == level
                  and report.get('launcher_calls') == calls
                  and (reason is None or reason in report.get('detail','')))
            if not ok: failed.append(label)
            print(f'{"PASS" if ok else "FAIL"}: {label}', flush=True)
        run.data['failed_cases'] = failed
        run.finish('failed' if failed else 'passed')
        return int(bool(failed))
    except BaseException as exc:
        run.finish('interrupted', error=exc)
        raise


if __name__ == '__main__':
    sys.exit(main())
