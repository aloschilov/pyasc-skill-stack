"""Run inside the pinned compiler image; attest runtime bytes before lowering."""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workers'))
import local_compile_gate as gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--function', required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--task-dir', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    pin = json.loads(Path(__file__).with_name('single-runtime.json').read_text())
    assert hashlib.sha256(args.runtime.read_bytes()).hexdigest() == pin['runtime_bundle_sha256']
    import asc
    site = Path(asc.__file__).parent.parent
    with zipfile.ZipFile(args.runtime) as z:
        names = [n for n in z.namelist() if n.startswith(('asc/', 'asctile/')) and n.endswith(('.py', '.so'))]
        assert len(names) > 0
        for name in names:
            assert (site / name).read_bytes() == z.read(name), 'Compiler runtime differs: ' + name
    gate.PYASC_COMMIT = pin['pyasc_commit']
    original_compile = gate.compile_specialization
    def observed_compile(jit, prepared):
        print('Lowering specialization:', jit.fn.__name__, prepared[2], flush=True)
        result = original_compile(jit, prepared)
        print('Lowering completed:', jit.fn.__name__, flush=True)
        return result
    gate.compile_specialization = observed_compile
    report = gate.evaluate(args.candidate, args.function, args.task_dir / 'cases.yaml',
                           json.loads((args.task_dir / 'proto.yaml').read_text()))
    report.update(runtime_bundle_sha256=pin['runtime_bundle_sha256'], runtime_files_verified=len(names),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({k: report[k] for k in ('status', 'cases', 'compile_passed', 'pyasc_commit', 'runtime_files_verified')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
