"""Model-free simulator acceptance of the executor's numerical adapters."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/tools'))
from generation_adapters import task_contract
from generation_executor import external, qualify, save
from generation_plan import build_plan, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True, type=Path)
    args = parser.parse_args()
    root = external(args.results_dir); root.mkdir(parents=True, exist_ok=False)
    plan = build_plan(); save(root/'plan.json', plan)
    cases = []
    start = time.monotonic()
    for cell, filename in zip(plan['cells'], ('add.py', 'reduce.py', 'broadcast.py')):
        contract = task_contract(ROOT, cell)
        for wrong in (False, True):
            directory = root/(cell['id']+('-wrong' if wrong else '')); directory.mkdir()
            source = (ROOT/'tests/generation/fixtures'/filename).read_text()
            if wrong:
                # Still launches a real kernel: wrong result must fail numerics.
                source = source.replace('ax+bx', 'ax-bx').replace('reduced, b', 'reduced+1, b').replace('row, b', 'row+1, b')
            (directory/'candidate.py').write_text(source)
            try:
                report = qualify(directory, plan, contract, ROOT)
                accepted = report['status']=='passed'
                failure = None
            except (ValueError, OSError, TimeoutError) as exc:
                accepted = False; failure = str(exc)
            report = json.loads((directory/'qualification.json').read_text()) if (directory/'qualification.json').exists() else {}
            real = report.get('phases', {}).get('execution', {}).get('report', {})
            rejected_numerics = bool(real.get('launcher_calls', 0)>0 and real.get('numerical_failure'))
            passed = accepted if not wrong else not accepted and rejected_numerics
            cases.append({'case': directory.name, 'passed': passed, 'expected': 'numerical-rejection' if wrong else 'numerically_verified',
                          'candidate_sha256': digest((directory/'candidate.py').read_bytes()), 'error': failure})
            print(json.dumps(cases[-1]), flush=True)
    save(root/'report.json', {'kind': 'fixed-fixture-adapter-regression', 'cases': cases,
        'duration_seconds': time.monotonic()-start, 'passed': all(c['passed'] for c in cases),
        'model_invoked': False, 'hardware_invoked': False})
    save(root/'files.json', {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p.name!='files.json'})
    return 0 if all(c['passed'] for c in cases) else 1


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
