"""Opt-in paired CI bridge. Reuse the shared executor, ledger and public archive."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

from generation_ci import collect, read
from generation_compare import require
from generation_executor import external, save
from generation_pairs import build_pairs, run_pairs, validate_pairs
from generation_plan import ROOT, write_plan


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    request = {'experiment': 'paired', 'action': os.environ.get('GENERATION_ACTION', 'plan'),
        'approve_plan': os.environ.get('APPROVE_PLAN', ''),
        'state_dir': str(external(os.environ.get('GENERATION_STATE_DIR', str(Path.home()/'.local/state/pyasc-generation')))),
        'context': {k: os.environ.get(k) for k in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT',
            'GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_WORKFLOW_REF', 'GITHUB_EVENT_NAME')},
        'created_at': datetime.now(timezone.utc).isoformat()}
    save(root/'ci.json', request)
    require(request['action'] in ('plan', 'execute'), 'Use plan or execute')
    require(not os.environ.get('BASELINE_REF') and not os.environ.get('BASELINE_SHA256'), 'Paired controls are predeclared, not a historical baseline')
    context = request['context']
    if request['action'] == 'execute':
        require(context['GITHUB_EVENT_NAME'] in ('workflow_dispatch', 'schedule')
            and context['GITHUB_REPOSITORY'] == 'aloschilov/pyasc-skill-stack'
            and context['GITHUB_WORKFLOW_REF'] == 'aloschilov/pyasc-skill-stack/.github/workflows/generation-paired.yml@'+os.environ.get('GITHUB_REF', ''),
            'Paired execution requires the explicit controller')
    if context['GITHUB_EVENT_NAME'] == 'schedule':
        sys.path.insert(0, str(ROOT/'scripts'))
        from generation_operations import paired_due, policy
        config = policy(datetime.now(timezone.utc), paired=True)
        require(request['action'] == 'execute' and config['enabled']
            and os.environ.get('GITHUB_REF') == 'refs/heads/main', 'Paired scheduling is not activated')
        due, history = paired_due(config, datetime.now(timezone.utc))
        request.update(periodic_policy=config, cadence_history=history); save(root/'ci.json', request)
        require(due, 'Paired interval already consumed; no automatic retry')
    plan = build_pairs()
    plan['checkout_revision'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if context['GITHUB_SHA']: require(plan['checkout_revision'] == context['GITHUB_SHA'], 'Checkout differs from dispatched revision')
    plan = write_plan(plan, root/'plan')
    require(request['approve_plan'] == (plan['plan_sha256'] if request['action'] == 'execute' else ''), 'Reviewed paired plan digest mismatch')
    save(root/'result.json', {'status': 'prepared' if request['action'] == 'execute' else 'not-run',
        'plan_sha256': plan['plan_sha256'], 'model_invoked': False, 'hardware_invoked': False})
    return 0


def run(root):
    request = read(root/'ci.json')
    require(request['action'] == 'execute', 'Execution not requested')
    plan = validate_pairs(root/'plan/plan.json')
    require(plan['plan_sha256'] == request['approve_plan'], 'Execution approval mismatch')
    subprocess.run(['opencode', '--version'], check=True, timeout=20)
    subprocess.run(['docker', 'image', 'inspect', plan['common']['runtime']['image']], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    report = run_pairs(plan, root/'run', external(request['state_dir']))
    save(root/'result.json', {'status': report['status'], 'experiment': 'paired',
        'plan_sha256': plan['plan_sha256'], 'model_invoked': True if report['status'] == 'complete' else None, 'hardware_invoked': False,
        'trial_denominator': report['trial_denominator'],
        'cells': {k: v['outcome'] if v['status'] == 'valid' else v['status'] for k, v in report['observations'].items()},
        'note': 'Complete means an auditable experiment, not that every candidate passed or that skills helped.'})
    return 0 if report['status'] == 'complete' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'execute', 'collect'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = external(args.root)
    if args.action == 'prepare' and root.exists() or args.action == 'execute' and (root/'run').exists():
        print('Existing paired execution is never replayed', file=sys.stderr); return 2
    try:
        return {'prepare': prepare, 'execute': run, 'collect': collect}[args.action](root)
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        if root.is_dir() and args.action != 'collect':
            save(root/'result.json', {'status': 'blocked', 'model_invoked': None, 'hardware_invoked': False,
                'alerts': ['paired-ci-step-failed'], 'error_type': type(exc).__name__, 'reason': reason})
        print('Paired CI stopped: '+reason, file=sys.stderr); return 2


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.exit(main())
