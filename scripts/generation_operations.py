"""Opt-in generation planning and independent GET-only coverage observation.

No model, Docker, workflow dispatch or issue creation. Notification is an explicit
idempotent update of one preconfigured, marked alert issue; never a retrying POST.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/tools'))
from generation_ci import baseline_at, unpack
from generation_compare import Bundle, compare, require, utc
from generation_executor import external, save
from generation_plan import build_plan, canonical, digest, write_plan

REPO = 'aloschilov/pyasc-skill-stack'
WORKFLOW = '.github/workflows/generation-nightly.yml'
POLICY = ROOT/'tests/generation/nightly.json'
MARKER = '<!-- pyasc-generation-health:v1 -->'


def read(path):
    return json.loads(Path(path).read_text())


def policy(now):
    value = read(POLICY)
    require(set(value) == {'schema_version', 'enabled', 'enabled_since', 'baseline_ref',
        'baseline_sha256', 'max_age_hours', 'max_run_hours', 'alert_issue'}, 'Unexpected nightly policy fields')
    require(type(value['schema_version']) is int and value['schema_version'] == 1
        and type(value['enabled']) is bool, 'Invalid nightly policy version/enable flag')
    require(re.fullmatch(r'release/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value['baseline_ref']), 'Nightly needs an explicit durable Release baseline')
    require(re.fullmatch(r'[0-9a-f]{64}', value['baseline_sha256']), 'Invalid baseline checksum')
    for field, minimum, maximum in [('max_age_hours', 1, 168), ('max_run_hours', 1, 24)]:
        require(type(value[field]) is int and minimum <= value[field] <= maximum, 'Invalid '+field)
    issue = value['alert_issue']
    require(issue is None or type(issue) is int and issue > 0, 'Invalid alert issue')
    if value['enabled']:
        require(utc(value['enabled_since']) <= now, 'Activation timestamp must not be in the future')
        require(issue is not None, 'Configure the dedicated alert issue before activation')
    elif value['enabled_since'] is not None:
        utc(value['enabled_since'])
    return value


def plan(root, config, mode, approval):
    event = os.environ.get('GITHUB_EVENT_NAME', 'workflow_dispatch')
    repository = os.environ.get('GITHUB_REPOSITORY', REPO)
    require(repository == REPO, 'Unexpected repository')
    require(event in ('workflow_dispatch', 'schedule', 'pull_request'), 'Unexpected trigger')
    if event == 'pull_request':
        require(mode == 'plan' and not approval, 'Pull requests are planning-only')
    if event == 'schedule':
        require(os.environ.get('GITHUB_REF') == 'refs/heads/main', 'Scheduled execution is main-only')
        require(os.environ.get('GITHUB_WORKFLOW_REF') == REPO+'/'+WORKFLOW+'@refs/heads/main', 'Unexpected scheduled caller')
        mode = 'execute' if config['enabled'] else 'none'
    value = build_plan()
    value['checkout_revision'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    require(not os.environ.get('GITHUB_SHA') or os.environ['GITHUB_SHA'] == value['checkout_revision'], 'Checkout/CI revision mismatch')
    value = write_plan(value, root/'plan')
    if mode == 'execute' and event != 'schedule':
        require(approval == value['plan_sha256'], 'Manual execution requires the exact reviewed plan digest')
    result = {'status': 'prepared' if mode == 'execute' else 'disabled' if mode == 'none' else 'not-run', 'action': mode,
        'approve_plan': value['plan_sha256'] if mode == 'execute' else '',
        'baseline_ref': config['baseline_ref'] if mode == 'execute' else '',
        'baseline_sha256': config['baseline_sha256'] if mode == 'execute' else '',
        'event': event, 'nightly_enabled': config['enabled'], 'model_invoked': False,
        'policy': config, 'policy_sha256': digest(POLICY.read_bytes()),
        'controller_sha256': digest(Path(__file__).read_bytes())}
    save(root/'decision.json', result)
    if os.environ.get('GITHUB_OUTPUT'):
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as out:
            for key in ('action', 'approve_plan', 'baseline_ref', 'baseline_sha256'):
                out.write(key+'='+result[key]+'\n')
    return result


def api(endpoint, payload=None):
    command = ['gh', 'api', '--method', 'GET' if payload is None else 'PATCH', endpoint]
    if payload is not None: command += ['--input', '-']
    return json.loads(subprocess.check_output(command, input=None if payload is None else json.dumps(payload),
        text=True, stderr=subprocess.PIPE, timeout=60))


def download(root, config, run=None):
    directory = root/('baseline' if run is None else str(run['id'])+'-'+str(run['run_attempt']))
    directory.mkdir()
    command = (['gh', 'release', 'download', config['baseline_ref'].split('/')[1], '--pattern', 'generation.tar.gz']
        if run is None else ['gh', 'run', 'download', str(run['id']), '--name', f'generation-evidence-{run["id"]}-{run["run_attempt"]}'])
    subprocess.run(command+['--repo', REPO, '--dir', str(directory)], check=True, timeout=120,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    sha = config['baseline_sha256'] if run is None else (directory/'generation.tar.gz.sha256').read_text().split()[0]
    unpack(directory/'generation.tar.gz', directory/'evidence', sha)
    return directory/'evidence', sha


def observation(run, evidence, baseline, config, now, manual=False, paired=False):
    ci, result = read(evidence/'ci.json'), read(evidence/'result.json')
    context = ci['context']; event = 'workflow_dispatch' if manual else 'schedule'
    path = run['path'] if manual else WORKFLOW
    allowed = ('.github/workflows/generation-paired.yml',) if paired else ('.github/workflows/ci.yml', WORKFLOW)
    require(path in allowed and (not paired or manual), 'Unexpected generation workflow')
    require(run['event'] == event and run['path'] == path, 'Unexpected workflow origin')
    require(manual or run['head_branch'] == 'main', 'Scheduled coverage must be on main')
    require(context['GITHUB_RUN_ID'] == str(run['id']) and context['GITHUB_RUN_ATTEMPT'] == str(run['run_attempt']), 'Run/attempt mismatch')
    require(context['GITHUB_REPOSITORY'] == REPO and context['GITHUB_EVENT_NAME'] == event
        and context['GITHUB_WORKFLOW_REF'] == REPO+'/'+path+'@refs/heads/'+run['head_branch'], 'Artifact origin mismatch')
    if paired:
        from generation_pairs import report_pairs
        plan = read(evidence/'run/plan.json'); state = read(evidence/'run/state.json')
        require(ci.get('experiment') == 'paired' and ci['action'] == 'execute'
            and result['status'] == 'complete' and result['model_invoked'] is True
            and result['hardware_invoked'] is False, 'Not a completed paired experiment')
        require(ci['approve_plan'] == plan['plan_sha256'] == result['plan_sha256'], 'Paired approval mismatch')
        require(context['GITHUB_SHA'] == run['head_sha'] == plan['checkout_revision'], 'Paired revision mismatch')
        require(utc(run['created_at']) <= utc(state['started_at']) <= utc(state['finished_at']) <= now,
            'Invalid paired execution timestamps')
        report = report_pairs(evidence/'run')
        require(report['status'] == 'complete' and result['trial_denominator'] == report['trial_denominator']
            and result['cells'] == {k: v['outcome'] for k, v in report['observations'].items()}, 'Incomplete/inconsistent paired result')
        return {'status': 'passed', 'kind': 'paired-context',
            'plan_sha256': report['plan_sha256'], 'reporter_sha256': report['reporter_sha256'],
            'alerts': ['stale-coverage'] if now-utc(state['finished_at']) > timedelta(hours=config['max_age_hours']) else [],
            'trial_denominator': report['trial_denominator'], 'comparisons': report['comparisons'],
            'general_skills_benefit_claim': False, 'performance_claim': False,
            'note': 'Verified complete paired experiment; does not require every candidate to pass.'}, state['finished_at']
    require(ci['action'] == 'execute' and result['model_invoked'] is True
        and result['hardware_invoked'] is False and result['compared'] is True, 'Not a compared generation run')
    require(ci['baseline_ref'] == config['baseline_ref'] and ci['baseline_sha256'] == config['baseline_sha256'], 'Baseline selection changed')
    candidate = Bundle(evidence/'run')
    require(context['GITHUB_SHA'] == run['head_sha'] == candidate.plan['checkout_revision'] == candidate.state['checkout_revision'], 'Revision mismatch')
    require(utc(run['created_at']) <= utc(candidate.state['started_at']), 'Execution predates its run')
    report = compare(baseline, candidate, now=now, max_age_hours=config['max_age_hours'])
    require(result['status'] == 'passed' and candidate.state['status'] == 'passed', 'Run did not pass')
    return report, candidate.state['finished_at']


def monitor(root, config, now, inspect_run=None, paired=False):
    require(not paired or inspect_run, 'Paired observation requires an explicit RUN_ID/ATTEMPT; scheduling is not activated')
    report = {'status': 'disabled', 'scope': 'manual-inspection' if inspect_run else 'scheduled-coverage',
        'experiment': 'paired' if paired else 'single',
        'checked_at': now.isoformat(), 'policy': config, 'policy_sha256': digest(POLICY.read_bytes()),
        'controller_sha256': digest(Path(__file__).read_bytes()),
        'alerts': [], 'latest_run': None, 'last_qualified': None, 'observations': [], 'model_invoked': False}
    if not config['enabled'] and not inspect_run: return report
    baseline = None
    if not paired:
        base_dir, _ = download(root, config)
        baseline = baseline_at(base_dir, REPO, config['baseline_ref'])
    if inspect_run:
        require(re.fullmatch(r'[1-9][0-9]*/[1-9][0-9]*', inspect_run), 'Use RUN_ID/ATTEMPT')
        run_id, attempt = inspect_run.split('/')
        run = api(f'repos/{REPO}/actions/runs/{run_id}/attempts/{attempt}')
        runs = [run]
    else:
        # A daily schedule yields at most seven runs in this bounded window.
        # A truncated/unexpected history never silently establishes freshness.
        since = max(utc(config['enabled_since']), now-timedelta(hours=config['max_age_hours']))
        query = f'repos/{REPO}/actions/workflows/generation-nightly.yml/runs?'+urlencode({
            'event': 'schedule', 'branch': 'main', 'per_page': 100, 'created': '>='+since.strftime('%Y-%m-%dT%H:%M:%SZ')})
        history = api(query); save(root/'history.json', history)
        runs = sorted(history['workflow_runs'], key=lambda r: (r['created_at'], r['id']), reverse=True)
        if history['total_count'] > len(runs): report['alerts'].append('history-truncated')
    for run in runs[:20]:
        require(utc(run['created_at']) <= now, 'Run timestamp is in the future')
        if not inspect_run:
            require(run['event'] == 'schedule' and run['path'] == WORKFLOW and run['head_branch'] == 'main'
                and utc(run['created_at']) >= utc(config['enabled_since']), 'Unexpected scheduled run')
        item = {k: run[k] for k in ('id', 'run_attempt', 'head_sha', 'created_at', 'status', 'conclusion', 'html_url')}
        report['observations'].append(item)
        if report['latest_run'] is None: report['latest_run'] = item
        if run['status'] != 'completed':
            if (now-utc(run['created_at'])).total_seconds() > (5 if paired else config['max_run_hours'])*3600:
                report['alerts'].append('run-stalled')
            continue
        if run['conclusion'] != 'success':
            report['alerts'].append('run-failed'); continue
        try:
            evidence, sha = download(root, config, run)
            comparison, finished = observation(run, evidence, baseline, config, now, manual=bool(inspect_run), paired=paired)
            item.update(archive_sha256=sha, comparison=comparison['status'])
            if paired: item['paired_summary'] = comparison
            report['alerts'] += comparison['alerts']
            if comparison['status'] == 'passed':
                report['last_qualified'] = dict(item, finished_at=finished)
                break
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            item['error_type'] = type(exc).__name__; report['alerts'].append('unverifiable-run')
    if not report['last_qualified']:
        if len(runs) > 20: report['alerts'].append('inspection-limit')
        if inspect_run or now-utc(config['enabled_since']) >= timedelta(hours=config['max_age_hours']):
            report['alerts'].append('stale-coverage')
    report['alerts'] = sorted(set(report['alerts']))
    report['status'] = 'alert' if report['alerts'] else 'verified-manual' if inspect_run else 'healthy' if report['last_qualified'] else 'pending'
    return report


def notify(report, config, now=None):
    now = now or datetime.now(timezone.utc)
    require(config['enabled'] and config['alert_issue'] is not None, 'Notification is not configured/enabled')
    require(report['scope'] == 'scheduled-coverage', 'Manual inspection cannot send a schedule alert')
    require(report['policy'] == config and report['policy_sha256'] == digest(POLICY.read_bytes()), 'Observation policy changed')
    require(timedelta(0) <= now-utc(report['checked_at']) <= timedelta(minutes=15), 'Observation too old for delivery')
    if report['status'] not in ('healthy', 'alert'): return {'status': 'not-needed'}
    endpoint = f'repos/{REPO}/issues/{config["alert_issue"]}'
    issue = api(endpoint)
    require(not issue.get('pull_request') and (issue.get('body') or '').startswith(MARKER), 'Refusing an unmarked alert issue')
    # Do not include observation time/age in the fingerprint: unchanged findings
    # must not produce hourly edits. Issue body updates are idempotent PATCHes.
    value = {key: report[key] for key in ('status', 'alerts')}
    for key in ('latest_run', 'last_qualified'):
        run = report[key]
        value[key] = {k: run.get(k) for k in ('id', 'run_attempt', 'head_sha', 'conclusion', 'finished_at')} if run else None
    fingerprint = digest(canonical(value))
    body = MARKER+'\n<!-- observation:'+fingerprint+' -->\n\nGeneration coverage: **'+report['status']+'**.\n\n'
    body += 'Alerts: '+(', '.join(report['alerts']) or 'none')+'.\n\n'
    for key in ('latest_run', 'last_qualified'):
        if report[key]: body += key.replace('_', ' ').capitalize()+': '+report[key]['html_url']+'\n\n'
    body += 'This dedicated issue is updated only when findings change. Manual inspections do not establish scheduled coverage.\n'
    desired = {'body': body, 'state': 'open' if report['status'] == 'alert' else 'closed'}
    if issue['body'] == body and issue['state'] == desired['state']: return {'status': 'unchanged', 'issue': issue['html_url']}
    try: api(endpoint, desired)
    except (OSError, ValueError, subprocess.SubprocessError): pass  # Reconcile, never resend blindly.
    confirmed = api(endpoint)
    require(all(confirmed[k] == v for k, v in desired.items()), 'Alert update unconfirmed; inspect before retry')
    return {'status': 'confirmed', 'issue': confirmed['html_url'], 'fingerprint': fingerprint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'monitor', 'notify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'execute'), default='plan')
    parser.add_argument('--approve-plan', default='')
    parser.add_argument('--inspect-run', help='GET-only inspection of RUN_ID/ATTEMPT; never scheduled acceptance or notification')
    parser.add_argument('--paired', action='store_true', help='Inspect a paired experiment; requires --inspect-run')
    args = parser.parse_args()
    if args.paired and (args.action != 'monitor' or not args.inspect_run):
        parser.error('--paired requires monitor --inspect-run RUN_ID/ATTEMPT')
    root = external(args.root)
    if args.action != 'notify': root.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc)
    config = None
    try:
        config = policy(now)
        if args.action == 'plan': result = plan(root, config, args.mode, args.approve_plan)
        elif args.action == 'notify': result = notify(read(root/'health.json'), config)
        else: result = monitor(root, config, now, args.inspect_run, paired=args.paired)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        result = {'status': 'alert', 'scope': 'manual-inspection' if args.inspect_run else 'scheduled-coverage',
            'checked_at': now.isoformat(),
            'alerts': ['observation-unavailable'], 'error_type': type(exc).__name__,
            'latest_run': None, 'last_qualified': None, 'model_invoked': False}
        if config is not None: result.update(policy=config, policy_sha256=digest(POLICY.read_bytes()))
        if args.action != 'monitor':
            save(root/'error.json', result); print(json.dumps(result)); return 2
    if args.action == 'monitor': save(root/'health.json', result)
    elif args.action == 'notify': save(root/'delivery.json', result)
    print(json.dumps(result))
    return 1 if args.action == 'monitor' and result['status'] == 'alert' else 0


if __name__ == '__main__': sys.exit(main())
