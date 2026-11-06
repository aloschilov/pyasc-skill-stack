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
PAIRED_WORKFLOW = '.github/workflows/generation-paired.yml'
PAIRED_POLICY = ROOT/'tests/generation/paired.json'
MODELS_WORKFLOW = '.github/workflows/generation-models.yml'
MODELS_POLICY = ROOT/'tests/generation/models.json'
MARKER = '<!-- pyasc-generation-health:v1 -->'


def read(path):
    return json.loads(Path(path).read_text())


def experiment_name(paired=False):
    # Preserve the existing boolean API; the new model lane uses its own
    # controller, cadence history, observer policy and notification marker.
    require(paired is False or paired is True or paired == 'models', 'Unknown coverage lane')
    return 'models' if paired == 'models' else 'paired' if paired else 'single'


def workflow_path(paired=False):
    return {'single': WORKFLOW, 'paired': PAIRED_WORKFLOW, 'models': MODELS_WORKFLOW}[experiment_name(paired)]


def policy_path(paired=False):
    return {'single': POLICY, 'paired': PAIRED_POLICY, 'models': MODELS_POLICY}[experiment_name(paired)]


def policy(now, paired=False):
    value = read(policy_path(paired))
    if paired:
        require(type(value['schema_version']) is int and value['schema_version'] in ((3,) if paired == 'models' else (1, 2)), 'Invalid paired policy version')
        value = value['periodic']
        require(set(value) == {'enabled', 'enabled_since', 'min_interval_hours', 'max_age_hours',
            'max_run_hours', 'alert_issue'}, 'Unexpected paired periodic policy fields')
        interval = value['min_interval_hours']
        require(type(interval) is int and 24 <= interval <= 720, 'Invalid paired interval')
    else:
        require(set(value) == {'schema_version', 'enabled', 'enabled_since', 'baseline_ref',
            'baseline_sha256', 'max_age_hours', 'max_run_hours', 'alert_issue'}, 'Unexpected nightly policy fields')
        require(type(value['schema_version']) is int and value['schema_version'] == 1, 'Invalid nightly policy version')
        require(re.fullmatch(r'release/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value['baseline_ref']), 'Nightly needs an explicit durable Release baseline')
        require(re.fullmatch(r'[0-9a-f]{64}', value['baseline_sha256']), 'Invalid baseline checksum')
    require(type(value['enabled']) is bool, 'Invalid policy enable flag')
    for field, minimum, maximum in [('max_age_hours', interval if paired else 1, interval+72 if paired else 168),
                                    ('max_run_hours', 10 if paired else 1, 24)]:
        require(type(value[field]) is int and minimum <= value[field] <= maximum, 'Invalid '+field)
    issue = value['alert_issue']
    require(issue is None or type(issue) is int and issue > 0, 'Invalid alert issue')
    if value['enabled']:
        require(utc(value['enabled_since']) <= now, 'Activation timestamp must not be in the future')
        require(issue is not None, 'Configure the dedicated alert issue before activation')
    elif value['enabled_since'] is not None:
        utc(value['enabled_since'])
    return value


def deferred_paired_run(run):
    """A successful controller-only skip is not another execution attempt."""
    if run.get('status') != 'completed' or run.get('conclusion') != 'success': return False, None
    endpoint = f'repos/{REPO}/actions/runs/{run["id"]}/attempts/{run["run_attempt"]}/jobs?per_page=100'
    jobs = api(endpoint)
    rows = jobs['jobs']
    require(type(jobs['total_count']) is int and jobs['total_count'] == len(rows), 'Incomplete paired job history')
    names = {job['name']: job['conclusion'] for job in rows}
    receipt = {'endpoint':endpoint, 'total_count':jobs['total_count'],
        'jobs':[{key:job[key] for key in ('name','conclusion')} for job in rows]}
    return len(rows) == 2 and names == {'prepare': 'success', 'paired': 'skipped'}, receipt


def paired_due(config, now, paired=True):
    """GET-only cadence guard. Failed/cancelled attempts also consume the interval."""
    require(os.environ.get('GITHUB_RUN_ATTEMPT') == '1', 'Scheduled paired reruns require a new explicit manual plan')
    current = os.environ.get('GITHUB_RUN_ID', '')
    require(re.fullmatch(r'[1-9][0-9]*', current), 'Scheduled run identity required')
    cutoff = max(utc(config['enabled_since']), now-timedelta(hours=config['min_interval_hours']))
    since = cutoff.replace(microsecond=0)  # GitHub's created filter is second-granular.
    query = f'repos/{REPO}/actions/workflows/{Path(workflow_path(paired)).name}/runs?'+urlencode({
        'event': 'schedule', 'branch': 'main', 'per_page': 100, 'created': '>='+since.strftime('%Y-%m-%dT%H:%M:%SZ')})
    history = api(query); runs = history['workflow_runs']
    require(type(history['total_count']) is int and history['total_count'] == len(runs), 'Incomplete paired cadence history')
    ids = [str(run['id']) for run in runs]
    require(len(ids) == len(set(ids)) and ids.count(current) == 1, 'Current run absent/duplicated in cadence history')
    for run in runs:
        require(run['event'] == 'schedule' and run['path'] == workflow_path(paired) and run['head_branch'] == 'main'
            and since <= utc(run['created_at']) <= now, 'Unexpected paired cadence history')
    recent = [run for run in runs if str(run['id']) != current and utc(run['created_at']) > cutoff]
    require(len(recent) <= 20, 'Paired cadence inspection limit exceeded')
    deferred = []; history['job_checks'] = {}
    for run in recent:
        skipped, receipt = deferred_paired_run(run)
        if receipt: history['job_checks'][str(run['id'])] = receipt
        if skipped: deferred.append(run['id'])
    history['deferred_runs'] = deferred
    return all(run['id'] in deferred for run in recent), history


def plan(root, config, mode, approval, paired=False, now=None):
    now = now or datetime.now(timezone.utc)
    workflow = workflow_path(paired)
    skip_reason = None
    event = os.environ.get('GITHUB_EVENT_NAME', 'workflow_dispatch')
    repository = os.environ.get('GITHUB_REPOSITORY', REPO)
    require(repository == REPO, 'Unexpected repository')
    require(event in ('workflow_dispatch', 'schedule', 'pull_request'), 'Unexpected trigger')
    if event == 'pull_request':
        require(mode == 'plan' and not approval, 'Pull requests are planning-only')
    if event == 'schedule':
        require(os.environ.get('GITHUB_REF') == 'refs/heads/main', 'Scheduled execution is main-only')
        require(os.environ.get('GITHUB_WORKFLOW_REF') == REPO+'/'+workflow+'@refs/heads/main', 'Unexpected scheduled caller')
        mode = 'execute' if config['enabled'] else 'none'
        if mode == 'none': skip_reason = 'policy-disabled'
        if paired and config['enabled']:
            due, history = paired_due(config, now, paired=paired); save(root/'cadence-history.json', history)
            if not due: mode = 'none'; skip_reason = 'interval-not-elapsed'
    if paired:
        from generation_pairs import build_pairs
        value = build_pairs(experiment=experiment_name(paired))
    else: value = build_plan()
    value['checkout_revision'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    require(not os.environ.get('GITHUB_SHA') or os.environ['GITHUB_SHA'] == value['checkout_revision'], 'Checkout/CI revision mismatch')
    value = write_plan(value, root/'plan')
    if mode == 'execute' and event != 'schedule':
        require(approval == value['plan_sha256'], 'Manual execution requires the exact reviewed plan digest')
    result = {'status': 'prepared' if mode == 'execute' else 'deferred' if skip_reason == 'interval-not-elapsed' else 'disabled' if mode == 'none' else 'not-run',
        'action': mode, 'skip_reason': skip_reason,
        'approve_plan': value['plan_sha256'] if mode == 'execute' else '',
        'baseline_ref': config['baseline_ref'] if mode == 'execute' and not paired else '',
        'baseline_sha256': config['baseline_sha256'] if mode == 'execute' and not paired else '',
        'experiment': experiment_name(paired),
        'event': event, 'nightly_enabled': config['enabled'], 'model_invoked': False,
        'policy': config, 'policy_sha256': digest(policy_path(paired).read_bytes()),
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
    path = run['path'] if manual else workflow_path(paired)
    allowed = (workflow_path(paired),) if paired else ('.github/workflows/ci.yml', WORKFLOW)
    require(path in allowed, 'Unexpected generation workflow')
    require(run['event'] == event and run['path'] == path, 'Unexpected workflow origin')
    require(manual or run['head_branch'] == 'main', 'Scheduled coverage must be on main')
    require(context['GITHUB_RUN_ID'] == str(run['id']) and context['GITHUB_RUN_ATTEMPT'] == str(run['run_attempt']), 'Run/attempt mismatch')
    require(context['GITHUB_REPOSITORY'] == REPO and context['GITHUB_EVENT_NAME'] == event
        and context['GITHUB_WORKFLOW_REF'] == REPO+'/'+path+'@refs/heads/'+run['head_branch'], 'Artifact origin mismatch')
    if paired:
        from generation_pairs import report_pairs
        plan = read(evidence/'run/plan.json'); state = read(evidence/'run/state.json')
        require(ci.get('experiment') == experiment_name(paired) and ci['action'] == 'execute'
            and result['status'] == 'complete' and result['model_invoked'] is True
            and result['hardware_invoked'] is False, 'Not a completed paired experiment')
        require((plan['schema_version'] == 2) == (paired == 'models'), 'Comparison lane does not match the retained plan')
        require(ci['approve_plan'] == plan['plan_sha256'] == result['plan_sha256'], 'Paired approval mismatch')
        require(context['GITHUB_SHA'] == run['head_sha'] == plan['checkout_revision'], 'Paired revision mismatch')
        require(utc(run['created_at']) <= utc(state['started_at']) <= utc(state['finished_at']) <= now,
            'Invalid paired execution timestamps')
        report = report_pairs(evidence/'run')
        require(report['status'] == 'complete' and result['trial_denominator'] == report['trial_denominator']
            and result['cells'] == {k: v['outcome'] for k, v in report['observations'].items()}, 'Incomplete/inconsistent paired result')
        return {'status': 'passed', 'kind': 'model-comparison' if paired == 'models' else 'paired-context',
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
    workflow = workflow_path(paired)
    report = {'status': 'disabled', 'scope': 'manual-inspection' if inspect_run else 'scheduled-coverage',
        'experiment': experiment_name(paired),
        'checked_at': now.isoformat(), 'policy': config, 'policy_sha256': digest(policy_path(paired).read_bytes()),
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
        # Both lanes use bounded policy windows. Truncated/unexpected history
        # never silently establishes freshness.
        since = max(utc(config['enabled_since']), now-timedelta(hours=config['max_age_hours']))
        query = f'repos/{REPO}/actions/workflows/{Path(workflow).name}/runs?'+urlencode({
            'event': 'schedule', 'branch': 'main', 'per_page': 100, 'created': '>='+since.strftime('%Y-%m-%dT%H:%M:%SZ')})
        history = api(query); save(root/'history.json', history)
        runs = sorted(history['workflow_runs'], key=lambda r: (r['created_at'], r['id']), reverse=True)
        if history['total_count'] > len(runs): report['alerts'].append('history-truncated')
    for run in runs[:20]:
        require(utc(run['created_at']) <= now, 'Run timestamp is in the future')
        if not inspect_run:
            require(run['event'] == 'schedule' and run['path'] == workflow and run['head_branch'] == 'main'
                and utc(run['created_at']) >= utc(config['enabled_since']), 'Unexpected scheduled run')
        item = {k: run[k] for k in ('id', 'run_attempt', 'head_sha', 'created_at', 'status', 'conclusion', 'html_url')}
        report['observations'].append(item)
        if report['latest_run'] is None: report['latest_run'] = item
        if run['status'] != 'completed':
            if (now-utc(run['created_at'])).total_seconds() > config['max_run_hours']*3600:
                report['alerts'].append('run-stalled')
            continue
        if run['conclusion'] != 'success':
            report['alerts'].append('run-failed'); continue
        try:
            if paired and not inspect_run:
                skipped, receipt = deferred_paired_run(run)
                if receipt: item['job_check'] = receipt
                if skipped: item.update(comparison='not-run', reason='controller-deferred'); continue
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
        if (inspect_run and runs[0]['status'] == 'completed' or not inspect_run
                and now-utc(config['enabled_since']) >= timedelta(hours=config['max_age_hours'])):
            report['alerts'].append('stale-coverage')
    report['alerts'] = sorted(set(report['alerts']))
    report['status'] = ('alert' if report['alerts'] else 'pending' if not report['last_qualified']
        else 'verified-manual' if inspect_run else 'healthy')
    return report


def notify(report, config, now=None):
    now = now or datetime.now(timezone.utc)
    paired = 'models' if report.get('experiment') == 'models' else report.get('experiment') == 'paired'
    marker = MARKER.replace(':v1', ':'+experiment_name(paired)+':v1') if paired else MARKER
    require(report['scope'] == 'scheduled-coverage', 'Manual inspection cannot send a schedule alert')
    require(report['policy'] == config and report['policy_sha256'] == digest(policy_path(paired).read_bytes()), 'Observation policy changed')
    if not config['enabled']: return {'status': 'disabled'}
    require(config['alert_issue'] is not None, 'Notification is not configured/enabled')
    require(timedelta(0) <= now-utc(report['checked_at']) <= timedelta(minutes=15), 'Observation too old for delivery')
    if report['status'] not in ('healthy', 'alert'): return {'status': 'not-needed'}
    endpoint = f'repos/{REPO}/issues/{config["alert_issue"]}'
    issue = api(endpoint)
    require(not issue.get('pull_request') and (issue.get('body') or '').startswith(marker), 'Refusing an unmarked alert issue')
    # Do not include observation time/age in the fingerprint: unchanged findings
    # must not produce hourly edits. Issue body updates are idempotent PATCHes.
    value = {key: report[key] for key in ('status', 'alerts')}
    value['experiment'] = experiment_name(paired)
    for key in ('latest_run', 'last_qualified'):
        run = report[key]
        value[key] = {k: run.get(k) for k in ('id', 'run_attempt', 'head_sha', 'conclusion', 'finished_at')} if run else None
    fingerprint = digest(canonical(value))
    body = marker+'\n<!-- observation:'+fingerprint+' -->\n\n'+('Paired experiment' if paired else 'Generation')+' coverage: **'+report['status']+'**.\n\n'
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
    lanes = parser.add_mutually_exclusive_group()
    lanes.add_argument('--paired', action='store_true', help='Use the paired context policy')
    lanes.add_argument('--models', action='store_true', help='Use the paired model policy')
    args = parser.parse_args()
    lane = 'models' if args.models else args.paired
    root = external(args.root)
    if args.action != 'notify': root.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc)
    config = None
    try:
        config = policy(now, paired=lane)
        if args.action == 'plan': result = plan(root, config, args.mode, args.approve_plan, paired=lane, now=now)
        elif args.action == 'notify': result = notify(read(root/'health.json'), config)
        else: result = monitor(root, config, now, args.inspect_run, paired=lane)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        result = {'status': 'alert', 'scope': 'manual-inspection' if args.inspect_run else 'scheduled-coverage',
            'checked_at': now.isoformat(),
            'alerts': ['observation-unavailable'], 'error_type': type(exc).__name__,
            'latest_run': None, 'last_qualified': None, 'model_invoked': False}
        if config is not None: result.update(experiment=experiment_name(lane),
            policy=config, policy_sha256=digest(policy_path(lane).read_bytes()))
        if args.action != 'monitor':
            save(root/'error.json', result); print(json.dumps(result)); return 2
    if args.action == 'monitor': save(root/'health.json', result)
    elif args.action == 'notify': save(root/'delivery.json', result)
    print(json.dumps(result))
    return 1 if args.action == 'monitor' and result['status'] == 'alert' else 0


if __name__ == '__main__': sys.exit(main())
