"""Compare two calendar days of scheduled, first-attempt simulator qualifications."""
import argparse
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zlib
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/tools'))
from generation_ci import unpack
from generation_compare import Bundle, require, utc
from generation_plan import canonical, digest

REPOSITORY = 'aloschilov/pyasc-skill-stack'
WORKFLOW = '.github/workflows/generation-nightly.yml'
ZONE = ZoneInfo('Europe/Moscow')


def gh(*args):
    return subprocess.check_output(['gh', *args], timeout=120, text=True)


def runs_for_dates(first, last, workflow='generation-nightly.yml', event='schedule'):
    start = datetime.combine(first, time.min, ZONE).astimezone(timezone.utc)
    end = datetime.combine(last + timedelta(days=1), time.min, ZONE).astimezone(timezone.utc) - timedelta(seconds=1)
    bounds = '..'.join(t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in (start, end))
    page = json.loads(gh('api', f'repos/{REPOSITORY}/actions/workflows/{workflow}/runs'
                         f'?event={event}&branch=main&created={bounds}&per_page=100'))
    require(len(page['workflow_runs']) == page['total_count'], 'Incomplete workflow inventory')
    return page['workflow_runs']


def read_run(run):
    rid, attempt = run['id'], run['run_attempt']
    require(type(rid) is int and type(attempt) is int and min(rid, attempt) > 0, 'Invalid run identity')
    with tempfile.TemporaryDirectory() as directory:
        local = Path(directory)
        gh('run', 'download', str(rid), '--repo', REPOSITORY, '--name', f'generation-evidence-{rid}-{attempt}',
           '--dir', str(local))
        archive_sha = (local/'generation.tar.gz.sha256').read_text().split()[0]
        evidence = local/'verified'
        unpack(local/'generation.tar.gz', evidence, archive_sha)
        ci = json.loads((evidence/'ci.json').read_bytes())
        result = json.loads((evidence/'result.json').read_bytes())
        context = ci['context']
        expected = {'GITHUB_REPOSITORY': REPOSITORY, 'GITHUB_RUN_ID': str(rid), 'GITHUB_RUN_ATTEMPT': str(attempt),
                    'GITHUB_EVENT_NAME': 'schedule', 'GITHUB_SHA': run['head_sha'],
                    'GITHUB_WORKFLOW_REF': f'{REPOSITORY}/{WORKFLOW}@refs/heads/main'}
        require(all(context.get(k) == v for k,v in expected.items()) and ci['action'] == 'execute',
                'Scheduled execution identity differs')
        bundle = Bundle(evidence/'run')
        require(bundle.plan['checkout_revision'] == bundle.state['checkout_revision'] == run['head_sha']
                and ci['approve_plan'] == bundle.plan['plan_sha256'] == result['plan_sha256']
                and result['hardware_invoked'] is False and result['model_invoked'] is True
                and result['cells'] == bundle.cells, 'Qualification attribution differs')
        require(bundle.state['status'] in ('passed', 'failed'), 'Incomplete cohort')
        attempts = [cell['attempts'][0] for cell in bundle.state['cells']]
        sessions = set()
        # Retain model failures in the denominator; an unknown provider or interrupted
        # request cannot establish a skill effect. Bundle validates numerical winners.
        expected_model = bundle.plan['model']['expected_response']
        for cell, attempt in zip(bundle.state['cells'], attempts):
            require(attempt['status'] in ('failed', 'passed'), 'Invalid first attempt')
            # The executor records sessions in the per-attempt attribution.
            prefix = cell['id']+'/attempt-1/'
            session = bundle.json(prefix+'attribution.json')['session_id']
            require(isinstance(session, str) and session and session not in sessions, 'Reused first attempt')
            sessions.add(session)
            usage = attempt['usage']
            require(usage['usage_complete'] is True and usage['requests']
                    and all(r['status'] == 'complete' and r['usage']['observed_model'] == expected_model
                            for r in usage['requests']), 'Missing first-attempt attribution')
        skills = {k:v for k,v in bundle.plan['source_sha256'].items() if k.startswith('skills/')}
        require(bool(skills), 'Missing skill inputs')
        return {'revision': run['head_sha'], 'archive_sha256': archive_sha, 'run_id': rid, 'attempt': run['run_attempt'],
                'protocol': digest(canonical(bundle.compatibility())), 'model': expected_model,
                'skills': digest(canonical(skills)), 'runtime': bundle.plan['runtime']['source_revision'],
                'cells': {c['id']: a['status'] for c,a in zip(bundle.state['cells'], attempts)},
                'sessions': sessions, 'run_date': utc(run['created_at']).astimezone(ZONE).date().isoformat()}


def compare(before, after):
    if before['protocol'] != after['protocol'] or before['cells'].keys() != after['cells'].keys():
        return {'status':'incompatible'}
    if before['archive_sha256'] == after['archive_sha256'] or before['sessions'] & after['sessions']:
        return {'status':'incompatible'}
    gained = [k for k,v in after['cells'].items() if v == 'passed' and before['cells'][k] == 'failed']
    lost = [k for k,v in after['cells'].items() if v == 'failed' and before['cells'][k] == 'passed']
    if before['skills'] == after['skills']:
        status = 'same-skills'
    elif gained and lost:
        status = 'mixed'
    elif gained:
        status = 'improved'
    elif lost:
        status = 'regressed'
    else:
        status = 'tie'
    return {'status':status,'gained':gained,'lost':lost}


def report(today, lookup=runs_for_dates, reader=read_run):
    yesterday = today - timedelta(days=1)
    dates = [yesterday, today]
    out = {'dates':[d.isoformat() for d in dates], 'timezone':str(ZONE), 'runs':[None,None], 'status':'unavailable',
           'reasons':['Workflow inventory unavailable'] * 2}
    try:
        runs = lookup(*dates)
    except (subprocess.SubprocessError, OSError, LookupError, ValueError, TypeError):
        return out
    selected = []
    for i,day in enumerate(dates):
        rows = [r for r in runs if r['event']=='schedule' and r['path']==WORKFLOW and r['head_branch']=='main'
                and utc(r['created_at']).astimezone(ZONE).date() == day]
        if len(rows) != 1:
            out['reasons'][i] = 'No scheduled run' if not rows else 'Multiple scheduled runs'
            selected.append(None)
            continue
        run, = rows
        out['runs'][i] = {'run_id':run['id'], 'attempt':run['run_attempt'], 'revision':run['head_sha']}
        if run['status'] != 'completed':
            out['reasons'][i] = 'Scheduled run in progress'
            selected.append(None)
            continue
        try:
            verified = reader(run)
        except (OSError, LookupError, ValueError, TypeError, subprocess.SubprocessError, EOFError, tarfile.TarError, zlib.error):
            out['reasons'][i] = 'Incomplete or unverifiable execution evidence'
            selected.append(None)
            continue
        out['reasons'][i] = None
        out['runs'][i] = {k:v for k,v in verified.items() if k != 'sessions'}
        selected.append(verified)
    if all(selected):
        out.update(compare(*selected))
    return out


def hardware_pair(dates, records):
    """Compare qualified first attempts in the pinned public CANNBench catalog."""
    by_day = [[], []]
    for row in records:
        fresh = row.get('first_hardware')
        if row.get('kind') != 'hardware' or row.get('availability') != 'available' or not fresh:
            continue
        day = utc(fresh['generated_at']).astimezone(ZONE).date().isoformat()
        if day in dates:
            by_day[dates.index(day)].append(row)
    if any(len(rows) != 1 for rows in by_day):
        return {'status':'unavailable'}
    a,b = (rows[0] for rows in by_day)
    if a['first_hardware']['protocol'] != b['first_hardware']['protocol'] or a['ci_url'] == b['ci_url']:
        return {'status':'incompatible'}
    left,right = ({t['id']:t for t in r['tasks']} for r in (a,b))
    official = {f'level1/swi_glu_{i}' for i in range(1, 21)}
    if left.keys() != official or right.keys() != official or any(left[k]['baseline_us'] != right[k]['baseline_us'] for k in left):
        return {'status':'incompatible'}
    from statistics import geometric_mean
    return {'status':'measured', 'count':len(left), 'previous_over_current':geometric_mean(
        left[k]['elapsed_us']/right[k]['elapsed_us'] for k in left),
        'skills_changed':a['first_hardware']['skills'] != b['first_hardware']['skills'],
        'scores':[a['service_score'], b['service_score']], 'sources':[a['release_url'], b['release_url']]}


LABELS = {'unavailable':'Insufficient verified observations', 'incompatible':'Execution conditions differ',
          'same-skills':'Same skill bytes', 'tie':'Same first-attempt outcomes', 'mixed':'Gains and losses',
          'improved':'More first-attempt qualifications', 'regressed':'Fewer first-attempt qualifications'}


def summary(daily, transfer=None, swiglu=None):
    lines = ['## Daily skill quality', '', ' → '.join(daily['dates'])+' (Europe/Moscow)',
             '', '**'+LABELS[daily['status']]+'**', '']
    for i, (day, run) in enumerate(zip(daily['dates'], daily['runs'])):
        if run and 'cells' in run:
            n = sum(v == 'passed' for v in run['cells'].values())
            lines.append(f"- {day}: {n}/{len(run['cells'])} first attempts qualified "
                         f"([run {run['run_id']}](https://github.com/{REPOSITORY}/actions/runs/{run['run_id']})).")
        else:
            lines.append(f"- {day}: {daily['reasons'][i]}.")
    hw = daily['hardware']
    lines += ['', (f"SwiGLU device time, previous/current: {hw['previous_over_current']:.3f}× across {hw['count']} cases; "
                   f"service scores {hw['scores'][0]:.4f} → {hw['scores'][1]:.4f}; "
                   f"skill inputs {'changed' if hw['skills_changed'] else 'unchanged'}." if hw['status'] == 'measured'
                   else 'CANNBench performance: no matched pair of qualified first-attempt runs in the public catalog.')]
    if transfer and transfer['status'] == 'available':
        a, b = (list(transfer['cases'][arm].values()) for arm in ('A','B'))
        lines += ['', f"Retained [held-out skill comparison]({transfer['url']}): "
                      f"{a.count('passed')}/{len(a)} → {b.count('passed')}/{len(b)} first attempts passed CaModel."]
    if swiglu and swiglu['status'] == 'available':
        lines += ['', f"Retained [SwiGLU tiling experiment]({swiglu['release']}): {swiglu['parent_over_wide']:.2f}× "
                      "on six wide cases at eight blocks. All 20 passed on CANNBench. This measures one candidate edit."]
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date', type=date.fromisoformat, default=datetime.now(ZONE).date())
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    result = report(args.date)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['dates'], result['status'])
