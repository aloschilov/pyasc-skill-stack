"""Manual CI bridge: plan, execute, compare and archive; no schedule or NPU path."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tarfile

from generation_compare import Bundle, compare, compatibility, require
from generation_executor import execute, external, save, validate_plan
from generation_plan import ROOT, build_plan, digest, write_plan

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
PUBLIC_NAMES = ('ci.json', 'result.json', 'comparison.json', 'plan', 'run')
BASELINE_REF = r'(?:[1-9][0-9]*/[1-9][0-9]*|release/[A-Za-z0-9][A-Za-z0-9_.-]{0,127})'


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    request = {'action': os.environ.get('GENERATION_ACTION', 'plan'),
        'approve_plan': os.environ.get('APPROVE_PLAN', ''),
        'baseline_ref': os.environ.get('BASELINE_REF', ''),
        'baseline_sha256': os.environ.get('BASELINE_SHA256', ''),
        'state_dir': str(external(os.environ.get('GENERATION_STATE_DIR', str(Path.home()/'.local/state/pyasc-generation')))),
        'context': {key: os.environ.get(key) for key in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT',
            'GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_WORKFLOW_REF', 'GITHUB_EVENT_NAME')},
        'created_at': datetime.now(timezone.utc).isoformat()}
    save(root/'ci.json', request)
    require(request['action'] in ('plan', 'execute'), 'Use plan or execute')
    require(bool(request['baseline_ref']) == bool(request['baseline_sha256']), 'Baseline run and archive checksum are a pair')
    if request['baseline_ref']:
        require(re.fullmatch(BASELINE_REF, request['baseline_ref']), 'Baseline must be RUN_ID/ATTEMPT or release/TAG')
        require(re.fullmatch(r'[0-9a-f]{64}', request['baseline_sha256']), 'Invalid baseline archive checksum')
    plan = build_plan()
    plan['checkout_revision'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if request['context']['GITHUB_SHA']:
        require(plan['checkout_revision'] == request['context']['GITHUB_SHA'], 'Checkout differs from dispatched revision')
    plan = write_plan(plan, root/'plan')
    if request['action'] == 'execute':
        require(request['approve_plan'] == plan['plan_sha256'], 'Reviewed plan digest does not match this checkout')
    else:
        require(not request['approve_plan'] and not request['baseline_ref'], 'Plan-only mode does not execute or compare')
    save(root/'result.json', {'status': 'not-run' if request['action'] == 'plan' else 'prepared',
        'plan_sha256': plan['plan_sha256'], 'model_invoked': False, 'hardware_invoked': False})
    return 0


def unpack(archive, destination, expected):
    """No extractall, links, duplicate paths or executable snapshot imports."""
    require(archive.stat().st_size <= MAX_ARCHIVE_BYTES, 'Archive too large')
    require(digest(archive.read_bytes()) == expected, 'Baseline archive checksum mismatch')
    require(not destination.exists(), 'Never replace baseline evidence')
    with tarfile.open(archive, 'r:gz') as source:
        files = {}; total = 0
        for member in source:
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and '..' not in path.parts
                and 'private' not in path.parts and member.name == path.as_posix(), 'Unsafe archive member')
            require(path.parts[0] in (*PUBLIC_NAMES, 'archive-index.json'), 'Unexpected archive member')
            require(member.name not in files and len(files) < 20000, 'Duplicate/oversized archive index')
            total += member.size
            require(total <= MAX_ARCHIVE_BYTES, 'Unpacked archive too large')
            files[member.name] = source.extractfile(member).read()
    index = json.loads(files.pop('archive-index.json'))
    require(set(index) == set(files) and all(digest(data) == index[name] for name, data in files.items()), 'Archive member checksum mismatch')
    destination.mkdir(parents=True)
    for name, data in files.items():
        target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle: handle.write(data)


def fetch(root):
    request = read(root/'ci.json')
    require(request['action'] == 'execute', 'Baseline retrieval is execute-only')
    kind, identifier = request['baseline_ref'].split('/')
    repository = request['context']['GITHUB_REPOSITORY']
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository or ''), 'Explicit repository required')
    downloaded = root/'baseline-download'; downloaded.mkdir()
    command = (['gh', 'release', 'download', identifier, '--pattern', 'generation.tar.gz'] if kind == 'release' else
               ['gh', 'run', 'download', kind, '--name', f'generation-evidence-{kind}-{identifier}'])
    subprocess.run(command+['--repo', repository, '--dir', str(downloaded)], check=True, timeout=120)
    unpack(downloaded/'generation.tar.gz', root/'baseline', request['baseline_sha256'])
    baseline = Bundle(root/'baseline/run')
    require(baseline.state['status'] == 'passed', 'Selected baseline did not qualify')
    origin_record = read(root/'baseline/ci.json'); origin = origin_record['context']
    require(origin_record['action'] == 'execute' and read(root/'baseline/result.json')['status'] in ('qualified', 'passed'), 'Baseline was not an accepted execution')
    require(origin['GITHUB_REPOSITORY'] == repository and origin['GITHUB_EVENT_NAME'] == 'workflow_dispatch', 'Baseline origin mismatch')
    require(origin['GITHUB_SHA'] == baseline.plan.get('checkout_revision') == baseline.state.get('checkout_revision'), 'Baseline revision mismatch')
    if kind != 'release':
        require(origin['GITHUB_RUN_ID'] == kind and origin['GITHUB_RUN_ATTEMPT'] == identifier, 'Baseline run/attempt mismatch')
    request['baseline_index_sha256'] = baseline.sha256
    save(root/'ci.json', request)
    return 0


def run(root):
    request = read(root/'ci.json')
    require(request['action'] == 'execute', 'Execution not requested')
    plan = validate_plan(root/'plan/plan.json')
    require(plan['plan_sha256'] == request['approve_plan'], 'Execution approval mismatch')
    # Preflight before the durable executor latch or any provider request.
    version = subprocess.check_output(['opencode', '--version'], text=True, timeout=20).strip()
    subprocess.run(['docker', 'image', 'inspect', plan['runtime']['image']], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    baseline = None
    if request['baseline_ref']:
        baseline = Bundle(root/'baseline/run', request['baseline_index_sha256'])
        require(baseline.state['status'] == 'passed', 'Baseline is incomplete')
        # A comparison must be possible before spending on a fresh candidate.
        require(baseline.compatibility() == compatibility(plan, version), 'Baseline is incompatible with the approved plan/client')
    state = execute(plan, root/'run', external(request['state_dir']))
    attempts = [a for c in state['cells'] for a in c['attempts']]
    observed_request = any(a.get('usage', {}).get('requests') for a in attempts)
    result = {'status': state['status'], 'model_invoked': True if observed_request else None if attempts else False,
        'hardware_invoked': False, 'cells': {c['id']: c['status'] for c in state['cells']},
        'plan_sha256': plan['plan_sha256'], 'compared': baseline is not None}
    if baseline:
        comparison = compare(baseline, Bundle(root/'run', None))
        save(root/'comparison.json', comparison); result.update(status=comparison['status'], alerts=comparison['alerts'])
    elif state['status'] == 'passed':
        Bundle(root/'run')  # Do not trust a lone PASS string for qualification.
        result['status'] = 'qualified'  # Candidate for review, not an auto-promoted baseline.
    save(root/'result.json', result)
    return 0 if result['status'] in ('passed', 'qualified') else 1 if result['status'] in ('failed', 'regression') else 2


def collect(root):
    root.mkdir(parents=True, exist_ok=True)
    if not (root/'result.json').exists() or read(root/'result.json')['status'] == 'prepared':
        save(root/'result.json', {'status': 'incomplete', 'model_invoked': None, 'hardware_invoked': False,
            'alerts': ['missing-terminal-result'], 'note': 'Inspect retained executor state; do not replay.'})
    result = read(root/'result.json')
    out = root/'public'; out.mkdir()
    archive = out/'generation.tar.gz'; index = {}; total = 0
    with tarfile.open(archive, 'x:gz') as dest:
        for name in PUBLIC_NAMES:
            path = root/name
            paths = path.rglob('*') if path.is_dir() else [path]
            for file in sorted(paths):
                if not file.is_file() or file.is_symlink() or 'private' in file.relative_to(root).parts: continue
                require(not any((root/p).is_symlink() for p in file.relative_to(root).parents), 'Symlinked public directory')
                total += file.stat().st_size
                require(total <= MAX_ARCHIVE_BYTES, 'Public evidence too large')
                data = file.read_bytes(); relative = file.relative_to(root).as_posix()
                entry = tarfile.TarInfo(relative); entry.size = len(data); entry.mode = 0o600
                dest.addfile(entry, io.BytesIO(data)); index[relative] = digest(data)
        data = (json.dumps(index, indent=2)+'\n').encode()
        entry = tarfile.TarInfo('archive-index.json'); entry.size = len(data)
        dest.addfile(entry, io.BytesIO(data))
    checksum = digest(archive.read_bytes())
    (out/'generation.tar.gz.sha256').write_text(checksum+'  generation.tar.gz\n')
    plan = read(root/'plan/plan.json') if (root/'plan/plan.json').exists() else {}
    summary = ('# Generation simulator result\n\nStatus: **'+result['status']+'**.\n\n'
        'Plan: `'+plan.get('plan_sha256', 'unavailable')+'`\n\n'
        'Archive SHA256: `'+checksum+'`\n\n'
        'No hardware execution or scheduled qualification. A plan is not a generation result; '
        'a qualified run is not automatically the baseline.\n')
    if result.get('cells'):
        summary += '\n| Cell | Outcome |\n|---|---|\n'+''.join('| '+name+' | '+status+' |\n' for name, status in result['cells'].items())
    if result.get('alerts'): summary += '\nAlerts: '+', '.join(result['alerts'])+'\n'
    (out/'SUMMARY.md').write_text(summary)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as handle: handle.write(summary)
    print(json.dumps({'status': result['status'], 'archive_sha256': checksum, 'members': len(index)}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'fetch', 'execute', 'collect'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = external(args.root)
    # Never change a previous accepted result on accidental prepare/execute replay.
    if (args.action == 'prepare' and root.exists()) or (args.action in ('fetch', 'execute') and (root/'run').exists()):
        print('Use a new CI run directory; existing execution is never replayed', file=sys.stderr)
        return 2
    try:
        return {'prepare': prepare, 'fetch': fetch, 'execute': run, 'collect': collect}[args.action](root)
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) if type(exc) is ValueError else type(exc).__name__
        if root.is_dir() and args.action != 'collect':
            save(root/'result.json', {'status': 'blocked', 'model_invoked': None, 'hardware_invoked': False,
                'alerts': ['ci-step-failed'], 'error_type': type(exc).__name__, 'reason': reason})
        print('Generation CI stopped: '+reason, file=sys.stderr)
        return 2


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
