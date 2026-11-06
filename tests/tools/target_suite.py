"""Bounded, immutable simulator shards for the complete pinned target census.

Resume runs only unattempted cases. Existing attempts, including failures, are
never overwritten or replayed. Full-suite and selected-subset completeness are
separate claims; neither discovery nor successful compilation earns numerics.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from target_inventory import ROOT, CONTRACT, canonical, derive, digest, normalize, read, require, save, sources, validate
from target_case import proves_case

HELPERS = ('tests/tools/target_inventory.py', 'tests/tools/target_case.py', 'tests/tools/target_suite.py', CONTRACT)


def inputs(root=ROOT):
    return dict(sources(root), **{name: digest((root/name).read_bytes()) for name in HELPERS})


def external(path):
    path = Path(path).expanduser().resolve()
    require(path != ROOT and ROOT not in path.parents, 'Keep results outside checkout')
    return path


def build_plan(collection, selected=None, reason=None, shard_size=8, seconds=30, shard_seconds=900, root=ROOT):
    receipt = read(collection/'result.json'); ids = normalize(read(collection/'nodeids.json'))
    inventory = validate(ids, root)
    require(receipt['status'] == 'collected' and receipt['cleanup_confirmed'] and receipt['inputs_unchanged']
        and receipt['runtime_image'] == inventory['runtime_image'] and receipt['source_sha256'] == sources(root)
        and receipt['case_count'] == len(ids), 'Unqualified collection receipt')
    for name, expected in receipt['artifact_sha256'].items():
        require(Path(name).name == name and digest((collection/name).read_bytes()) == expected, 'Collection artifact drift')
    chosen = ids if selected is None else normalize(selected)
    require(set(chosen) <= set(ids), 'Unknown selected case')
    require(set(chosen) == set(ids) or isinstance(reason, str) and bool(reason.strip()), 'Explain excluded cases')
    require(type(shard_size) is int and 1 <= shard_size <= 16 and type(seconds) is int and 1 <= seconds <= 300
        and type(shard_seconds) is int and 1 <= shard_seconds <= 3600, 'Invalid shard limits')
    # Each phase reserves a cleanup interval as well as its subprocess timeout.
    require(shard_size*2*(seconds+20) <= shard_seconds, 'Shard cannot contain its worst-case reservations')
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    plan = {'schema_version': 1, 'kind': 'target-suite-plan', 'inventory': inventory,
        'case_ids': ids, 'selected': chosen, 'excluded_reason': reason if len(chosen) != len(ids) else None,
        'shards': [chosen[n:n+shard_size] for n in range(0, len(chosen), shard_size)],
        'limits': {'case_phase_seconds': seconds, 'shard_seconds': shard_seconds, 'cleanup_seconds': 20,
                   'cpus': 1, 'memory_gib': 2},
        'source_sha256': inputs(root), 'checkout_revision': revision.stdout.strip() or None,
        'collection_sha256': digest(canonical(receipt)), 'backend': 'Model', 'hardware_invoked': False}
    plan['plan_sha256'] = digest(canonical(plan)); return plan


def check_plan(plan, root=ROOT):
    unsigned = dict(plan); claimed = unsigned.pop('plan_sha256')
    require(claimed == digest(canonical(unsigned)) and plan['schema_version'] == 1
        and plan['kind'] == 'target-suite-plan' and plan['backend'] == 'Model'
        and plan['hardware_invoked'] is False, 'Invalid target plan')
    require(plan['source_sha256'] == inputs(root) and plan['inventory'] == derive(plan['case_ids'], root), 'Plan/source drift')
    ids = normalize(plan['case_ids']); selected = normalize(plan['selected'])
    require(ids == plan['case_ids'] and selected == plan['selected'] and set(selected) <= set(ids), 'Invalid selected inventory')
    require([node for shard in plan['shards'] for node in shard] == selected
        and all(isinstance(s, list) and 1 <= len(s) <= 16 for s in plan['shards']), 'Missing/duplicate sharded cases')
    limits = plan['limits']
    require(limits['cpus'] == 1 and limits['memory_gib'] == 2 and limits['cleanup_seconds'] == 20
        and type(limits['case_phase_seconds']) is int and 1 <= limits['case_phase_seconds'] <= 300
        and type(limits['shard_seconds']) is int and 1 <= limits['shard_seconds'] <= 3600
        and all(len(s)*2*(limits['case_phase_seconds']+20) <= limits['shard_seconds'] for s in plan['shards']), 'Invalid plan limits')
    require(set(ids) == set(selected) or isinstance(plan['excluded_reason'], str) and bool(plan['excluded_reason'].strip()),
            'Missing exclusion reason')


def case_key(nodeid):
    return digest(nodeid.encode())


def phase(directory, plan, source, nodeid, mode):
    name = 'pyasc-target-'+uuid.uuid4().hex; limits = plan['limits']; env = read(source/'reference/images/ENV.json')
    argv = ['docker', 'run', '--pull', 'never', '--name', name, '--network', 'none', '--cpus', '1', '--memory', '2g',
        '--ulimit', 'nofile=65536:65536', '-v', str(source)+':/repo:ro', '-v', str(directory)+':/evidence', '-w', '/repo',
        '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
        '-e', 'PYASC_CACHE_DIR=/tmp/cache', '-e', 'PYASC_DUMP_PATH=/tmp/dumps',
        plan['inventory']['runtime_image'], env['image']['python_bin'], '/repo/tests/tools/target_case.py',
        '--nodeid', nodeid, '--output', '/evidence/'+mode+'.json']
    if mode == 'control': argv.append('--suppress-launch')
    record = {'status': 'starting', 'command': argv, 'container': name, 'timeout_seconds': limits['case_phase_seconds']}
    path = directory/(mode+'.command.json'); save(path, record); start = time.monotonic(); proc = None
    try:
        with (directory/(mode+'.stdout')).open('wb') as out, (directory/(mode+'.stderr')).open('wb') as err:
            proc = subprocess.Popen(argv, stdout=out, stderr=err)
            record.update(status='running', pid=proc.pid); save(path, record)
            record['exit_code'] = proc.wait(timeout=limits['case_phase_seconds'])
            record['status'] = 'completed'
    except subprocess.TimeoutExpired:
        record['status'] = 'timed-out'
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: record['client_cleanup_error'] = 'TimeoutExpired'
        record['cleanup_confirmed'] = False
        try:
            inspected = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name], capture_output=True, timeout=5)
            if inspected.returncode == 0: record['container_state'] = json.loads(inspected.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            record['inspection_error'] = type(exc).__name__
        try:
            removed = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10)
            record['cleanup_confirmed'] = removed.returncode == 0 or b'No such container' in removed.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            record['cleanup_error'] = type(exc).__name__
        record['seconds'] = time.monotonic()-start; save(path, record)
    return record


def run_case(directory, plan, source, nodeid):
    directory.mkdir(); result = {'nodeid': nodeid, 'plan_sha256': plan['plan_sha256'], 'status': 'running', 'phases': {}}
    save(directory/'result.json', result)
    try:
        for mode in ('real', 'control'):
            cmd = phase(directory, plan, source, nodeid, mode); result['phases'][mode] = cmd
            require(cmd['cleanup_confirmed'], 'Unresolved container cleanup')
            if cmd['status'] == 'timed-out': result.update(status='timed-out', reason=mode+' exceeded its phase budget'); break
            if not (directory/(mode+'.json')).is_file():
                result.update(status='blocked', reason=mode+' produced no worker receipt; exit '+str(cmd.get('exit_code'))); break
            receipt = read(directory/(mode+'.json'))
            unsupported = any(p.get('outcome') == 'skipped' or p.get('xfail') for p in receipt['phases'].values())
            if unsupported:
                result.update(status='unsupported', reason=receipt['phases']); break
            if mode == 'real' and cmd['exit_code'] != 0:
                call_failed = cmd['exit_code'] == 1 and receipt['phases'].get('call', {}).get('outcome') == 'failed'
                result.update(status='failed' if call_failed else 'blocked',
                              reason={'pytest_exit_code': cmd['exit_code'], 'phases': receipt['phases']}); break
        else:
            proves_case(read(directory/'real.json'), read(directory/'control.json'))
            result.update(status='qualified', reason='Official numerical assertions and rejecting suppressed-launch control')
    except ValueError as exc:
        result.update(status='blocked', reason=str(exc))
    finally:
        result['artifact_sha256'] = {p.name: digest(p.read_bytes()) for p in directory.iterdir() if p.name != 'result.json'}
        save(directory/'result.json', result)
    return result


def run_shard(plan, directory, number, root=ROOT):
    directory = external(directory); check_plan(plan, root)
    require(type(number) is int and 0 <= number < len(plan['shards']), 'Unknown shard (zero based)')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'executor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory/'plan.json').exists(): require(read(directory/'plan.json') == plan, 'Cannot change a resumed plan')
        else:
            require(not any(p.name != 'executor.lock' for p in directory.iterdir()), 'Nonempty results directory without plan')
            save(directory/'plan.json', plan)
            for name, expected in plan['source_sha256'].items():
                data = (root/name).read_bytes(); require(digest(data) == expected, 'Source changed while exporting')
                target = directory/'source'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
            (directory/'cases').mkdir()
        check_plan(plan, directory/'source')
        # A previous interrupted attempt must be reconciled, never replayed.
        for attempt in (directory/'cases').iterdir():
            require((attempt/'result.json').is_file() and read(attempt/'result.json')['status'] != 'running', 'Unresolved previous case')
            for cmd in attempt.glob('*.command.json'): require(read(cmd).get('cleanup_confirmed'), 'Unresolved previous container')
        deadline = time.monotonic()+plan['limits']['shard_seconds']
        for node in plan['shards'][number]:
            attempt = directory/'cases'/case_key(node)
            if attempt.exists(): continue
            require(time.monotonic()+2*(plan['limits']['case_phase_seconds']+20) <= deadline, 'Shard budget exhausted before next case')
            result = run_case(attempt, plan, directory/'source', node)
            require(all(c.get('cleanup_confirmed') for c in result['phases'].values()), 'Cannot continue with unresolved container')
            if result['status'] == 'blocked': break
    return report(directory)


def report(directory):
    plan = read(directory/'plan.json'); source = directory/'source'; check_plan(plan, source)
    known = {case_key(n): n for n in plan['selected']}; actual = list((directory/'cases').iterdir())
    require(all(p.is_dir() and p.name in known for p in actual), 'Unexpected/duplicate case report')
    outcomes = {}
    for node in plan['case_ids']:
        if node not in plan['selected']:
            outcomes[node] = {'status': 'excluded', 'reason': plan['excluded_reason']}; continue
        attempt = directory/'cases'/case_key(node)
        if not attempt.exists(): outcomes[node] = {'status': 'not-run', 'reason': 'Selected shard/case not yet attempted'}; continue
        if not (attempt/'result.json').is_file():
            outcomes[node] = {'status': 'unresolved', 'reason': 'Attempt directory lacks a case receipt'}; continue
        value = read(attempt/'result.json')
        require(value['nodeid'] == node and value['plan_sha256'] == plan['plan_sha256'], 'Case plan/identity mismatch')
        if value['status'] == 'running':
            outcomes[node] = {'status': 'unresolved', 'reason': 'Interrupted case; reconcile its owned process/container'}; continue
        require(value['status'] in ('qualified', 'failed', 'blocked', 'unsupported', 'timed-out'), 'Unknown case status')
        require(set(value['artifact_sha256']) == {p.name for p in attempt.iterdir() if p.name != 'result.json'}
            and all(p.is_file() and not p.is_symlink() for p in attempt.iterdir()), 'Incomplete case artifact index')
        for name, expected in value['artifact_sha256'].items():
            require(Path(name).name == name and digest((attempt/name).read_bytes()) == expected, 'Case artifact drift')
        if value['status'] == 'qualified':
            require(set(value['phases']) == {'real', 'control'} and all(c.get('cleanup_confirmed') for c in value['phases'].values()),
                    'Missing phase/cleanup evidence')
            for mode, code in (('real', 0), ('control', 1)):
                cmd = read(attempt/(mode+'.command.json'))
                require(cmd == value['phases'][mode] and cmd['status'] == 'completed' and cmd['exit_code'] == code
                    and plan['inventory']['runtime_image'] in cmd['command'], 'Phase command/runtime mismatch')
            real, control = read(attempt/'real.json'), read(attempt/'control.json')
            require(real['nodeid'] == node and real['source_sha256'] == sources(source), 'Worker source/identity mismatch')
            env = read(source/'reference/images/ENV.json')
            require(real['runtime']['pyasc'] == env['version'] and real['runtime']['torch'] == env['dev_image']['torch']
                and real['runtime']['pytest'] == env['dev_image']['pytest'], 'Worker runtime mismatch')
            proves_case(real, control)
        outcomes[node] = {'status': value['status'], 'reason': value.get('reason')}
    counts = dict(Counter(v['status'] for v in outcomes.values()))
    value = {'schema_version': 1, 'kind': 'target-numerical-coverage', 'plan_sha256': plan['plan_sha256'],
        'upstream_revision': plan['inventory']['upstream_revision'], 'runtime_image': plan['inventory']['runtime_image'],
        'required_cases': len(plan['case_ids']), 'selected_cases': len(plan['selected']), 'counts': counts,
        'all_selected_qualified': all(outcomes[n]['status'] == 'qualified' for n in plan['selected']),
        'full_suite_qualified': counts.get('qualified', 0) == len(plan['case_ids']), 'outcomes': outcomes,
        'compile_only_coverage': 'not measured by this lane', 'npu_coverage': 'not run', 'model_generation_coverage': 'not measured'}
    save(directory/'coverage.json', value); return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan'); p.add_argument('--collection', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selected', type=Path); p.add_argument('--excluded-reason'); p.add_argument('--shard-size', type=int, default=8)
    p.add_argument('--case-seconds', type=int, default=30); p.add_argument('--shard-seconds', type=int, default=900)
    p = sub.add_parser('run'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True)
    p = sub.add_parser('report'); p.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'plan':
        path = external(args.output); require(not path.exists(), 'Use a new plan file'); path.parent.mkdir(parents=True, exist_ok=True)
        plan = build_plan(args.collection, read(args.selected) if args.selected else None, args.excluded_reason,
                          args.shard_size, args.case_seconds, args.shard_seconds); save(path, plan)
        print(json.dumps({'plan_sha256': plan['plan_sha256'], 'cases': len(plan['case_ids']), 'selected': len(plan['selected']), 'shards': len(plan['shards'])})); return 0
    result = run_shard(read(args.plan), args.root, args.shard) if args.action == 'run' else report(external(args.root))
    print(json.dumps({k:v for k,v in result.items() if k != 'outcomes'}))
    # The full-suite completeness gate is deliberately red for partial qualification.
    return 0 if result['full_suite_qualified'] else 1


if __name__ == '__main__': sys.exit(main())
