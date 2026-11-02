#!/usr/bin/env python3
"""One external SwiGlu hardware check: prepare, submit once, collect by GET.

No generation campaign, remote shell, automatic retries or in-repo evidence.
The runtime bundle is supplied separately; never download it implicitly.
"""
import argparse
import base64
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'workers'), str(HERE / 'ci')]
from hardware_gate import evaluate
from submission import save, reconcile, TERMINAL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def external(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('Keep results/state outside checkout')
    return path


def client(credentials):
    if credentials:
        from run import load_vm_credentials
        load_vm_credentials(credentials)
    from evalqueue import EvalQueue
    return EvalQueue()


def task_snapshot(queue):
    before = next(b for b in queue._client.list_benchmarks()['benchmarks']
                  if b['slug'] == queue.benchmark_slug)
    version = before['submission_version']
    if not re.fullmatch(r'\d+\.\d+\.\d+', version) or version not in before['accepted_submission_versions']:
        raise ValueError('Unsupported submission version')
    spec = next(o for o in before['multi_operators'] if o['operator'] == 'SwiGlu')
    task = queue._client.get_benchmark_task(queue.benchmark_slug, 'SwiGlu')
    if (task['benchmark_slug'] != queue.benchmark_slug or task['operator'] != 'SwiGlu'
            or task['function_name'] != 'swi_glu' or spec['function_name'] != 'swi_glu'):
        raise ValueError('Unexpected task identity')
    ids = [spec['rel_path'] + '_' + str(c['case_id']) for c in task['cases_yaml']]
    if len(ids) != spec['case_count'] or len(ids) != len(set(ids)) or not ids:
        raise ValueError('Invalid official case inventory')
    after = next(b for b in queue._client.list_benchmarks()['benchmarks']
                 if b['slug'] == queue.benchmark_slug)
    if before != after:
        raise ValueError('Catalog changed during capture')
    return {'benchmark_slug': queue.benchmark_slug, 'benchmark_version': version,
            'required_operators': 1, 'required_cases': len(ids),
            'operators': [dict(spec, case_ids=ids)]}, task


def make_bundle(runtime, expected_hash, sources, version, destination):
    """Keep only runtime, licenses and the selected kernel; rebuild wheel RECORD."""
    if sha(runtime) != expected_hash:
        raise ValueError('Runtime bundle hash mismatch')
    if not runtime.name.endswith('-cp312-cp312-linux_x86_64.whl'):
        raise ValueError('This smoke supports CPython 3.12 / Linux x86_64 runtime wheels')
    contents = {}
    metadata = f'cann_bench-{version}.dist-info/'
    with zipfile.ZipFile(runtime) as archive:
        for n in archive.namelist():
            if n.endswith('/'):
                continue
            if '..' in Path(n).parts or n.startswith('/'):
                raise ValueError('Unsafe wheel member')
            if n.startswith(('asc/', 'asctile/', 'pybind11/')):
                contents[n] = archive.read(n)
            elif '.dist-info/licenses/' in n:
                contents[metadata + 'licenses/' + n.split('.dist-info/licenses/', 1)[1]] = archive.read(n)
    if (not all(n in contents for n in ('asc/__init__.py', 'asctile/__init__.py', 'pybind11/__init__.py'))
            or not any(n.startswith('asc/_C/libpyasc') and n.endswith('.so') for n in contents)
            or not any('PYASC-LICENSE' in n for n in contents)):
        raise ValueError('Expected self-contained PyASC runtime, pybind11 and licenses')
    for name in ('swi_glu.py', '_pyasc_runtime.py'):
        data = (sources / name).read_bytes()
        compile(data, name, 'exec')
        contents['cann_bench/' + name] = data
    contents['cann_bench/__init__.py'] = b'from .swi_glu import swi_glu\n__all__ = ["swi_glu"]\n'
    contents[metadata + 'METADATA'] = (f'Metadata-Version: 2.1\nName: cann_bench\nVersion: {version}\n'
                                      'Requires-Python: >=3.12,<3.13\n\n').encode()
    contents[metadata + 'WHEEL'] = b'Wheel-Version: 1.0\nGenerator: pyasc-cannbench-smoke\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n'
    record = io.StringIO(); writer = csv.writer(record, lineterminator='\n')
    for name, data in sorted(contents.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        writer.writerow((name, 'sha256=' + digest, len(data)))
    writer.writerow((metadata + 'RECORD', '', ''))
    contents[metadata + 'RECORD'] = record.getvalue().encode()
    wheel_name = f'cann_bench-{version}-cp312-cp312-linux_x86_64.whl'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as wheel:
        for name, data in sorted(contents.items()):
            wheel.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), data, compress_type=zipfile.ZIP_DEFLATED)
    wheel_bytes = buf.getvalue()
    digest = hashlib.sha256(wheel_bytes).hexdigest()
    build = ('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'
             f"printf '%s  %s\\n' '{digest}' 'dist/{wheel_name}' | sha256sum --check\n")
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, data in [('build.sh', build.encode()), ('dist/' + wheel_name, wheel_bytes)]:
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o100755 if name == 'build.sh' else 0o100644) << 16
            bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
    return digest


def prepare(args, queue):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    catalog, task = task_snapshot(queue)
    schemes = queue._client._get_json('/api/workspace/aggregation-schemes/options')['schemes']
    if not any(item['code'] == args.aggregation_token for item in schemes):
        raise ValueError('Select an aggregation scheme owned by or shared with this account')
    save(args.run_dir / 'catalog.json', catalog)
    save(args.run_dir / 'task.json', task)
    sources = args.run_dir / 'candidate'; sources.mkdir()
    for name in ('swi_glu.py', '_pyasc_runtime.py'):
        (sources / name).write_bytes((args.kernel_dir / name).read_bytes())
    wheel_hash = make_bundle(args.runtime_bundle, args.runtime_sha256, sources,
                             catalog['benchmark_version'], args.run_dir / 'submission.zip')
    manifest = {'prepared_at': now(), 'scope': 'one-operator hardware smoke; not fresh generation',
                'target_hardware': queue.target_hardware, 'catalog': catalog,
                'aggregation_token': args.aggregation_token,
                'runtime_bundle_sha256': args.runtime_sha256, 'wheel_sha256': wheel_hash,
                'files': {name: sha(args.run_dir / name) for name in
                          ('submission.zip', 'catalog.json', 'task.json', 'candidate/swi_glu.py', 'candidate/_pyasc_runtime.py')},
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                                 (Path(__file__), HERE/'workers/evalqueue.py', HERE/'ci/hardware_gate.py')},
                'command': sys.argv}
    save(args.run_dir / 'manifest.json', manifest)
    return manifest


def verify(run):
    run = run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha(path) != digest:
            raise ValueError('Prepared input changed: ' + name)
    return manifest


def submit_once(queue, run, ledger):
    manifest = verify(run)
    digest = manifest['files']['submission.zip']
    record = ledger / (digest + '.json')
    previous = []
    if record.exists():
        state = json.loads(record.read_text())
        if state.get('phase') == 'rejected':
            if state['manifest_sha256'] == sha(run/'manifest.json'):
                raise ValueError('Rejected request: prepare corrected inputs before submitting again')
            previous = state.get('previous_intents', []) + [state]
        else:
            if state['manifest_sha256'] != sha(run/'manifest.json'):
                raise ValueError('Archive already has an intent; collect from ' + state['run_dir'])
            save(run/'state.json', state)
            return state  # Never repeat a POST, including unknown delivery.
    if queue.target_hardware != manifest['target_hardware'] or queue.benchmark_slug != manifest['catalog']['benchmark_slug']:
        raise ValueError('Submission destination changed')
    for path in ledger.glob('*.json'):
        if json.loads(path.read_text()).get('phase') not in ('finished', 'rejected'):
            raise ValueError('Reconcile existing intent before another submission: ' + str(path))
    queue._check_credit()
    jobs = queue._client.list_jobs(limit=100).get('jobs', [])
    if any(j.get('status') not in TERMINAL for j in jobs):
        raise ValueError('An account job is active or unknown; collect it first')
    current, task = task_snapshot(queue)
    if current != manifest['catalog'] or task != json.loads((run/'task.json').read_text()):
        raise ValueError('Task/version changed since prepare; prepare a new run')
    state = {'phase': 'upload-intent', 'created_at': now(), 'run_dir': str(run),
             'manifest_sha256': sha(run/'manifest.json'), 'archive_sha256': digest,
             'benchmark_slug': queue.benchmark_slug, 'job_tag': 'pyasc-smoke-' + digest[:24],
             'job_id': None, 'previous_intents': previous}
    save(record, state); save(run/'state.json', state)
    # Intent is durable before POST. Exceptions leave it unresolved; collect uses GET only.
    try:
        posted = queue._submit_streaming(str(run/'submission.zip'), ['swi_glu'], state['job_tag'],
                                         aggregation_token=manifest['aggregation_token'])
    except Exception as exc:
        from evalqueue import SubmissionRejected
        state['phase'] = 'rejected' if isinstance(exc, SubmissionRejected) else 'ambiguous'
        save(record, state); save(run/'state.json', state)
        # Retain diagnostics privately, redacting the credential even if an SDK echoes it.
        detail = str(exc).replace(queue._client.token, '[REDACTED]')
        detail = re.sub(r'https?://[^\s\"<>]+', '[URL]', detail)
        save(run/'upload-error.json', {'at': now(), 'type': type(exc).__name__, 'detail': detail})
        raise
    state['job_id'] = posted.get('job_id') or (posted.get('job') or {}).get('id')
    state['submission_id'] = posted.get('submission_id') or (posted.get('submission') or {}).get('id')
    if not state['job_id']:
        raise ValueError('Unknown submission delivery; collect, do not resubmit')
    state['phase'] = 'submitted'
    save(record, state); save(run/'state.json', state)
    return state


def collect(queue, run, ledger):
    manifest = verify(run)
    record = ledger / (manifest['files']['submission.zip'] + '.json')
    state = json.loads(record.read_text())
    if state['manifest_sha256'] != sha(run/'manifest.json'):
        raise ValueError('Manifest differs from submission intent')
    job_id = state.get('job_id') or reconcile(queue._client, state)
    payload = queue._client.get_job(job_id)
    job = payload.get('job', payload)
    if job.get('id') != job_id or job.get('job_tag') != state['job_tag']:
        raise ValueError('Hardware job identity differs from submission intent')
    if state.get('submission_id') and job.get('submission_id') != state['submission_id']:
        raise ValueError('Hardware submission identity mismatch')
    state.update(job_id=job_id, submission_id=job.get('submission_id'), status=job.get('status'), checked_at=now())
    state['phase'] = 'finished' if state['status'] in TERMINAL else 'submitted'
    save(record, state); save(run/'state.json', state)
    # Keep service response private and outside Git; emit only the gate summary.
    save(run/'job.json', payload)
    if state['phase'] != 'finished':
        return 2, state
    report = evaluate(manifest['catalog'], [payload], manifest['target_hardware'])
    report.update(scope=manifest['scope'], archive_sha256=state['archive_sha256'],
                  job_id=job_id, submission_id=state['submission_id'],
                  queued_at=job.get('queued_at'), started_at=job.get('started_at'), finished_at=job.get('finished_at'))
    save(run/'report.json', report)
    return (0 if report['gate_passed'] else 1), report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('prepare', 'submit', 'collect'))
    ap.add_argument('--run-dir', type=external, required=True)
    ap.add_argument('--state-dir', type=external, default=Path.home()/'.local/state/pyasc-cannbench-smoke')
    ap.add_argument('--credentials-file', type=Path)
    ap.add_argument('--runtime-bundle', type=Path)
    ap.add_argument('--runtime-sha256')
    ap.add_argument('--aggregation-token', default=os.environ.get('CANNBENCH_AGGREGATION_TOKEN'))
    ap.add_argument('--kernel-dir', type=Path, default=HERE/'smoke-kernel')
    args = ap.parse_args()
    if args.action == 'prepare' and (not args.runtime_bundle or not args.runtime_sha256 or not args.aggregation_token):
        ap.error('prepare requires --runtime-bundle, --runtime-sha256 and --aggregation-token')
    queue = client(args.credentials_file)
    if args.action == 'prepare':
        result = prepare(args, queue); code = 0
    else:
        args.state_dir.mkdir(parents=True, exist_ok=True)
        with (args.state_dir/'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'submit':
                result = submit_once(queue, args.run_dir, args.state_dir); code = 0
            else:
                code, result = collect(queue, args.run_dir, args.state_dir)
    print(json.dumps(result, indent=2)); return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        # SDK errors can include response bodies; do not print credentials/URLs.
        detail = str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__
        print('CANNBench smoke stopped: ' + detail + '. Inspect retained state; never blindly repeat submit.', file=sys.stderr)
        sys.exit(1)
