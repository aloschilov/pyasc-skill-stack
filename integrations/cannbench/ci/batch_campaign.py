"""Durable quota-bounded batches of one fully qualified evaluator bundle."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import submission
from generation_gate import sha256


def validate_base(archive, catalog):
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read('BUNDLE.json'))
        qualification = json.loads(source.read('qualification.json'))
        names = {op['function_name'] for op in catalog['operators']}
        if (set(manifest['operators']) != names or manifest['catalog_sha256'] != catalog['catalog_sha256']
                or manifest['benchmark_version'] != catalog['benchmark_version']
                or qualification.get('status') != 'passed'
                or qualification.get('catalog_sha256') != catalog['catalog_sha256']
                or qualification.get('wheel_sha256') != manifest['wheel_sha256']):
            raise ValueError('Full qualified bundle identity mismatch')
        reports = qualification.get('operators') or []
        if (len(reports) != len(names) or {r.get('operator') for r in reports} != names):
            raise ValueError('Incomplete installed-wheel qualification')
        for spec in catalog['operators']:
            report = next(r for r in reports if r['operator'] == spec['function_name'])
            cases = report.get('case_results') or []
            if (report.get('status') != 'passed' or report.get('compile_passed') != spec['case_count']
                    or report.get('dispatch_passed') != spec['case_count']
                    or len(cases) != spec['case_count']
                    or {c.get('case_id') for c in cases} != set(spec['case_ids'])
                    or any(c.get('compile') != 'passed' or c.get('dispatch') != 'passed' for c in cases)):
                raise ValueError('Incomplete installed-wheel case coverage')
        wheels = [n for n in source.namelist() if n.startswith('dist/') and n.endswith('.whl')]
        if len(wheels) != 1 or hashlib.sha256(source.read(wheels[0])).hexdigest() != manifest['wheel_sha256']:
            raise ValueError('Qualified wheel hash mismatch')
        return manifest


def initialize(bundle_root, catalog):
    """Freeze full bundle outside the disposable Actions checkout."""
    submission.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    archive = bundle_root.parent / 'campaign-bundle.zip'
    expected = {str(p.relative_to(bundle_root)): p for p in bundle_root.rglob('*') if p.is_file()}
    if not archive.exists():
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as target:
            for name, path in sorted(expected.items()):
                target.write(path, name)
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != set(expected) or any(source.read(n) != p.read_bytes() for n, p in expected.items()):
            raise ValueError('Existing archive differs from qualified bundle')
    manifest = validate_base(archive, catalog)
    digest = sha256(archive)
    with submission.shared_locks('upload'):
        pointer = submission.STATE_ROOT / 'campaign-current.json'
        if pointer.is_file():
            prior_file = Path(json.loads(pointer.read_text())['state_file'])
            prior = json.loads(prior_file.read_text())
            if (prior['archive_sha256'] != digest
                    and any(b.get('phase') != 'finished' for b in prior['batches'])):
                raise RuntimeError('Existing campaign has unresolved hardware jobs; reconcile first')
        root = submission.STATE_ROOT / 'campaigns' / digest
        root.mkdir(parents=True, exist_ok=True)
        durable = root / 'bundle.zip'
        if not durable.exists():
            # Do not publish a partially copied archive after a crash.
            temporary = root / 'bundle.zip.tmp'
            with temporary.open('wb') as output, archive.open('rb') as source:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(durable)
        if sha256(durable) != digest:
            raise ValueError('Durable bundle changed')
        path = root / 'campaign.json'
        if not path.exists():
            submission.save(path, {'schema': 1, 'phase': 'campaign', 'bundle': manifest,
                'archive_sha256': digest, 'archive_path': str(durable), 'batches': [],
                'benchmark_slug': catalog['benchmark_slug'], 'benchmark_version': catalog['benchmark_version']})
        submission.save(submission.STATE_ROOT / 'campaign-current.json', {'state_file': str(path)})
    return path


def _batch_archive(state, selected, root):
    metadata = {'catalog_sha256': state['bundle']['catalog_sha256'],
                'wheel_sha256': state['bundle']['wheel_sha256'], 'selected_operators': selected}
    destination = root / ('batch-%03d.zip' % (len(state['batches']) + 1))
    with zipfile.ZipFile(state['archive_path']) as source, zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            target.writestr(info, source.read(info.filename))
        target.writestr('BATCH.json', json.dumps(metadata, sort_keys=True))
    # fsync archive before durable intent, so ambiguous POST always has bytes.
    with destination.open('rb') as stream:
        os.fsync(stream.fileno())
    return destination, metadata


def validate_batches(state, catalog):
    """Bind every operator partition and all upload bytes to the frozen bundle."""
    if sha256(state['archive_path']) != state['archive_sha256']:
        raise ValueError('Durable bundle changed')
    if validate_base(Path(state['archive_path']), catalog) != state['bundle']:
        raise ValueError('Campaign bundle mismatch')
    selected = []
    with zipfile.ZipFile(state['archive_path']) as base:
        for batch in state['batches']:
            if sha256(batch['archive_path']) != batch['archive_sha256']:
                raise ValueError('Batch archive changed')
            expected = {'catalog_sha256': state['bundle']['catalog_sha256'],
                        'wheel_sha256': state['bundle']['wheel_sha256'],
                        'selected_operators': batch['selected_operators']}
            if batch.get('binding') != expected or not batch['selected_operators']:
                raise ValueError('Batch partition binding mismatch')
            with zipfile.ZipFile(batch['archive_path']) as uploaded:
                if (len(uploaded.namelist()) != len(base.namelist()) + 1
                        or set(uploaded.namelist()) != set(base.namelist()) | {'BATCH.json'}
                        or json.loads(uploaded.read('BATCH.json')) != expected
                        or any(uploaded.read(n) != base.read(n) for n in base.namelist())):
                    raise ValueError('Batch bytes differ from qualified bundle')
            selected.extend(batch['selected_operators'])
    names = {op['function_name'] for op in catalog['operators']}
    if len(selected) != len(set(selected)) or not set(selected).issubset(names):
        raise ValueError('Duplicate or unknown campaign operators')
    return selected


def advance(state_file, catalog, repo_root, *, allow_submit=False):
    """GET every existing batch first; at most one new POST within fresh quota.

    An unresolved intent prevents all new POSTs. Failed batches stay charged and
    visible; repairs require a new qualified bundle, not automatic resubmission.
    """
    queue = submission.site_queue(repo_root)
    payloads = []
    with submission.shared_locks('upload'):
        state = json.loads(state_file.read_text())
        selected = validate_batches(state, catalog)
        active = False
        for batch in state['batches']:
            if not batch.get('job_id'):
                batch['job_id'] = submission.reconcile(queue._client, batch)
                submission.save(state_file, state)
            payload = queue._client.get_job(batch['job_id'])
            status = payload.get('job', payload).get('status')
            batch.update(phase='finished' if status in submission.TERMINAL else 'submitted', status=status)
            active = active or status not in submission.TERMINAL
            payloads.append(payload)
        names = [op['function_name'] for op in catalog['operators']]
        if len(selected) != len(set(selected)) or not set(selected).issubset(names):
            raise ValueError('Duplicate or unknown campaign operators')
        remaining = [name for name in names if name not in selected]
        state['remaining_operators'] = remaining
        submission.save(state_file, state)
        if not allow_submit or active or not remaining:
            return payloads, state
        credits = queue._client.get_credits().get('credits') or {}
        balance = len(remaining) if credits.get('unlimited') is True else credits.get('remaining', 0)
        if not isinstance(balance, int) or isinstance(balance, bool) or balance < 0:
            raise ValueError('Invalid credit balance')
        if not balance:
            return payloads, state
        recent = queue._client.list_jobs(limit=100).get('jobs', [])
        if any(j.get('status') in ('queued', 'running', 'compiling', 'evaluating') for j in recent):
            return payloads, state
        chosen = remaining[:balance]
        archive, metadata = _batch_archive(state, chosen, state_file.parent)
        digest = sha256(archive)
        batch = {'phase': 'upload-intent', 'archive_path': str(archive), 'archive_sha256': digest,
                 'benchmark_slug': state['benchmark_slug'], 'job_tag': 'pyasc-batch-' + digest[:24],
                 'job_id': None, 'selected_operators': chosen, 'binding': metadata}
        state['batches'].append(batch)
        state['remaining_operators'] = remaining[len(chosen):]
        submission.save(state_file, state)  # Intent MUST precede POST.
        try:
            posted = queue._submit_streaming(str(archive), chosen, batch['job_tag'])
            batch['job_id'] = posted.get('job_id') or (posted.get('job') or {}).get('id')
            if not batch['job_id']:
                raise RuntimeError('Missing job identity')
        except Exception:
            batch['phase'] = 'ambiguous'
            submission.save(state_file, state)
            batch['job_id'] = submission.reconcile(queue._client, batch)
        batch['phase'] = 'submitted'
        submission.save(state_file, state)
        payloads.append(queue._client.get_job(batch['job_id']))
        return payloads, state
