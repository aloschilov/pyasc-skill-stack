"""One private reference-DAG diagnostic; durable intent and GET-only recovery."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
COMPARISONS = Path('/home/aloschilov/workspace/pyasc-skill-stack/integrations/cannbench/comparisons')
sys.path.insert(0, str(COMPARISONS/'gelu-adaptive-20260910/queue'))
from controller import ReadOnlyClient

TAG = 'gelu-reference-v2-9069108-20261007'
TERMINAL = {'succeeded', 'failed', 'runtime_failed', 'compile_failed',
            'correctness_failed', 'cancelled', 'canceled'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value, exclusive=False):
    data = json.dumps(value, indent=2) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stream.name, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def gates(artifact=ROOT):
    info = json.loads((artifact/'package.json').read_text())
    assert digest(artifact/'diagnostic.zip') == info['archive_sha256']
    assert digest(ROOT/'kernel.py') == info['kernel_sha256']
    for name, expected in info['members'].items():
        assert digest(artifact/'bundle'/name) == expected, name
    qualified = json.loads((artifact/'x86-qualification/x86-results.json').read_text())
    assert qualified['passed'] and len(qualified['dispatches']) == 80
    assert len(qualified['specializations']) == 6
    routes = []
    for dtype in ('float16', 'bfloat16', 'float32'):
        for mode in ('none', 'tanh'):
            label = 'verified' if dtype == 'float32' else 'qualification'
            path = ROOT/'evidence'/f'{label}-{dtype}-{mode}'/'result.json'
            result = json.loads(path.read_text())
            assert result['completed'] and result['compiled'] and result['ub_fit']
            assert result['source_sha256'] == info['kernel_sha256']
            assert all(c['repeat_equal'] and c['guards_intact'] for c in result['checks'])
            if (dtype, mode) == ('float32', 'none'):
                # Explicitly diagnostic: do not disguise the known stress failure.
                assert not result['accuracy_passed']
                assert all(s['passed'] for c in result['checks'] for s in c['segment_checks']
                           if s['input'] != 'negative_tail')
            else:
                assert result['accuracy_passed']
            routes.append(dict(dtype=dtype, mode=mode, evidence=str(path),
                               evidence_sha256=digest(path), accuracy=result['accuracy_passed']))
    for dtype in ('float16', 'bfloat16'):
        result = json.loads((ROOT/'evidence'/f'multitile-{dtype}-tanh'/'result.json').read_text())
        assert result['completed'] and result['accuracy_passed']
        assert result['source_sha256'] == info['kernel_sha256']
    multicore = json.loads((ROOT/'evidence/multicore-float16-none/result.json').read_text())
    assert multicore['completed'] and multicore['accuracy_passed']
    assert multicore['source_sha256'] == info['kernel_sha256']
    return info, routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-only', action='store_true')
    parser.add_argument('--repair', action='store_true', help='Distinct manifest-only build fix; never retries the first archive')
    args = parser.parse_args()
    artifact = ROOT/'repair' if args.repair else ROOT
    tag = TAG+'-buildfix' if args.repair else TAG
    out = artifact/'remote'
    out.mkdir(mode=0o700, exist_ok=True)
    os.chmod(out, 0o700)  # Raw API payloads may contain private scheme metadata.
    path = out/'state.json'
    with (COMPARISONS/'gelu-geometry-strategy-20260908/artifacts/remote.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = ReadOnlyClient().client
        client.timeout, client.retries = 60, 1
        assert client.base_url.rstrip('/') == 'https://cannbench.com'
        if args.poll_only:
            state = json.loads(path.read_text())
            if not state.get('job_id'):
                matches = [j for j in client.list_jobs(limit=100)['jobs'] if j.get('job_tag') == tag]
                assert len(matches) == 1, 'No uniquely accepted job: NEVER automatically repeat POST'
                state.update(job_id=matches[0]['id'], state='accepted')
                save(path, state)
        else:
            assert not path.exists(), 'Intent already consumed; --poll-only is the only recovery'
            info, routes = gates(artifact)
            if args.repair:
                first = json.loads((ROOT/'remote/job.json').read_text())['job']
                assert first['status'] == 'compile_failed' and first['job_completed']
                assert info['original_archive_sha256'] == digest(ROOT/'diagnostic.zip')
                assert info['changed_members'] == ['source.sha256.txt']
                tests = json.loads((artifact/'entrypoint-tests.json').read_text())
                assert len(tests) == 7 and all((t['returncode'] == 0) == t['expected_success'] for t in tests)
            jobs = client.list_jobs(limit=100)['jobs']
            save(out/'jobs-before.json', jobs)
            assert all(j.get('job_completed') or j.get('status') in TERMINAL for j in jobs), 'Active jobs'
            assert not any(j.get('job_tag') == tag for j in jobs)
            credits = client.get_credits()
            save(out/'credits-before.json', credits)
            assert credits['credits']['remaining'] > 0
            scheme = json.loads((COMPARISONS/'gelu-case15-topdown-20260911/hardware-20260915/recovery/scheme.json').read_text())['scheme']
            assert scheme['organization'] == 'huawei'
            pools = client._get_json('/api/submissions/runner-pools', {'aggregation_token': scheme['code']})
            save(out/'pools-before.json', pools)
            pool = next(p for p in pools['pools'] if p['id'] == 'system-shared')
            assert pool.get('online_runner_counts', {}).get('950pr', 0) > 0
            # Current public pool snapshot and MCP 0.8 submit contract:
            # system-shared is platform-managed, not an independent environment.
            # Do not reuse the historical independent-environment consent field.
            assert pool.get('is_system') is True
            assert pool.get('is_independent_environment') is False
            state = dict(tag=tag, state='intent_consumed', created=time.time(),
                archive_sha256=info['archive_sha256'], kernel_sha256=info['kernel_sha256'],
                offline_qualification_sha256=digest(artifact/'x86-qualification/x86-results.json'),
                routes=routes, diagnostic=True, attempt_limit=1, retry_count=0,
                rationale='Requested reference-algorithm hardware evaluation. Known dense-tail FP32 exact stress failure; sampled official ranges passed. Not a fully correctness-qualified release.',
                selected_operators=['gelu'], is_private=True)
            save(path, state, exclusive=True)
            with tempfile.NamedTemporaryFile(mode='w', prefix='cannbench-headers-') as headers:
                os.chmod(headers.name, 0o600)
                headers.write('Authorization: Bearer '+client.token+'\nX-BenchSite-Client: pyasc-skill-stack/1\nX-BenchSite-MCP-Tool: submit_kernel\n')
                headers.flush()
                command = ['curl', '--disable', '--http1.1', '--progress-bar', '--show-error', '--fail-with-body',
                    '--connect-timeout', '30', '--max-time', '10800', '--retry', '0', '--header', '@'+headers.name,
                    '--output', str(out/'response.json'), '--write-out', '%{json}']
                fields = dict(source='mcp', target_hardware='950pr', selected_operators='["gelu"]',
                    benchmark_slug='official-tasks', job_tag=tag, is_private='true', aggregation_token=scheme['code'],
                    requested_pool_id='system-shared')
                for name, value in fields.items():
                    command += ['--form-string', name+'='+value]
                command += ['--form', 'file=@'+str(artifact/'diagnostic.zip')+';type=application/octet-stream',
                            'https://cannbench.com/api/submissions']
                with (out/'progress.log').open('x') as progress, (out/'curl-stats.json').open('x') as stats:
                    process = subprocess.Popen(command, stdout=stats, stderr=progress)
                    state.update(state='uploading', upload_pid=process.pid)
                    save(path, state)
                    print('One diagnostic upload started; durable intent saved.', flush=True)
                    code = process.wait()
            state.update(curl_exit=code, upload_finished=time.time(), state='needs_reconciliation')
            save(path, state)
            if code:
                raise RuntimeError('Upload outcome uncertain; GET reconciliation only')
            response = json.loads((out/'response.json').read_text())
            job = response.get('job', {})
            assert job.get('id', '').startswith('job_'), 'Ambiguous response; GET reconciliation only'
            state.update(state='accepted', job_id=job['id'])
            save(path, state)
        payload = client.get_job(state['job_id'])
        job = payload.get('job', payload)
        assert job['job_tag'] == tag and job['selected_operators'] == ['gelu'] and job['is_private']
        save(out/'job.json', payload)
        state.update(state=job['status'], observed=time.time())
        save(path, state)
        print(json.dumps(dict(job_id=job['id'], status=job['status'],
                             url='https://cannbench.com/workspace/jobs/'+job['id'])), flush=True)


if __name__ == '__main__':
    main()
