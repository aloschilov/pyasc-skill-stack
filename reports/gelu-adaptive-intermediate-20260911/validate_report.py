"""Validate published report using only its bundled data; no device or API."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'evidence.json').read_text())
text = (ROOT / 'README.md').read_text()
assert data['stop']['heartbeat_deleted'] and not data['stop']['active_remote_jobs']
assert len(data['runs']) == 11 and data['attempts_consumed'] == 8
assert len([r for r in data['runs'] if r['job_id']]) == 7
assert len(data['navigation']) == 20
assert {r['case_id'] for r in data['navigation']} == set(range(1, 21))
for label, source in data['sources'].items():
    path = ROOT / source['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], label
    ast.parse(path.read_text())
case_rows = 0
for run in data['runs']:
    if run.get('job_id'):
        assert run['url'] in text
        assert run['source_sha256'] == hashlib.sha256((ROOT / run['kernel']).read_bytes()).hexdigest()
    summary = run.get('summary')
    if run.get('status') == 'compile_failed': assert summary is None
    if not summary: continue
    rows = summary['rows']
    assert {r['case_id'] for r in rows} == set(range(1, 21)) and len(rows) == 20
    assert summary['passed'] == sum(r['accuracy'] for r in rows)
    case_rows += len(rows)
    for row in rows:
        if row['speedup'] is not None:
            assert math.isclose(row['speedup'], row['reference_us'] / row['elapsed_us'], rel_tol=1e-12)
    if summary['gm_all20'] is not None:
        assert all(r['accuracy'] and r['status'] == 'success' for r in rows)
        assert math.isclose(summary['gm_all20'], math.exp(sum(math.log(r['speedup']) for r in rows) / 20), rel_tol=1e-12)
    else:
        assert not all(r['accuracy'] for r in rows)
assert case_rows == 100
states = [row['status'] for route in data['local_ranking']['routes'].values() for row in route['all_configurations']]
assert len(states) == 25 and states.count('qualified_local_ranking') == 15
assert states.count('rejected') == 6 and states.count('running_or_pending') == 4
assert len(data['local_phase_records']) == 81
for link in re.findall(r'\]\(([^)]+)\)', text):
    if not link.startswith(('https://', '#')):
        assert (ROOT / link.split('#')[0]).is_file(), link
forbidden = {'api_key', 'authorization', 'access_token', 'refresh_token', 'aggregation_token', 'user_id'}
def inspect(value):
    if isinstance(value, dict):
        assert not (set(map(str.lower, value)) & forbidden)
        for child in value.values(): inspect(child)
    elif isinstance(value, list):
        for child in value: inspect(child)
inspect(data)
print('PASS: stop state;7 jobs;100 measured-case rows;20 shapes;25 local configurations;81 phases;GM;source hashes;links;no credential/account fields')
