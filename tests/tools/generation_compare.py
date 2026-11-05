"""Offline regression comparison of two retained generation runs; never execute them."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from generation_plan import ROOT, canonical, digest
from numerical_checks import proves_numerics

POLICY = 'generation-no-lost-pass-v1'
SHA256 = re.compile(r'[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value):
    require(isinstance(value, str), 'Missing timestamp')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(stamp.utcoffset() is not None, 'Timestamp needs a timezone')
    return stamp.astimezone(timezone.utc)


class Bundle:
    """Verify public evidence without importing or running snapshotted Python."""
    def __init__(self, directory, expected=None):
        self.root = Path(directory).resolve()
        require(not (self.root/'files.json').is_symlink(), 'Symlink in evidence index')
        raw = (self.root/'files.json').read_bytes()
        self.sha256 = digest(raw)
        require(expected is None or self.sha256 == expected, 'Baseline files.json digest mismatch')
        self.files = json.loads(raw)
        require(isinstance(self.files, dict) and self.files, 'Empty evidence index')
        for name, sha in self.files.items():
            require(isinstance(sha, str) and SHA256.fullmatch(sha), 'Invalid file digest')
            self.read(name)
        self.plan = self.json('plan.json')
        unsigned = dict(self.plan); signature = unsigned.pop('plan_sha256')
        require(digest(canonical(unsigned)) == signature, 'Plan digest mismatch')
        require(self.plan['schema_version'] == 2 and self.plan['kind'] == 'generation-plan', 'Unsupported plan')
        for name, sha in self.plan['source_sha256'].items():
            require(digest(self.read('source/'+name)) == sha, 'Source snapshot mismatch: '+name)
        self.state = self.json('state.json')
        require(self.state['plan_sha256'] == signature, 'State/plan mismatch')
        require(self.state['status'] in ('passed', 'failed', 'blocked'), 'Run is not terminal')
        ids = [c['id'] for c in self.plan['cells']]
        require(ids and len(ids) == len(set(ids)) and all(re.fullmatch(r'[a-z][a-z0-9_-]*', i) for i in ids), 'Invalid cell inventory')
        require([c['id'] for c in self.state['cells']] == ids, 'Missing, extra or reordered cell results')
        self.cells = {c['id']: c['status'] for c in self.state['cells']}
        require(set(self.cells.values()) <= {'passed', 'failed', 'blocked', 'not-run'}, 'Invalid cell outcome')
        if self.state['status'] != 'blocked':
            expected_status = 'passed' if set(self.cells.values()) == {'passed'} else 'failed'
            require(self.state['status'] == expected_status and self.state.get('inputs_unchanged') is True, 'Inconsistent terminal state')
            require(set(self.cells.values()) <= {'passed', 'failed'}, 'Incomplete terminal run')
        self.version = self.read('opencode-version.stdout').decode().strip()
        require(bool(self.version), 'Missing OpenCode version')
        self.command('', 'opencode-version')
        self.sessions = set()
        for cell in self.state['cells']:
            attempts = cell['attempts']
            require(len(attempts) <= self.plan['limits']['attempts_per_cell'], 'Attempt limit exceeded')
            require([a['number'] for a in attempts] == list(range(1, len(attempts)+1)), 'Nonsequential attempts')
            if cell['status'] == 'not-run':
                require(not attempts, 'Not-run cell contains attempts')
            if cell['status'] == 'failed':
                require(attempts and all(a['status'] == 'failed' and a.get('error') for a in attempts), 'Missing failure evidence')
            if cell['status'] == 'passed':
                require(attempts and attempts[-1]['status'] == 'passed' and all(a['status'] == 'failed' for a in attempts[:-1]), 'Invalid winning attempt')
                self.passed(cell['id']+f'/attempt-{attempts[-1]["number"]}/', attempts[-1])
        # Detect index replacement during collection. The input is retained
        # evidence, not a live-run monitoring or adversarial filesystem API.
        require(digest((self.root/'files.json').read_bytes()) == self.sha256, 'Evidence index changed')

    def read(self, name):
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and 'private' not in path.parts, 'Unsafe evidence path')
        target = self.root/path
        require(target.resolve().is_relative_to(self.root) and not any(p.is_symlink() for p in [target, *target.parents] if p != self.root), 'Symlink in evidence path')
        require(name in self.files, 'Required evidence is not indexed: '+name)
        data = target.read_bytes()
        require(digest(data) == self.files[name], 'File digest mismatch: '+name)
        return data

    def json(self, name):
        return json.loads(self.read(name))

    def command(self, prefix, name):
        record = self.json(prefix+name+'.command.json')
        require(record['status'] == 'completed' and record['exit_code'] == 0, 'Unsuccessful '+name)
        for stream in ('stdout', 'stderr'):
            require(digest(self.read(prefix+name+'.'+stream)) == record[stream+'_sha256'], 'Command log mismatch')

    def passed(self, prefix, attempt):
        sha = attempt['candidate_sha256']
        require(digest(self.read(prefix+'candidate.py')) == digest(self.read(prefix+'project/candidate.py')) == sha, 'Candidate changed')
        report = self.json(prefix+'qualification.json')
        require(report['status'] == 'passed' and report['backend'] == 'Model' and report['inputs_unchanged'] is True
                and report['verification_level'] == 'numerically_verified' and report['static']['status'] == 'passed', 'Missing qualification')
        require(report['input_sha256']['candidate.py'] == sha, 'Qualification candidate mismatch')
        for name, expected in report['input_sha256'].items():
            require(digest(self.read(prefix+name)) == expected, 'Qualification input mismatch')
        contract = self.json(prefix+'adapter.json')
        phases = report['phases']; real = phases['execution']['report']; control = phases['control']
        for name in ('jit', 'execution'):
            phase = phases[name]
            require(phase['exit_code'] == 0 and phase['report']['launcher_calls'] == len(contract['seeds']), 'Missing launches')
        require(real['mock_launch'] is False and len(real['numerical_checks']) == len(contract['seeds'])*len(contract['arguments'])
                and proves_numerics(real, control['report'], control['exit_code']), 'Missing numerical/negative-control proof')
        require(self.json(prefix+'container-cleanup.json')['confirmed'] is True, 'Cleanup unconfirmed')
        for name in ('generation', 'qualification'):
            self.command(prefix, name)
        attribution = self.json(prefix+'attribution.json')
        require(attribution['requested_model'] == self.plan['model']['requested']
                and attribution['observed_model'] == self.plan['model']['expected_response']
                and sorted(attribution['loaded_skills']) == sorted(self.plan['skills']), 'Missing model/skill attribution')
        session = attribution['session_id']
        require(isinstance(session, str) and session and session not in self.sessions, 'Reused/missing session')
        self.sessions.add(session)
        usage = self.json(prefix+'usage.json')
        require(usage == attempt['usage'] and usage['usage_complete'] is True and usage['error'] is None, 'Unknown winning usage')
        require(0 < len(usage['requests']) <= self.plan['limits']['rounds_per_attempt'], 'Missing/excess requests')
        for request in usage['requests']:
            require(request['status'] == 'complete' and request['usage']['observed_model'] == self.plan['model']['expected_response'], 'Unknown/different response model')

    def compatibility(self):
        # Skills are the changed treatment, not a reason to discard a comparison.
        # Exclude only this offline reporter; all execution/verifier/profile/task
        # bytes must match. Even harmless executor edits require a new baseline.
        return {key: self.plan[key] for key in ('schema_version', 'model', 'runtime', 'cells', 'skills', 'limits', 'required_gates')} | {
            'opencode_version': self.version,
            'source_sha256': {name: sha for name, sha in self.plan['source_sha256'].items()
                if not name.startswith('skills/') and name != 'tests/tools/generation_compare.py'}}


def compare(baseline, candidate, now=None, max_age_hours=36):
    require(max_age_hours > 0, 'Freshness window must be positive')
    require(baseline.state['status'] == 'passed', 'Baseline must be a complete accepted run')
    require(baseline.sha256 != candidate.sha256 and not baseline.sessions.intersection(candidate.sessions), 'Same run/session is not a fresh comparison')
    before, after = baseline.compatibility(), candidate.compatibility()
    mismatches = [key for key in before if before[key] != after[key]]
    now = now or datetime.now(timezone.utc)
    result = {'schema_version': 1, 'policy': POLICY, 'baseline_sha256': baseline.sha256,
        'checked_at': now.isoformat(), 'reporter_sha256': digest(Path(__file__).read_bytes()),
        'candidate_sha256': candidate.sha256, 'baseline_revision': baseline.state.get('checkout_revision'),
        'candidate_revision': candidate.state.get('checkout_revision'), 'compatible': not mismatches,
        'incompatible_fields': mismatches, 'status': 'incompatible', 'alerts': [],
        'denominator': len(baseline.cells), 'passed': None, 'regressions': [], 'coverage_gaps': [],
        'cells': candidate.cells, 'max_age_hours': max_age_hours,
        'changed_skill_files': sorted(name for name in set(baseline.plan['source_sha256']) | set(candidate.plan['source_sha256'])
            if name.startswith('skills/') and baseline.plan['source_sha256'].get(name) != candidate.plan['source_sha256'].get(name))}
    if mismatches:
        result['alerts'] = ['incompatible-run']; return result
    try:
        start, end = utc(candidate.state['started_at']), utc(candidate.state['finished_at'])
        require(start <= end <= now, 'Invalid run timestamps')
        result['age_hours'] = (now-end).total_seconds()/3600
        if result['age_hours'] > max_age_hours: result['alerts'].append('stale-coverage')
    except (KeyError, TypeError, ValueError):
        result['alerts'].append('unknown-coverage-time')
    result['regressions'] = [name for name, status in candidate.cells.items() if status == 'failed']
    result['coverage_gaps'] = [name for name, status in candidate.cells.items() if status in ('blocked', 'not-run')]
    result['passed'] = sum(status == 'passed' for status in candidate.cells.values())
    if result['regressions']: result['alerts'].append('lost-pass')
    if result['coverage_gaps'] or candidate.state['status'] == 'blocked': result['alerts'].append('incomplete-run')
    result['status'] = 'regression' if result['regressions'] else 'incomplete' if result['alerts'] else 'passed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True, help='Reviewed SHA256 of baseline files.json, not a moving latest pointer')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--max-age-hours', type=int, default=36)
    args = parser.parse_args()
    report = args.report.expanduser().resolve()
    require(report != ROOT and ROOT not in report.parents, 'Report must stay outside checkout')
    require(not any(p.resolve() == report or p.resolve() in report.parents for p in (args.baseline, args.candidate)), 'Report must stay outside input bundles')
    require(not report.exists(), 'Never overwrite a comparison report')
    try:
        baseline = Bundle(args.baseline, args.baseline_sha256)
        candidate = Bundle(args.candidate)
        result = compare(baseline, candidate, max_age_hours=args.max_age_hours)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        result = {'schema_version': 1, 'policy': POLICY, 'status': 'invalid-evidence',
                  'checked_at': datetime.now(timezone.utc).isoformat(), 'reporter_sha256': digest(Path(__file__).read_bytes()),
                  'alerts': ['invalid-evidence'], 'error': str(exc)}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as handle: handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'passed' else 1 if result['status'] == 'regression' else 2


if __name__ == '__main__':
    sys.exit(main())
