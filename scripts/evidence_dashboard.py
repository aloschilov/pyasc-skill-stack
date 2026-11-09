"""Read-only projection of retained qualification reports; never execute archive code.

This is a presentation layer, not another numerical validator. The catalog pins
intentionally public archives. Counts come from their reports, never from Git.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'aloschilov/pyasc-skill-stack'
MAX_ARCHIVE = 128 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
MAX_JSON = 8 * 1024 * 1024


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key'); result[key] = value
        return result
    def invalid(_): raise ValueError('Non-finite JSON number')
    require(len(raw) <= MAX_JSON, 'JSON too large')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def timestamp(value):
    require(isinstance(value, str), 'Missing timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    # Old CANNBench UTC timestamps omit the offset.
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def finite(value, positive=False):
    require(type(value) in (float, int) and math.isfinite(value) and (not positive or value > 0), 'Invalid metric')
    return value


class Archive:
    """Bounded tar reader: no filesystem extraction, imports, symlinks or execution."""
    def __init__(self, raw, expected=None):
        require(len(raw) <= MAX_ARCHIVE and (expected is None or digest(raw) == expected), 'Archive checksum/size mismatch')
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            unpacked = stream.read(MAX_EXPANDED+1)
        require(len(unpacked) <= MAX_EXPANDED, 'Expanded archive too large')
        self.tar = tarfile.open(fileobj=io.BytesIO(unpacked), mode='r:')
        self.members = {}; total = 0
        for member in self.tar:
            name = member.name.rstrip('/')
            path = PurePosixPath(name)
            require(name and not path.is_absolute() and '..' not in path.parts and str(path) == name
                and name not in self.members and (member.isfile() or member.isdir()), 'Unsafe/duplicate archive member')
            require(not member.sparse and member.size >= 0, 'Sparse or invalid member')
            total += member.size
            require(total <= MAX_EXPANDED and len(self.members) < 20000, 'Expanded archive too large')
            self.members[name] = member

    def raw(self, name, limit=MAX_JSON):
        member = self.members[name]
        require(member.isfile() and member.size <= limit, 'Invalid member size/type')
        return self.tar.extractfile(member).read(limit+1)

    def json(self, name): return decode(self.raw(name))

    def one(self, suffix):
        names = [n for n in self.members if n.endswith('/'+suffix) or n == suffix]
        require(len(names) == 1, 'Missing/ambiguous archive record')
        return names[0]

    def index(self, name):
        index = self.json(name); prefix = name.rsplit('/', 1)[0]+'/' if '/' in name else ''
        require(isinstance(index, dict) and index, 'Empty archive index')
        for member, expected in index.items():
            require(isinstance(member, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'Invalid member hash')
            require(digest(self.raw(prefix+member, MAX_EXPANDED)) == expected, 'Indexed member changed')


def catalog(path):
    value = decode(Path(path).read_bytes())
    require(value['schema_version'] == 1 and value['repository'] == REPOSITORY
        and type(value['max_age_hours']) is int and 1 <= value['max_age_hours'] <= 8760, 'Unknown catalog')
    require(isinstance(value['sources'], list) and 0 < len(value['sources']) <= 50, 'Invalid source count')
    ids = set()
    for row in value['sources']:
        require(set(row) == {'id', 'kind', 'release', 'asset', 'sha256', 'run_id'}, 'Unknown source field')
        require(row['kind'] in ('paired', 'models', 'review', 'target', 'hardware'), 'Unknown lane')
        for key in ('id', 'release', 'asset'):
            require(isinstance(row[key], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,150}', row[key]), 'Invalid source identifier')
        require(row['id'] not in ids and re.fullmatch(r'[0-9a-f]{64}', row['sha256']), 'Duplicate source/invalid hash')
        require(row['run_id'] is None if row['kind'] == 'target' else type(row['run_id']) is int and row['run_id'] > 0,
                'Missing CI identity (manual target coverage is separate)')
        ids.add(row['id'])
    return value


def fetch(url):
    # No token, cookies or service credentials. Only catalog-derived public GitHub URLs.
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'pyasc-evidence-dashboard'}), timeout=60) as response:
        raw = response.read(MAX_ARCHIVE+1)
    require(len(raw) <= MAX_ARCHIVE, 'Download too large'); return raw


def checked_plan(plan):
    unsigned = dict(plan); sha = unsigned.pop('plan_sha256')
    raw = json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(digest(raw) == sha, 'Plan hash mismatch')
    return plan


def profile_settings(config):
    provider, model = config['model'].split('/')
    selected = config['provider'][provider]['models'][model]
    options, limits = selected.get('options', {}), selected.get('limit', {})
    require(set(options) <= {'thinking', 'reasoningEffort', 'reasoning_effort'}, 'Unknown model settings schema')
    for key in ('reasoningEffort', 'reasoning_effort'):
        if key in options: require(options[key] in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'), 'Invalid reasoning setting')
    if 'thinking' in options:
        require(set(options['thinking']) == {'type'} and options['thinking']['type'] in ('enabled', 'disabled', 'adaptive'), 'Invalid thinking setting')
    require(set(limits) <= {'context', 'output'} and all(type(v) is int and v > 0 for v in limits.values()), 'Invalid model limits')
    return {'options': options, 'limits': limits, 'meaning': 'Archived client configuration; provider compliance is not asserted.'}


def public_models(plan):
    models = plan.get('models') or {'writer': plan['common']['model']}
    return [{'arm': name, 'requested': m['requested'], 'expected_response': m['expected_response'],
        'profile': m['profile'], 'provider_revision': None,
        'profile_sha256': m.get('profile_sha256')}
        for name, m in models.items()]


def generated(archive, entry):
    outer = archive.one('files.json'); archive.index(outer)
    prefix = outer.rsplit('/', 1)[0]+'/'
    ci = archive.json(prefix+'ci-run.json'); verification = archive.json(prefix+'verification.json')
    require(verification['run'] == entry['run_id'] and verification['head'] == ci['headSha'], 'CI identity mismatch')
    bundle = Archive(archive.raw(prefix+'generation.tar.gz', MAX_ARCHIVE), verification['archive_sha256'])
    plan, report = checked_plan(bundle.json('run/plan.json')), bundle.json('run/report.json')
    require(plan['checkout_revision'] == ci['headSha'] and report['plan_sha256'] == plan['plan_sha256'], 'Report/source mismatch')
    require(plan['hardware_invoked'] is False and ci['status'] == 'completed', 'Unfinished or wrong lane')
    common = plan['common']; result = {
        'category': 'fresh-generated', 'level': 'simulator-numerical', 'stack_revision': ci['headSha'],
        'runtime': common['runtime'], 'models': public_models(plan),
        'skill_hashes': {k:v for k,v in common['source_sha256'].items() if k.startswith('skills/')},
        'tasks': [{'id': c['id'], 'module': c['module'], 'cases': c['cases']} for c in common['cells']],
        'finished_at': bundle.json('run/state.json').get('finished_at'), 'comparisons': {},
        'service_score': None, 'baseline_time_ratio': None, 'provider_cost': None}
    for model in result['models']:
        path = 'integrations/opencode/profiles/'+model['profile']+'.json'
        expected = common['source_sha256'].get(path)
        candidates = [n for n in bundle.members if n.endswith('/source/'+path)]
        if expected and candidates:
            raw = bundle.raw(candidates[0]); require(digest(raw) == expected, 'Profile source hash mismatch')
            config = decode(raw); require(config['model'] == model['requested'], 'Profile/model mismatch')
            model.update(profile_sha256=expected, configured_settings=profile_settings(config))
    if entry['kind'] in ('paired', 'models'):
        require(plan['kind'] == 'paired-generation-plan' and plan['schema_version'] in (1, 2), 'Unknown paired schema')
        observations = report['observations']; trials = plan['trials']
        require(set(observations) == {t['id'] for t in trials} and len(trials) == report['trial_denominator'], 'Missing/duplicate trials')
        counts = dict(Counter(o.get('outcome', o['status']) for o in observations.values()))
        require(counts == verification['outcomes'], 'Contradictory trial counts')
        complete = report['status'] == 'complete'
        require(not complete or all(o['status'] == 'valid' for o in observations.values()), 'Invalid complete comparison')
        for name, comparison in report['comparisons'].items():
            valid = complete and comparison['valid_pairs'] == comparison['denominator'] and not comparison['coverage_gaps']
            delta = comparison['qualification_delta']
            require(valid or delta is None, 'Partial comparison has an effect estimate')
            if valid:
                rows = comparison['paired_rows']
                require(len(rows) == comparison['denominator'] and rows, 'Invalid pair denominator')
                expected = sum(finite(r['qualification_delta']) for r in rows)/len(rows)
                require(abs(expected-finite(delta)) < 1e-12 and -1 <= delta <= 1, 'Contradictory effect estimate')
                uncertainty = comparison['uncertainty']; interval = uncertainty['interval']
                radius = math.sqrt(2*math.log(40)/len(rows))
                expected_interval = [max(-1, delta-radius), min(1, delta+radius)]
                require(uncertainty['method'] == 'Hoeffding 95%, independent paired trials, D in [-1,1]'
                    and len(interval) == 2 and all(math.isclose(finite(a), b, abs_tol=1e-12)
                        for a,b in zip(interval, expected_interval)), 'Invalid uncertainty')
                result['comparisons'][name] = {'pairs': len(rows), 'qualification_delta': delta, 'interval': interval}
        result.update(status=report['status'], counts=counts, required=len(trials), repetitions=plan['spec']['repetitions'],
            qualified=counts.get('qualified', 0), observed_models=sorted({o['model']['observed'] for o in observations.values()
                if isinstance(o.get('model'), dict) and o['model'].get('observed')}),
            elapsed_seconds=verification['elapsed_seconds'], checks=verification['accepted_numerical_checks'],
            launches=verification['accepted_execution_launches'], controls=verification['rejecting_controls'])
    else:
        require(plan['kind'] == 'review-generation-plan' and plan['schema_version'] == 1, 'Unknown review schema')
        require(report['status'] == 'accepted' and report['validated_acceptance'] is True
            and report['candidate_sha256'] == verification['report']['candidate_sha256'], 'Review is not accepted')
        result.update(status='complete', counts={'qualified': 1}, required=1, qualified=1, repetitions=1,
            candidate_sha256=report['candidate_sha256'], observed_models=[report['observed_model']],
            elapsed_seconds=report['elapsed_seconds'], checks=verification['output_input_checks'],
            launches=verification['execution_launches'], controls=int(verification['suppressed_launch_control_exit'] == 1),
            independent_model=report['independent_model'], repair_attempts=report['repair_attempts'])
    require(result['qualified'] == 0 or result['checks'] > 0 and result['launches'] > 0 and result['controls'] > 0,
            'Missing numerical/control evidence')
    # Retain only public attribution fields; native sessions and provider payloads are omitted.
    usage = [bundle.json(n) for n in bundle.members if n.endswith('/usage.json') and '/private/' not in n]
    requests = [r for u in usage for r in u.get('requests', [])]
    result['observed_models'] = sorted({r['usage']['observed_model'] for r in requests if r.get('usage')})
    # Reuse the trusted public allowlist; do not copy arbitrary artifact fields.
    import sys
    sys.path.insert(0, str(ROOT/'tests/tools'))
    from generation_models import inspect_request_settings
    settings = set()
    for request in requests:
        if request.get('request_settings') is not None:
            value = request['request_settings']
            # Validate the original receipt before deduplicating for display;
            # synthesizing a request loses its independently recorded cap.
            inspect_request_settings(request, value['max_tokens'], True)
            settings.add(json.dumps(value, sort_keys=True))
    result['wire_settings'] = [json.loads(v) for v in sorted(settings)]
    result['unknown_requests'] = sum('usage' not in r for r in requests)
    if result['status'] == 'complete': require(result['unknown_requests'] == 0, 'Complete cohort has unknown usage')
    return result


def target(archive, entry):
    archive.index(archive.one('qualification-target-subset-files.json'))
    name = archive.one('coverage.json'); prefix = name.rsplit('/', 1)[0]+'/'
    coverage = archive.json(name); plan = checked_plan(archive.json(prefix+'plan.json'))
    verification = archive.json(archive.one('verification-'+plan['checkout_revision'][:7]+'.json'))
    require(coverage['schema_version'] == 1 and coverage['kind'] == 'target-numerical-coverage'
        and coverage['plan_sha256'] == plan['plan_sha256'], 'Target report/plan mismatch')
    outcomes = coverage['outcomes']; counts = dict(Counter(row['status'] for row in outcomes.values()))
    require(counts == coverage['counts'] and set(outcomes) == set(plan['case_ids'])
        and len(outcomes) == coverage['required_cases'], 'Target denominator mismatch')
    qualified = counts.get('qualified', 0)
    require(coverage['full_suite_qualified'] == (qualified == len(outcomes))
        and qualified == verification['selected_qualified_cases'], 'Contradictory target acceptance')
    return {'category': 'upstream-fixture', 'level': 'simulator-numerical', 'status': 'complete' if coverage['full_suite_qualified'] else 'partial',
        'required': len(outcomes), 'qualified': qualified, 'counts': counts, 'repetitions': 1,
        'stack_revision': plan['checkout_revision'], 'runtime': {'backend': 'Model', 'image': coverage['runtime_image'],
            'source_revision': coverage['upstream_revision']}, 'models': [], 'observed_models': [], 'skill_hashes': {},
        'tasks': [{'id': n, 'status': outcomes[n]['status']} for n in plan['selected']],
        'finished_at': verification['checked_at'], 'comparisons': {}, 'elapsed_seconds': verification['total_phase_seconds'],
        'checks': verification['official_assertions'], 'launches': verification['actual_model_launches'],
        'controls': verification['rejecting_controls'], 'service_score': None, 'baseline_time_ratio': None,
        'excluded_reason': plan['excluded_reason'], 'manual_run': True}


def hardware(archive, entry):
    acceptance = archive.json('evidence/acceptance.json'); origin = archive.json('evidence/run.json')
    report = archive.json('evidence/package/report.json'); generation = archive.json('evidence/generation.json')
    require(acceptance['passed'] is True and report['schema_version'] == 1 and report['source'] == 'cannbench-hardware'
        and report['gate_passed'] is True and not report['errors'], 'Hardware acceptance missing')
    require(int(origin['origin_run_id']) == entry['run_id'] and origin['repository'] == REPOSITORY
        and acceptance['origin'] == origin and acceptance['generation'] == generation, 'Hardware lineage mismatch')
    cases = [c for op in report['operators'] for c in op['cases']]
    require(len({c['case_id'] for c in cases}) == len(cases) == report['required_cases'] == report['correct_cases'] == report['measured_cases'],
            'Hardware case coverage mismatch')
    ratios = []
    for case in cases:
        require(case['status'] == 'success', 'Unsuccessful hardware case')
        ratios.append(finite(case['baseline_perf_us'], True)/finite(case['elapsed_us'], True))
    ratio = math.exp(sum(math.log(v) for v in ratios)/len(ratios))
    require(math.isclose(ratio, finite(report['true_geometric_mean_speedup'], True), rel_tol=1e-12), 'Timing ratio differs from cases')
    require(acceptance['hardware']['operators'] == report['operators'], 'Hardware summaries disagree')
    candidate = generation['candidate_sha256']
    require(digest(archive.raw('evidence/'+generation['original_candidate'])) == candidate, 'Candidate hash mismatch')
    require(digest(archive.raw('evidence/model-config.json')) == generation['model_configuration_sha256'], 'Model configuration hash mismatch')
    return {'category': 'fresh-generated', 'level': 'npu-numerical-and-timing', 'status': 'complete',
        'stack_revision': origin['skill_stack_commit'], 'runtime': {k:origin['pyasc'][k]
            for k in ('pyasc_commit', 'runtime_bundle_sha256', 'compile_image_id')},
        'models': [{'requested': generation['model'], 'provider_revision': generation['provider_model_revision'],
            'profile': origin['pyasc']['profile'], 'profile_sha256': generation['model_configuration_sha256'],
            'configured_settings': profile_settings(archive.json('evidence/model-config.json'))}],
        'observed_models': [], 'skill_hashes': generation['skill_hashes'], 'candidate_sha256': candidate,
        'tasks': [{'id': c['case_id'], 'elapsed_us': c['elapsed_us'], 'baseline_us': c['baseline_perf_us']} for c in cases],
        'required': len(cases), 'qualified': len(cases), 'counts': {'qualified': len(cases)}, 'repetitions': 1,
        'benchmark': report['benchmark_slug']+' '+report['benchmark_version'], 'hardware': report['hardware'],
        'finished_at': report['finished_at'], 'comparisons': {}, 'service_score': finite(acceptance['service_score']),
        'baseline_time_ratio': ratio, 'elapsed_seconds': (timestamp(report['finished_at'])-timestamp(report['claimed_at'])).total_seconds(),
        'queue_seconds': (timestamp(report['claimed_at'])-timestamp(report['queued_at'])).total_seconds(),
        'note': 'One retained candidate; no cross-model or overall skills-effect claim. Provider revision remains unknown.'}


def summarize(archive, entry):
    if entry['kind'] in ('paired', 'models', 'review'): return generated(archive, entry)
    if entry['kind'] == 'target': return target(archive, entry)
    if entry['kind'] == 'hardware': return hardware(archive, entry)
    raise ValueError('Unknown lane')


MAX_KERNEL_SOURCE = 128 * 1024


def kernel_sources(archive, entry):
    """Project only named, hash-bound candidate Python files, never archive code execution.

    Missing source is independent of numerical acceptance: it cannot create a new
    pass or remove a failed trial from the original cohort denominator.
    """
    if entry['kind'] == 'target':
        return {'status': 'not-applicable', 'files': [], 'missing': []}
    files, missing = [], []

    def source(bundle, member, sha, identity, outcome, twin=None):
        require(isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha), 'Missing source hash')
        raw = bundle.raw(member, MAX_KERNEL_SOURCE)
        require(digest(raw) == sha and (twin is None or digest(bundle.raw(twin, MAX_KERNEL_SOURCE)) == sha), 'Candidate source mismatch')
        text = raw.decode('utf-8')
        require(text and all(ord(c) >= 32 or c in '\n\r\t' for c in text), 'Invalid source text')
        require(sum(len(f['text'].encode('utf-8')) for f in files)+len(raw) <= 4*MAX_JSON, 'Source projection too large')
        files.append({'id': identity, 'outcome': outcome, 'member': member, 'sha256': sha, 'text': text})

    def identifier(value):
        require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,150}', value), 'Invalid candidate identifier')
        return value

    try:
        if entry['kind'] == 'hardware':
            generation = archive.json('evidence/generation.json')
            member = generation['original_candidate']
            require(re.fullmatch(r'generation/attempt-[1-9][0-9]*/original-candidate\.py', member), 'Unknown hardware candidate path')
            source(archive, 'evidence/'+member, generation['candidate_sha256'], 'original-candidate', 'qualified')
        else:
            bundle = Archive(archive.raw(archive.one('generation.tar.gz'), MAX_ARCHIVE))
            plan = checked_plan(bundle.json('run/plan.json'))
            report = bundle.json('run/report.json')
            if entry['kind'] == 'review':
                state = bundle.json('run/state.json')
                require(state['accepted_candidate_sha256'] == report['candidate_sha256'], 'Review source binding mismatch')
                cycles = [c for c in state['cycles'] if c['status'] == 'accepted' and c['candidate_sha256'] == report['candidate_sha256']]
                require(len(cycles) == 1, 'Ambiguous accepted candidate')
                member = cycles[0]['candidate']
                require(re.fullmatch(r'cycle-[1-9][0-9]*/writer/candidate\.py', member), 'Unknown review candidate path')
                source(bundle, 'run/'+member, report['candidate_sha256'], 'accepted-candidate', 'qualified',
                       'run/'+member.replace('/writer/', '/verification/'))
            else:
                require(len(plan['trials']) <= 200, 'Too many source trials')
                for trial in plan['trials']:
                    identity, cell = identifier(trial['id']), identifier(trial['cell'])
                    prefix = 'run/'+identity+'/'
                    state = bundle.json(prefix+'state.json')
                    child_plan = checked_plan(bundle.json(prefix+'plan.json'))
                    require(state['plan_sha256'] == child_plan['plan_sha256'] and child_plan['checkout_revision'] == plan['checkout_revision'], 'Trial source binding mismatch')
                    cells = [c for c in state['cells'] if c['id'] == cell]
                    require(len(cells) == 1 and len(cells[0]['attempts']) <= 10, 'Invalid source attempts')
                    attempts = cells[0]['attempts']
                    if not attempts:
                        missing.append({'id': identity, 'reason': 'No retained candidate attempt'})
                    for attempt in attempts:
                        number = attempt['number']
                        require(type(number) is int and 1 <= number <= 10, 'Invalid attempt number')
                        label = identity+'/attempt-'+str(number)
                        if not attempt.get('candidate_sha256'):
                            record = attempt.get('source_availability', {})
                            if record.get('status') == 'incomplete':
                                require(attempt['status'] in ('failed', 'blocked') and
                                        record.get('reason') == 'generation-incomplete', 'Invalid partial source')
                                base = prefix+cell+'/attempt-'+str(number)+'/'
                                source(bundle, base+'partial-candidate.py', record['sha256'], label,
                                       'incomplete', base+'project/candidate.py')
                            else:
                                reasons = {'no-file': 'No candidate file was retained',
                                           'invalid-file': 'Output was not a regular file within the size limit',
                                           'invalid-text': 'Output was not a readable source file'}
                                missing.append({'id': label, 'reason': reasons.get(record.get('reason'),
                                    'No hash-bound candidate retained for this attempt')})
                            continue
                        require(attempt['status'] in ('passed', 'failed'), 'Unfinished candidate source')
                        member = prefix+cell+'/attempt-'+str(number)+'/candidate.py'
                        source(bundle, member, attempt['candidate_sha256'], label,
                               'qualified' if attempt['status'] == 'passed' else 'failed',
                               member.replace('/candidate.py', '/project/candidate.py'))
        require(files or missing, 'No source entries')
        require(len({f['id'] for f in files}) == len(files), 'Duplicate source entry')
        return {'status': 'partial' if missing else 'available', 'files': files, 'missing': missing}
    except (OSError, ValueError, LookupError, TypeError, AttributeError, ArithmeticError,
            EOFError, RecursionError, zlib.error, tarfile.TarError):
        # Never include parser exception strings or partially trusted source bytes.
        return {'status': 'unavailable', 'files': [], 'missing': [],
                'reason': 'Candidate source could not be retrieved or validated'}


def validate_projection(value):
    require(value['category'] in ('fresh-generated', 'upstream-fixture')
        and value['level'] in ('simulator-numerical', 'npu-numerical-and-timing')
        and value['status'] in ('complete', 'incomplete', 'partial'), 'Unknown public result schema')
    for key in ('required', 'qualified', 'repetitions'):
        require(type(value[key]) is int and value[key] >= (0 if key == 'qualified' else 1), 'Invalid result denominator')
    counts = value['counts']
    allowed = {'qualified', 'candidate-failed', 'budget-exhausted', 'not-run', 'interrupted', 'timeout',
        'unresolved-infrastructure', 'simulator-unavailable', 'invalid-instrumentation', 'running',
        'blocked', 'failed', 'unsupported', 'timed-out', 'excluded', 'unresolved'}
    require(isinstance(counts, dict) and set(counts) <= allowed
        and all(type(n) is int and n >= 0 for n in counts.values())
        and sum(counts.values()) == value['required'] and counts.get('qualified', 0) == value['qualified'], 'Contradictory public counts')
    require(finite(value['elapsed_seconds']) >= 0 and re.fullmatch(r'[0-9a-f]{40}', value['stack_revision']), 'Invalid time or revision')


def build(catalog_path, output, loader=fetch, now=None):
    spec = catalog(catalog_path); now = now or datetime.now(timezone.utc)
    output = Path(output).resolve()
    require(output != ROOT and ROOT not in output.parents and not output.exists(), 'Use a new external output directory')
    records = []
    for entry in spec['sources']:
        release = 'https://github.com/'+REPOSITORY+'/releases/tag/'+entry['release']
        url = 'https://github.com/'+REPOSITORY+'/releases/download/'+entry['release']+'/'+entry['asset']
        record = {'id': entry['id'], 'kind': entry['kind'], 'release_url': release, 'archive_url': url,
            'archive_sha256': entry['sha256'], 'ci_url': 'https://github.com/'+REPOSITORY+'/actions/runs/'+str(entry['run_id']) if entry['run_id'] else None}
        try:
            archive = Archive(loader(url), entry['sha256'])
            projection = summarize(archive, entry)
            validate_projection(projection)
            age = (now-timestamp(projection['finished_at'])).total_seconds()/3600
            require(age >= -1, 'Future result timestamp')
            record.update(projection, availability='available', freshness='stale' if age > spec['max_age_hours'] else 'recent',
                          kernel_sources=kernel_sources(archive, entry))
        except (OSError, ValueError, LookupError, TypeError, AttributeError, ArithmeticError,
                EOFError, RecursionError, zlib.error, tarfile.TarError) as exc:
            # External payloads/error strings are never copied into public pages.
            record.update(availability='unavailable', freshness='unknown', error='Evidence could not be retrieved or validated',
                          error_type=type(exc).__name__)
        records.append(record)
    payload = {'schema_version': 1, 'built_at': now.isoformat(), 'max_age_hours': spec['max_age_hours'],
        'catalog_sha256': digest(Path(catalog_path).read_bytes()), 'records': records}
    output.mkdir(parents=True)
    for path in (ROOT/'dashboard').glob('*.html'): shutil.copy2(path, output/path.name)
    for path in (ROOT/'dashboard').glob('*.js'): shutil.copy2(path, output/path.name)
    for path in (ROOT/'dashboard').glob('*.css'): shutil.copy2(path, output/path.name)
    workflow_dir = output / 'workflows'
    workflow_dir.mkdir()
    assets = ['viewer.css', 'viewer.js']
    for version in ('asc2-july-detailed', 'asctile-november-detailed'):
        assets.extend(version + extension for extension in ('.html', '.bpmn', '.svg'))
    for asset in assets:
        shutil.copy2(ROOT / 'dashboard' / 'workflows' / asset, workflow_dir / asset)
    (output/'data.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT/'dashboard/sources.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = build(args.catalog, args.output)
    print(json.dumps({'built_at': result['built_at'], 'records': len(result['records']),
        'unavailable': sum(r['availability'] != 'available' for r in result['records'])}))


if __name__ == '__main__': main()
