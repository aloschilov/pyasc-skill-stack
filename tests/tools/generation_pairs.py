"""Predeclared paired experiments using the shared executor and numerical proof."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from generation_compare import Bundle, require
from generation_executor import Blocked, configure, execute, exclusive, external, save
from generation_plan import ROOT, build_plan, canonical, digest, exact_keys, resource, write_plan
from generation_protocols import BASELINE, CONTRASTS, PROTOCOLS, audit_trace, context_files, prompt

SPEC = 'tests/generation/paired.json'


def read(path):
    return json.loads(Path(path).read_text())


def ordered_trials(cells, spec):
    # Hash ordering is reproducible across Python versions; the seed affects
    # order only, never claims control of provider sampling.
    def rank(label): return digest((str(spec['order_seed'])+'/'+label).encode())
    labels = sorted(PROTOCOLS, key=lambda p: rank('protocol/'+p))
    cells = sorted(cells, key=lambda c: rank('cell/'+c['id']))
    blocks = []
    for repetition in range(spec['repetitions']):
        for i, cell in enumerate(cells):
            rotation = (i+repetition//2) % len(labels)
            order = labels[rotation:]+labels[:rotation]
            if repetition % 2: order = list(reversed(order))
            blocks.append((cell['id'], repetition+1, order))
    blocks.sort(key=lambda b: rank(f'block/{b[0]}/{b[1]}'))
    return [{'id': f'{cell}-r{rep:02d}-{pid.lower()}', 'cell': cell, 'repetition': rep, 'protocol': pid}
            for cell, rep, order in blocks for pid in order]


def build_pairs(root=ROOT):
    spec = json.loads(resource(root, SPEC))
    exact_keys(spec, ('schema_version', 'matrix', 'protocols', 'repetitions', 'order_seed',
                     'max_output_tokens', 'max_execution_seconds'), 'paired specification')
    require(type(spec['schema_version']) is int and spec['schema_version'] == 1, 'Unknown paired schema')
    require(spec['protocols'] == list(PROTOCOLS), 'Declare all four protocols in canonical order')
    reps = spec['repetitions']
    require(type(reps) is int and 2 <= reps <= 8 and reps % 2 == 0, 'Use an even number of repetitions, 2..8')
    require(type(spec['order_seed']) is int and 0 <= spec['order_seed'] < 2**32, 'Explicit order seed required')
    common = build_plan(root, spec['matrix'])
    for name in (SPEC, BASELINE, '.github/workflows/generation-paired.yml'):
        common['source_sha256'][name] = digest(resource(root, name))
    trials = ordered_trials(common['cells'], spec)
    limits = common['limits']; attempts = len(trials)*limits['attempts_per_cell']
    maximum = {'model_attempts': attempts,
        'output_tokens': attempts*limits['rounds_per_attempt']*limits['output_tokens_per_response'],
        'execution_seconds': attempts*(limits['generation_seconds_per_attempt']+limits['qualification_seconds_per_attempt'])}
    for field in ('output_tokens', 'execution_seconds'):
        cap = spec['max_'+field]
        require(type(cap) is int and cap > 0 and maximum[field] <= cap, 'Paired aggregate '+field+' budget exceeded')
    require(spec['max_execution_seconds'] <= 14400, 'Paired execution must fit the 300-minute CI job including setup/collection')
    return {'schema_version': 1, 'kind': 'paired-generation-plan', 'spec': spec, 'common': common,
        'protocols': PROTOCOLS, 'contrasts': CONTRASTS, 'trials': trials, 'maximum_work': maximum,
        'order_design': 'seeded-mirrored-rotations-v1', 'generation_seed': None,
        'generation_seed_note': 'Only execution order is seeded; the provider has no qualified reproducible sampling seed.',
        'schedule_enabled': False, 'hardware_invoked': False}


def validate_pairs(path):
    signed = read(path); unsigned = dict(signed); signature = unsigned.pop('plan_sha256')
    require(digest(canonical(unsigned)) == signature, 'Paired plan digest mismatch')
    unsigned.pop('checkout_revision', None)
    require(canonical(unsigned) == canonical(build_pairs()), 'Paired plan differs from current inputs')
    return signed


def trial_plan(plan, trial):
    common = deepcopy(plan['common']); selected = common['skills']
    common['cells'] = [c for c in common['cells'] if c['id'] == trial['cell']]
    require(len(common['cells']) == 1, 'Unknown paired task')
    common['skills'] = selected if PROTOCOLS[trial['protocol']]['skills'] else []
    common['intervention'] = {'id': trial['protocol'], 'definition': PROTOCOLS[trial['protocol']],
        'selected_skills': selected, 'experiment_sha256': plan['plan_sha256'], 'trial': trial}
    common['checkout_revision'] = plan.get('checkout_revision')
    limits = common['limits']; attempts = limits['attempts_per_cell']
    common['maximum_work'] = {'model_attempts': attempts,
        'output_tokens': attempts*limits['rounds_per_attempt']*limits['output_tokens_per_response'],
        'execution_seconds': attempts*(limits['generation_seconds_per_attempt']+limits['qualification_seconds_per_attempt'])}
    common['plan_sha256'] = digest(canonical(common))
    return common


def seal(root):
    save(root/'files.json', {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in root.rglob('*')
        if p.is_file() and not p.is_symlink() and 'private' not in p.relative_to(root).parts and p != root/'files.json'})


def run_pairs(plan, output, state_dir):
    output = external(output)
    require(not output.exists(), 'Use a new paired results directory; never replay partial experiments')
    with exclusive(state_dir, output) as lease:
        output.mkdir(parents=True); save(output/'plan.json', plan)
        state = {'status': 'running', 'plan_sha256': plan['plan_sha256'],
            'started_at': datetime.now(timezone.utc).isoformat(), 'reserved_output_tokens': 0,
            'observed_output_tokens': 0, 'reserved_execution_seconds': 0,
            'trials': [dict(t, status='not-run') for t in plan['trials']]}
        save(output/'state.json', state); start = time.monotonic()
        limits = plan['common']['limits']; n = limits['attempts_per_cell']
        tokens = n*limits['rounds_per_attempt']*limits['output_tokens_per_response']
        seconds = n*(limits['generation_seconds_per_attempt']+limits['qualification_seconds_per_attempt'])
        sessions = set(); schema = None
        try:
            for trial, item in zip(plan['trials'], state['trials']):
                require(state['reserved_output_tokens']+tokens <= plan['spec']['max_output_tokens'], 'Paired token budget exhausted')
                require(state['reserved_execution_seconds']+seconds <= plan['spec']['max_execution_seconds']
                    and time.monotonic()-start+seconds <= plan['spec']['max_execution_seconds']+1, 'Paired time budget exhausted')
                item['status'] = 'running'; state['reserved_output_tokens'] += tokens
                state['reserved_execution_seconds'] += seconds; save(output/'state.json', state)
                result = execute(trial_plan(plan, trial), output/trial['id'], state_dir, _lease=lease)
                item['status'] = result['status']; state['observed_output_tokens'] += result['observed_output_tokens']
                save(output/'state.json', state)
                if result['status'] == 'blocked': raise Blocked('Paired trial unresolved; remaining trials not run')
                try:
                    observed = inspect_trial(output/trial['id'], trial_plan(plan, trial), result['status'])
                    require(not sessions.intersection(observed['sessions']), 'Reused paired session')
                    require(schema is None or schema == observed['tool_schema_sha256'], 'Tool schemas differ between protocols')
                    sessions.update(observed['sessions']); schema = observed['tool_schema_sha256']
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    item.update(status='invalid-instrumentation', reason=str(exc)); save(output/'state.json', state)
                    raise Blocked('Invalid paired instrumentation; remaining trials not run') from exc
            state['status'] = 'completed'  # A completed experiment may contain genuine candidate failures.
        except BaseException as exc:
            state.update(status='blocked', error=type(exc).__name__+': '+str(exc))
        finally:
            state['finished_at'] = datetime.now(timezone.utc).isoformat(); save(output/'state.json', state); seal(output)
        report = report_pairs(output)
        save(output/'report.json', report); seal(output)
        if state['status'] == 'completed' and report['status'] == 'complete': lease.unlink()
        return report


def inspect_trial(root, expected, status):
    """A failed kernel is a valid outcome only with complete generation provenance."""
    bundle = Bundle(root)
    require(bundle.plan == expected, 'Trial plan/provenance mismatch')
    require(bundle.state['checkout_revision'] == expected['checkout_revision'], 'Trial revision mismatch')
    require(bundle.state['status'] == status and status in ('passed', 'failed'), 'Unresolved/inconsistent trial status')
    cell = bundle.state['cells'][0]; rows = []; schemas = set(); sessions = set()
    for attempt in cell['attempts']:
        prefix = cell['id']+f'/attempt-{attempt["number"]}/'
        isolation = bundle.json(prefix+'isolation.json')
        require(isolation['fresh_project'] and isolation['fresh_private']
            and isolation['context_sha256'] == context_files(expected), 'Unisolated or unexpected context')
        config = bundle.json(prefix+'model-config.json'); provider = expected['model']['requested'].split('/')[0]
        expected_config = configure(bundle.json('context/integrations/opencode/profiles/'+expected['model']['profile']+'.json'),
            expected, Path(isolation['context']), Path(isolation['project']))
        options = config['provider'][provider]['options']
        require(options['apiKey'] == '{env:PYASC_GATE_TOKEN}' and options['baseURL'].startswith('http://127.0.0.1:'), 'Non-isolated provider endpoint')
        expected_config['provider'][provider]['options'] = options
        require(config == expected_config and isolation['permission'] == config['permission'], 'Model/configuration/permissions drift')
        from generation_adapters import task_contract
        expected_prompt = prompt(task_contract(ROOT, expected['cells'][0]), expected['intervention']['id'], expected['skills'], Path(isolation['context']))
        require(bundle.read(prefix+'prompt.txt').decode() == expected_prompt, 'Prompt intervention drift')
        events = bundle.read(prefix+'generation.stdout').decode()
        # Rebind retained absolute access paths to the archived context below.
        trace = audit_trace(events, Path(isolation['project']), Path(isolation['context']), expected,
                            source_context=root/'context')
        attribution = bundle.json(prefix+'attribution.json')
        for key, value in trace.items(): require(attribution.get(key) == value, 'Native protocol attribution mismatch')
        require(trace['session_id'] not in sessions, 'Reused attempt session'); sessions.add(trace['session_id'])
        bundle.command(prefix, 'generation')
        usage = bundle.json(prefix+'usage.json')
        require(usage == attempt['usage'] and usage['usage_complete'] and usage['error'] is None, 'Unknown trial usage')
        require(0 < len(usage['requests']) <= expected['limits']['rounds_per_attempt'], 'Missing/excess provider requests')
        for req in usage['requests']:
            require(req['status'] == 'complete' and req['usage']['observed_model'] == expected['model']['expected_response'], 'Request model/outcome mismatch')
            require(req['audit']['protocol'] == expected['intervention']['id'], 'Request protocol mismatch')
            if req['audit']['tool_schema_sha256']: schemas.add(req['audit']['tool_schema_sha256'])
            for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                require(type(req['usage'][field]) is int and req['usage'][field] >= 0, 'Invalid measured token usage')
            require(req['usage']['prompt_tokens']+req['usage']['completion_tokens'] == req['usage']['total_tokens']
                and req['usage']['completion_tokens'] <= expected['limits']['output_tokens_per_response'], 'Measured token limit/accounting mismatch')
        require(usage['observed_output_tokens'] == sum(r['usage']['completion_tokens'] for r in usage['requests']), 'Output accounting mismatch')
        require(any(req['audit']['prompt_present'] for req in usage['requests']), 'Prompt never reached the provider')
        require({s for req in usage['requests'] for s in req['audit']['visible_skills']} == set(expected['skills']), 'Actual skill context mismatch')
        require(expected['intervention']['id'] != 'P4' or any(req['audit']['baseline_visible'] for req in usage['requests']), 'Baseline context never reached the provider')
        if attempt['status'] == 'failed':
            require(attempt['failure_stage'] == 'qualifying', 'Generation failure is not a qualified control outcome')
            sha = attempt['candidate_sha256']
            require(digest(bundle.read(prefix+'candidate.py')) == digest(bundle.read(prefix+'project/candidate.py')) == sha, 'Failed candidate changed')
            if prefix+'qualification.json' in bundle.files:
                q = bundle.json(prefix+'qualification.json')
                require(q['status'] == 'failed' and q['inputs_unchanged'] and q['backend'] == 'Model', 'Unclassified evaluator failure')
                require(bundle.json(prefix+'container-cleanup.json')['confirmed'], 'Failed container cleanup unknown')
                require(q['phases'].get('jit', {}).get('exit_code') == 1 or
                    q['phases'].get('execution', {}).get('exit_code') == 1, 'Control/verifier failure is not candidate failure')
            else:
                from generation_adapters import static_check, task_contract
                rejected = False
                try: static_check(bundle.read(prefix+'candidate.py').decode(), task_contract(ROOT, expected['cells'][0]))
                except (ValueError, SyntaxError): rejected = True
                require(rejected, 'Missing failed qualification evidence')
        if prefix+'qualification.json' in bundle.files:
            from generation_adapters import wrapper
            contract = task_contract(ROOT, expected['cells'][0])
            require(bundle.json(prefix+'adapter.json') == contract, 'Numerical task/oracle contract drift')
            for name, check in [('probe.py', True), ('compile_probe.py', False)]:
                require(bundle.read(prefix+name).decode() == wrapper(contract, check=check), 'Numerical evaluator changed')
        for value in (attempt['duration_seconds'], usage['provider_cost']):
            require(value is None or type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid duration/cost')
        require(attempt['duration_seconds'] is not None, 'Missing duration')
        rows.append({'attempts': 1, 'seconds': attempt['duration_seconds'],
            'output_tokens': usage['observed_output_tokens'],
            'input_tokens': sum(req['usage']['prompt_tokens'] for req in usage['requests']),
            'cost': usage['provider_cost']})
    require(len(schemas) == 1 and rows, 'Tool schema absent or changed within a trial')
    return {'status': 'valid', 'qualified': cell['status'] == 'passed', 'sessions': sorted(sessions),
        'outcome': 'qualified' if cell['status'] == 'passed' else 'candidate-failed',
        'tool_schema_sha256': schemas.pop(), 'attempts': len(rows), 'repair_attempts': 0,
        'seconds': sum(r['seconds'] for r in rows), 'output_tokens': sum(r['output_tokens'] for r in rows),
        'input_tokens': sum(r['input_tokens'] for r in rows),
        'cost': sum(r['cost'] for r in rows) if all(r['cost'] is not None for r in rows) else None}


def summarize(plan, observations):
    """Paired denominator is fixed in advance. Invalid controls never imply gain."""
    ids = [t['id'] for t in plan['trials']]
    require(set(observations) == set(ids), 'Missing/extra trial records')
    pairs = {}
    for t in plan['trials']: pairs.setdefault((t['cell'], t['repetition']), {})[t['protocol']] = observations[t['id']]
    comparisons = {}
    for treatment, control in CONTRASTS:
        rows = []; gaps = []
        for (cell, rep), legs in pairs.items():
            a, b = legs[treatment], legs[control]
            if a['status'] != 'valid' or b['status'] != 'valid':
                gaps.append({'cell': cell, 'repetition': rep, 'treatment': a['status'], 'control': b['status']}); continue
            rows.append({'cell': cell, 'repetition': rep, 'qualification_delta': int(a['qualified'])-int(b['qualified']),
                **{field+'_delta': a[field]-b[field] if a[field] is not None and b[field] is not None else None
                   for field in ('attempts', 'seconds', 'output_tokens', 'input_tokens', 'cost')}})
        n = len(pairs); complete = not gaps
        delta = sum(r['qualification_delta'] for r in rows)/n if complete else None
        # Hoeffding for independent paired D in [-1,1], including non-identical
        # fixed tasks. Deliberately conservative at this small sample size.
        radius = math.sqrt(2*math.log(40)/n)
        comparisons[treatment+'-'+control] = {'denominator': n, 'valid_pairs': len(rows), 'coverage_gaps': gaps,
            'paired_rows': rows, 'qualification_delta': delta,
            'uncertainty': {'method': 'Hoeffding 95%, independent paired trials, D in [-1,1]',
                'interval': [max(-1, delta-radius), min(1, delta+radius)] if complete else None,
                'limitations': 'Conditional on independent trials for these fixed tasks/model. Provider correlation and other tasks/models are not covered.'},
            'mean_deltas': {field: sum(r[field+'_delta'] for r in rows)/n if complete and all(r[field+'_delta'] is not None for r in rows) else None
                for field in ('attempts', 'seconds', 'output_tokens', 'input_tokens', 'cost')}}
    return {'status': 'complete' if all(o['status'] == 'valid' for o in observations.values()) else 'incomplete',
        'trial_denominator': len(ids), 'observations': observations, 'comparisons': comparisons,
        'skills_effect_contrast': 'P6-P3', 'general_skills_benefit_claim': False, 'performance_claim': False,
        'note': 'Generation/qualification duration is not kernel performance. P6-P4 replaces context packages. Unknown cost remains null.'}


def report_pairs(root):
    root = Path(root).resolve(); plan = read(root/'plan.json'); unsigned = dict(plan); sha = unsigned.pop('plan_sha256')
    require(digest(canonical(unsigned)) == sha and plan['kind'] == 'paired-generation-plan', 'Invalid paired plan')
    require(plan['schema_version'] == 1 and plan['protocols'] == PROTOCOLS
        and plan['contrasts'] == [list(c) for c in CONTRASTS], 'Unrecognized paired definitions')
    require(plan['trials'] == ordered_trials(plan['common']['cells'], plan['spec']), 'Predeclared order changed')
    index = read(root/'files.json')
    require({'plan.json', 'state.json'} <= set(index), 'Required paired records are not indexed')
    public = [p for p in root.rglob('*') if 'private' not in p.relative_to(root).parts and p != root/'files.json']
    require(not any(p.is_symlink() for p in public), 'Symlink in paired evidence')
    require({p.relative_to(root).as_posix() for p in public if p.is_file()} == set(index), 'Incomplete paired index')
    for name, expected in index.items():
        path = root/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and 'private' not in Path(name).parts
            and path.resolve().is_relative_to(root.resolve()) and not any(p.is_symlink() for p in [path, *path.parents]), 'Unsafe paired index')
        require(digest(path.read_bytes()) == expected, 'Paired evidence changed: '+name)
    state = read(root/'state.json'); require(state['plan_sha256'] == sha, 'Paired state/plan mismatch')
    require([{k: t[k] for k in ('id', 'cell', 'repetition', 'protocol')} for t in state['trials']] == plan['trials'], 'Trial inventory/order changed')
    observations = {}; sessions = set(); schema = None
    for trial, item in zip(plan['trials'], state['trials']):
        if item['status'] not in ('passed', 'failed'):
            value = {'status': item['status'], 'outcome': item['status']}
            if item.get('reason'): value['reason'] = item['reason']
            record = root/trial['id']/'state.json'
            if record.exists() and item['status'] != 'invalid-instrumentation':
                error = read(record).get('error', '')
                value.update(reason=error, outcome='timeout' if error.startswith('TimeoutError:') else
                    'interrupted' if error.startswith('KeyboardInterrupt:') else
                    'simulator-unavailable' if error == 'Blocked: Simulator unavailable' else 'unresolved-infrastructure')
            observations[trial['id']] = value; continue
        try:
            value = inspect_trial(root/trial['id'], trial_plan(plan, trial), item['status'])
            require(not sessions.intersection(value['sessions']), 'Reused paired session')
            require(schema is None or schema == value['tool_schema_sha256'], 'Tool schemas differ between protocols')
            schema = value['tool_schema_sha256']; sessions.update(value['sessions'])
            observations[trial['id']] = value
        except (OSError, ValueError, KeyError, TypeError) as exc:
            observations[trial['id']] = {'status': 'invalid-instrumentation', 'reason': str(exc)}
    result = summarize(plan, observations)
    if state['status'] != 'completed':
        result.update(status='incomplete', alerts=['experiment-unresolved'])
        for comparison in result['comparisons'].values():
            comparison['qualification_delta'] = None
            comparison['uncertainty']['interval'] = None
            comparison['mean_deltas'] = {k: None for k in comparison['mean_deltas']}
    return dict(result, plan_sha256=sha, experiment_status=state['status'],
        checked_at=datetime.now(timezone.utc).isoformat(), reporter_sha256=digest(Path(__file__).read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'execute', 'report'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--approve-plan', default='')
    parser.add_argument('--state-dir', type=Path, default=Path.home()/'.local/state/pyasc-generation')
    args = parser.parse_args(); root = external(args.root)
    if args.action == 'plan':
        plan = build_pairs(); plan['checkout_revision'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        result = write_plan(plan, root)
        print(json.dumps({'status': 'not-run', 'plan_sha256': result['plan_sha256'], 'maximum_work': result['maximum_work']})); return 0
    if args.action == 'report': result = report_pairs(root)
    else:
        plan = validate_pairs(args.plan); require(args.approve_plan == plan['plan_sha256'], 'Exact paired plan approval required')
        result = run_pairs(plan, root, args.state_dir)
    print(json.dumps(result)); return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.exit(main())
