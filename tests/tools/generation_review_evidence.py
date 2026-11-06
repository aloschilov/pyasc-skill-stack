"""Recheck retained review acceptance using trusted code, without model/runtime calls."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from generation_adapters import static_check, task_contract, wrapper
from generation_compare import require, utc
from generation_plan import canonical, digest
from generation_protocols import provider_prompt
from generation_review import (audit_role, binding, build_review, role_config, role_plan,
    role_prompt, validate_review, verifier_feedback)
from numerical_checks import proves_numerics


class Evidence:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        require(not (self.root/'files.json').is_symlink(), 'Symlinked evidence index')
        self.index = json.loads((self.root/'files.json').read_text())
        paths = [p for p in self.root.rglob('*') if 'private' not in p.relative_to(self.root).parts and p != self.root/'files.json']
        require(not any(p.is_symlink() for p in paths), 'Symlink in review evidence')
        require({p.relative_to(self.root).as_posix() for p in paths if p.is_file()} == set(self.index), 'Incomplete review index')
        for name in self.index: self.bytes(name)

    def bytes(self, name):
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and 'private' not in p.parts, 'Unsafe review evidence path')
        require(name in self.index, 'Missing indexed review evidence: '+name)
        data = (self.root/p).read_bytes()
        require(digest(data) == self.index[name], 'Review evidence hash mismatch: '+name)
        return data

    def json(self, name):
        return json.loads(self.bytes(name))

    def command(self, prefix, name, exit_code=0):
        value = self.json(prefix+name+'.command.json')
        require(value['status'] == 'completed' and value['exit_code'] == exit_code, 'Unsuccessful/unresolved '+name)
        seconds = value['duration_seconds']
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, 'Invalid command duration')
        for stream in ('stdout', 'stderr'):
            require(digest(self.bytes(prefix+name+'.'+stream)) == value[stream+'_sha256'], 'Changed command log')
        return value


def inspect_role(evidence, prefix, plan, role, contract, inputs):
    selected = role_plan(plan, role); limits = selected['limits']
    isolation = evidence.json(prefix+'isolation.json')
    require(isolation['role'] == role and isolation['fresh_session'] is True, 'Unexpected role context')
    project, context = Path(isolation['project']), Path(isolation['context'])
    require(project.is_absolute() and context.is_absolute(), 'Missing original isolated paths')
    expected = {n: sha for n, sha in selected['source_sha256'].items() if any(n.startswith('skills/'+s+'/') for s in selected['skills'])}
    expected.update({'inputs/'+n: digest(data) for n, data in dict(inputs, **{'task.json': canonical(contract)}).items()})
    require(isolation['context_sha256'] == expected, 'Role context differs from handoff/skills')
    for name, sha in expected.items(): require(digest(evidence.bytes(prefix+'context/'+name)) == sha, 'Handoff changed')
    message = role_prompt(selected, role, contract, context, inputs)
    require(evidence.bytes(prefix+'prompt.txt').decode() == message, 'Role prompt drift')
    config = evidence.json(prefix+'model-config.json'); provider = selected['model']['requested'].split('/')[0]
    expected_config = role_config(selected, role, evidence.root/'source', context, project, expected)
    options = config['provider'][provider]['options']
    require(options['apiKey'] == '{env:PYASC_GATE_TOKEN}' and options['baseURL'].startswith('http://127.0.0.1:'), 'Role bypassed shared provider gate')
    expected_config['provider'][provider]['options'] = options
    require(config == expected_config and isolation['permission'] == config['permission'], 'Role configuration/permission drift')
    usage = evidence.json(prefix+'usage.json')
    require(usage['usage_complete'] is True and usage['error'] is None, 'Incomplete role usage')
    requests = usage['requests']; require(0 < len(requests) <= limits['rounds_per_attempt'], 'Role request budget/provenance missing')
    for request in requests:
        row = request['usage']; audit = request['audit']
        require(request['status'] == 'complete' and row['observed_model'] == selected['model']['expected_response'], 'Different/unknown role response model')
        require(audit['role'] == role and audit['provided_prompt_sha256'] == digest(message.encode())
            and audit['provider_prompt_sha256'] == digest(provider_prompt(message).encode()), 'Role request prompt drift')
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            require(type(row[key]) is int and row[key] >= 0, 'Invalid role token usage')
        require(row['total_tokens'] == row['prompt_tokens']+row['completion_tokens']
            and row['completion_tokens'] <= limits['output_tokens_per_response'], 'Role accounting/budget mismatch')
    require(usage['observed_output_tokens'] == sum(r['usage']['completion_tokens'] for r in requests), 'Role output accounting mismatch')
    trace = audit_role(evidence.root/prefix, selected, role, project, context, inputs, expected,
        message, usage, source_context=evidence.root/prefix/'context')
    require(evidence.json(prefix+'attribution.json') == trace, 'Role native attribution changed')
    cmd = evidence.command(prefix, 'generation')
    require(cmd['argv'] == ['opencode', 'run', '--pure', '--format', 'json', '--dir', str(project),
        '-m', selected['model']['requested'], message], 'Role executed another client/command')
    for name in (['review.json'] if role == 'reviewer' else ['candidate.py', 'design.md']):
        require(evidence.bytes(prefix+name) == evidence.bytes(prefix+'project/'+name), 'Role output changed after capture')
    return trace, {'output_tokens': usage['observed_output_tokens'],
        'input_tokens': sum(r['usage']['prompt_tokens'] for r in requests), 'seconds': cmd['duration_seconds']}


def inspect_verifier(evidence, prefix, contract, candidate, accepted):
    require(evidence.bytes(prefix+'candidate.py') == candidate, 'Verifier received another candidate')
    if prefix+'qualification.json' not in evidence.index:
        rejected = False
        try: static_check(candidate.decode(), contract)
        except (ValueError, SyntaxError): rejected = True
        require(not accepted and rejected, 'Missing numerical proof')
        return
    q = evidence.json(prefix+'qualification.json')
    require(q['backend'] == 'Model' and q['inputs_unchanged'] is True, 'Untrusted verifier backend/inputs')
    require(q['input_sha256']['candidate.py'] == digest(candidate), 'Verifier candidate binding mismatch')
    for name, sha in q['input_sha256'].items(): require(digest(evidence.bytes(prefix+name)) == sha, 'Verifier input changed')
    require(evidence.json(prefix+'adapter.json') == contract, 'Numerical task/oracle changed')
    for name, check in [('probe.py', True), ('compile_probe.py', False)]:
        require(evidence.bytes(prefix+name).decode() == wrapper(contract, check=check), 'Numerical evaluator changed')
    require(evidence.json(prefix+'container-cleanup.json')['confirmed'] is True, 'Unconfirmed verifier cleanup')
    phases = q['phases']
    if accepted:
        require(q['status'] == 'passed' and q['verification_level'] == 'numerically_verified'
            and q['static']['status'] == 'passed', 'Missing numerical acceptance')
        for name in ('jit', 'execution'):
            require(phases[name]['exit_code'] == 0 and phases[name]['report']['launcher_calls'] == len(contract['seeds']), 'Missing real verification launches')
        real, control = phases['execution']['report'], phases['control']
        require(real['mock_launch'] is False and len(real['numerical_checks']) == len(contract['seeds'])*len(contract['arguments'])
            and proves_numerics(real, control['report'], control['exit_code']), 'Missing numerical/negative control proof')
    else:
        require(q['status'] == 'failed' and (phases.get('jit', {}).get('exit_code') == 1
            or phases.get('execution', {}).get('exit_code') == 1), 'Verifier/control failure is not candidate rejection')
    evidence.command(prefix, 'qualification', 0 if accepted else 1)


def report_review(directory):
    e = Evidence(directory); plan = e.json('plan.json'); unsigned = dict(plan); signature = unsigned.pop('plan_sha256')
    require(digest(canonical(unsigned)) == signature, 'Invalid review plan')
    unsigned.pop('checkout_revision', None)
    for name, sha in plan['common']['source_sha256'].items(): require(digest(e.bytes('source/'+name)) == sha, 'Review source hash mismatch')
    from generation_models import inspect_binding
    binding_version = inspect_binding(plan['common']['model'], plan['common']['source_sha256'],
        lambda name: e.bytes('source/'+name))
    expected = build_review(e.root/'source', tuning=plan['tuning'])
    if binding_version is None:
        # Read old receipts with their original scope, without a new credential
        # binding claim. Current execution validation has no such downgrade.
        for field in ('profile_sha256', 'configuration'): expected['common']['model'].pop(field)
    require(canonical(unsigned) == canonical(expected), 'Review specification/source drift')
    state = e.json('state.json')
    require(state['plan_sha256'] == signature and state['status'] in ('accepted', 'rejected', 'blocked'), 'Nonterminal review state')
    result = {'status': state['status'], 'validated_acceptance': False, 'plan_sha256': signature,
        'model_binding_schema_version': binding_version,
        'hardware_invoked': False, 'performance_claim': False, 'independent_model': False,
        'reporter_sha256': digest(Path(__file__).read_bytes()), 'checked_at': datetime.now(timezone.utc).isoformat()}
    if state['status'] != 'accepted':
        return dict(result, note='Retained failure/interruption; no accepted candidate or tuning promotion.')
    require(state['checkout_revision'] == plan['checkout_revision'], 'Review checkout revision mismatch')
    require(utc(state['started_at']) <= utc(state['finished_at']) <= datetime.now(timezone.utc), 'Invalid review timestamps')
    require(state['independent_session'] is True and state['independent_model'] is False and state['hardware_invoked'] is False, 'Overstated review scope')
    e.command('', 'opencode-version'); require(e.bytes('opencode-version.stdout').strip(), 'Missing client version')
    image = plan['common']['runtime']['image']; image_command = e.command('', 'image-inspect')
    require(image_command['argv'] == ['docker', 'image', 'inspect', image]
        and image in e.json('image-inspect.stdout')[0]['RepoDigests'], 'Runtime digest differs from plan')
    rows = state['cycles']; require(0 < len(rows) <= 1+plan['spec']['max_repairs'], 'Invalid repair count')
    contract = task_contract(e.root/'source', plan['common']['cells'][0])
    tuning_inputs = None
    if plan['tuning'] is not None:
        from generation_tuning import validate_handoff, handoff_inputs
        validate_handoff(plan['tuning'], plan, source=e.root/'source')
        tuning_inputs = handoff_inputs(plan['tuning'])
    previous = None; feedback = {}; sessions = set(); metrics = []
    for number, row in enumerate(rows, 1):
        prefix = f'cycle-{number}/'; require(row['number'] == number, 'Nonsequential review cycles')
        require(row['status'] in ('rejected', 'verification-failed', 'accepted') and (row['status'] == 'accepted') == (number == len(rows)), 'Invalid acceptance transition')
        require(row['parent_candidate_sha256'] == (previous['candidate_sha256'] if previous else None), 'Repair lineage changed')
        inputs = (tuning_inputs or {}) if previous is None else {'prior-candidate.py': e.bytes(previous['candidate']),
            'prior-design.md': e.bytes(previous['design']), 'feedback.json': canonical(feedback)}
        writing_role = 'tuner' if previous is None and tuning_inputs is not None else 'writer'
        writer, metric = inspect_role(e, prefix+'writer/', plan, writing_role, contract, inputs); metrics.append(metric)
        require(row['writer'] == writer, 'Writer attribution mismatch')
        candidate, design = e.bytes(prefix+'writer/candidate.py'), e.bytes(prefix+'writer/design.md')
        bound = binding(candidate, design, contract)
        if plan['tuning'] is not None: require(bound['candidate_sha256'] != plan['tuning']['binding']['candidate_sha256'], 'Unchanged tuning proposal')
        require(row['candidate'] == prefix+'writer/candidate.py' and row['design'] == prefix+'writer/design.md'
            and row['candidate_sha256'] == bound['candidate_sha256'] and row['binding'] == bound, 'Candidate/design binding changed')
        require(not previous or bound['candidate_sha256'] != previous['candidate_sha256'], 'Unchanged repair')
        inputs = {'candidate.py': candidate, 'design.md': design, 'binding.json': canonical(bound)}
        if feedback.get('review'): inputs['prior-review.json'] = canonical(feedback['review'])
        reviewer, metric = inspect_role(e, prefix+'reviewer/', plan, 'reviewer', contract, inputs); metrics.append(metric)
        require(row['reviewer'] == reviewer, 'Reviewer attribution mismatch')
        for phase in (writer, reviewer):
            require(phase['session_id'] not in sessions, 'Reused role session'); sessions.add(phase['session_id'])
        verdict = validate_review(e.json(prefix+'reviewer/review.json'), bound, feedback.get('review'))
        require(row['review'] == verdict, 'Verdict changed')
        feedback = {'review': verdict}
        if row['status'] == 'rejected':
            require(verdict['verdict'] == 'reject' and not any(n.startswith(prefix+'verification/') for n in e.index), 'Rejected review reached verifier')
        else:
            require(verdict['verdict'] == 'accept', 'Review rejection cannot accept a kernel')
            inspect_verifier(e, prefix+'verification/', contract, candidate, row['status'] == 'accepted')
            if row['status'] == 'verification-failed':
                feedback['verification'] = verifier_feedback(e.root/prefix/'verification', row['verification_error'])
        previous = row
    require(state['accepted_candidate_sha256'] == previous['candidate_sha256'], 'Accepted candidate changed')
    output_tokens = sum(m['output_tokens'] for m in metrics)
    expected_tokens = len(rows)*sum(plan['spec'][r]['rounds_per_attempt']*plan['spec'][r]['output_tokens_per_response']
        for r in ('writer', 'reviewer'))
    expected_seconds = len(rows)*(sum(plan['spec'][r]['generation_seconds_per_attempt'] for r in ('writer', 'reviewer'))
        +plan['common']['limits']['qualification_seconds_per_attempt'])
    require(type(state['reserved_output_tokens']) is int and state['reserved_output_tokens'] == expected_tokens
        and type(state['reserved_execution_seconds']) is int and state['reserved_execution_seconds'] == expected_seconds,
        'Reservations differ from completed review cycles')
    require(state['usage_complete'] is True and state['observed_output_tokens'] == output_tokens
        and output_tokens <= state['reserved_output_tokens'] <= plan['spec']['max_output_tokens'], 'Run usage mismatch')
    require(state['reserved_execution_seconds'] <= plan['spec']['max_execution_seconds'], 'Run time budget mismatch')
    return dict(result, validated_acceptance=True, candidate_sha256=previous['candidate_sha256'], repair_attempts=len(rows)-1,
        sessions=sorted(sessions), observed_model=plan['common']['model']['expected_response'],
        output_tokens=output_tokens, input_tokens=sum(m['input_tokens'] for m in metrics),
        generation_seconds=sum(m['seconds'] for m in metrics),
        elapsed_seconds=(utc(state['finished_at'])-utc(state['started_at'])).total_seconds(),
        note='Fresh-session static review plus unchanged-byte simulator numerical and negative-control evidence; no independent-model or device-performance claim.')
